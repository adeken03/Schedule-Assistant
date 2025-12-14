"""Lightweight FastAPI wrapper on the existing scheduler database.

This is additive and keeps the current high-cohesion schema untouched.
Endpoints use one-table-at-a-time reads and service-layer assembly to mirror
the assignment’s intent without refactoring the DB.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
import datetime
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

# Ensure legacy absolute imports (e.g., "import database") still resolve.
APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import database  # noqa: E402
from database import (  # noqa: E402
    SessionLocal,
    EmployeeSessionLocal,
    Modifier,
    WeekContext,
    WeekDailyProjection,
    apply_saved_modifier_to_week,
    init_database,
    get_active_policy,
    get_or_create_week,
    get_or_create_week_context,
    get_shifts_for_week,
    get_week_summary,
    list_modifiers_for_week,
    record_audit_log,
    save_week_daily_projection_values,
    set_week_status,
)
from generator.api import generate_schedule_for_week  # noqa: E402
from policy import ensure_default_policy  # noqa: E402
from database import Policy, upsert_policy, Employee  # noqa: E402
from sqlalchemy import delete
from sqlalchemy.orm import Session
from validation import validate_week_schedule  # noqa: E402


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_database()
    ensure_default_policy(SessionLocal)
    yield


app = FastAPI(title="Schedule Assistant API", version="0.1", lifespan=lifespan)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_employee_db():
    db = EmployeeSessionLocal()
    try:
        yield db
    finally:
        db.close()


def _parse_week_start(value: str) -> datetime.date:
    try:
        return datetime.date.fromisoformat(value)
    except ValueError:
        raise HTTPException(status_code=400, detail="weekStart must be YYYY-MM-DD")


def _serialize_shifts(shifts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    payload = []
    for item in shifts:
        copy: Dict[str, Any] = dict(item)
        for key in ("start", "end"):
            if isinstance(copy.get(key), datetime.datetime):
                copy[key] = copy[key].isoformat()
        payload.append(copy)
    return payload


def _audit(db: Session, actor: str, action: str, target: Optional[str], payload: Optional[Dict[str, Any]] = None) -> None:
    record_audit_log(db, user_id=actor, action=action, target_type="API", target_id=None, payload=payload)


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/api/v1/auth/login")
def login(payload: Dict[str, Any], db=Depends(get_db)) -> JSONResponse:
    username = (payload.get("username") or "").strip()
    password = payload.get("password") or ""
    # Minimal stub: matches the seeded default login noted in README.
    if username != "it_assistant" or password != "letmein":
        raise HTTPException(status_code=401, detail="Invalid credentials")
    _audit(db, actor=username, action="LOGIN_SUCCESS", target=username, payload={})
    return JSONResponse(content={"token": "stub-token", "user": username, "role": "IT_Assistant"})


@app.get("/api/v1/weeks/{week_start}/summary")
def week_summary(week_start: str, db=Depends(get_db)) -> JSONResponse:
    start_date = _parse_week_start(week_start)
    summary = get_week_summary(db, start_date)
    return JSONResponse(content=jsonable_encoder(summary))


@app.get("/api/v1/weeks/{week_start}/modifiers")
def week_modifiers(week_start: str, db=Depends(get_db)) -> JSONResponse:
    start_date = _parse_week_start(week_start)
    modifiers = list_modifiers_for_week(db, start_date)
    return JSONResponse(content=jsonable_encoder({"week_start": start_date.isoformat(), "modifiers": modifiers}))


@app.get("/api/v1/weeks/{week_start}/shifts")
def week_shifts(
    week_start: str,
    employee_id: Optional[int] = Query(None),
    role: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    db=Depends(get_db),
    employee_db=Depends(get_employee_db),
) -> JSONResponse:
    start_date = _parse_week_start(week_start)
    shifts = get_shifts_for_week(
        db,
        start_date,
        employee_id=employee_id,
        role=role,
        status=status,
        employee_session=employee_db,
    )
    return JSONResponse(
        content=jsonable_encoder({"week_start": start_date.isoformat(), "shifts": _serialize_shifts(shifts)})
    )


@app.get("/api/v1/schedules/{week_start}/validate")
def validate_schedule_endpoint(week_start: str, db=Depends(get_db)) -> JSONResponse:
    start_date = _parse_week_start(week_start)
    with EmployeeSessionLocal() as employee_db:
        report = validate_week_schedule(db, start_date, employee_session=employee_db)
    return JSONResponse(content=jsonable_encoder(report))


@app.post("/api/v1/weeks/{week_start}/projection")
def upsert_week_projection(
    week_start: str,
    payload: Dict[str, Any],
    db=Depends(get_db),
) -> JSONResponse:
    start_date = _parse_week_start(week_start)
    store_code = (payload.get("store_code") or "default").strip() or "default"
    days = payload.get("days") or []
    modifiers_payload = payload.get("modifiers") or []

    # Ensure context and projections exist.
    week = get_or_create_week(db, start_date)
    context = get_or_create_week_context(db, week.iso_year, week.iso_week, week.label)
    week.context_id = context.id
    db.commit()

    # Upsert day projections.
    values: Dict[int, Dict[str, Any]] = {}
    for entry in days:
        try:
            day_idx = int(entry.get("day_of_week"))
        except Exception:
            continue
        values[day_idx] = {
            "projected_sales_amount": float(entry.get("projected_sales_amount") or 0.0),
            "projected_notes": entry.get("projected_notes") or "",
        }
    if values:
        save_week_daily_projection_values(db, context.id, values)

    # Replace modifiers for this context.
    db.execute(delete(Modifier).where(Modifier.week_id == context.id))
    for mod in modifiers_payload:
        try:
            day_idx = int(mod.get("day_of_week"))
            start_time = datetime.time.fromisoformat(mod.get("start_time"))
            end_time = datetime.time.fromisoformat(mod.get("end_time"))
        except Exception:
            continue
        db.add(
            Modifier(
                week_id=context.id,
                title=mod.get("title") or "Event",
                modifier_type=mod.get("modifier_type") or "increase",
                day_of_week=day_idx,
                start_time=start_time,
                end_time=end_time,
                pct_change=int(mod.get("pct_change") or 0),
                notes=mod.get("notes") or "",
                created_by=mod.get("created_by") or "api",
            )
        )
    db.commit()
    _audit(db, actor=payload.get("actor") or "api", action="WEEK_PROJECTION_SAVE", target=str(context.id), payload={})

    projections = get_week_summary(db, start_date)
    return JSONResponse(content=jsonable_encoder(projections))


@app.post("/api/v1/modifiers/apply-template")
def apply_modifier_template(
    payload: Dict[str, Any],
    db=Depends(get_db),
) -> JSONResponse:
    template_id = payload.get("template_id")
    week_start_raw = payload.get("week_start")
    created_by = (payload.get("created_by") or "api").strip() or "api"
    if template_id is None or week_start_raw is None:
        raise HTTPException(status_code=400, detail="template_id and week_start are required")
    start_date = _parse_week_start(str(week_start_raw))
    week = get_or_create_week(db, start_date)
    if not week.context_id:
        context = get_or_create_week_context(db, week.iso_year, week.iso_week, week.label)
        week.context_id = context.id
        db.commit()
    try:
        modifier = apply_saved_modifier_to_week(db, template_id, week.context_id, created_by=created_by)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return JSONResponse(
        content=jsonable_encoder(
            {
                "modifier_id": modifier.id,
                "week_context_id": week.context_id,
                "title": modifier.title,
                "type": modifier.modifier_type,
                "day_of_week": modifier.day_of_week,
                "start_time": modifier.start_time.isoformat() if modifier.start_time else None,
                "end_time": modifier.end_time.isoformat() if modifier.end_time else None,
                "pct_change": modifier.pct_change,
            }
        )
    )


@app.post("/api/v1/schedules/generate")
def generate_schedule(payload: Dict[str, Any], db=Depends(get_db)) -> JSONResponse:
    week_start_raw = payload.get("weekStart") or payload.get("week_start")
    actor = (payload.get("actor") or "api").strip() or "api"
    if not week_start_raw:
        raise HTTPException(status_code=400, detail="weekStart is required")
    start_date = _parse_week_start(str(week_start_raw))
    try:
        result = generate_schedule_for_week(
            SessionLocal,
            start_date,
            actor,
            employee_session_factory=EmployeeSessionLocal,
        )
    except Exception as exc:  # pragma: no cover - surface generator errors
        raise HTTPException(status_code=500, detail=f"schedule generation failed: {exc}") from exc
    return JSONResponse(content=jsonable_encoder(result))


@app.post("/api/v1/schedules/{week_start}/publish")
def publish_schedule(week_start: str, payload: Dict[str, Any] | None = None, db=Depends(get_db)) -> JSONResponse:
    start_date = _parse_week_start(week_start)
    actor = (payload or {}).get("actor") or "api"
    week = set_week_status(db, start_date, status="exported")
    _audit(db, actor=actor, action="WEEK_PUBLISH", target=str(week.id), payload={})
    return JSONResponse(content=jsonable_encoder({"week_id": week.id, "status": week.status, "label": week.label}))


@app.get("/api/v1/policy/active")
def active_policy(db=Depends(get_db)) -> JSONResponse:
    policy = get_active_policy(db)
    if not policy:
        raise HTTPException(status_code=404, detail="No active policy found")
    payload = {
        "id": policy.id,
        "name": policy.name,
        "params": policy.params_dict(),
        "lastEditedBy": policy.lastEditedBy,
        "lastEditedAt": policy.lastEditedAt.isoformat() if policy.lastEditedAt else None,
    }
    return JSONResponse(content=jsonable_encoder(payload))


@app.put("/api/v1/policy/active")
def set_active_policy(payload: Dict[str, Any], db=Depends(get_db)) -> JSONResponse:
    name = payload.get("name")
    params = payload.get("params") or {}
    actor = (payload.get("actor") or "api").strip() or "api"
    if not name:
        raise HTTPException(status_code=400, detail="name is required")
    policy = upsert_policy(db, name=name, params_dict=params, edited_by=actor)
    _audit(db, actor=actor, action="POLICY_EDIT", target=str(policy.id), payload={"name": policy.name})
    return JSONResponse(
        content=jsonable_encoder(
            {
                "id": policy.id,
                "name": policy.name,
                "params": policy.params_dict(),
                "lastEditedBy": policy.lastEditedBy,
                "lastEditedAt": policy.lastEditedAt.isoformat() if policy.lastEditedAt else None,
            }
        )
    )


@app.post("/api/v1/employees/{employee_id}/roles-wages")
def update_employee_roles_wages(
    employee_id: int,
    payload: Dict[str, Any],
    db=Depends(get_db),
    employee_db=Depends(get_employee_db),
) -> JSONResponse:
    employee: Optional[Employee] = employee_db.get(Employee, employee_id)
    if not employee:
        raise HTTPException(status_code=404, detail="Employee not found")
    roles = payload.get("roles") or []
    desired_hours = payload.get("desired_hours")
    status = payload.get("status")
    if roles:
        employee.role_list = [str(r) for r in roles]
    if desired_hours is not None:
        try:
            employee.desired_hours = int(desired_hours)
        except Exception:
            pass
    if status:
        employee.status = str(status)
    employee_db.commit()
    employee_db.refresh(employee)
    _audit(
        db,
        actor=payload.get("actor") or "api",
        action="EMP_UPDATE",
        target=str(employee.id),
        payload={"roles": roles, "desired_hours": desired_hours, "status": status},
    )
    return JSONResponse(
        content=jsonable_encoder(
            {
                "id": employee.id,
                "name": employee.full_name,
                "roles": employee.role_list,
                "desired_hours": employee.desired_hours,
                "status": employee.status,
            }
        )
    )
