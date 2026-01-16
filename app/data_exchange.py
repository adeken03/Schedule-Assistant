from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from sqlalchemy import delete, select

from database import (
    Employee,
    EmployeeRoleWage,
    EmployeeUnavailability,
    Modifier,
    Policy,
    SavedModifier,
    Shift,
    WeekContext,
    EmployeeSessionLocal,
    get_all_weeks,
    get_or_create_week,
    get_or_create_week_context,
    get_active_policy,
    get_week_daily_projections,
    get_week_modifiers,
    list_saved_modifiers,
    save_week_daily_projection_values,
    set_week_status,
    save_employee_role_wages,
    upsert_policy,
)
from exporter import DATA_DIR as EXPORT_DIR
from roles import defined_roles
from wages import export_wages, import_wages, wage_amounts

EXPORT_DIR.mkdir(parents=True, exist_ok=True)
VALID_ROLES = set(defined_roles())


def _timestamp() -> str:
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def _week_info_from_date(week_start: datetime.date) -> Dict[str, int | str]:
    iso_year, iso_week, _ = week_start.isocalendar()
    return {
        "iso_year": iso_year,
        "iso_week": iso_week,
        "label": f"{iso_year} W{iso_week:02d}",
        "week_start": week_start.isoformat(),
    }


# ---------------------------------------------------------------------------
# Employee import/export


def export_employees(employee_session) -> Path:
    payload: List[Dict] = []
    employees = employee_session.scalars(select(Employee).order_by(Employee.full_name.asc())).all()
    for employee in employees:
        entry = {
            "full_name": employee.full_name,
            "desired_hours": employee.desired_hours,
            "status": employee.status,
            "notes": employee.notes,
            "start_month": employee.start_month,
            "start_year": employee.start_year,
            "roles": employee.role_list,
            "role_wages": [
                {"role": wage.role, "wage": wage.wage, "confirmed": bool(getattr(wage, "confirmed", False))}
                for wage in getattr(employee, "role_wages", []) or []
            ],
            "unavailability": [
                {
                    "day_of_week": row.day_of_week,
                    "start_time": row.start_time.isoformat(timespec="minutes"),
                    "end_time": row.end_time.isoformat(timespec="minutes"),
                }
                for row in employee.unavailability
            ],
        }
        payload.append(entry)
    filename = EXPORT_DIR / f"employees_{_timestamp()}.json"
    filename.write_text(
        json.dumps({"generated_at": datetime.datetime.utcnow().isoformat(), "employees": payload}, indent=2),
        encoding="utf-8",
    )
    return filename


def import_employees(employee_session, file_path: Path, *, replace_existing: bool = True) -> Tuple[int, int]:
    data = json.loads(file_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Employees file must be a JSON object.")
    employees = data.get("employees")
    if not isinstance(employees, list):
        raise ValueError("Employees file must include an 'employees' list.")
    if replace_existing:
        employee_session.execute(delete(EmployeeRoleWage))
        employee_session.execute(delete(EmployeeUnavailability))
        employee_session.execute(delete(Employee))
    created = 0
    updated = 0
    for payload in employees:
        if not isinstance(payload, dict):
            continue
        name = payload.get("full_name")
        if not name:
            continue
        raw_roles = payload.get("roles", [])
        if isinstance(raw_roles, str):
            raw_roles = [role.strip() for role in raw_roles.split(",") if role.strip()]
        elif not isinstance(raw_roles, list):
            raw_roles = []
        roles = [role for role in raw_roles if role in VALID_ROLES]
        if not roles:
            continue
        stmt = select(Employee).where(Employee.full_name == name)
        employee = employee_session.scalars(stmt).first()
        if not employee:
            employee = Employee(full_name=name)
            employee_session.add(employee)
            employee_session.flush()
            created += 1
        else:
            updated += 1
        employee.role_list = roles
        desired_default = employee.desired_hours or 25
        try:
            employee.desired_hours = int(payload.get("desired_hours", desired_default))
        except (TypeError, ValueError):
            employee.desired_hours = desired_default
        status = payload.get("status", employee.status or "active")
        employee.status = status if isinstance(status, str) and status else (employee.status or "active")
        employee.notes = payload.get("notes", "") or ""
        employee.start_month = payload.get("start_month")
        employee.start_year = payload.get("start_year")

        unavailability_payload = payload.get("unavailability")
        if isinstance(unavailability_payload, list):
            employee_session.execute(
                delete(EmployeeUnavailability).where(EmployeeUnavailability.employee_id == employee.id)
            )
            for entry in unavailability_payload:
                if not isinstance(entry, dict):
                    continue
                try:
                    start_time = datetime.time.fromisoformat(entry["start_time"])
                    end_time = datetime.time.fromisoformat(entry["end_time"])
                    day = int(entry["day_of_week"])
                except (KeyError, TypeError, ValueError):
                    continue
                employee_session.add(
                    EmployeeUnavailability(
                        employee_id=employee.id,
                        day_of_week=day,
                        start_time=start_time,
                        end_time=end_time,
                    )
                )
        role_wage_entries = payload.get("role_wages")
        if isinstance(role_wage_entries, list):
            role_wage_map = {
                entry["role"]: entry.get("wage", 0.0)
                for entry in role_wage_entries
                if isinstance(entry, dict) and entry.get("role")
            }
            save_employee_role_wages(employee_session, employee.id, role_wage_map, commit=False)
    employee_session.commit()
    return created, updated


# ---------------------------------------------------------------------------
# Week projections & modifiers


def export_week_projections(session, week: WeekContext) -> Path:
    projections = get_week_daily_projections(session, week.id)
    payload = [
        {
            "day_of_week": row.day_of_week,
            "projected_sales_amount": row.projected_sales_amount,
            "projected_notes": row.projected_notes,
        }
        for row in projections
    ]
    filename = EXPORT_DIR / f"week_{week.iso_year}W{week.iso_week}_projections_{_timestamp()}.json"
    filename.write_text(
        json.dumps({"week": {"id": week.id, "label": week.label}, "projections": payload}, indent=2),
        encoding="utf-8",
    )
    return filename


def import_week_projections(session, week: WeekContext, file_path: Path) -> int:
    data = json.loads(file_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Projections file must be a JSON object.")
    projections = data.get("projections")
    if not isinstance(projections, list):
        raise ValueError("Projections file must include a 'projections' list.")
    values: Dict[int, Dict[str, Any]] = {}
    for entry in projections:
        if not isinstance(entry, dict):
            continue
        if "day_of_week" not in entry:
            continue
        try:
            day_index = int(entry["day_of_week"])
        except (TypeError, ValueError):
            continue
        try:
            amount = float(entry.get("projected_sales_amount", 0.0) or 0.0)
        except (TypeError, ValueError):
            amount = 0.0
        notes = entry.get("projected_notes", "")
        if not isinstance(notes, str):
            notes = ""
        values[day_index] = {
            "projected_sales_amount": amount,
            "projected_notes": notes,
        }
    save_week_daily_projection_values(session, week.id, values)
    return len(values)


def export_week_modifiers(session, week: WeekContext) -> Path:
    modifiers = get_week_modifiers(session, week.id)
    saved = list_saved_modifiers(session)
    payload = [
        {
            "title": item.title,
            "modifier_type": item.modifier_type,
            "day_of_week": item.day_of_week,
            "start_time": item.start_time.isoformat(timespec="minutes"),
            "end_time": item.end_time.isoformat(timespec="minutes"),
            "pct_change": item.pct_change,
            "notes": item.notes,
        }
        for item in modifiers
    ]
    saved_payload = [
        {
            "title": item.title,
            "modifier_type": item.modifier_type,
            "day_of_week": item.day_of_week,
            "start_time": item.start_time.isoformat(timespec="minutes"),
            "end_time": item.end_time.isoformat(timespec="minutes"),
            "pct_change": item.pct_change,
            "notes": item.notes,
            "created_by": item.created_by,
        }
        for item in saved
    ]
    filename = EXPORT_DIR / f"week_{week.iso_year}W{week.iso_week}_modifiers_{_timestamp()}.json"
    filename.write_text(
        json.dumps(
            {"week": {"id": week.id, "label": week.label}, "modifiers": payload, "saved_modifiers": saved_payload},
            indent=2,
        ),
        encoding="utf-8",
    )
    return filename


def import_week_modifiers(session, week: WeekContext, file_path: Path, *, created_by: str) -> int:
    data = json.loads(file_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Modifiers file must be a JSON object.")
    modifiers = data.get("modifiers")
    if not isinstance(modifiers, list):
        raise ValueError("Modifiers file must include a 'modifiers' list.")
    saved_modifiers = data.get("saved_modifiers")
    session.execute(delete(Modifier).where(Modifier.week_id == week.id))
    added = 0
    for entry in modifiers:
        if not isinstance(entry, dict):
            continue
        try:
            start_time = datetime.time.fromisoformat(entry["start_time"])
            end_time = datetime.time.fromisoformat(entry["end_time"])
            day = int(entry["day_of_week"])
        except (KeyError, TypeError, ValueError):
            continue
        modifier = Modifier(
            week_id=week.id,
            title=entry.get("title") or "Imported modifier",
            modifier_type=entry.get("modifier_type", "increase"),
            day_of_week=day,
            start_time=start_time,
            end_time=end_time,
            pct_change=int(entry.get("pct_change", 0)),
            notes=entry.get("notes", "") or "",
            created_by=created_by,
        )
        session.add(modifier)
        added += 1
    if isinstance(saved_modifiers, list) and saved_modifiers:
        existing_saved = list_saved_modifiers(session)
        existing_keys = {
            (
                item.title or "",
                item.modifier_type or "",
                int(item.day_of_week),
                item.start_time,
                item.end_time,
                int(item.pct_change),
                item.notes or "",
            )
            for item in existing_saved
        }
        for entry in saved_modifiers:
            if not isinstance(entry, dict):
                continue
            try:
                start_time = datetime.time.fromisoformat(entry["start_time"])
                end_time = datetime.time.fromisoformat(entry["end_time"])
                day = int(entry["day_of_week"])
                pct = int(entry.get("pct_change", 0))
            except (KeyError, TypeError, ValueError):
                continue
            key = (
                entry.get("title") or "",
                entry.get("modifier_type", "increase"),
                day,
                start_time,
                end_time,
                pct,
                entry.get("notes", "") or "",
            )
            if key in existing_keys:
                continue
            saved = SavedModifier(
                title=entry.get("title") or "Imported modifier",
                modifier_type=entry.get("modifier_type", "increase"),
                day_of_week=day,
                start_time=start_time,
                end_time=end_time,
                pct_change=pct,
                notes=entry.get("notes", "") or "",
                created_by=entry.get("created_by") or created_by,
            )
            session.add(saved)
            existing_keys.add(key)
    session.commit()
    return added


# ---------------------------------------------------------------------------
# Week schedule (shifts)


def export_week_schedule(session, week_start: datetime.date, *, employee_session=None) -> Path:
    week = get_or_create_week(session, week_start)
    shifts = session.scalars(select(Shift).where(Shift.week_id == week.id)).all()
    employees: Dict[int, str] = {}
    if employee_session:
        employees = {emp.id: emp.full_name for emp in employee_session.scalars(select(Employee))}
    payload = [
        {
            "role": shift.role,
            "start": shift.start.isoformat(),
            "end": shift.end.isoformat(),
            "location": shift.location,
            "notes": shift.notes,
            "status": shift.status,
            "labor_rate": shift.labor_rate,
            "labor_cost": shift.labor_cost,
            "employee_name": employees.get(shift.employee_id),
        }
        for shift in shifts
    ]
    filename = EXPORT_DIR / f"week_{week.iso_year}W{week.iso_week}_shifts_{_timestamp()}.json"
    filename.write_text(
        json.dumps({"week": _week_info_from_date(week_start), "shifts": payload}, indent=2),
        encoding="utf-8",
    )
    return filename


def import_week_schedule(session, week_start: datetime.date, file_path: Path, *, employee_session=None) -> int:
    data = json.loads(file_path.read_text(encoding="utf-8"))
    week = get_or_create_week(session, week_start)
    session.execute(delete(Shift).where(Shift.week_id == week.id))
    name_to_id: Dict[str, int] = {}
    if employee_session:
        name_to_id = {
            employee.full_name: employee.id
            for employee in employee_session.scalars(select(Employee)).all()
        }
    added = 0
    for entry in data.get("shifts", []):
        try:
            start = datetime.datetime.fromisoformat(entry["start"])
            end = datetime.datetime.fromisoformat(entry["end"])
        except (KeyError, ValueError):
            continue
        role = entry.get("role")
        if not role:
            continue
        employee_name = entry.get("employee_name")
        employee_id = name_to_id.get(employee_name) if employee_name else None
        shift = Shift(
            week_id=week.id,
            employee_id=employee_id,
            role=role,
            start=start,
            end=end,
            location=entry.get("location", "") or "",
            notes=entry.get("notes", "") or "",
            status=entry.get("status", "draft"),
            labor_rate=float(entry.get("labor_rate", 0.0) or 0.0),
            labor_cost=float(entry.get("labor_cost", 0.0) or 0.0),
        )
        session.add(shift)
        added += 1
    set_week_status(session, week_start, "draft")
    return added


# ---------------------------------------------------------------------------
# Role wage import/export


def export_role_wages_dataset() -> Path:
    filename = EXPORT_DIR / f"role_wages_{_timestamp()}.json"
    return export_wages(filename)


def apply_role_wages_to_employees(
    *,
    employee_session=None,
    wage_map: Dict[str, float] | None = None,
    overwrite: bool = True,
) -> int:
    wage_map = wage_map or wage_amounts()
    if not wage_map:
        return 0
    close_session = False
    if employee_session is None:
        employee_session = EmployeeSessionLocal()
        close_session = True
    existing_ids: set[int] = set()
    if not overwrite:
        stmt = select(EmployeeRoleWage.employee_id).distinct()
        existing_ids = {row[0] for row in employee_session.execute(stmt) if row[0] is not None}
    employees = employee_session.scalars(select(Employee)).all()
    updated = 0
    for employee in employees:
        if not overwrite and employee.id in existing_ids:
            continue
        roles = [role for role in employee.role_list if role]
        if not roles:
            continue
        mapping = {role: wage_map.get(role, 0.0) for role in roles}
        save_employee_role_wages(employee_session, employee.id, mapping, commit=False)
        updated += 1
    employee_session.commit()
    if close_session:
        employee_session.close()
    return updated


def import_role_wages_dataset(file_path: Path) -> int:
    imported = import_wages(file_path)
    apply_role_wages_to_employees(wage_map=wage_amounts())
    return imported


# ---------------------------------------------------------------------------
# Policy import/export


def export_policy_dataset(session) -> Path:
    policy = get_active_policy(session)
    if not policy:
        raise ValueError("No active policy found to export.")
    payload = {
        "name": policy.name,
        "params": policy.params_dict(),
    }
    filename = EXPORT_DIR / f"policy_{_timestamp()}.json"
    filename.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return filename


def import_policy_dataset(session, file_path: Path, *, edited_by: str = "import") -> Policy:
    data = json.loads(file_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Policy file must be a JSON object.")
    params = data.get("params") if isinstance(data.get("params"), dict) else None
    if params is None:
        params = {k: v for k, v in data.items() if k != "name"}
    params = dict(params)
    params.pop("name", None)
    name = data.get("name") or params.get("policy_name") or "Imported Policy"
    return upsert_policy(session, name, params, edited_by=edited_by)


# ---------------------------------------------------------------------------
# Copy helpers (no files)


def copy_week_dataset(
    session,
    source_week: WeekContext,
    target_week: WeekContext,
    dataset: str,
    *,
    actor: str,
    employee_session=None,
) -> Dict[str, int]:
    """Copy projections/modifiers/shifts between weeks. Returns summary counts."""
    dataset = dataset.lower()
    if dataset == "projections":
        projections = get_week_daily_projections(session, source_week.id)
        values = {
            item.day_of_week: {
                "projected_sales_amount": item.projected_sales_amount,
                "projected_notes": item.projected_notes,
            }
            for item in projections
        }
        save_week_daily_projection_values(session, target_week.id, values)
        return {"projections": len(values)}
    if dataset == "modifiers":
        session.execute(delete(Modifier).where(Modifier.week_id == target_week.id))
        clones = get_week_modifiers(session, source_week.id)
        count = 0
        for source in clones:
            session.add(
                Modifier(
                    week_id=target_week.id,
                    title=source.title,
                    modifier_type=source.modifier_type,
                    day_of_week=source.day_of_week,
                    start_time=source.start_time,
                    end_time=source.end_time,
                    pct_change=source.pct_change,
                    notes=f"Copied from {source.week.label} by {actor}",
                    created_by=actor,
                )
            )
            count += 1
        session.commit()
        return {"modifiers": count}
    if dataset == "shifts":
        target_date = datetime.date.fromisocalendar(target_week.iso_year, target_week.iso_week, 1)
        source_date = datetime.date.fromisocalendar(source_week.iso_year, source_week.iso_week, 1)
        source_schedule = get_or_create_week(session, source_date)
        export_path = EXPORT_DIR / f"temp_copy_{_timestamp()}.json"
        employees: Dict[int, str] = {}
        if employee_session:
            employees = {
                employee.id: employee.full_name
                for employee in employee_session.scalars(select(Employee))
            }
        export_path.write_text(
            json.dumps({
                "week": _week_info_from_date(source_date),
                "shifts": [
                    {
                        "role": shift.role,
                        "start": shift.start.isoformat(),
                        "end": shift.end.isoformat(),
                        "location": shift.location,
                        "notes": shift.notes,
                        "status": shift.status,
                        "labor_rate": shift.labor_rate,
                        "labor_cost": shift.labor_cost,
                        "employee_name": employees.get(shift.employee_id),
                    }
                    for shift in session.scalars(select(Shift).where(Shift.week_id == source_schedule.id))
                ],
            }),
            encoding="utf-8",
        )
        count = import_week_schedule(session, target_date, export_path, employee_session=employee_session)
        export_path.unlink(missing_ok=True)
        return {"shifts": count}
    raise ValueError(f"Unsupported dataset '{dataset}' for copy operation.")


def get_weeks_summary(session) -> List[Dict[str, str]]:
    """Return list of week labels for copy prompts."""
    weeks = get_all_weeks(session)
    summary = []
    for week in weeks:
        summary.append(
            {
                "label": week.label,
                "iso_year": week.iso_year,
                "iso_week": week.iso_week,
                "id": week.id,
            }
        )
    return summary
