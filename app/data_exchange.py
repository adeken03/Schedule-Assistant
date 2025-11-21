from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Dict, List, Tuple

from sqlalchemy import delete, select

from database import (
    Employee,
    EmployeeUnavailability,
    Modifier,
    Shift,
    WeekContext,
    get_all_weeks,
    get_or_create_week,
    get_or_create_week_context,
    get_week_daily_projections,
    get_week_modifiers,
    save_week_daily_projection_values,
    set_week_status,
)
from exporter import DATA_DIR as EXPORT_DIR
from roles import defined_roles
from wages import export_wages, import_wages

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


def export_employees(session) -> Path:
    payload: List[Dict] = []
    employees = session.scalars(select(Employee).order_by(Employee.full_name.asc())).all()
    for employee in employees:
        entry = {
            "full_name": employee.full_name,
            "desired_hours": employee.desired_hours,
            "status": employee.status,
            "notes": employee.notes,
            "start_month": employee.start_month,
            "start_year": employee.start_year,
            "roles": employee.role_list,
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


def import_employees(session, file_path: Path) -> Tuple[int, int]:
    data = json.loads(file_path.read_text(encoding="utf-8"))
    employees = data.get("employees", [])
    created = 0
    updated = 0
    for payload in employees:
        name = payload.get("full_name")
        if not name:
            continue
        roles = [role for role in payload.get("roles", []) if role in VALID_ROLES]
        if not roles:
            continue
        stmt = select(Employee).where(Employee.full_name == name)
        employee = session.scalars(stmt).first()
        if not employee:
            employee = Employee(full_name=name)
            session.add(employee)
            session.flush()
            created += 1
        else:
            updated += 1
        employee.role_list = roles
        employee.desired_hours = int(payload.get("desired_hours", employee.desired_hours or 25))
        employee.status = payload.get("status", employee.status or "active")
        employee.notes = payload.get("notes", "") or ""
        employee.start_month = payload.get("start_month")
        employee.start_year = payload.get("start_year")

        # Replace unavailability rows
        session.execute(delete(EmployeeUnavailability).where(EmployeeUnavailability.employee_id == employee.id))
        for entry in payload.get("unavailability", []):
            try:
                start_time = datetime.time.fromisoformat(entry["start_time"])
                end_time = datetime.time.fromisoformat(entry["end_time"])
                day = int(entry["day_of_week"])
            except (KeyError, ValueError):
                continue
            session.add(
                EmployeeUnavailability(
                    employee_id=employee.id,
                    day_of_week=day,
                    start_time=start_time,
                    end_time=end_time,
                )
            )
    session.commit()
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
    values = {
        int(entry["day_of_week"]): {
            "projected_sales_amount": float(entry.get("projected_sales_amount", 0.0)),
            "projected_notes": entry.get("projected_notes", ""),
        }
        for entry in data.get("projections", [])
        if "day_of_week" in entry
    }
    save_week_daily_projection_values(session, week.id, values)
    return len(values)


def export_week_modifiers(session, week: WeekContext) -> Path:
    modifiers = get_week_modifiers(session, week.id)
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
    filename = EXPORT_DIR / f"week_{week.iso_year}W{week.iso_week}_modifiers_{_timestamp()}.json"
    filename.write_text(
        json.dumps({"week": {"id": week.id, "label": week.label}, "modifiers": payload}, indent=2),
        encoding="utf-8",
    )
    return filename


def import_week_modifiers(session, week: WeekContext, file_path: Path, *, created_by: str) -> int:
    data = json.loads(file_path.read_text(encoding="utf-8"))
    session.execute(delete(Modifier).where(Modifier.week_id == week.id))
    added = 0
    for entry in data.get("modifiers", []):
        try:
            start_time = datetime.time.fromisoformat(entry["start_time"])
            end_time = datetime.time.fromisoformat(entry["end_time"])
            day = int(entry["day_of_week"])
        except (KeyError, ValueError):
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
    session.commit()
    return added


# ---------------------------------------------------------------------------
# Week schedule (shifts)


def export_week_schedule(session, week_start: datetime.date) -> Path:
    week = get_or_create_week(session, week_start)
    shifts = session.scalars(select(Shift).where(Shift.week_id == week.id)).all()
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
            "employee_name": shift.employee.full_name if shift.employee else None,
        }
        for shift in shifts
    ]
    filename = EXPORT_DIR / f"week_{week.iso_year}W{week.iso_week}_shifts_{_timestamp()}.json"
    filename.write_text(
        json.dumps({"week": _week_info_from_date(week_start), "shifts": payload}, indent=2),
        encoding="utf-8",
    )
    return filename


def import_week_schedule(session, week_start: datetime.date, file_path: Path) -> int:
    data = json.loads(file_path.read_text(encoding="utf-8"))
    week = get_or_create_week(session, week_start)
    session.execute(delete(Shift).where(Shift.week_id == week.id))
    name_to_id = {
        employee.full_name: employee.id
        for employee in session.scalars(select(Employee)).all()
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


def import_role_wages_dataset(file_path: Path) -> int:
    return import_wages(file_path)


# ---------------------------------------------------------------------------
# Copy helpers (no files)


def copy_week_dataset(session, source_week: WeekContext, target_week: WeekContext, dataset: str, *, actor: str) -> Dict[str, int]:
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
        export_path = EXPORT_DIR / f"temp_copy_{_timestamp()}.json"
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
                        "employee_name": shift.employee.full_name if shift.employee else None,
                    }
                    for shift in session.scalars(select(Shift).where(Shift.week_id == source_week.id))
                ],
            }),
            encoding="utf-8",
        )
        count = import_week_schedule(session, target_date, export_path)
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
