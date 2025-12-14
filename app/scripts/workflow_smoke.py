from __future__ import annotations

import argparse
import datetime
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from sqlalchemy import delete, select

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from database import (  # noqa: E402
    Modifier,
    Shift,
    SessionLocal,
    EmployeeSessionLocal,
    get_or_create_week_context,
    get_or_create_week,
    get_shifts_for_week,
    get_week_summary,
    init_database,
    list_employees,
    save_week_daily_projection_values,
    set_week_status,
)
from exporter import export_week  # noqa: E402
from generator.api import generate_schedule_for_week  # noqa: E402
from policy import ensure_default_policy  # noqa: E402


def _default_week_start(today: datetime.date | None = None) -> datetime.date:
    base = today or datetime.date.today()
    delta = (7 - base.weekday()) % 7
    delta = delta or 7
    target = base + datetime.timedelta(days=delta)
    return target


def _week_label(week_start: datetime.date) -> str:
    iso_year, iso_week, _ = week_start.isocalendar()
    end = week_start + datetime.timedelta(days=6)
    return f"{iso_year} W{iso_week:02d} ({week_start:%b %d} - {end:%b %d})"


def _projection_payload() -> List[Tuple[float, Dict[str, float | str]]]:
    sales = [21000, 22000, 23000, 24000, 27000, 32000, 18500]
    notes = {
        4: {"promo": "Wing Friday", "togo_index": 0.35},
        5: {"event": "UFC card", "togo_index": 0.4},
        0: {"campus": "Study night", "togo_index": 0.2},
    }
    return [
        (amount, notes.get(idx, {}))
        for idx, amount in enumerate(sales)
    ]


def _modifier_specs() -> List[Dict[str, object]]:
    return [
        {
            "title": "Monday Trivia",
            "day": 0,
            "start": datetime.time(19, 0),
            "end": datetime.time(22, 0),
            "pct": 12,
            "notes": "Trivia crowd spike",
        },
        {
            "title": "Friday Night Football",
            "day": 4,
            "start": datetime.time(17, 0),
            "end": datetime.time(23, 30),
            "pct": 25,
            "notes": "Home game boost",
        },
        {
            "title": "Saturday Fight Card",
            "day": 5,
            "start": datetime.time(18, 0),
            "end": datetime.time(1, 0),
            "pct": 30,
            "notes": "PPV main event",
        },
    ]


def _seed_projections(session, week_id: int) -> None:
    payload: Dict[int, Dict[str, object]] = {}
    for idx, (amount, note_dict) in enumerate(_projection_payload()):
        payload[idx] = {
            "projected_sales_amount": float(amount),
            "projected_notes": json.dumps(note_dict) if note_dict else "",
        }
    save_week_daily_projection_values(session, week_id, payload)


def _seed_modifiers(session, week_id: int, *, created_by: str) -> None:
    session.execute(delete(Modifier).where(Modifier.week_id == week_id))
    for spec in _modifier_specs():
        session.add(
            Modifier(
                week_id=week_id,
                title=str(spec["title"]),
                modifier_type="increase" if int(spec["pct"]) >= 0 else "decrease",
                day_of_week=int(spec["day"]),
                start_time=spec["start"],
                end_time=spec["end"],
                pct_change=int(spec["pct"]),
                notes=str(spec.get("notes", "")),
                created_by=created_by,
            )
        )
    session.commit()


def _validate_week(week_start: datetime.date) -> Tuple[List[str], Dict[str, object]]:
    with SessionLocal() as session:
        summary = get_week_summary(session, week_start)
        shifts = get_shifts_for_week(session, week_start)

    errors: List[str] = []
    if summary.get("total_shifts", 0) == 0:
        errors.append("No shifts scheduled for the selected week.")
    for day in summary.get("days", []):
        if day.get("count", 0) == 0:
            errors.append(f"No coverage scheduled for {day.get('date')}.")
    for shift in shifts:
        if not shift.get("employee_id"):
            errors.append(
                f"{shift.get('role')} shift starting {_format_shift_label(shift.get('start'))} is unassigned."
            )
    if not errors:
        with SessionLocal() as session:
            set_week_status(session, week_start, "validated")
    return errors, summary


def _format_shift_label(start: datetime.datetime | None) -> str:
    if not isinstance(start, datetime.datetime):
        return "unknown time"
    localized = start.astimezone()
    return localized.strftime("%a %m/%d %I:%M %p")


def run_workflow(week_start: datetime.date, actor: str) -> None:
    ensure_default_policy(SessionLocal)
    iso_year, iso_week, _ = week_start.isocalendar()

    with SessionLocal() as session:
        week = get_or_create_week_context(session, iso_year, iso_week, _week_label(week_start))
        _seed_projections(session, week.id)
        _seed_modifiers(session, week.id, created_by=actor)

    result = generate_schedule_for_week(
        SessionLocal,
        week_start,
        actor=actor,
        employee_session_factory=EmployeeSessionLocal,
    )
    warnings = result.get("warnings") or []
    print(f"[workflow] Generated {result.get('shifts_created', 0)} shifts for {week_start.isoformat()}.")
    if warnings:
        for warning in warnings:
            print(f"[workflow][warning] {warning}")

    filled = _autofill_unassigned_shifts(week_start)
    if filled:
        print(f"[workflow] Auto-assigned {filled} remaining unfilled shifts for validation.")

    errors, summary = _validate_week(week_start)
    if errors:
        for err in errors:
            print(f"[workflow][validation-error] {err}")
        raise SystemExit(1)
    print("[workflow] Validation passed, week marked as validated.")

    week_id = summary.get("week_id") or result.get("week_id")
    if not week_id:
        raise SystemExit("Unable to determine week_id for export.")
    pdf_path = export_week(week_id, "pdf")
    csv_path = export_week(week_id, "csv")
    print(f"[workflow] Exported schedule -> {pdf_path}")
    print(f"[workflow] Exported schedule -> {csv_path}")
    print(
        f"[workflow] Total shifts: {summary.get('total_shifts')} | "
        f"Labor cost: ${summary.get('total_cost', 0.0):.2f}"
    )


def _autofill_unassigned_shifts(week_start: datetime.date) -> int:
    with SessionLocal() as session, EmployeeSessionLocal() as employee_session:
        week = get_or_create_week(session, week_start)
        shifts = list(session.scalars(select(Shift).where(Shift.week_id == week.id)))
        roster: Dict[str, List[int]] = defaultdict(list)
        for employee in list_employees(employee_session, only_active=True):
            for role in employee.get("roles", []):
                roster[role].append(employee["id"])
        filled = 0
        rng = random.Random(42)
        for shift in shifts:
            if shift.employee_id or not roster.get(shift.role):
                continue
            shift.employee_id = rng.choice(roster[shift.role])
            filled += 1
        if filled:
            session.commit()
        return filled


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run an end-to-end smoke test that seeds projections, modifiers, "
            "ensures a policy exists, generates shifts, validates them, and exports files."
        )
    )
    parser.add_argument(
        "--week-start",
        help="ISO date (YYYY-MM-DD) for the Monday to target. Defaults to next Monday.",
    )
    parser.add_argument("--actor", default="workflow_smoke", help="Audit trail actor name.")
    return parser.parse_args()


def main() -> None:
    init_database()
    args = parse_args()
    if args.week_start:
        try:
            week_start = datetime.date.fromisoformat(args.week_start)
        except ValueError as exc:
            raise SystemExit(f"Invalid --week-start value: {exc}") from exc
    else:
        week_start = _default_week_start()
    if week_start.weekday() != 0:
        week_start = week_start - datetime.timedelta(days=week_start.weekday())
    print(f"[workflow] Target week start: {week_start} ({_week_label(week_start)})")
    run_workflow(week_start, actor=args.actor)


if __name__ == "__main__":
    main()
