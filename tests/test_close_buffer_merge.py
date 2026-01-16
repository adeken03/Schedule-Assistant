from __future__ import annotations

import datetime
import sys
from pathlib import Path

from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

APP_DIR = Path(__file__).resolve().parents[1] / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import database as db  # noqa: E402
from database import (  # noqa: E402
    Base,
    Employee,
    EmployeeBase,
    PolicyBase,
    ProjectionsBase,
    Shift,
    get_or_create_week,
    get_or_create_week_context,
    save_week_daily_projection_values,
    shift_display_date,
    upsert_policy,
)
from generator.api import generate_schedule_for_week  # noqa: E402
from policy import build_default_policy  # noqa: E402
from wages import reset_wages_to_defaults  # noqa: E402


def _setup_engines():
    schedule_engine = create_engine("sqlite:///:memory:", future=True)
    employee_engine = create_engine("sqlite:///:memory:", future=True)
    projection_engine = create_engine("sqlite:///:memory:", future=True)

    db.schedule_engine = schedule_engine
    db.SessionLocal = sessionmaker(bind=schedule_engine, expire_on_commit=False, future=True)
    db.employee_engine = employee_engine
    db.EmployeeSessionLocal = sessionmaker(bind=employee_engine, expire_on_commit=False, future=True)
    db.policy_engine = schedule_engine
    db.PolicySessionLocal = db.SessionLocal
    db.projections_engine = projection_engine
    db.ProjectionSessionLocal = sessionmaker(bind=projection_engine, expire_on_commit=False, future=True)

    Base.metadata.create_all(schedule_engine)
    EmployeeBase.metadata.create_all(employee_engine)
    PolicyBase.metadata.create_all(schedule_engine)
    ProjectionsBase.metadata.create_all(projection_engine)

    return db.SessionLocal, db.EmployeeSessionLocal


def _seed_policy(session_factory) -> None:
    policy = build_default_policy()
    policy["global"]["max_consecutive_days"] = 7
    policy["pre_engine"]["fallback"]["allow_mgr_fallback"] = False
    with session_factory() as session:
        upsert_policy(session, policy.get("name", "Default Policy"), policy, edited_by="test")


def _seed_employees(session) -> None:
    roles = [
        "Bartender,Bartender - Opener,Bartender - Closer",
        "Bartender",
        "Server - Opener,Server - Dining",
        "Server - Opener,Server - Dining",
        "Server - Dining Preclose",
        "Server - Dining Closer",
        "Server - Cocktail Preclose",
        "Server - Cocktail Closer",
        "Server - Dining",
        "Server - Cocktail",
        "HOH - Opener",
        "HOH - Expo",
        "HOH - Southwest",
        "HOH - Chip",
        "HOH - Shake",
        "HOH - Grill",
        "Cashier",
    ]
    for idx, role in enumerate(roles):
        session.add(
            Employee(
                full_name=f"Auto {role} {idx}",
                roles=role,
                desired_hours=32,
                status="active",
            )
        )
    session.commit()


def _seed_sales(session, week_start: datetime.date) -> None:
    week = get_or_create_week(session, week_start)
    ctx = get_or_create_week_context(session, week.iso_year, week.iso_week, week.label)
    values = {day_idx: {"projected_sales_amount": 1000.0, "projected_notes": "{}"} for day_idx in range(7)}
    save_week_daily_projection_values(session, ctx.id, values)
    session.commit()


def test_server_close_buffer_not_merged_on_sunday() -> None:
    SessionLocal, EmployeeSessionLocal = _setup_engines()
    schedule_session = SessionLocal()
    employee_session = EmployeeSessionLocal()

    reset_wages_to_defaults()
    _seed_policy(SessionLocal)
    _seed_employees(employee_session)

    week_start = datetime.date(2025, 6, 2)  # Monday
    _seed_sales(schedule_session, week_start)

    result = generate_schedule_for_week(
        SessionLocal,
        week_start,
        actor="close-buffer-test",
        employee_session_factory=EmployeeSessionLocal,
        max_attempts=1,
    )
    assert result.get("shifts_created", 0) > 0

    shifts = list(schedule_session.execute(select(Shift)).scalars())
    sunday = week_start + datetime.timedelta(days=6)

    cocktail_buffer_shifts = [
        shift
        for shift in shifts
        if shift.role == "Server - Cocktail Closer"
        and shift_display_date(shift.start, shift.location) == sunday
        and "close buffer" in (shift.notes or "").lower()
    ]
    assert cocktail_buffer_shifts, "Expected a cocktail close-buffer shift on Sunday"

    longest_minutes = max((shift.end - shift.start).total_seconds() / 60 for shift in cocktail_buffer_shifts)
    assert longest_minutes <= 90, "Close-buffer job code should not absorb the full closer shift on Sunday"

