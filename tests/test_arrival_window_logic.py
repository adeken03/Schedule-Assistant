from __future__ import annotations

import datetime
import sys
from pathlib import Path
from typing import Dict

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


def _seed_sales(session, week_start: datetime.date, *, amount: float) -> None:
    week = get_or_create_week(session, week_start)
    ctx = get_or_create_week_context(session, week.iso_year, week.iso_week, week.label)
    values: Dict[int, Dict[str, float | str]] = {}
    for day_idx in range(7):
        values[day_idx] = {"projected_sales_amount": float(amount), "projected_notes": "{}"}
    save_week_daily_projection_values(session, ctx.id, values)
    session.commit()


def test_hoh_grill_starts_within_arrival_windows_on_high_days() -> None:
    SessionLocal, EmployeeSessionLocal = _setup_engines()
    schedule_session = SessionLocal()
    employee_session = EmployeeSessionLocal()

    reset_wages_to_defaults()
    _seed_policy(SessionLocal)
    _seed_employees(employee_session)

    week_start = datetime.date(2025, 6, 2)  # Monday
    _seed_sales(schedule_session, week_start, amount=10_000.0)  # forces high tier (split HOH)

    result = generate_schedule_for_week(
        SessionLocal,
        week_start,
        actor="arrival-window-test",
        employee_session_factory=EmployeeSessionLocal,
        max_attempts=1,
    )
    assert result.get("shifts_created", 0) > 0

    shifts = list(schedule_session.execute(select(Shift)).scalars())
    monday = week_start

    am_grill = sorted(
        [s for s in shifts if s.role == "HOH - Grill" and (s.location or "").strip() == "AM" and s.start.date() == monday],
        key=lambda s: s.start,
    )
    assert am_grill, "Expected an AM HOH - Grill shift on a high-tier day"
    assert am_grill[0].start.time() <= datetime.time(12, 30)

    pm_grill = sorted(
        [s for s in shifts if s.role == "HOH - Grill" and (s.location or "").strip() == "PM" and s.start.date() == monday],
        key=lambda s: s.start,
    )
    assert pm_grill, "Expected a PM HOH - Grill shift on a high-tier day"
    assert pm_grill[0].start.time() <= datetime.time(17, 30)

