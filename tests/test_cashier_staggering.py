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
    # Keep budgets generous so plan trimming doesn't drop extra cashiers.
    policy["global"]["labor_budget_pct"] = 0.9
    policy["global"]["labor_budget_tolerance_pct"] = 0.5
    policy["pre_engine"]["fallback"]["allow_mgr_fallback"] = False
    with session_factory() as session:
        upsert_policy(session, policy.get("name", "Default Policy"), policy, edited_by="test")


def _seed_employees(session) -> None:
    """
    Seed minimal required roles plus multiple cashiers so multi-cashier periods can assign cleanly.
    """
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
        "Cashier",
    ]
    for idx, role in enumerate(roles):
        session.add(
            Employee(
                full_name=f"Auto {role} {idx}",
                roles=role,
                desired_hours=60,
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


def test_cashier_shifts_are_staggered_by_30_minutes() -> None:
    SessionLocal, EmployeeSessionLocal = _setup_engines()
    schedule_session = SessionLocal()
    employee_session = EmployeeSessionLocal()

    reset_wages_to_defaults()
    _seed_policy(SessionLocal)
    _seed_employees(employee_session)

    week_start = datetime.date(2025, 11, 3)  # Monday (matches screenshot week)
    _seed_sales(schedule_session, week_start, amount=10_000.0)  # high tier => 2 cashiers AM + PM

    result = generate_schedule_for_week(
        SessionLocal,
        week_start,
        actor="cashier-stagger-test",
        employee_session_factory=EmployeeSessionLocal,
        max_attempts=1,
    )
    assert result.get("shifts_created", 0) > 0

    shifts = list(schedule_session.execute(select(Shift)).scalars())
    thursday = week_start + datetime.timedelta(days=3)

    pm_cashiers = sorted(
        [
            shift
            for shift in shifts
            if shift.role == "Cashier" and (shift.location or "").strip().upper() == "PM" and shift.start.date() == thursday
        ],
        key=lambda shift: shift.start,
    )
    assert len(pm_cashiers) >= 2, "Expected at least two PM cashier shifts on a high-tier Thursday"

    delta_minutes = int((pm_cashiers[1].start - pm_cashiers[0].start).total_seconds() // 60)
    assert delta_minutes >= 30, f"Expected cashier starts staggered by >=30m, got {delta_minutes}m"
