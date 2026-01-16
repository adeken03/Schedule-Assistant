from __future__ import annotations

import datetime
import sys
from pathlib import Path
from typing import Dict, Tuple

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


def _setup_engines() -> Tuple:
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
    policy["role_groups"] = {
        "Kitchen": {"allocation_pct": 0.05, "allow_cuts": True, "cut_buffer_minutes": 25},
        "Servers": {"allocation_pct": 0.39, "allow_cuts": True, "cut_buffer_minutes": 35},
        "Bartenders": {"allocation_pct": 0.12, "allow_cuts": False, "always_on": True, "cut_buffer_minutes": 0},
        "Cashier": {"allocation_pct": 0.15, "allow_cuts": True, "cut_buffer_minutes": 25},
    }
    with session_factory() as session:
        upsert_policy(session, policy.get("name", "Default Policy"), policy, edited_by="test")


def _seed_employees(session) -> None:
    bartender_roles = "Bartender,Bartender - Opener,Bartender - Closer"
    for idx in range(4):
        session.add(Employee(full_name=f"Auto Bartender {idx}", roles=bartender_roles, desired_hours=60, status="active"))

    server_roles = (
        "Server - Opener,Server - Dining,Server - Cocktail,"
        "Server - Dining Preclose,Server - Dining Closer,"
        "Server - Cocktail Preclose,Server - Cocktail Closer"
    )
    for idx in range(14):
        session.add(Employee(full_name=f"Auto Server {idx}", roles=server_roles, desired_hours=60, status="active"))

    kitchen_roles = "HOH - Opener,HOH - Expo,HOH - Southwest,HOH - Chip,HOH - Shake,HOH - Grill"
    for idx in range(12):
        session.add(Employee(full_name=f"Auto Kitchen {idx}", roles=kitchen_roles, desired_hours=60, status="active"))

    for idx in range(4):
        session.add(Employee(full_name=f"Auto Cashier {idx}", roles="Cashier", desired_hours=60, status="active"))

    session.commit()


def _seed_sales(session, week_start: datetime.date, *, amount: float) -> None:
    week = get_or_create_week(session, week_start)
    ctx = get_or_create_week_context(session, week.iso_year, week.iso_week, week.label)
    values: Dict[int, Dict[str, float | str]] = {}
    for day_idx in range(7):
        values[day_idx] = {"projected_sales_amount": float(amount), "projected_notes": "{}"}
    save_week_daily_projection_values(session, ctx.id, values)
    session.commit()


def test_kitchen_required_roles_are_cuttable_but_expo_is_protected() -> None:
    SessionLocal, EmployeeSessionLocal = _setup_engines()
    schedule_session = SessionLocal()
    employee_session = EmployeeSessionLocal()

    reset_wages_to_defaults()
    _seed_policy(SessionLocal)
    _seed_employees(employee_session)

    week_start = datetime.date(2025, 6, 2)  # Monday
    _seed_sales(schedule_session, week_start, amount=3000.0)  # low tier to allow trimming slack

    result = generate_schedule_for_week(
        SessionLocal,
        week_start,
        actor="kitchen-budget-cut-test",
        employee_session_factory=EmployeeSessionLocal,
        max_attempts=1,
    )
    assert result.get("shifts_created", 0) > 0

    kitchen_cut_events = [
        entry
        for entry in result.get("cut_insights", [])
        if entry.get("role_group") == "Kitchen" and entry.get("minutes_trimmed")
    ]
    assert kitchen_cut_events, "Expected kitchen budget cuts to be applied"

    shifts = list(schedule_session.execute(select(Shift)).scalars())
    non_expo_hoh = [
        shift
        for shift in shifts
        if (shift.role or "").lower().startswith("hoh -")
        and "expo" not in (shift.role or "").lower()
        and "opener" not in (shift.role or "").lower()
    ]
    assert non_expo_hoh, "Expected HOH station shifts to be generated"

    cut_required = [
        shift
        for shift in non_expo_hoh
        if "required coverage" in (shift.notes or "").lower() and "cut" in (shift.notes or "").lower()
    ]
    assert cut_required, "Expected at least one required (non-expo) kitchen shift to be trimmed"

    expo_shifts = [shift for shift in shifts if "expo" in (shift.role or "").lower()]
    assert expo_shifts, "Expected expo shifts to be generated"
    assert all("cut" not in (shift.notes or "").lower() for shift in expo_shifts), "Expo shifts should never be cut"

