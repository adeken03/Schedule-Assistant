from __future__ import annotations

import datetime
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

APP_DIR = Path(__file__).resolve().parents[1] / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from database import (  # noqa: E402
    Base,
    Employee,
    EmployeeBase,
    EmployeeUnavailability,
    PolicyBase,
    ProjectionsBase,
    get_or_create_week,
    get_or_create_week_context,
    save_week_daily_projection_values,
    upsert_policy,
)
import database as db  # noqa: E402
from generator.api import generate_schedule_for_week  # noqa: E402
from policy import build_default_policy, ensure_default_policy  # noqa: E402
from wages import reset_wages_to_defaults  # noqa: E402
from validation import validate_week_schedule  # noqa: E402


def _seed_policy(session_factory) -> None:
    ensure_default_policy(session_factory)
    # Relax constraints for tests to avoid fallback noise.
    policy = build_default_policy()
    policy["global"]["max_consecutive_days"] = 7
    policy["pre_engine"]["fallback"]["allow_mgr_fallback"] = False
    with session_factory() as session:
        upsert_policy(session, policy.get("name", "Default Policy"), policy, edited_by="test")


def _seed_employees(session) -> List[int]:
    """Create one employee for every required BWW role."""
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
    employees: List[int] = []
    for idx, role in enumerate(roles):
        employee = Employee(
            full_name=f"Auto {role} {idx}",
            roles=role,
            desired_hours=32,
            status="active",
        )
        session.add(employee)
        session.flush()
        employees.append(employee.id)
    session.commit()
    return employees


def _seed_unavailability(session, employee_ids: Iterable[int]) -> None:
    """Keep availability wide open but ensure we exercise the unavailability logic a bit."""
    for emp_id in employee_ids:
        entry = EmployeeUnavailability(
            employee_id=emp_id,
            day_of_week=0,  # Monday early morning to avoid close buffers
            start_time=datetime.time(0, 0),
            end_time=datetime.time(6, 0),
        )
        session.add(entry)
    session.commit()


def _seed_sales(session) -> datetime.date:
    iso_year, iso_week, _ = datetime.date.today().isocalendar()
    week_start = datetime.date.fromisocalendar(iso_year, iso_week, 1)
    week = get_or_create_week(session, week_start)
    context = get_or_create_week_context(session, week.iso_year, week.iso_week, week.label)
    values: Dict[int, Dict[str, float]] = {}
    for day_idx in range(7):
        values[day_idx] = {"projected_sales_amount": 1000.0, "projected_notes": "{}"}
    save_week_daily_projection_values(session, context.id, values)
    session.commit()
    return week_start


def test_generate_and_validate_schedule_smoke() -> None:
    """End-to-end smoke: seed policy/employees/projections, generate, validate."""
    schedule_engine = create_engine("sqlite:///:memory:", future=True)
    employee_engine = create_engine("sqlite:///:memory:", future=True)
    projection_engine = create_engine("sqlite:///:memory:", future=True)

    # Wire shared engines so db helpers operate on the same in-memory stores.
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

    session = db.SessionLocal()
    employee_session = db.EmployeeSessionLocal()

    reset_wages_to_defaults()
    _seed_policy(db.SessionLocal)
    employee_ids = _seed_employees(employee_session)
    _seed_unavailability(employee_session, employee_ids)
    week_start = _seed_sales(session)

    result = generate_schedule_for_week(
        db.SessionLocal,
        week_start,
        actor="smoke-test",
        employee_session_factory=db.EmployeeSessionLocal,
        max_attempts=1,
    )
    assert result.get("shifts_created", 0) > 0, f"Generator created no shifts: {result}"

    validation = validate_week_schedule(session, week_start, employee_session=employee_session)
    issues = validation.get("issues") or []
    assert not issues, f"Validation issues found: {issues}"

    session.close()
    employee_session.close()
