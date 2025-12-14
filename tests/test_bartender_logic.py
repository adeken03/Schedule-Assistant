from __future__ import annotations

import datetime
from typing import Dict, Iterable, List

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

import database as db
from database import (
    Base,
    Employee,
    EmployeeBase,
    EmployeeUnavailability,
    PolicyBase,
    ProjectionsBase,
    Shift,
    get_or_create_week,
    get_or_create_week_context,
    save_week_daily_projection_values,
)
from generator.api import generate_schedule_for_week
from policy import ensure_default_policy
from wages import reset_wages_to_defaults

UTC = datetime.timezone.utc


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
    ensure_default_policy(session_factory)


def _seed_employees(session) -> List[int]:
    """Bartender opener/closer continuity plus minimal required roles."""
    employees: List[int] = []
    # One bartender who can open/AM/close.
    bart_roles = ["Bartender", "Bartender - Opener", "Bartender - Closer"]
    bartender = Employee(full_name="Bartender A", roles=",".join(bart_roles), desired_hours=32, status="active")
    session.add(bartender)
    session.flush()
    employees.append(bartender.id)
    # Backup bartender.
    backup = Employee(full_name="Bartender B", roles="Bartender", desired_hours=30, status="active")
    session.add(backup)
    session.flush()
    employees.append(backup.id)
    # Required servers.
    required_servers = [
        "Server - Opener",
        "Server - Dining",
        "Server - Dining Preclose",
        "Server - Dining Closer",
        "Server - Cocktail",
        "Server - Cocktail Preclose",
        "Server - Cocktail Closer",
    ]
    for idx, role in enumerate(required_servers):
        emp = Employee(full_name=f"Server {idx}", roles=role, desired_hours=30, status="active")
        session.add(emp)
        session.flush()
        employees.append(emp.id)
    # HOH core.
    kitchen_roles = ["HOH - Opener", "HOH - Expo", "HOH - Southwest", "HOH - Chip", "HOH - Shake", "HOH - Grill"]
    for idx, role in enumerate(kitchen_roles):
        emp = Employee(full_name=f"Kitchen {idx}", roles=role, desired_hours=32, status="active")
        session.add(emp)
        session.flush()
        employees.append(emp.id)
    # Cashier.
    cashier = Employee(full_name="Cashier", roles="Cashier", desired_hours=28, status="active")
    session.add(cashier)
    session.flush()
    employees.append(cashier.id)
    session.commit()
    return employees


def _seed_unavailability(session, employee_ids: Iterable[int]) -> None:
    for emp_id in employee_ids:
        entry = EmployeeUnavailability(
            employee_id=emp_id,
            day_of_week=6,
            start_time=datetime.time(0, 0),
            end_time=datetime.time(6, 0),
        )
        session.add(entry)
    session.commit()


def _seed_sales(session) -> datetime.date:
    iso_year, iso_week, _ = datetime.date.today().isocalendar()
    week_start = datetime.date.fromisocalendar(iso_year, iso_week, 1)
    week = get_or_create_week(session, week_start)
    ctx = get_or_create_week_context(session, week.iso_year, week.iso_week, week.label)
    values: Dict[int, Dict[str, float]] = {}
    for day_idx in range(7):
        values[day_idx] = {"projected_sales_amount": 1200.0, "projected_notes": "{}"}
    save_week_daily_projection_values(session, ctx.id, values)
    session.commit()
    return week_start


def _find_shifts(shifts, role_contains: str, day: datetime.date):
    role_lower = role_contains.lower()
    return [
        s
        for s in shifts
        if role_lower in (s.role or "").lower()
        and s.start.date() in {day, day + datetime.timedelta(days=1)}
    ]


def test_bartender_opener_is_fixed_and_continuous():
    SessionLocal, EmployeeSessionLocal = _setup_engines()
    session = SessionLocal()
    employee_session = EmployeeSessionLocal()
    reset_wages_to_defaults()
    _seed_policy(db.SessionLocal)
    ids = _seed_employees(employee_session)
    _seed_unavailability(employee_session, ids)
    week_start = _seed_sales(session)

    result = generate_schedule_for_week(
        db.SessionLocal,
        week_start,
        actor="bartender-test",
        employee_session_factory=EmployeeSessionLocal,
        max_attempts=1,
    )
    assert result.get("shifts_created", 0) > 0
    # Pull raw shifts for Monday using a fresh session to avoid stale caches.
    session.close()
    session = SessionLocal()
    shifts = list(session.execute(select(Shift)).scalars())
    monday = week_start
    opener_shifts = [s for s in _find_shifts(shifts, "Bartender - Opener", monday) if s.start.date() == monday]
    assert len(opener_shifts) == 1, "Expected exactly one bartender opener"
    opener = opener_shifts[0]
    assert opener.start.time() == datetime.time(10, 30)
    assert opener.end.time() == datetime.time(11, 0)
    # The same employee should immediately work the AM bartender shift starting at 11:00.
    am_shifts = _find_shifts(shifts, "Bartender", monday)
    follow = [s for s in am_shifts if s.employee_id == opener.employee_id and s.start.time() >= datetime.time(11, 0)]
    assert follow, "Opener must roll into an AM bartender shift"
    am = sorted(follow, key=lambda s: s.start)[0]
    assert am.start.time() == datetime.time(11, 0)
    assert am.start.date() == monday


def test_bartender_opener_rolls_into_am_all_week():
    SessionLocal, EmployeeSessionLocal = _setup_engines()
    session = SessionLocal()
    employee_session = EmployeeSessionLocal()
    reset_wages_to_defaults()
    _seed_policy(db.SessionLocal)
    ids = _seed_employees(employee_session)
    _seed_unavailability(employee_session, ids)
    week_start = _seed_sales(session)

    result = generate_schedule_for_week(
        db.SessionLocal,
        week_start,
        actor="bartender-test",
        employee_session_factory=EmployeeSessionLocal,
        max_attempts=1,
    )
    assert result.get("shifts_created", 0) > 0
    session.close()
    session = SessionLocal()
    shifts = list(session.execute(select(Shift)).scalars())

    for offset in range(7):
        day = week_start + datetime.timedelta(days=offset)
        opener_shifts = [
            s for s in shifts if (s.role or "").lower().startswith("bartender - opener") and s.start.date() == day
        ]
        assert opener_shifts, f"Expected a bartender opener on {day}"
        opener = opener_shifts[0]
        am_shifts = _find_shifts(shifts, "Bartender", day)
        follow = [
            s
            for s in am_shifts
            if s.employee_id == opener.employee_id and s.start.date() == day and s.start.time() >= datetime.time(11, 0)
        ]
        assert follow, f"Opener must roll into AM bartender shift on {day}"


def test_bartender_opener_does_not_close_same_day_and_hours_cap():
    SessionLocal, EmployeeSessionLocal = _setup_engines()
    session = SessionLocal()
    employee_session = EmployeeSessionLocal()
    reset_wages_to_defaults()
    _seed_policy(db.SessionLocal)
    ids = _seed_employees(employee_session)
    _seed_unavailability(employee_session, ids)
    week_start = _seed_sales(session)

    result = generate_schedule_for_week(
        db.SessionLocal,
        week_start,
        actor="bartender-test",
        employee_session_factory=EmployeeSessionLocal,
        max_attempts=1,
    )
    assert result.get("shifts_created", 0) > 0
    session.close()
    session = SessionLocal()
    shifts = list(session.execute(select(Shift)).scalars())

    for offset in range(7):
        day = week_start + datetime.timedelta(days=offset)
        day_shifts = [s for s in shifts if s.start.date() == day and "bartender" in (s.role or "").lower()]
        opener_ids = {
            s.employee_id for s in day_shifts if (s.role or "").lower().startswith("bartender - opener")
        }
        closer_ids = {s.employee_id for s in day_shifts if "close" in (s.location or "").lower()}
        assert not (opener_ids & closer_ids), f"Bartender opener should not also close on {day}"
        hours_by_emp = {}
        for s in day_shifts:
            duration = (s.end - s.start).total_seconds() / 3600
            hours_by_emp[s.employee_id] = hours_by_emp.get(s.employee_id, 0.0) + duration
        for emp_id, total_hours in hours_by_emp.items():
            non_close_hours = sum(
                (s.end - s.start).total_seconds() / 3600
                for s in day_shifts
                if s.employee_id == emp_id and (s.location or "").lower() not in {"close"}
            )
            if non_close_hours > 0:
                assert total_hours <= 9.0, f"Employee {emp_id} over daily cap with non-close work on {day}"


def test_bartender_closer_no_redundant_leadin_when_full_pm():
    SessionLocal, EmployeeSessionLocal = _setup_engines()
    session = SessionLocal()
    employee_session = EmployeeSessionLocal()
    reset_wages_to_defaults()
    _seed_policy(db.SessionLocal)
    ids = _seed_employees(employee_session)
    _seed_unavailability(employee_session, ids)
    week_start = _seed_sales(session)

    result = generate_schedule_for_week(
        db.SessionLocal,
        week_start,
        actor="bartender-test",
        employee_session_factory=EmployeeSessionLocal,
        max_attempts=1,
    )
    assert result.get("shifts_created", 0) > 0
    session.close()
    session = SessionLocal()
    shifts = list(session.execute(select(Shift)).scalars())

    for offset in range(7):
        day = week_start + datetime.timedelta(days=offset)
        closers = [
            s for s in shifts if (s.role or "").lower().startswith("bartender") and (s.location or "").lower() == "close"
        ]
        for closer in closers:
            duration = (closer.end - closer.start).total_seconds() / 3600
            if duration >= 6.0:
                leadins = [
                    s
                    for s in shifts
                    if s.employee_id == closer.employee_id
                    and s.start.date() == closer.start.date()
                    and "auto lead-in for closer" in (s.notes or "").lower()
                ]
                assert not leadins, f"Should not create lead-in when closer already works full PM on {day}"

def test_bartender_closer_runs_pm_into_buffer():
    SessionLocal, EmployeeSessionLocal = _setup_engines()
    session = SessionLocal()
    employee_session = EmployeeSessionLocal()
    reset_wages_to_defaults()
    _seed_policy(db.SessionLocal)
    ids = _seed_employees(employee_session)
    _seed_unavailability(employee_session, ids)
    week_start = _seed_sales(session)

    result = generate_schedule_for_week(
        db.SessionLocal,
        week_start,
        actor="bartender-test",
        employee_session_factory=EmployeeSessionLocal,
        max_attempts=1,
    )
    assert result.get("shifts_created", 0) > 0
    session.close()
    session = SessionLocal()
    shifts = list(session.execute(select(Shift)).scalars())
    monday = week_start
    buffer_shifts = _find_shifts(shifts, "Bartender - Closer", monday)
    assert buffer_shifts, "Closer buffer shift missing"
    buffer = buffer_shifts[0]
    closer_core = [
        s
        for s in shifts
        if s.employee_id == buffer.employee_id
        and (s.role or "").lower().startswith("bartender")
        and s.start.date() == monday
        and s.end.time() == buffer.start.time()
    ]
    assert closer_core, "Closer must work PM shift into the buffer"
    pm = closer_core[0]
    assert pm.start.time() >= datetime.time(15, 0), "Closer PM shift should start in the afternoon"
    assert pm.end == buffer.start, "Closer PM shift must end exactly at close time"


def test_bartender_close_buffer_does_not_create_extra_closer():
    SessionLocal, EmployeeSessionLocal = _setup_engines()
    session = SessionLocal()
    employee_session = EmployeeSessionLocal()
    reset_wages_to_defaults()
    _seed_policy(db.SessionLocal)
    ids = _seed_employees(employee_session)
    _seed_unavailability(employee_session, ids)
    week_start = _seed_sales(session)

    result = generate_schedule_for_week(
        db.SessionLocal,
        week_start,
        actor="bartender-test",
        employee_session_factory=EmployeeSessionLocal,
        max_attempts=1,
    )
    assert result.get("shifts_created", 0) > 0
    session.close()
    session = SessionLocal()
    shifts = list(session.execute(select(Shift)).scalars())
    monday = week_start
    closer_shifts = sorted(
        _find_shifts(shifts, "Bartender - Closer", monday), key=lambda s: (s.start, s.end)
    )
    assert len(closer_shifts) == 1, "Only the close buffer should exist for bartender closer role"
    buffer = closer_shifts[0]
    duration = buffer.end - buffer.start
    assert duration <= datetime.timedelta(hours=1), "Buffer should be a short close-out segment, not a PM shift"
