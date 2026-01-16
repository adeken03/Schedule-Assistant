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
from roles import role_group  # noqa: E402
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
    with session_factory() as session:
        upsert_policy(session, policy.get("name", "Default Policy"), policy, edited_by="test")


def _seed_employees(session) -> None:
    bartender_roles = "Bartender,Bartender - Opener,Bartender - Closer"
    for idx in range(8):
        session.add(
            Employee(
                full_name=f"Auto Bartender {idx}",
                roles=bartender_roles,
                desired_hours=60,
                status="active",
            )
        )

    for idx in range(4):
        session.add(
            Employee(
                full_name=f"Auto Cashier {idx}",
                roles="Cashier",
                desired_hours=60,
                status="active",
            )
        )

    kitchen_roles = "HOH - Opener,HOH - Expo,HOH - Southwest,HOH - Chip,HOH - Shake,HOH - Grill"
    for idx in range(12):
        session.add(
            Employee(
                full_name=f"Auto Kitchen {idx}",
                roles=kitchen_roles,
                desired_hours=60,
                status="active",
            )
        )

    server_roles = (
        "Server - Opener,Server - Dining,Server - Cocktail,"
        "Server - Dining Preclose,Server - Dining Closer,"
        "Server - Cocktail Preclose,Server - Cocktail Closer"
    )
    for idx in range(25):
        session.add(
            Employee(
                full_name=f"Auto Server {idx}",
                roles=server_roles,
                desired_hours=60,
                status="active",
            )
        )
    session.commit()


def _seed_employees_sparse_kitchen_expo(session) -> None:
    bartender_roles = "Bartender,Bartender - Opener,Bartender - Closer"
    for idx in range(8):
        session.add(
            Employee(
                full_name=f"Sparse Bartender {idx}",
                roles=bartender_roles,
                desired_hours=60,
                status="active",
            )
        )

    for idx in range(4):
        session.add(
            Employee(
                full_name=f"Sparse Cashier {idx}",
                roles="Cashier",
                desired_hours=60,
                status="active",
            )
        )

    session.add(
        Employee(
            full_name="Sparse Expo Opener",
            roles="HOH - Opener,HOH - Expo",
            desired_hours=60,
            status="active",
        )
    )
    for idx in range(8):
        session.add(
            Employee(
                full_name=f"Sparse Kitchen {idx}",
                roles="HOH - All Roles",
                desired_hours=60,
                status="active",
            )
        )

    server_roles = (
        "Server - Opener,Server - Dining,Server - Cocktail,"
        "Server - Dining Preclose,Server - Dining Closer,"
        "Server - Cocktail Preclose,Server - Cocktail Closer"
    )
    for idx in range(25):
        session.add(
            Employee(
                full_name=f"Sparse Server {idx}",
                roles=server_roles,
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


def test_open_and_close_buffers_have_continuity_assignments() -> None:
    SessionLocal, EmployeeSessionLocal = _setup_engines()
    schedule_session = SessionLocal()
    employee_session = EmployeeSessionLocal()

    reset_wages_to_defaults()
    _seed_policy(SessionLocal)
    _seed_employees(employee_session)

    week_start = datetime.date(2025, 6, 2)  # Monday
    _seed_sales(schedule_session, week_start, amount=9000.0)

    result = generate_schedule_for_week(
        SessionLocal,
        week_start,
        actor="open-close-continuity-test",
        employee_session_factory=EmployeeSessionLocal,
        max_attempts=1,
    )
    assert result.get("shifts_created", 0) > 0

    shifts = list(schedule_session.execute(select(Shift)).scalars())
    tolerance = datetime.timedelta(minutes=10)

    open_shifts = [shift for shift in shifts if (shift.location or "").strip().lower() == "open"]
    assert open_shifts, "Expected at least one opener buffer shift"
    for opener in open_shifts:
        assert opener.employee_id is not None, f"Opener shift unassigned: {opener.role}"
        group = role_group(opener.role)
        has_followup = any(
            candidate is not opener
            and candidate.employee_id == opener.employee_id
            and role_group(candidate.role) == group
            and (candidate.location or "").strip().lower() != "open"
            and abs(candidate.start - opener.end) <= tolerance
            for candidate in shifts
        )
        assert has_followup, f"{opener.role} opener should have a same-group follow-up shift"

    close_buffers = [shift for shift in shifts if "close buffer" in (shift.notes or "").lower()]
    assert close_buffers, "Expected at least one close-buffer shift"
    for buffer_shift in close_buffers:
        assert buffer_shift.employee_id is not None, f"Close-buffer shift unassigned: {buffer_shift.role}"
        group = role_group(buffer_shift.role)
        has_leadin = any(
            candidate is not buffer_shift
            and candidate.employee_id == buffer_shift.employee_id
            and role_group(candidate.role) == group
            and abs(candidate.end - buffer_shift.start) <= tolerance
            for candidate in shifts
        )
        assert has_leadin, f"{buffer_shift.role} close buffer should have a same-group lead-in shift"


def test_server_opener_gets_11am_am_shift_and_is_first_out() -> None:
    SessionLocal, EmployeeSessionLocal = _setup_engines()
    schedule_session = SessionLocal()
    employee_session = EmployeeSessionLocal()

    reset_wages_to_defaults()
    _seed_policy(SessionLocal)
    _seed_employees(employee_session)

    week_start = datetime.date(2025, 6, 2)  # Monday
    _seed_sales(schedule_session, week_start, amount=9000.0)

    result = generate_schedule_for_week(
        SessionLocal,
        week_start,
        actor="server-opener-continuity-test",
        employee_session_factory=EmployeeSessionLocal,
        max_attempts=1,
    )
    assert result.get("shifts_created", 0) > 0

    shifts = list(schedule_session.execute(select(Shift)).scalars())
    tolerance = datetime.timedelta(minutes=10)

    server_openers = [
        shift
        for shift in shifts
        if (shift.location or "").strip().lower() == "open" and (shift.role or "").strip() == "Server - Opener"
    ]
    assert server_openers, "Expected a Server - Opener open-buffer shift"

    for opener in server_openers:
        assert opener.employee_id is not None
        op_date = opener.start.date()
        followups = [
            shift
            for shift in shifts
            if shift.employee_id == opener.employee_id
            and role_group(shift.role) == "Servers"
            and (shift.location or "").strip().upper() == "AM"
            and abs(shift.start - opener.end) <= tolerance
        ]
        assert followups, "Server opener should have an AM follow-up shift starting at open"
        followup = sorted(followups, key=lambda s: s.start)[0]
        assert followup.start.time() == datetime.time(11, 0)

        server_am = [
            shift
            for shift in shifts
            if role_group(shift.role) == "Servers"
            and (shift.location or "").strip().upper() == "AM"
            and shift.start.date() == op_date
        ]
        other_ends = [shift.end for shift in server_am if shift.id != followup.id]
        assert other_ends, "Expected additional AM server shifts"
        earliest_other_end = min(other_ends)
        assert followup.end <= earliest_other_end + tolerance


def test_sparse_expo_still_assigns_expo_closer_and_buffers() -> None:
    SessionLocal, EmployeeSessionLocal = _setup_engines()
    schedule_session = SessionLocal()
    employee_session = EmployeeSessionLocal()

    reset_wages_to_defaults()
    _seed_policy(SessionLocal)
    _seed_employees_sparse_kitchen_expo(employee_session)

    week_start = datetime.date(2025, 6, 2)  # Monday
    _seed_sales(schedule_session, week_start, amount=9000.0)

    result = generate_schedule_for_week(
        SessionLocal,
        week_start,
        actor="sparse-expo-test",
        employee_session_factory=EmployeeSessionLocal,
        max_attempts=1,
    )
    assert result.get("shifts_created", 0) > 0

    shifts = list(schedule_session.execute(select(Shift)).scalars())

    expo_close_buffers = [shift for shift in shifts if "expo close buffer" in (shift.notes or "").lower()]
    assert expo_close_buffers, "Expected HOH expo close-buffer shifts"
    assert all(shift.employee_id is not None for shift in expo_close_buffers), "Expo close-buffer shifts must be assigned"

    by_employee = {}
    for shift in shifts:
        if not shift.employee_id:
            continue
        by_employee.setdefault(shift.employee_id, []).append(shift)

    for employee_id, bucket in by_employee.items():
        ordered = sorted(bucket, key=lambda sh: sh.start)
        for left, right in zip(ordered, ordered[1:]):
            assert left.end <= right.start, f"Employee {employee_id} has overlapping shifts"


def test_hoh_expo_opener_and_closer_pairing() -> None:
    SessionLocal, EmployeeSessionLocal = _setup_engines()
    schedule_session = SessionLocal()
    employee_session = EmployeeSessionLocal()

    reset_wages_to_defaults()
    _seed_policy(SessionLocal)
    _seed_employees(employee_session)

    week_start = datetime.date(2025, 6, 2)  # Monday
    _seed_sales(schedule_session, week_start, amount=9000.0)

    result = generate_schedule_for_week(
        SessionLocal,
        week_start,
        actor="hoh-open-close-pairing-test",
        employee_session_factory=EmployeeSessionLocal,
        max_attempts=1,
    )
    assert result.get("shifts_created", 0) > 0

    shifts = list(schedule_session.execute(select(Shift)).scalars())
    tolerance = datetime.timedelta(minutes=10)

    hoh_openers = [
        shift
        for shift in shifts
        if (shift.role or "").strip() == "HOH - Opener" and (shift.location or "").strip().lower() == "open"
    ]
    assert len(hoh_openers) >= 7, "Expected a HOH opener buffer shift for each day"

    for opener in hoh_openers:
        assert opener.employee_id is not None
        expo_am = [
            shift
            for shift in shifts
            if shift.employee_id == opener.employee_id
            and (shift.role or "").strip() == "HOH - Expo"
            and (shift.location or "").strip().upper() == "AM"
            and abs(shift.start - opener.end) <= tolerance
        ]
        assert expo_am, "HOH opener buffer should pair into an AM Expo shift"
        followup = sorted(expo_am, key=lambda s: s.start)[0]
        assert followup.start.time() == datetime.time(11, 0)
        assert followup.end.time() == datetime.time(16, 0)

    expo_pm_shifts = [
        shift
        for shift in shifts
        if (shift.role or "").strip() == "HOH - Expo" and (shift.location or "").strip().upper() == "PM"
    ]
    assert len(expo_pm_shifts) >= 7, "Expected a PM Expo shift for each day"

    for expo_pm in expo_pm_shifts:
        assert expo_pm.employee_id is not None
        buffers = [
            shift
            for shift in shifts
            if shift.employee_id == expo_pm.employee_id
            and (shift.role or "").strip() == "HOH - Expo"
            and "close buffer" in (shift.notes or "").lower()
            and abs(shift.start - expo_pm.end) <= tolerance
        ]
        assert buffers, "PM Expo should have a post-close buffer shift"
        buffer_shift = sorted(buffers, key=lambda s: s.start)[0]
        assert (buffer_shift.end - buffer_shift.start) <= datetime.timedelta(minutes=35)
