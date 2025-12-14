from __future__ import annotations

import datetime
import json
import sys
from pathlib import Path

import pytest
from sqlalchemy import delete, select, create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

APP_DIR = Path(__file__).resolve().parents[1] / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import database as db  # noqa: E402
from database import (  # noqa: E402
    Base,
    Employee,
    EmployeeBase,
    EmployeeUnavailability,
    ProjectionsBase,
    Policy,
    Shift,
    WeekContext,
    WeekDailyProjection,
    WeekProjectionContext,
    get_or_create_week,
    get_or_create_week_context,
    get_week_daily_projections,
)
from data_exchange import (  # noqa: E402
    copy_week_dataset,
    export_employees,
    export_policy_dataset,
    export_role_wages_dataset,
    export_week_modifiers,
    export_week_projections,
    export_week_schedule,
    import_employees,
    import_policy_dataset,
    import_role_wages_dataset,
    import_week_modifiers,
    import_week_projections,
    import_week_schedule,
)
import wages  # noqa: E402


@pytest.fixture()
def memory_db(monkeypatch, tmp_path):
    """Single in-memory engine shared across schedule/employee/projection tables for exports."""
    schedule_engine = create_engine(
        "sqlite://",
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    projection_engine = create_engine(
        "sqlite://",
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Session = sessionmaker(bind=schedule_engine, expire_on_commit=False, future=True)
    ProjectionSession = sessionmaker(bind=projection_engine, expire_on_commit=False, future=True)

    # Point the database module at the in-memory engines so helper functions use them.
    monkeypatch.setattr(db, "schedule_engine", schedule_engine)
    monkeypatch.setattr(db, "employee_engine", schedule_engine)
    monkeypatch.setattr(db, "policy_engine", schedule_engine)
    monkeypatch.setattr(db, "projections_engine", projection_engine)
    monkeypatch.setattr(db, "SessionLocal", Session)
    monkeypatch.setattr(db, "EmployeeSessionLocal", Session)
    monkeypatch.setattr(db, "PolicySessionLocal", Session)
    monkeypatch.setattr(db, "ProjectionSessionLocal", ProjectionSession)

    Base.metadata.create_all(schedule_engine)
    EmployeeBase.metadata.create_all(schedule_engine)
    ProjectionsBase.metadata.create_all(projection_engine)

    # Keep exports in a temp folder to avoid polluting the repo.
    monkeypatch.setattr("data_exchange.EXPORT_DIR", tmp_path, raising=False)
    monkeypatch.setattr(wages, "DATA_DIR", tmp_path, raising=False)
    monkeypatch.setattr(wages, "WAGES_FILE", tmp_path / "role_wages.json", raising=False)

    session = Session()
    employee_session = Session()
    projection_session = ProjectionSession()

    try:
        yield {
            "session": session,
            "employee_session": employee_session,
            "projection_session": projection_session,
            "engine": schedule_engine,
            "tmp": tmp_path,
        }
    finally:
        session.close()
        employee_session.close()
        projection_session.close()
        schedule_engine.dispose()
        projection_engine.dispose()


def _seed_week(session) -> WeekContext:
    week = get_or_create_week(session, datetime.date(2024, 4, 1))
    iso_year, iso_week, _ = week.week_start_date.isocalendar()
    context = get_or_create_week_context(session, iso_year, iso_week, week.label)
    week.context_id = context.id
    session.commit()
    return context


def test_employee_export_import_round_trip(memory_db) -> None:
    session = memory_db["session"]
    employee_session = memory_db["employee_session"]

    employee = Employee(full_name="Test User", desired_hours=30, status="active", notes="note")
    employee.role_list = ["Server - Dining"]
    employee_session.add(employee)
    employee_session.flush()
    employee_session.add(
        EmployeeUnavailability(
            employee_id=employee.id,
            day_of_week=1,
            start_time=datetime.time(9, 0),
            end_time=datetime.time(12, 0),
        )
    )
    employee_session.commit()

    export_path = export_employees(employee_session)
    # Drop employees to ensure import recreates them.
    employee_session.execute(delete(EmployeeUnavailability))
    employee_session.execute(delete(Employee))
    employee_session.commit()

    created, updated = import_employees(employee_session, export_path)

    employees = employee_session.scalars(select(Employee)).all()
    assert created == 1
    assert updated == 0
    assert len(employees) == 1
    assert employees[0].full_name == "Test User"
    assert employees[0].role_list == ["Server - Dining"]
    availability = employee_session.scalars(select(EmployeeUnavailability)).all()
    assert len(availability) == 1
    assert availability[0].day_of_week == 1


def test_projection_export_import_preserves_notes(memory_db) -> None:
    session = memory_db["session"]
    projection_session = memory_db["projection_session"]
    week = _seed_week(session)
    values = {
        0: {"projected_sales_amount": 500.0, "projected_notes": "AM push"},
        1: {"projected_sales_amount": 100.0, "projected_notes": ""},
    }
    db.save_week_daily_projection_values(session, week.id, values, projection_session=projection_session)

    export_path = export_week_projections(session, week)
    # Clear projections to force re-import
    projection_session.execute(delete(WeekDailyProjection))
    projection_session.commit()

    imported = import_week_projections(session, week, export_path)
    ctx_id = projection_session.scalar(
        select(WeekProjectionContext.id).where(WeekProjectionContext.schedule_context_id == week.id)
    )
    projections = projection_session.scalars(
        select(WeekDailyProjection).where(WeekDailyProjection.projection_context_id == ctx_id)
    ).all()

    assert imported == 7
    notes = {row.day_of_week: row.projected_notes for row in projections}
    assert notes[0] == "AM push"
    assert notes[1] == ""


def test_modifier_export_import_round_trip(memory_db) -> None:
    session = memory_db["session"]
    week = _seed_week(session)
    modifier = db.Modifier(
        week_id=week.id,
        title="Event",
        modifier_type="increase",
        day_of_week=5,
        start_time=datetime.time(18, 0),
        end_time=datetime.time(20, 0),
        pct_change=15,
        notes="Big game",
        created_by="tester",
    )
    session.add(modifier)
    session.commit()

    export_path = export_week_modifiers(session, week)
    session.execute(delete(db.Modifier))
    session.commit()

    added = import_week_modifiers(session, week, export_path, created_by="tester")
    stored = session.scalars(select(db.Modifier)).all()

    assert added == 1
    assert len(stored) == 1
    assert stored[0].title == "Event"
    assert stored[0].notes == "Big game"


def test_shift_export_import_with_employee_names(memory_db) -> None:
    session = memory_db["session"]
    employee_session = memory_db["employee_session"]
    week_start = datetime.date(2024, 4, 1)
    week = get_or_create_week(session, week_start)
    employee = Employee(full_name="Closer", desired_hours=20, status="active", notes="")
    employee.role_list = ["Server - Dining"]
    employee_session.add(employee)
    employee_session.commit()

    shift = Shift(
        week_id=week.id,
        employee_id=employee.id,
        role="Server - Dining",
        start=datetime.datetime(2024, 4, 1, 16, 0),
        end=datetime.datetime(2024, 4, 1, 22, 0),
        location="Mid",
        notes="Primary",
        status="draft",
        labor_rate=15.0,
        labor_cost=90.0,
    )
    session.add(shift)
    session.commit()

    export_path = export_week_schedule(session, week_start, employee_session=employee_session)
    session.execute(delete(Shift))
    session.commit()

    imported = import_week_schedule(session, week_start, export_path, employee_session=employee_session)
    stored = session.scalars(select(Shift)).all()

    assert imported == 1
    assert len(stored) == 1
    assert stored[0].employee_id == employee.id
    assert stored[0].notes == "Primary"


def test_copy_week_dataset_duplicates_projections(memory_db) -> None:
    session = memory_db["session"]
    projection_session = memory_db["projection_session"]
    source_week = _seed_week(session)
    target_week = get_or_create_week_context(session, 2024, 2, "2024 W02")
    db.save_week_daily_projection_values(
        session,
        source_week.id,
        {0: {"projected_sales_amount": 200.0, "projected_notes": "carry"}},
        projection_session=projection_session,
    )

    summary = copy_week_dataset(
        session,
        source_week,
        target_week,
        "projections",
        actor="tester",
        employee_session=memory_db["employee_session"],
    )

    target_ctx_id = projection_session.scalar(
        select(WeekProjectionContext.id).where(WeekProjectionContext.schedule_context_id == target_week.id)
    )
    target_rows = projection_session.scalars(
        select(WeekDailyProjection).where(WeekDailyProjection.projection_context_id == target_ctx_id)
    ).all()
    assert summary == {"projections": 7}
    assert target_rows[0].projected_sales_amount == 200.0
    assert target_rows[0].projected_notes == "carry"


def test_copy_week_dataset_copies_shifts_when_ids_diverge(memory_db) -> None:
    session = memory_db["session"]

    # Pre-create WeekContext rows before WeekSchedule so the autoincrement ids diverge.
    get_or_create_week_context(session, 2024, 1, "2024 W01")
    source_ctx = get_or_create_week_context(session, 2024, 2, "2024 W02")

    week_start = datetime.date.fromisocalendar(2024, 2, 1)
    source_schedule = get_or_create_week(session, week_start)
    assert source_schedule.context_id == source_ctx.id

    session.add(
        Shift(
            week_id=source_schedule.id,
            employee_id=None,
            role="Server - Dining",
            start=datetime.datetime(2024, 1, 8, 16, 0),
            end=datetime.datetime(2024, 1, 8, 22, 0),
            location="Mid",
            notes="Copy me",
            status="draft",
            labor_rate=15.0,
            labor_cost=90.0,
        )
    )
    session.commit()

    target_ctx = get_or_create_week_context(session, 2024, 3, "2024 W03")
    summary = copy_week_dataset(session, source_ctx, target_ctx, "shifts", actor="tester")

    target_start = datetime.date.fromisocalendar(2024, 3, 1)
    target_schedule = get_or_create_week(session, target_start)
    imported = session.scalars(select(Shift).where(Shift.week_id == target_schedule.id)).all()

    assert summary == {"shifts": 1}
    assert len(imported) == 1
    assert imported[0].role == "Server - Dining"
    assert imported[0].notes == "Copy me"


def test_role_wages_export_import_round_trip(memory_db) -> None:
    # Seed a known wage value, export, clobber, and import back.
    wages.reset_wages_to_defaults()
    data = wages.load_wages()
    data["Server - Dining"]["wage"] = 22.5
    data["Server - Dining"]["confirmed"] = True
    wages.save_wages(data)

    export_path = export_role_wages_dataset()

    corrupted = wages.load_wages()
    corrupted["Server - Dining"]["wage"] = 0.0
    corrupted["Server - Dining"]["confirmed"] = False
    wages.save_wages(corrupted)

    imported = import_role_wages_dataset(export_path)
    restored = wages.load_wages()

    assert imported > 0
    assert restored["Server - Dining"]["wage"] == 22.5
    assert restored["Server - Dining"]["confirmed"] is True


def test_policy_export_import_round_trip(memory_db) -> None:
    session = memory_db["session"]
    policy_payload = {"description": "export me", "roles": {"Server": {"hourly_wage": 15}}}
    upsert_policy = db.upsert_policy  # type: ignore[attr-defined]

    upsert_policy(session, "Baseline", policy_payload, edited_by="tester")
    export_path = export_policy_dataset(session)

    session.execute(delete(Policy))
    session.commit()

    imported = import_policy_dataset(session, export_path, edited_by="tester")
    reloaded = db.get_active_policy(session)

    assert imported.name == "Baseline"
    assert reloaded is not None
    assert reloaded.params_dict().get("description") == "export me"
