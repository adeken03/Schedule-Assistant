from __future__ import annotations

import datetime
import sys
import unittest

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1] / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from database import (  # noqa: E402
    Base,
    EmployeeBase,
    PolicyBase,
    ProjectionsBase,
    Employee,
    EmployeeUnavailability,
    Shift,
    get_or_create_week,
    upsert_policy,
)
import database as db  # noqa: E402
from policy import build_default_policy  # noqa: E402
from validation import validate_week_schedule  # noqa: E402

UTC = datetime.timezone.utc


class ScheduleValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.schedule_engine = create_engine("sqlite:///:memory:", future=True)
        Base.metadata.create_all(self.schedule_engine)
        self.employee_engine = create_engine("sqlite:///:memory:", future=True)
        EmployeeBase.metadata.create_all(self.employee_engine)
        self.projection_engine = create_engine("sqlite:///:memory:", future=True)
        ProjectionsBase.metadata.create_all(self.projection_engine)
        session_factory = sessionmaker(bind=self.schedule_engine, expire_on_commit=False, future=True)
        employee_session_factory = sessionmaker(bind=self.employee_engine, expire_on_commit=False, future=True)
        projection_session_factory = sessionmaker(bind=self.projection_engine, expire_on_commit=False, future=True)
        # Use the schedule engine for policies in tests to simplify table setup.
        db.policy_engine = self.schedule_engine
        db.PolicySessionLocal = session_factory
        db.projections_engine = self.projection_engine
        db.ProjectionSessionLocal = projection_session_factory
        PolicyBase.metadata.create_all(db.policy_engine)
        self.session = session_factory()
        self.employee_session = employee_session_factory()
        self.week_start = datetime.date(2024, 4, 1)
        upsert_policy(self.session, "Baseline", build_default_policy(), edited_by="tests")

    def tearDown(self) -> None:
        self.session.close()
        self.employee_session.close()
        self.schedule_engine.dispose()
        self.employee_engine.dispose()
        self.projection_engine.dispose()

    def test_reports_unavailability_conflicts(self) -> None:
        employee = self._add_employee("Server A", ["Server"])
        self._add_unavailability(employee, day=0, start="12:00", end="14:00")
        self._add_shift(
            role="Server",
            start=datetime.datetime(2024, 4, 1, 12, 30, tzinfo=UTC),
            end=datetime.datetime(2024, 4, 1, 15, 0, tzinfo=UTC),
            employee=employee,
        )
        report = validate_week_schedule(self.session, self.week_start, employee_session=self.employee_session)

        self.assertTrue(any(issue["type"] == "availability" for issue in report["issues"]))

    def test_flags_missing_required_openers(self) -> None:
        # Schedule only kitchen staff so the server opener anchor fails.
        self._add_shift(
            role="Kitchen Opener",
            start=datetime.datetime(2024, 4, 1, 10, 0, tzinfo=UTC),
            end=datetime.datetime(2024, 4, 1, 18, 0, tzinfo=UTC),
            employee=None,
        )

        report = validate_week_schedule(self.session, self.week_start, employee_session=self.employee_session)

        self.assertTrue(
            any(issue["type"] == "coverage" and issue.get("group") == "Servers" for issue in report["issues"])
        )

    def test_warns_when_server_dining_exceeds_limit(self) -> None:
        employees = [self._add_employee(f"Server {idx}", ["Server - Dining"]) for idx in range(7)]
        for employee in employees:
            self._add_shift(
                role="Server - Dining",
                start=datetime.datetime(2024, 4, 1, 16, 0, tzinfo=UTC),
                end=datetime.datetime(2024, 4, 1, 21, 0, tzinfo=UTC),
                employee=employee,
            )

        report = validate_week_schedule(self.session, self.week_start, employee_session=self.employee_session)

        warnings = [warning for warning in report["warnings"] if warning["type"] == "concurrency"]
        self.assertTrue(any(warning.get("role") == "Server - Dining" for warning in warnings))

    def test_warns_when_employee_exceeds_40_hours(self) -> None:
        employee = self._add_employee("Full Time Worker", ["Server"])
        # Add shifts totaling 50 hours across the week (exceeds default 48-hour limit)
        shifts_hours = [8, 8, 8, 8, 8, 10]
        start_day = datetime.datetime(2024, 4, 1, 10, 0, tzinfo=UTC)
        for day_offset, hours in enumerate(shifts_hours):
            self._add_shift(
                role="Server",
                start=start_day + datetime.timedelta(days=day_offset),
                end=start_day + datetime.timedelta(days=day_offset, hours=hours),
                employee=employee,
            )

        report = validate_week_schedule(self.session, self.week_start, employee_session=self.employee_session)

        hours_warnings = [w for w in report["warnings"] if w["type"] == "weekly_hours"]
        self.assertEqual(len(hours_warnings), 1)
        self.assertEqual(hours_warnings[0]["employee_id"], employee.id)
        self.assertEqual(hours_warnings[0]["hours"], 50.0)
        self.assertEqual(hours_warnings[0]["limit"], 48)

    def test_no_warning_for_employee_at_48_hours(self) -> None:
        employee = self._add_employee("Full Time Worker", ["Server"])
        # Add shifts totaling exactly 48 hours (at policy limit, should not warn)
        shifts_hours = [8, 8, 8, 8, 8, 8]
        start_day = datetime.datetime(2024, 4, 1, 10, 0, tzinfo=UTC)
        for day_offset, hours in enumerate(shifts_hours):
            self._add_shift(
                role="Server",
                start=start_day + datetime.timedelta(days=day_offset),
                end=start_day + datetime.timedelta(days=day_offset, hours=hours),
                employee=employee,
            )

        report = validate_week_schedule(self.session, self.week_start, employee_session=self.employee_session)

        hours_warnings = [w for w in report["warnings"] if w["type"] == "weekly_hours"]
        self.assertEqual(len(hours_warnings), 0)

    def test_no_warning_for_employee_under_48_hours(self) -> None:
        employee = self._add_employee("Part Time Worker", ["Server"])
        # Add shifts totaling 30 hours (well under 48-hour policy limit)
        shifts_hours = [6, 6, 6, 6, 6, 0]
        start_day = datetime.datetime(2024, 4, 1, 10, 0, tzinfo=UTC)
        for day_offset, hours in enumerate(shifts_hours):
            if hours > 0:
                self._add_shift(
                    role="Server",
                    start=start_day + datetime.timedelta(days=day_offset),
                    end=start_day + datetime.timedelta(days=day_offset, hours=hours),
                    employee=employee,
                )

        report = validate_week_schedule(self.session, self.week_start, employee_session=self.employee_session)

        hours_warnings = [w for w in report["warnings"] if w["type"] == "weekly_hours"]
        self.assertEqual(len(hours_warnings), 0)

    def _add_employee(self, name: str, roles: list[str]) -> Employee:
        employee = Employee(full_name=name, roles=", ".join(roles), desired_hours=30, status="active", notes="")
        self.employee_session.add(employee)
        self.employee_session.commit()
        return employee

    def _add_unavailability(self, employee: Employee, *, day: int, start: str, end: str) -> None:
        start_time = datetime.time.fromisoformat(start)
        end_time = datetime.time.fromisoformat(end)
        window = EmployeeUnavailability(
            employee_id=employee.id,
            day_of_week=day,
            start_time=start_time,
            end_time=end_time,
        )
        self.employee_session.add(window)
        self.employee_session.commit()

    def _add_shift(
        self,
        *,
        role: str,
        start: datetime.datetime,
        end: datetime.datetime,
        employee: Employee | None,
    ) -> Shift:
        week = get_or_create_week(self.session, self.week_start)
        shift = Shift(
            week_id=week.id,
            employee_id=employee.id if employee else None,
            role=role,
            start=start,
            end=end,
            status="draft",
        )
        self.session.add(shift)
        self.session.commit()
        return shift


if __name__ == "__main__":
    unittest.main()
