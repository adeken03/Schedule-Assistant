from __future__ import annotations

import copy
import datetime
import sys
from pathlib import Path
import unittest
from unittest import mock

import pytest

from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

APP_DIR = Path(__file__).resolve().parents[1] / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from database import (  # noqa: E402
    Base,
    PolicyBase,
    EmployeeBase,
    ProjectionsBase,
    Employee,
    EmployeeUnavailability,
    Shift,
    get_or_create_week,
    get_or_create_week_context,
    get_week_daily_projections,
    save_week_daily_projection_values,
    shift_display_date,
)
import database as db  # noqa: E402
from generator.api import generate_schedule_for_week as api_generate_schedule_for_week  # noqa: E402
from generator.engine import BlockDemand, ScheduleGenerator  # noqa: E402
from policy import SHIFT_PRESET_DEFAULTS  # noqa: E402


class ScheduleGeneratorTests(unittest.TestCase):
    """Regression tests for the core scheduling heuristics."""

    def setUp(self) -> None:
        self.schedule_engine = create_engine("sqlite:///:memory:", future=True)
        Base.metadata.create_all(self.schedule_engine)
        self.employee_engine = create_engine("sqlite:///:memory:", future=True)
        EmployeeBase.metadata.create_all(self.employee_engine)
        self.projection_engine = create_engine("sqlite:///:memory:", future=True)
        ProjectionsBase.metadata.create_all(self.projection_engine)
        self.session_factory = sessionmaker(bind=self.schedule_engine, expire_on_commit=False, future=True)
        self.employee_session_factory = sessionmaker(bind=self.employee_engine, expire_on_commit=False, future=True)
        self.projection_session_factory = sessionmaker(bind=self.projection_engine, expire_on_commit=False, future=True)
        # For tests, keep policy DB on the same engine as schedule to simplify setup.
        db.policy_engine = self.schedule_engine
        db.PolicySessionLocal = self.session_factory
        db.projections_engine = self.projection_engine
        db.ProjectionSessionLocal = self.projection_session_factory

        self.session = self.session_factory()
        self.employee_session = self.employee_session_factory()
        self.week_start = datetime.date(2024, 4, 1)  # Monday

    def tearDown(self) -> None:
        self.session.close()
        self.employee_session.close()
        self.schedule_engine.dispose()
        self.employee_engine.dispose()
        self.projection_engine.dispose()

    @pytest.mark.skip(reason="Legacy generic expectations; superseded by BWW engine behavior")
    def test_respects_employee_unavailability(self) -> None:
        policy = self._policy(daily_boost={"Mon": 1})
        unavailable = self._add_employee(
            "Unavailable Server",
            ["Server"],
            desired_hours=30,
            unavailability=[(0, "09:00", "13:00")],
        )
        available = self._add_employee("Available Server", ["Server"], desired_hours=30)

        result = self._run_generator(policy)

        self.assertEqual(result["warnings"], [])
        monday_shifts = self._shifts_for_day(0)
        self.assertEqual(len(monday_shifts), 1)
        self.assertEqual(monday_shifts[0].employee_id, available.id)
        self.assertNotEqual(monday_shifts[0].employee_id, unavailable.id)

    @pytest.mark.skip(reason="Legacy generic expectations; superseded by BWW engine behavior")
    def test_respects_desired_hour_ceiling_when_capacity_exists(self) -> None:
        policy = self._policy(daily_boost={"Mon": 1, "Tue": 1, "Wed": 1, "Thu": 1})
        focused = self._add_employee("Core Coverage", ["Server"], desired_hours=8)
        flexible = self._add_employee("Flex Coverage", ["Server"], desired_hours=32)

        self._run_generator(policy)

        totals = self._hours_by_employee()
        ceiling = focused.desired_hours * policy["global"]["desired_hours_ceiling_pct"]
        self.assertIn(focused.id, totals)
        self.assertLessEqual(totals[focused.id], ceiling + 0.01)
        self.assertGreaterEqual(totals.get(flexible.id, 0.0), totals[focused.id] - 0.01)

    @pytest.mark.skip(reason="Legacy generic expectations; superseded by BWW engine behavior")
    def test_disallows_split_shifts_when_disabled(self) -> None:
        policy = self._policy(
            block_names=["Open", "Close"],
            daily_boost={"Mon": 1},
            global_overrides={"allow_split_shifts": False},
        )
        only_employee = self._add_employee("Solo Closer", ["Server"], desired_hours=40)

        result = self._run_generator(policy)

        monday_shifts = self._shifts_for_day(0)
        self.assertEqual(len(monday_shifts), 2)
        assigned = [shift for shift in monday_shifts if shift.employee_id == only_employee.id]
        unassigned = [shift for shift in monday_shifts if shift.employee_id is None]
        self.assertEqual(len(assigned), 1)
        self.assertEqual(len(unassigned), 1)
        self.assertTrue(any("No coverage" in warning for warning in result["warnings"]))

    def test_specialized_role_matches_base_role(self) -> None:
        policy = self._policy(role_name="Server - Dining", daily_boost={"Mon": 1})
        closer = self._add_employee("Dining Closer", ["Server - Dining Closer"], desired_hours=24)

        self._run_generator(policy)

        monday_ids = [shift.employee_id for shift in self._shifts_for_day(0)]
        self.assertIn(closer.id, monday_ids)

    @pytest.mark.skip(reason="Legacy generic expectations; superseded by BWW engine behavior")
    def test_threshold_rules_add_additional_staff(self) -> None:
        policy = self._policy(block_names=["Mid"], daily_boost={"Mon": 1})
        policy["roles"]["Server"]["blocks"]["Mid"]["base"] = 1
        policy["roles"]["Server"]["blocks"]["Mid"]["max"] = 2
        policy["roles"]["Server"]["thresholds"] = [{"metric": "demand_index", "gte": 0.5, "add": 1}]
        self._seed_sales({0: 1000.0, 1: 100.0})
        first = self._add_employee("Primary", ["Server"], desired_hours=40)
        second = self._add_employee("Support", ["Server"], desired_hours=40)

        self._run_generator(policy)

        monday = [shift for shift in self._shifts_for_day(0) if shift.role == "Server"]
        assigned = [shift for shift in monday if shift.employee_id in {first.id, second.id}]
        self.assertGreaterEqual(len(assigned), 2)

    @pytest.mark.skip(reason="Legacy generic expectations; superseded by BWW engine behavior")
    def test_open_block_limited_to_core_roles(self) -> None:
        block_windows = {
            "Open": ("10:30", "11:00"),
            "Mid": ("11:00", "16:00"),
            "PM": ("16:00", "21:00"),
            "Close": ("21:00", "21:35"),
        }
        policy = self._policy_template(block_windows, ["Kitchen Opener", "Bartender", "Server - Dining", "Cashier"])
        for role in policy["roles"].values():
            role["blocks"]["Open"]["base"] = 1
            role["blocks"]["Open"]["min"] = 1
            role["blocks"]["Open"]["max"] = 1
        self._add_employee("Kitchen Lead", ["Kitchen Opener"], desired_hours=40)
        self._add_employee("Morning Bartender", ["Bartender"], desired_hours=40)
        self._add_employee("Lead Server", ["Server - Dining"], desired_hours=40)
        self._add_employee("Front Counter", ["Cashier"], desired_hours=40)

        self._run_generator(policy)

        monday_shifts = self._shifts_for_day(0)
        openers = [
            shift
            for shift in monday_shifts
            if shift.start.astimezone().time() < datetime.time(11, 0)
        ]
        self.assertEqual(len(openers), 3)
        self.assertSetEqual(
            {shift.role for shift in openers},
            {"Kitchen Opener", "Bartender", "Server - Dining"},
        )

    @pytest.mark.skip(reason="Legacy generic expectations; superseded by BWW engine behavior")
    def test_anchor_caps_limit_openers_and_closers(self) -> None:
        block_windows = {
            "Open": ("10:30", "11:00"),
            "PM": ("11:00", "22:00"),
            "Close": ("22:00", "22:35"),
        }
        roles = [
            "Server - Dining",
            "Server - Cocktail",
            "Bartender",
            "Kitchen Opener",
            "Kitchen Closer",
            "Cashier - To-Go Specialist",
        ]
        policy = self._policy_template(block_windows, roles)
        policy["global"]["open_buffer_minutes"] = 31
        policy["global"]["round_to_minutes"] = 5
        policy["business_hours"] = {
            day: {"open": "11:00", "mid": "16:00", "close": "23:00"}
            for day in ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        }

        for role in ["Server - Dining", "Server - Cocktail", "Bartender"]:
            policy["roles"][role]["blocks"]["Open"].update({"base": 2, "min": 2, "max": 3})
        policy["roles"]["Kitchen Opener"]["blocks"]["Open"].update({"base": 1, "min": 1, "max": 2})
        for role in ["Server - Dining", "Server - Cocktail", "Bartender", "Kitchen Closer", "Cashier - To-Go Specialist"]:
            cfg = policy["roles"][role]["blocks"].get("Close")
            if cfg:
                cfg.update({"base": 2, "min": 2, "max": 3})

        self._add_employee("Dining Lead", ["Server - Dining"], desired_hours=40)
        self._add_employee("Cocktail Lead", ["Server - Cocktail"], desired_hours=40)
        self._add_employee("Bar Lead", ["Bartender"], desired_hours=40)
        self._add_employee("Opener", ["Kitchen Opener"], desired_hours=40)
        self._add_employee("Closer", ["Kitchen Closer"], desired_hours=40)
        self._add_employee("Counter", ["Cashier - To-Go Specialist"], desired_hours=20)

        self._run_generator(policy)

        monday_shifts = self._shifts_for_day(0)
        open_shifts = [shift for shift in monday_shifts if shift.location.lower() == "open"]
        close_shifts = [shift for shift in monday_shifts if shift.location.lower() == "close"]

        self.assertEqual(len([shift for shift in open_shifts if shift.role.startswith("Server")]), 1)
        self.assertNotIn("Cashier - To-Go Specialist", {shift.role for shift in close_shifts})
        self.assertEqual(
            {shift.role for shift in close_shifts},
            {"Bartender", "Server - Dining", "Server - Cocktail", "Kitchen Closer"},
        )
        kitchen_open = next(shift for shift in open_shifts if "Kitchen Opener" in shift.role)
        self.assertEqual(kitchen_open.start.astimezone().time(), datetime.time(10, 30))

    @pytest.mark.skip(reason="Legacy generic expectations; superseded by BWW engine behavior")
    def test_close_block_starts_at_close_time(self) -> None:
        block_windows = {"Close": ("22:00", "22:35")}
        policy = self._policy_template(block_windows, ["Server"])
        policy["roles"]["Server"]["blocks"]["Close"]["base"] = 1
        policy["roles"]["Server"]["blocks"]["Close"]["min"] = 1
        policy["roles"]["Server"]["blocks"]["Close"]["max"] = 1
        self._add_employee("Closer", ["Server"], desired_hours=32)

        self._run_generator(policy)

        monday_shifts = [shift for shift in self._shifts_for_day(0) if shift.role == "Server"]
        self.assertTrue(monday_shifts)
        start_times = {shift.start.astimezone().time() for shift in monday_shifts if shift.location.lower() == "close"}
        self.assertIn(datetime.time(22, 0), start_times)
        end_times = {shift.end.astimezone().time() for shift in monday_shifts if shift.location.lower() == "close"}
        self.assertIn(datetime.time(22, 35), end_times)

    @pytest.mark.skip(reason="Legacy generic expectations; superseded by BWW engine behavior")
    def test_close_block_past_midnight_counts_same_day(self) -> None:
        block_windows = {"Close": ("24:00", "24:35")}
        policy = self._policy_template(block_windows, ["Server"])
        policy["roles"]["Server"]["blocks"]["Close"]["base"] = 1
        policy["roles"]["Server"]["blocks"]["Close"]["min"] = 1
        policy["roles"]["Server"]["blocks"]["Close"]["max"] = 1
        self._add_employee("Closer", ["Server"], desired_hours=32)

        self._run_generator(policy)

        monday_closers = [
            shift
            for shift in self._shifts_for_day(0)
            if shift.role == "Server" and shift.location == "Close"
        ]
        self.assertTrue(monday_closers)
        start_dates = {shift.start.date() for shift in monday_closers}
        self.assertEqual(start_dates, {self.week_start + datetime.timedelta(days=1)})
        start_times = {shift.start.astimezone().time() for shift in monday_closers}
        self.assertEqual(start_times, {datetime.time(0, 0)})
        end_times = {shift.end.astimezone().time() for shift in monday_closers}
        self.assertIn(datetime.time(0, 35), end_times)

    @pytest.mark.skip(reason="Legacy generic expectations; superseded by BWW engine behavior")
    def test_opener_receives_immediate_follow_up_shift(self) -> None:
        block_windows = {
            "Open": ("10:30", "11:00"),
            "Mid": ("11:00", "16:00"),
        }
        roles = ["Server - Dining", "Server - Dining Opener"]
        policy = self._policy_template(block_windows, roles)
        policy["roles"]["Server - Dining"]["blocks"]["Mid"].update({"base": 1, "min": 1, "max": 1})
        policy["roles"]["Server - Dining Opener"]["blocks"]["Open"].update({"base": 1, "min": 1, "max": 1})
        policy["roles"]["Server - Dining Opener"]["covers"] = ["Server - Dining"]
        opener = self._add_employee("Dedicated Opener", ["Server - Dining Opener"], desired_hours=40)
        self._add_employee("Midday Relief", ["Server - Dining"], desired_hours=40)

        self._run_generator(policy)

        monday = self._shifts_for_day(0)
        open_shift = next(shift for shift in monday if shift.role == "Server - Dining Opener")
        mid_shift = next(shift for shift in monday if shift.role == "Server - Dining")
        self.assertEqual(open_shift.employee_id, mid_shift.employee_id)
        self.assertEqual(open_shift.employee_id, opener.id)

    @pytest.mark.skip(reason="Legacy generic expectations; superseded by BWW engine behavior")
    def test_closer_requires_existing_assignment(self) -> None:
        block_windows = {
            "PM": ("12:00", "22:00"),
            "Close": ("22:00", "22:35"),
        }
        roles = ["Server - Dining", "Server - Dining Closer"]
        policy = self._policy_template(block_windows, roles)
        policy["roles"]["Server - Dining"]["blocks"]["PM"].update({"base": 1, "min": 1, "max": 1})
        policy["roles"]["Server - Dining Closer"]["blocks"]["Close"].update({"base": 1, "min": 1, "max": 1})
        policy["roles"]["Server - Dining Closer"]["covers"] = ["Server - Dining"]
        closer = self._add_employee("Closer With Shift", ["Server - Dining Closer"], desired_hours=40)
        self._add_employee(
            "Closer Without PM",
            ["Server - Dining Closer"],
            desired_hours=10,
            unavailability=[(0, "12:00", "21:45")],
        )

        self._run_generator(policy)

        monday = self._shifts_for_day(0)
        pm_shift = next(shift for shift in monday if shift.role == "Server - Dining")
        close_shift = next(shift for shift in monday if shift.role == "Server - Dining Closer")
        self.assertEqual(pm_shift.employee_id, close_shift.employee_id)
        self.assertEqual(close_shift.employee_id, closer.id)

    @pytest.mark.skip(reason="Legacy generic expectations; superseded by BWW engine behavior")
    def test_budget_trimmed_before_assignment(self) -> None:
        policy = self._policy(block_names=["Mid"], daily_boost={"Mon": 1})
        policy["roles"]["Server"]["blocks"]["Mid"].update({"base": 3, "min": 1, "max": 3})
        policy["roles"]["Server"]["hourly_wage"] = 20
        policy["global"]["labor_budget_pct"] = 0.1
        policy["global"]["labor_budget_tolerance_pct"] = 0.0
        policy["role_groups"] = {
            "Servers": {"allocation_pct": 1.0, "allow_cuts": True, "cut_buffer_minutes": 30}
        }

        self._seed_sales({0: 100.0})
        self._add_employee("Primary", ["Server"], desired_hours=40)
        self._add_employee("Support1", ["Server"], desired_hours=40)
        self._add_employee("Support2", ["Server"], desired_hours=40)

        result = self._run_generator(policy)

        monday = [shift for shift in self._shifts_for_day(0) if shift.role == "Server"]
        self.assertEqual(len(monday), 0, msg="Budget trimming should zero out over-budget coverage before assignment")
        self.assertTrue(result["warnings"] == [] or any("Budget shortfall" in warning for warning in result["warnings"]))

    @pytest.mark.skip(reason="Legacy generic expectations; superseded by BWW engine behavior")
    def test_budget_boost_increases_staffing_when_under_target(self) -> None:
        policy = self._policy(block_names=["Mid"], daily_boost={"Mon": 1})
        policy["roles"]["Server"]["blocks"]["Mid"].update({"base": 0, "min": 0, "max": 6})
        policy["roles"]["Server"]["hourly_wage"] = 25
        policy["timeblocks"]["Mid"] = {"start": "10:00", "end": "22:00"}
        policy["global"]["labor_budget_pct"] = 0.5
        policy["global"]["labor_budget_tolerance_pct"] = 0.25
        policy["pattern_templates"] = {"Servers": {}}
        policy["role_groups"] = {
            "Servers": {"allocation_pct": 1.0, "allow_cuts": True, "cut_buffer_minutes": 30}
        }
        self._seed_sales({0: 2000.0})
        for idx in range(12):
            self._add_employee(f"Server {idx}", ["Server"], desired_hours=40)

        self._run_generator(policy)

        monday = [shift for shift in self._shifts_for_day(0) if shift.role == "Server"]
        total_cost = sum(shift.labor_cost for shift in monday)
        self.assertGreaterEqual(len(monday), 3, msg="Budget boost should add extra coverage slots")
        self.assertGreaterEqual(total_cost, 750.0)
        self.assertLessEqual(total_cost, 1250.0)

    @pytest.mark.skip(reason="Legacy generic expectations; superseded by BWW engine behavior")
    def test_cut_notes_and_end_times_applied_before_insert(self) -> None:
        block_windows = {"Mid": ("10:00", "18:00")}
        policy = self._policy_template(block_windows, ["Server"])
        policy["roles"]["Server"]["blocks"]["Mid"].update({"base": 2, "min": 1, "max": 2, "cut_buffer_minutes": 90})
        policy["roles"]["Server"]["hourly_wage"] = 18
        policy["global"]["labor_budget_pct"] = 1.0
        policy["global"]["labor_budget_tolerance_pct"] = 0.2
        self._seed_sales({0: 2000.0})
        self._add_employee("Long Shift 1", ["Server"], desired_hours=40)
        self._add_employee("Long Shift 2", ["Server"], desired_hours=40)

        self._run_generator(policy)

        monday = [shift for shift in self._shifts_for_day(0) if shift.role == "Server"]
        self.assertEqual(len(monday), 2)
        self.assertTrue(any("cut around" in (shift.notes or "").lower() for shift in monday))
        self.assertTrue(any(shift.end.time() < datetime.time(18, 0) for shift in monday))

    @pytest.mark.skip(reason="Legacy generic expectations; superseded by BWW engine behavior")
    def test_recovery_pass_assigns_over_ceiling_employee(self) -> None:
        policy = self._policy(block_names=["Mid"], daily_boost={"Mon": 1})
        policy["roles"]["Server"]["blocks"]["Mid"].update({"base": 1, "min": 1, "max": 1})
        overflow_employee = self._add_employee("Over ceiling", ["Server"], desired_hours=2)

        result = self._run_generator(policy)

        monday = [shift for shift in self._shifts_for_day(0) if shift.role == "Server"]
        self.assertEqual(len(monday), 1)
        self.assertEqual(monday[0].employee_id, overflow_employee.id)
        self.assertFalse(any("No coverage" in warning for warning in result["warnings"]))

    def test_budget_lock_does_not_warn(self) -> None:
        policy = self._policy(block_names=["Close"], daily_boost={"Mon": 1}, role_name="Server - Dining Closer")
        block = policy["roles"]["Server - Dining Closer"]["blocks"]["Close"]
        block.update({"base": 1, "min": 1, "max": 1})
        policy["roles"]["Server - Dining Closer"]["allow_cuts"] = False
        policy["roles"]["Server - Dining Closer"]["hourly_wage"] = 30
        policy["global"]["labor_budget_pct"] = 0.01
        policy["global"]["labor_budget_tolerance_pct"] = 0.0
        self._add_employee("Closer Only", ["Server - Dining Closer"], desired_hours=40)

        result = self._run_generator(policy)

        monday = [shift for shift in self._shifts_for_day(0) if shift.role == "Server - Dining Closer"]
        self.assertEqual(len(monday), 2)
        self.assertTrue(any(shift.location.lower() == "close" for shift in monday))
        self.assertNotIn("Budget shortfall", " ".join(result["warnings"]))

    def test_fifo_adjustment_resolves_preferred_order(self) -> None:
        generator = ScheduleGenerator(self.session, self._policy(daily_boost={"Mon": 1}), actor="tests")
        demand = BlockDemand(
            day_index=0,
            date=self.week_start,
            start=datetime.datetime.combine(self.week_start, datetime.time(11, 0, tzinfo=datetime.timezone.utc)),
            end=datetime.datetime.combine(self.week_start, datetime.time(17, 0, tzinfo=datetime.timezone.utc)),
            role="Server",
            block_name="Mid",
            labels=["Mid"],
            need=3,
            priority=1.0,
            minimum=0,
            allow_cuts=True,
            always_on=False,
            role_group="Servers",
            hourly_rate=10.0,
        )
        employees = [
            {"assignments": {0: [(0, 60)]}},
            {"assignments": {0: [(30, 90)]}},
            {"assignments": {0: [(120, 180)]}},
        ]
        entries = [
            ({"start": demand.start, "_followup_locked": False}, employees[0]),
            ({"start": demand.start, "_followup_locked": False}, employees[1]),
            ({"start": demand.start, "_followup_locked": False}, employees[2]),
        ]
        base_end = demand.start + datetime.timedelta(hours=6)
        planned_end_times = [
            base_end,
            base_end - datetime.timedelta(hours=3),
            base_end - datetime.timedelta(hours=2),
        ]
        start_minutes, _ = generator._demand_window_minutes(demand)
        violations, locked_only = generator._fifo_violation_state(entries, planned_end_times, demand, start_minutes)
        self.assertTrue(violations)
        self.assertFalse(locked_only)
        generator._rebalance_fifo_entries(
            entries,
            planned_end_times,
            demand,
            start_minutes,
            datetime.timedelta(minutes=90),
            violations,
        )
        after_violations, _ = generator._fifo_violation_state(entries, planned_end_times, demand, start_minutes)
        self.assertFalse(after_violations)
        self.assertTrue(all(planned_end_times[i] <= planned_end_times[i + 1] for i in range(len(planned_end_times) - 1)))

    # Helpers -----------------------------------------------------------------

    def _run_generator(self, policy: dict) -> dict:
        engine = ScheduleGenerator(self.session, policy, actor="tests", employee_session=self.employee_session)
        return engine.generate(self.week_start)

    def _policy(
        self,
        *,
        block_names: list[str] | None = None,
        daily_boost: dict[str, int] | None = None,
        global_overrides: dict | None = None,
        role_name: str = "Server",
    ) -> dict:
        block_names = block_names or ["Open"]
        timeblocks = {
            "Open": {"start": "09:00", "end": "13:00"},
            "Mid": {"start": "11:00", "end": "16:00"},
            "Close": {"start": "16:00", "end": "22:00"},
        }
        role_blocks = {
            block: {"base": 0, "min": 0, "max": 1, "per_1000_sales": 0.0, "per_modifier": 0.0}
            for block in block_names
        }
        policy = {
            "global": {
                "max_hours_week": 40,
                "max_consecutive_days": 7,
                "round_to_minutes": 15,
                "allow_split_shifts": True,
                "desired_hours_floor_pct": 0.85,
                "desired_hours_ceiling_pct": 1.15,
                "open_buffer_minutes": 30,
                "close_buffer_minutes": 35,
                "labor_budget_pct": 0.27,
                "labor_budget_tolerance_pct": 0.08,
            },
            "timeblocks": {name: timeblocks[name] for name in block_names},
            "roles": {
                role_name: {
                    "enabled": True,
                    "priority": 1.0,
                    "max_weekly_hours": 40,
                    "daily_boost": daily_boost or {},
                    "blocks": role_blocks,
                }
            },
            "shift_presets": {
                "Servers": {
                    "am": [{"start": "11:00", "end": "16:00"}],
                    "pm": [{"start": "16:00", "end": "22:00"}],
                }
            },
        }
        if global_overrides:
            policy["global"].update(global_overrides)
        return policy

    def _policy_template(self, block_windows: dict[str, tuple[str, str]], role_names: list[str]) -> dict:
        timeblocks = {
            name: {"start": start, "end": end}
            for name, (start, end) in block_windows.items()
        }
        role_blocks = {
            block_name: {"base": 0, "min": 0, "max": 1, "per_1000_sales": 0.0, "per_modifier": 0.0}
            for block_name in block_windows
        }
        roles_payload = {}
        for role_name in role_names:
            roles_payload[role_name] = {
                "enabled": True,
                "priority": 1.0,
                "max_weekly_hours": 40,
                "daily_boost": {},
                "blocks": copy.deepcopy(role_blocks),
            }
        return {
            "global": {
                "max_hours_week": 40,
                "max_consecutive_days": 7,
                "round_to_minutes": 15,
                "allow_split_shifts": True,
                "desired_hours_floor_pct": 0.85,
                "desired_hours_ceiling_pct": 1.15,
                "open_buffer_minutes": 30,
                "close_buffer_minutes": 35,
                "labor_budget_pct": 0.27,
                "labor_budget_tolerance_pct": 0.08,
            },
            "timeblocks": timeblocks,
            "roles": roles_payload,
            "shift_presets": {
                "Servers": {
                    "am": [{"start": "11:00", "end": "16:00"}],
                    "pm": [{"start": "16:00", "end": "22:00"}],
                }
            },
        }

    def _add_employee(
        self,
        name: str,
        roles: list[str],
        desired_hours: int,
        *,
        unavailability: list[tuple[int, str, str]] | None = None,
    ) -> Employee:
        employee = Employee(full_name=name, desired_hours=desired_hours, status="active", notes="")
        employee.role_list = roles
        self.employee_session.add(employee)
        self.employee_session.commit()
        if unavailability:
            for day_index, start_label, end_label in unavailability:
                entry = EmployeeUnavailability(
                    employee_id=employee.id,
                    day_of_week=day_index,
                    start_time=self._time(start_label),
                    end_time=self._time(end_label),
                )
                self.employee_session.add(entry)
            self.employee_session.commit()
        return employee

    def _shifts_for_day(self, weekday_index: int) -> list[Shift]:
        rows = list(self.session.scalars(select(Shift)))
        return [
            shift
            for shift in rows
            if shift_display_date(shift.start, shift.location).weekday() == weekday_index
        ]

    def _hours_by_employee(self) -> dict[int, float]:
        totals: dict[int, float] = {}
        for shift in self.session.scalars(select(Shift)):
            if shift.employee_id is None:
                continue
            duration = (shift.end - shift.start).total_seconds() / 3600
            totals[shift.employee_id] = totals.get(shift.employee_id, 0.0) + duration
        return totals

    def _seed_sales(self, values: dict[int, float]) -> None:
        week = get_or_create_week(self.session, self.week_start)
        context = get_or_create_week_context(self.session, week.iso_year, week.iso_week, week.label)
        week.context_id = context.id
        self.session.commit()
        payload = {
            day_index: {"projected_sales_amount": amount, "projected_notes": ""}
            for day_index, amount in values.items()
        }
        save_week_daily_projection_values(self.session, context.id, payload)

    @staticmethod
    def _time(label: str) -> datetime.time:
        hours, minutes = [int(part) for part in label.split(":", 1)]
        return datetime.time(hour=hours, minute=minutes)


class GenerateApiTests(unittest.TestCase):
    @staticmethod
    def _session_factory():
        class _Context:
            def __enter__(self_inner):
                return object()

            def __exit__(self_inner, exc_type, exc, tb):
                return False

        return _Context()

    def test_retries_until_summary_contains_shifts(self) -> None:
        session_factory = self._session_factory
        with mock.patch("generator.api.load_active_policy", return_value={"global": {}, "roles": {}}), mock.patch(
            "generator.api.wage_amounts", return_value={}
        ), mock.patch("generator.api.ScheduleGenerator") as mock_engine:
            instance = mock_engine.return_value
            instance.generate.side_effect = [
                {"shifts_created": 0, "warnings": [], "projected_budget_total": 1000.0, "policy_budget_ratio": 0.0},
                {
                    "shifts_created": 4,
                    "warnings": [],
                    "projected_budget_total": 1000.0,
                    "policy_budget_ratio": 0.98,
                },
            ]
            result = api_generate_schedule_for_week(
                session_factory,
                datetime.date(2024, 4, 1),
                "tester",
                max_attempts=3,
                employee_session_factory=session_factory,
            )
            self.assertEqual(instance.generate.call_count, 2)
            self.assertEqual(result["shifts_created"], 4)
            self.assertEqual(result.get("attempts"), 2)
            self.assertGreaterEqual(result.get("budget_target_ratio", 0.0), 0.75)
            self.assertEqual(mock_engine.call_args_list[0].kwargs.get("cut_relax_level"), 0)
            self.assertEqual(mock_engine.call_args_list[1].kwargs.get("cut_relax_level"), 1)

    def test_raises_after_max_attempts(self) -> None:
        session_factory = self._session_factory
        with mock.patch("generator.api.load_active_policy", return_value={"global": {}, "roles": {}}), mock.patch(
            "generator.api.wage_amounts", return_value={}
        ), mock.patch("generator.api.ScheduleGenerator") as mock_engine:
            instance = mock_engine.return_value
            instance.generate.side_effect = RuntimeError("boom")
            with self.assertRaises(RuntimeError) as context:
                api_generate_schedule_for_week(
                    session_factory,
                    datetime.date(2024, 4, 1),
                    "tester",
                    max_attempts=2,
                    employee_session_factory=session_factory,
                )
            self.assertIn("after 2 attempts", str(context.exception))
            self.assertEqual(instance.generate.call_count, 2)

    def test_returns_best_summary_when_budget_target_unmet(self) -> None:
        session_factory = self._session_factory
        with mock.patch("generator.api.load_active_policy", return_value={"global": {}, "roles": {}}), mock.patch(
            "generator.api.wage_amounts", return_value={}
        ), mock.patch("generator.api.ScheduleGenerator") as mock_engine:
            instance = mock_engine.return_value
            instance.generate.side_effect = [
                {"shifts_created": 6, "warnings": [], "projected_budget_total": 1200.0, "policy_budget_ratio": 0.5},
                {"shifts_created": 8, "warnings": [], "projected_budget_total": 1200.0, "policy_budget_ratio": 0.6},
                {"shifts_created": 10, "warnings": [], "projected_budget_total": 1200.0, "policy_budget_ratio": 0.7},
            ]
            result = api_generate_schedule_for_week(
                session_factory,
                datetime.date(2024, 4, 1),
                "tester",
                max_attempts=3,
                employee_session_factory=session_factory,
            )
            self.assertEqual(result["shifts_created"], 10)
            self.assertEqual(result["attempts"], 3)
            self.assertTrue(any("budget target" in warning.lower() for warning in result.get("warnings", [])))
            self.assertAlmostEqual(result.get("policy_budget_ratio"), 0.7, places=3)


class PolicyPresetTests(unittest.TestCase):
    def test_shift_presets_are_copied_per_group(self) -> None:
        kitchen_original = copy.deepcopy(SHIFT_PRESET_DEFAULTS["Kitchen"])
        try:
            self.assertEqual(SHIFT_PRESET_DEFAULTS["Servers"], SHIFT_PRESET_DEFAULTS["Kitchen"])
            self.assertEqual(SHIFT_PRESET_DEFAULTS["Servers"], SHIFT_PRESET_DEFAULTS["Cashier"])
            original_start = SHIFT_PRESET_DEFAULTS["Servers"]["am"][0]["start"]
            SHIFT_PRESET_DEFAULTS["Kitchen"]["am"][0]["start"] = "10:00"
            self.assertEqual(SHIFT_PRESET_DEFAULTS["Servers"]["am"][0]["start"], original_start)
        finally:
            SHIFT_PRESET_DEFAULTS["Kitchen"] = kitchen_original


if __name__ == "__main__":
    unittest.main()
