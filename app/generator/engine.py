from __future__ import annotations

import copy
import datetime
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

from sqlalchemy import delete, select
from sqlalchemy.orm import selectinload

from database import (
    Employee,
    Shift,
    WeekSchedule,
    get_or_create_week,
    get_or_create_week_context,
    get_week_daily_projections,
    list_modifiers_for_week,
    record_audit_log,
    get_employee_role_wages,
    upsert_shift,
)
from policy import (
    PATTERN_TEMPLATES,
    SHIFT_PRESET_DEFAULTS,
    anchor_rules,
    build_default_policy,
    close_minutes,
    hourly_wage,
    open_minutes,
    resolve_policy_block,
    role_definition,
    parse_time_label,
    shift_length_limits,
    shift_length_rule,
    pre_engine_settings,
    required_roles,
    ROLE_GROUP_ALLOCATIONS,
)
from roles import is_manager_role, normalize_role, role_matches, role_group

UTC = datetime.timezone.utc
WEEKDAY_TOKENS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
HALF_HOUR = datetime.timedelta(minutes=30)
LABOR_PER_100_SALES = {"Servers": 0.18, "Bartenders": 0.05, "Kitchen": 0.2, "Cashier": 0.06}
MIN_STAFF_DEFAULTS = {"Servers": 1, "Server": 1, "Bartenders": 1, "Bartender": 1, "Kitchen": 2, "Cashier": 0}
SHIFT_TEMPLATE_CONFIG = {
    "Servers": [
        {"style": "Open", "time": "open-00:45", "hours": 7.5},
        {"style": "Lunch", "time": "11:00", "hours": 5.5},
        {"style": "Shoulder", "time": "14:00", "hours": 5.0},
        {"style": "Dinner", "time": "17:00", "hours": 6.5},
        {"style": "Late", "time": "20:00", "hours": 6.0},
    ],
    "Bartenders": [
        {"style": "Open", "time": "open-00:45", "hours": 8.0},
        {"style": "Mid", "time": "12:00", "hours": 6.0},
        {"style": "Dinner", "time": "17:00", "hours": 6.5},
        {"style": "Late", "time": "20:30", "hours": 6.0},
    ],
    "Kitchen": [
        {"style": "Prep", "time": "open-00:30", "hours": 8.0},
        {"style": "Mid", "time": "11:30", "hours": 6.5},
        {"style": "Dinner", "time": "17:00", "hours": 7.0},
        {"style": "Late", "time": "20:00", "hours": 6.0},
    ],
    "Cashier": [
        {"style": "Open", "time": "open-00:30", "hours": 6.0},
        {"style": "Dinner", "time": "17:00", "hours": 5.5},
        {"style": "Late", "time": "20:00", "hours": 5.0},
    ],
}
SHIFT_STYLE_ORDER = {"Open": 0, "Prep": 0, "Lunch": 1, "Mid": 1, "Shoulder": 2, "Dinner": 3, "Late": 4}
MIN_SHIFT_HOURS = 4.0
MAX_SHIFT_HOURS = 9.0


@dataclass
class BlockDemand:
    day_index: int
    date: datetime.date
    start: datetime.datetime
    end: datetime.datetime
    role: str
    block_name: str
    labels: List[str]
    need: int
    priority: float = 1.0
    minimum: int = 0
    allow_cuts: bool = True
    always_on: bool = False
    role_group: str = "Other"
    hourly_rate: float = 0.0
    recommended_cut: Optional[datetime.datetime] = None
    max_capacity: int = 0
    cut_score: float = 0.0
    cut_factors: Dict[str, float] = field(default_factory=dict)

    @property
    def duration_hours(self) -> float:
        return max(0.0, (self.end - self.start).total_seconds() / 3600)


class ScheduleGenerator:
    def __init__(
        self,
        session,
        policy: Dict,
        actor: str = "system",
        wage_overrides: Optional[Dict[str, float]] = None,
        *,
        cut_relax_level: int = 0,
        employee_session=None,
    ) -> None:
        self.session = session
        self.employee_session = employee_session
        self.policy = policy or {}
        self.actor = actor or "system"
        self.wage_overrides = wage_overrides or {}
        try:
            self.cut_relax_level: int = max(0, int(cut_relax_level))
        except (TypeError, ValueError):
            self.cut_relax_level = 0
        raw_roles = self.policy.get("roles") if isinstance(self.policy.get("roles"), dict) else {}
        self.roles_config: Dict[str, Dict] = {
            role: config for role, config in raw_roles.items() if not is_manager_role(role)
        }
        global_cfg = self.policy.get("global") or {}
        self.max_hours_per_week: float = float(global_cfg.get("max_hours_week", 40) or 40)
        self.max_consecutive_days: int = int(global_cfg.get("max_consecutive_days", 6) or 6)
        # Shift snapping fixed at 15-minute increments.
        self.round_to_minutes: int = 15
        self.allow_split_shifts: bool = True
        self.overtime_penalty: float = float(global_cfg.get("overtime_penalty", 1.5) or 1.5)
        desired_floor = float(global_cfg.get("desired_hours_floor_pct", 0.85) or 0.0)
        desired_ceiling = float(global_cfg.get("desired_hours_ceiling_pct", 1.15) or 0.0)
        self.desired_hours_floor_pct: float = self._clamp(desired_floor, 0.0, 1.0)
        min_ceiling = max(self.desired_hours_floor_pct + 0.05, 0.1)
        self.desired_hours_ceiling_pct: float = self._clamp(max(desired_ceiling, min_ceiling), min_ceiling, 2.0)
        # Open buffer is fixed at 30 minutes (10:30-11:00 window).
        self.open_buffer_minutes: int = 30
        self.close_buffer_minutes: int = int(global_cfg.get("close_buffer_minutes", 35) or 0)
        labor_pct = float(global_cfg.get("labor_budget_pct", 0.27) or 0.0)
        if labor_pct > 1.0:
            labor_pct /= 100.0
        self.labor_budget_pct = self._clamp(labor_pct, 0.05, 0.9)
        labor_tol = float(global_cfg.get("labor_budget_tolerance_pct", 0.08) or 0.0)
        if labor_tol > 1.0:
            labor_tol /= 100.0
        self.labor_budget_tolerance = self._clamp(labor_tol, 0.0, 0.5)
        self.budget_target_ratio = max(0.75, 1.0 - (self.labor_budget_tolerance / 2.0))
        self.pre_engine = pre_engine_settings(self.policy)
        self.pre_engine_staffing = self.pre_engine.get("staffing", {})
        self.required_role_labels: Set[str] = set(required_roles(self.policy))
        fallback_cfg = self.pre_engine.get("fallback", {})
        self.manager_fallback_allowed = bool(fallback_cfg.get("allow_mgr_fallback", False))
        self.manager_fallback_limits = {
            "am": int(fallback_cfg.get("am_limit", 1) or 0),
            "pm": int(fallback_cfg.get("pm_limit", 1) or 0),
        }
        self.manager_fallback_tag = fallback_cfg.get("tag", "MANAGER COVERING — REVIEW REQUIRED")
        self.manager_fallback_disallow = [str(entry).lower() for entry in fallback_cfg.get("disallow_roles", [])]
        self.manager_fallback_counts: Dict[int, Dict[str, int]] = {idx: {"am": 0, "pm": 0} for idx in range(7)}
        hoh_cfg = self.pre_engine_staffing.get("hoh", {})
        self.non_combinable_roles: Set[str] = {normalize_role(role) for role in hoh_cfg.get("non_combinable", [])}
        self.volume_thresholds = self.pre_engine_staffing.get("volume_thresholds", {})
        self.errors: List[str] = []

        self.employees: List[Dict[str, Any]] = []
        self.modifiers_by_day: Dict[int, List[Dict[str, Any]]] = {}
        self.day_contexts: List[Dict[str, Any]] = []
        self.role_group_settings: Dict[str, Dict[str, Any]] = self._load_role_group_settings()
        self.group_budget_by_day: List[Dict[str, float]] = []
        self.warnings: List[str] = []
        self.cut_insights: List[Dict[str, Any]] = []
        self.unfilled_slots: List[Dict[str, Any]] = []
        self.interchangeable_groups: Set[str] = {"Cashier"}
        self.random = random.Random()
        self.group_pressure: Dict[int, Dict[str, float]] = {}
        self.group_aliases = {"heart of house": "Kitchen", "cashier & takeout": "Cashier"}
        self.trim_aggressive_ratio: float = float(global_cfg.get("trim_aggressive_ratio", 1.0) or 1.0)
        self.anchors = anchor_rules(self.policy)
        order_mode = (self.anchors.get("open_close_order") or "prefer").strip().lower()
        self.open_close_order_mode = order_mode if order_mode in {"off", "prefer", "enforce"} else "prefer"
        self.cut_priority_settings = self._load_cut_priority_settings()
        self.cut_priority_rank: Dict[str, int] = self._build_cut_priority_rank()
        section_capacity = self.policy.get("section_capacity") if isinstance(self.policy, dict) else {}
        self.section_capacity: Dict[str, Dict[str, float]] = (
            section_capacity if isinstance(section_capacity, dict) else {}
        )
        self.non_cuttable_roles: Set[str] = {
            normalize_role(role) for role in (self.anchors.get("non_cuttable_roles") or [])
        }
        self.pattern_templates: Dict[str, Any] = {}
        raw_patterns = self.policy.get("pattern_templates") if isinstance(self.policy, dict) else {}
        if isinstance(raw_patterns, dict) and raw_patterns:
            self.pattern_templates = raw_patterns
        else:
            self.pattern_templates = PATTERN_TEMPLATES
        self.shift_presets = self.policy.get("shift_presets") if isinstance(self.policy, dict) else {}
        if not isinstance(self.shift_presets, dict) or not self.shift_presets:
            self.shift_presets = copy.deepcopy(SHIFT_PRESET_DEFAULTS)

    @staticmethod
    def _clamp(value: float, minimum: float, maximum: float) -> float:
        return max(minimum, min(maximum, value))

    @staticmethod
    def _to_float(value: Any, *, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _to_int(value: Any, *, default: int = 0) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def _round_minutes(self, minutes: float) -> int:
        """Round a minute value to the nearest configured step."""
        step = max(1, self.round_to_minutes)
        return int(round(minutes / step) * step)

    def generate(self, week_start_date: datetime.date) -> Dict[str, Any]:
        if not self.roles_config:
            return {
                "week_id": None,
                "days": [],
                "total_cost": 0.0,
                "warnings": ["Active policy does not define any eligible roles. Unable to generate schedule."],
                "cut_insights": [],
            }

        week = get_or_create_week(self.session, week_start_date)
        context = get_or_create_week_context(
            self.session,
            week.iso_year,
            week.iso_week,
            week.label,
        )
        week.context_id = context.id
        self.session.commit()

        self._reset_week(week)
        self.employees = self._load_employee_profiles()
        self.random.shuffle(self.employees)
        self.modifiers_by_day = self._load_modifiers(week.week_start_date)
        self.day_contexts = self._build_day_contexts(context, week.week_start_date)
        self.group_budget_by_day = self._build_group_budgets()
        self.cut_insights.clear()
        demands = self._compute_block_demands(week.week_start_date)
        assignments = self._assign(demands)
        self._apply_budget_cuts(assignments, demands)
        self._retry_unfilled_assignments(assignments)
        self._enforce_shift_continuity(assignments, week.week_start_date)
        self._warn_unpaired_openers()
        self._check_section_thresholds(assignments)
        self._record_required_role_gaps(assignments)

        created_ids: List[int] = []
        for payload in assignments:
            payload.update({"week_id": week.id, "status": "draft", "week_start": week.week_start_date})
            created_ids.append(upsert_shift(self.session, payload))

        summary = self._build_summary(week, assignments)
        summary["warnings"].extend(self.warnings)
        summary["errors"] = list(self.errors)
        summary["shifts_created"] = len(created_ids)
        record_audit_log(
            self.session,
            self.actor,
            "schedule_generate",
            target_type="WeekSchedule",
            target_id=week.id,
            payload={"shifts_created": len(created_ids)},
        )
        return summary

    def _reset_week(self, week: WeekSchedule) -> None:
        self.session.execute(delete(Shift).where(Shift.week_id == week.id))
        week.status = "draft"
        self.session.commit()
        self.unfilled_slots = []
        self.errors = []
        self.manager_fallback_counts = {idx: {"am": 0, "pm": 0} for idx in range(7)}

    def _load_employee_profiles(self) -> List[Dict[str, Any]]:
        stmt = (
            select(Employee)
            .where(Employee.status == "active")
            .options(selectinload(Employee.unavailability))
            .order_by(Employee.full_name.asc())
        )
        employees: List[Dict[str, Any]] = []
        self.employee_lookup: Dict[int, Dict[str, Any]] = {}
        source = self.employee_session or self.session
        rows = list(source.scalars(stmt))
        wage_overrides = get_employee_role_wages(self.employee_session or self.session, [emp.id for emp in rows if emp.id])
        for employee in rows:
            role_set = {role.strip() for role in employee.role_list if role.strip()}
            if not role_set:
                continue
            unavailability = {index: [] for index in range(7)}
            for entry in employee.unavailability:
                start_minutes = entry.start_time.hour * 60 + entry.start_time.minute
                end_minutes = entry.end_time.hour * 60 + entry.end_time.minute
                unavailability.setdefault(entry.day_of_week, []).append((start_minutes, end_minutes))
            desired_hours = max(0.0, float(employee.desired_hours or 0))
            desired_floor = desired_hours * self.desired_hours_floor_pct if desired_hours else 0.0
            desired_ceiling = desired_hours * self.desired_hours_ceiling_pct if desired_hours else self.max_hours_per_week
            desired_ceiling = min(self.max_hours_per_week, desired_ceiling or self.max_hours_per_week)
            if desired_ceiling < desired_floor:
                desired_ceiling = desired_floor
            record = {
                "id": employee.id,
                "name": employee.full_name,
                "roles": role_set,
                "desired_hours": desired_hours,
                "desired_floor": desired_floor,
                "desired_ceiling": desired_ceiling,
                "total_hours": 0.0,
                "assignments": {idx: [] for idx in range(7)},
                "day_last_block_end": {idx: None for idx in range(7)},
                "last_assignment_end": None,
                "unavailability": unavailability,
                "days_with_assignments": set(),
                "last_day_index": None,
                "consecutive_days": 0,
                "pending_open_links": {idx: [] for idx in range(7)},
                "wage_overrides": wage_overrides.get(employee.id, {}),
            }
            employees.append(record)
            if employee.id is not None:
                self.employee_lookup[employee.id] = record
        return employees

    def _load_modifiers(self, week_start: datetime.date) -> Dict[int, List[Dict[str, Any]]]:
        modifiers = list_modifiers_for_week(self.session, week_start)
        mapping: Dict[int, List[Dict[str, Any]]] = {idx: [] for idx in range(7)}
        for modifier in modifiers:
            day_idx = int(modifier.get("day_of_week", 0))
            start_time = modifier.get("start_time")
            end_time = modifier.get("end_time")
            pct = float(modifier.get("pct_change", 0) or 0)
            start_minutes = start_time.hour * 60 + start_time.minute if start_time else 0
            end_minutes = end_time.hour * 60 + end_time.minute if end_time else 24 * 60
            mapping.setdefault(day_idx, []).append(
                {
                    "start": start_minutes,
                    "end": end_minutes,
                    "pct": pct,
                    "multiplier": 1.0 + (pct / 100.0),
                }
            )
        return mapping

    def _load_role_group_settings(self) -> Dict[str, Dict[str, Any]]:
        specs = self.policy.get("role_groups") if isinstance(self.policy, dict) else {}
        mapping: Dict[str, Dict[str, Any]] = {}
        if not isinstance(specs, dict):
            specs = {}
        for name, raw in specs.items():
            if not isinstance(raw, dict):
                continue
            allocation = self._parse_allocation_pct(raw.get("allocation_pct"))
            mapping[name] = {
                "allocation_pct": allocation,
                "allow_cuts": bool(raw.get("allow_cuts", True)),
                "always_on": bool(raw.get("always_on", False)),
                "cut_buffer_minutes": int(raw.get("cut_buffer_minutes", 30) or 0),
            }
        if not mapping:
            mapping = copy.deepcopy(ROLE_GROUP_ALLOCATIONS)
        return mapping

    def _load_cut_priority_settings(self) -> Dict[str, Any]:
        """Normalize cut rotation + role ordering rules from the policy."""
        spec = self.anchors.get("cut_priority") if isinstance(self.anchors, dict) else {}
        settings: Dict[str, Any] = {
            "enabled": False,
            "include_unlisted": True,
            "sequence": [],
            "role_order": {},
        }
        if not isinstance(spec, dict):
            return settings
        settings["include_unlisted"] = bool(spec.get("include_unlisted", True))
        settings["enabled"] = bool(spec.get("enabled", False))

        raw_sequence = spec.get("sequence")
        normalized_sequence: List[Dict[str, Any]] = []
        if isinstance(raw_sequence, list):
            for entry in raw_sequence:
                normalized = self._normalize_cut_sequence_entry(entry)
                if normalized:
                    normalized_sequence.append(normalized)
        settings["sequence"] = normalized_sequence
        if not normalized_sequence:
            settings["enabled"] = False

        role_order_spec = spec.get("role_order")
        normalized_order: Dict[str, List[str]] = {}
        if isinstance(role_order_spec, dict):
            for group_name, roles in role_order_spec.items():
                canonical = self._canonical_group(group_name)
                normalized_roles = self._normalize_role_list(roles if isinstance(roles, list) else [roles])
                if canonical and normalized_roles:
                    normalized_order[canonical] = normalized_roles
        if "Servers" not in normalized_order:
            normalized_order["Servers"] = self._server_cut_preference()
        settings["role_order"] = normalized_order
        return settings

    def _normalize_cut_sequence_entry(self, entry: Any) -> Optional[Dict[str, Any]]:
        group_label = ""
        roles_field: List[Any] = []
        if isinstance(entry, str):
            if ":" in entry:
                group_label, role_part = entry.split(":", 1)
                roles_field = [role_part.strip()]
            else:
                group_label = entry
        elif isinstance(entry, dict):
            group_label = entry.get("group") or entry.get("name") or ""
            if isinstance(entry.get("roles"), list):
                roles_field = entry.get("roles")
            elif isinstance(entry.get("role"), str):
                roles_field = [entry.get("role")]
        if not group_label:
            return None
        canonical_group = self._canonical_group(group_label)
        normalized_roles = self._normalize_role_list(roles_field)
        return {"group": canonical_group, "roles": normalized_roles}

    @staticmethod
    def _normalize_role_list(values: Iterable[Any]) -> List[str]:
        seen: Set[str] = set()
        normalized: List[str] = []
        for value in values:
            label = normalize_role(value)
            if not label or label in seen:
                continue
            seen.add(label)
            normalized.append(label)
        return normalized

    def _build_cut_priority_rank(self) -> Dict[str, int]:
        """Convert the configured rotation into a numeric rank for earlier pull weighting."""
        base = {"Cashier": 0, "Servers": 1, "Kitchen": 2, "Bartenders": 3, "Other": 2}
        sequence = self.cut_priority_settings.get("sequence") or []
        assigned: Set[str] = set()
        for idx, entry in enumerate(sequence):
            group = entry.get("group")
            if not group or group in assigned:
                continue
            base[group] = idx
            assigned.add(group)
        return base

    @staticmethod
    def _parse_allocation_pct(value: Any) -> float:
        try:
            pct = float(value)
        except (TypeError, ValueError):
            return 0.0
        if pct > 1.0:
            pct = pct / 100.0
        return max(0.0, min(1.0, pct))

    def _build_day_contexts(self, context, week_start: datetime.date) -> List[Dict[str, Any]]:
        projections = get_week_daily_projections(self.session, context.id)
        projection_map = {projection.day_of_week: projection for projection in projections}
        contexts: List[Dict[str, Any]] = []
        adjusted_values: List[float] = []

        for day_index in range(7):
            projection = projection_map.get(day_index)
            sales = float(projection.projected_sales_amount) if projection else 0.0
            notes_payload = self._parse_projection_notes(projection.projected_notes if projection else "")
            modifier_multiplier = self._day_modifier_multiplier(day_index)
            adjusted_sales = sales * modifier_multiplier
            adjusted_values.append(adjusted_sales)
            contexts.append(
                {
                    "day_index": day_index,
                    "weekday_token": WEEKDAY_TOKENS[day_index],
                    "date": week_start + datetime.timedelta(days=day_index),
                    "sales": sales,
                    "notes": notes_payload,
                    "modifier_multiplier": modifier_multiplier,
                    "indices": {},
                }
            )

        max_sales = max(adjusted_values) if adjusted_values else 0.0
        if max_sales <= 0:
            max_sales = 1.0

        for ctx, adjusted in zip(contexts, adjusted_values):
            ctx["indices"] = self._compute_indices(ctx, adjusted, max_sales)
        return contexts

    def _build_group_budgets(self) -> List[Dict[str, float]]:
        budgets: List[Dict[str, float]] = []
        if not self.day_contexts:
            return budgets
        if not self.role_group_settings:
            return [{} for _ in self.day_contexts]
        for ctx in self.day_contexts:
            sales = float(ctx.get("sales", 0.0) or 0.0)
            modifier_multiplier = float(ctx.get("modifier_multiplier", 1.0) or 1.0)
            adjusted_sales = sales * modifier_multiplier
            total_budget = adjusted_sales * self.labor_budget_pct
            day_budget: Dict[str, float] = {}
            for group, spec in self.role_group_settings.items():
                pct = float(spec.get("allocation_pct", 0.0) or 0.0)
                if pct > 1.0:
                    pct /= 100.0
                pct = max(0.0, min(1.0, pct))
                if pct <= 0.0:
                    continue
                day_budget[group] = round(total_budget * pct, 2)
            budgets.append(day_budget)
        if not budgets:
            budgets = [{} for _ in self.day_contexts]
        return budgets

    def _compute_indices(self, day_ctx: Dict[str, Any], adjusted_sales: float, max_sales: float) -> Dict[str, float]:
        indices: Dict[str, float] = {}
        mapping = self.policy.get("demand_mapping") or {}
        base_index = adjusted_sales / max_sales if max_sales > 0 else 0.0
        base_index = max(0.0, min(1.5, base_index))
        indices["demand_index"] = round(base_index, 3)

        extra_data = day_ctx.get("notes", {})
        for name, spec in (mapping.get("indices") or {}).items():
            if name == "demand_index":
                continue
            value = None
            if isinstance(spec, dict):
                source_key = spec.get("source")
                if source_key and source_key in extra_data:
                    try:
                        value = float(extra_data[source_key])
                    except (TypeError, ValueError):
                        value = None
                role_weight = spec.get("roleWeight")
                if value is None and isinstance(role_weight, dict):
                    if role_weight:
                        value = base_index * max(role_weight.values())
            if value is None:
                value = base_index
            indices[name] = round(max(0.0, min(2.0, value)), 3)
        return indices

    def _day_modifier_multiplier(self, day_index: int) -> float:
        windows = self.modifiers_by_day.get(day_index, [])
        if not windows:
            return 1.0
        total = 0.0
        for window in windows:
            span = max(0.0, window["end"] - window["start"])
            fraction = span / (24 * 60)
            total += (window["pct"] / 100.0) * max(fraction, 0.1)
        return max(0.5, 1.0 + total)

    def _compute_block_demands(self, week_start: datetime.date) -> Dict[Tuple[int, str], Dict[str, Any]]:
        """
        Build a half-hour demand matrix per day and role group from projected sales.
        Coverage scales with adjusted sales, applies day-of-week/event weights, and
        enforces role minima so downstream shifts stay smooth around transitions.
        """
        matrix: Dict[Tuple[int, str], Dict[str, Any]] = {}
        roles_by_group = self._roles_by_group()
        for day_index in range(7):
            date_value = week_start + datetime.timedelta(days=day_index)
            day_start = datetime.datetime.combine(date_value, datetime.time.min, tzinfo=UTC)
            open_min = open_minutes(self.policy, date_value)
            close_min = close_minutes(self.policy, date_value)
            if close_min <= open_min:
                close_min += 24 * 60
            open_dt = day_start + datetime.timedelta(minutes=open_min)
            close_dt = day_start + datetime.timedelta(minutes=close_min)
            slots: List[Dict[str, Any]] = []
            slot_start = open_dt
            while slot_start < close_dt:
                slot_end = min(slot_start + HALF_HOUR, close_dt)
                slots.append({"start": slot_start, "end": slot_end})
                slot_start = slot_end
            ctx = self.day_contexts[day_index] if 0 <= day_index < len(self.day_contexts) else {}
            notes = ctx.get("notes", {}) if isinstance(ctx, dict) else {}
            adjusted_sales = float(ctx.get("sales", 0.0) or 0.0) * float(ctx.get("modifier_multiplier", 1.0) or 1.0)
            weights = self._slot_sales_weights(day_index, slots, notes)
            total_weight = sum(weights) or 1.0
            for group_name, role_names in roles_by_group.items():
                if not role_names:
                    continue
                min_staff = self._minimum_staff_for_group(group_name)
                labor_ratio = LABOR_PER_100_SALES.get(group_name, 0.12)
                targets: List[int] = []
                minima: List[int] = []
                for weight, slot in zip(weights, slots):
                    sales_for_slot = adjusted_sales * (weight / total_weight)
                    hours_needed = (sales_for_slot / 100.0) * labor_ratio
                    target = int(round(hours_needed / 0.5))
                    target = max(min_staff, target)
                    targets.append(target)
                    minima.append(min_staff)
                smoothed = self._smooth_targets(targets, minima)
                slot_records: List[Dict[str, Any]] = []
                for idx, slot in enumerate(slots):
                    slot_records.append(
                        {
                            "day_index": day_index,
                            "date": date_value,
                            "start": slot["start"],
                            "end": slot["end"],
                            "role_group": group_name,
                            "target": smoothed[idx],
                            "minimum": minima[idx],
                        }
                    )
                matrix[(day_index, group_name)] = {"slots": slot_records, "open": open_dt, "close": close_dt}
        self.current_slot_matrix = matrix
        return matrix

    def _roles_by_group(self) -> Dict[str, List[str]]:
        mapping: Dict[str, List[str]] = defaultdict(list)
        for role_name, cfg in self.roles_config.items():
            if not isinstance(cfg, dict) or not cfg.get("enabled", True):
                continue
            group_name = self._role_group_name(role_name, cfg)
            mapping[group_name].append(role_name)
        return mapping

    def _minimum_staff_for_group(self, group_name: str) -> int:
        canonical = self._canonical_group(group_name)
        return max(0, int(MIN_STAFF_DEFAULTS.get(canonical, MIN_STAFF_DEFAULTS.get(group_name, 1))))

    def _slot_sales_weights(
        self, day_index: int, slots: List[Dict[str, Any]], notes: Dict[str, Any]
    ) -> List[float]:
        """
        Estimate a time-of-day sales curve: lunch -> shoulder -> dinner -> late night.
        Event notes (BOGO/UFC/football) lift the relevant windows to keep coverage smooth.
        """
        if not slots:
            return []
        span_minutes = max(1.0, (slots[-1]["end"] - slots[0]["start"]).total_seconds() / 60.0)
        demand_index = 1.0
        if 0 <= day_index < len(self.day_contexts):
            demand_index = self.day_contexts[day_index].get("indices", {}).get("demand_index", 1.0)
        dow = WEEKDAY_TOKENS[day_index]
        is_weekend = dow in {"Fri", "Sat"}
        note_text = json.dumps(notes).lower() if notes else ""

        def bump(progress: float, center: float, width: float, amplitude: float) -> float:
            return amplitude * math.exp(-((progress - center) ** 2) / max(width, 1e-3))

        weights: List[float] = []
        for slot in slots:
            minutes_from_open = (slot["start"] - slots[0]["start"]).total_seconds() / 60.0
            progress = minutes_from_open / span_minutes
            base = 0.15
            base += bump(progress, 0.32, 0.028, 0.9)  # lunch peak
            base += bump(progress, 0.55, 0.045, 0.5)  # shoulder
            base += bump(progress, 0.72, 0.03, 1.3)  # dinner
            base += bump(progress, 0.9, 0.06, 0.45)  # late night
            if is_weekend:
                base *= 1.12
                base += bump(progress, 0.82, 0.05, 0.35)
            if dow == "Sun":
                base += bump(progress, 0.58, 0.04, 0.3)
            if "bogo" in note_text:
                base += bump(progress, 0.7, 0.035, 0.4)
            if "ufc" in note_text or "fight" in note_text:
                base += bump(progress, 0.92, 0.04, 0.6)
            if "football" in note_text or "nfl" in note_text:
                base += bump(progress, 0.6, 0.05, 0.35)
            weights.append(max(0.05, base * demand_index))
        return weights

    def _smooth_targets(self, targets: List[int], minima: List[int]) -> List[int]:
        """Soften dramatic slot-to-slot swings to avoid 4→1→5 coverage whiplash."""
        if not targets:
            return []
        smoothed = list(targets)
        for idx in range(1, len(smoothed)):
            smoothed[idx] = max(minima[idx], min(smoothed[idx], smoothed[idx - 1] + 2))
        for idx in range(len(smoothed) - 2, -1, -1):
            smoothed[idx] = max(minima[idx], min(smoothed[idx], smoothed[idx + 1] + 2))
        final: List[int] = []
        for idx, value in enumerate(smoothed):
            window = smoothed[max(0, idx - 1) : min(len(smoothed), idx + 2)]
            blended = (value * 2 + sum(window) / len(window)) / 3.0
            final.append(max(minima[idx], int(round(blended))))
        return final

    def _role_wage(self, role_name: str) -> float:
        if role_name in self.wage_overrides:
            try:
                override = float(self.wage_overrides[role_name])
                if override > 0:
                    return override
            except (TypeError, ValueError):
                pass
        return hourly_wage(self.policy, role_name, 0.0)

    def _employee_role_wage(self, employee: Optional[Dict[str, Any]], role_name: str) -> float:
        if employee:
            overrides = employee.get("wage_overrides") or {}
            target = normalize_role(role_name)
            for key, value in overrides.items():
                try:
                    wage = float(value)
                except (TypeError, ValueError):
                    continue
                if not wage:
                    continue
                if normalize_role(key) == target:
                    return wage
        return self._role_wage(role_name)

    def _adjust_block_window(
        self,
        role_name: str,
        block_name: str,
        date_value: datetime.date,
        start_dt: datetime.datetime,
        end_dt: datetime.datetime,
    ) -> Tuple[datetime.datetime, datetime.datetime]:
        normalized_role = normalize_role(role_name)
        block_label = (block_name or "").strip().lower()
        is_cashier = any(keyword in normalized_role for keyword in ("cashier", "takeout", "to-go"))
        opener_keywords = ("opener",)
        closer_keywords = ("closer",)
        if self.open_buffer_minutes and any(keyword in normalized_role for keyword in opener_keywords):
            buffer_minutes = self._round_minutes(max(0, self.open_buffer_minutes))
            open_min = open_minutes(self.policy, date_value)
            day_start = datetime.datetime.combine(date_value, datetime.time.min, tzinfo=UTC)
            open_dt = day_start + datetime.timedelta(minutes=open_min)
            buffered_start = open_dt - datetime.timedelta(minutes=buffer_minutes)
            if buffered_start < day_start:
                buffered_start = day_start
            if start_dt > buffered_start:
                start_dt = buffered_start
        # Apply pattern templates for Open/Mid/PM blocks to align with user-provided shift shapes
        # without moving the policy-driven start time.
        window_override = self._pattern_window(
            role_name,
            date_value,
            block_label,
            anchor_start=start_dt,
            anchor_end=end_dt,
        )
        if window_override:
            start_dt, end_dt = window_override
        return start_dt, end_dt


    def _day_sales_value(self, day_index: int) -> float:
        if 0 <= day_index < len(self.day_contexts):
            ctx = self.day_contexts[day_index]
            sales = float(ctx.get("sales", 0.0))
            modifier_multiplier = float(ctx.get("modifier_multiplier", 1.0))
            return sales * modifier_multiplier
        return 0.0

    def _pattern_window(
        self,
        role_name: str,
        date_value: datetime.date,
        block_label: str,
        *,
        anchor_start: Optional[datetime.datetime] = None,
        anchor_end: Optional[datetime.datetime] = None,
    ) -> Optional[Tuple[datetime.datetime, datetime.datetime]]:
        windows = self._pattern_windows(
            role_name,
            date_value,
            block_label,
            anchor_start=anchor_start,
            anchor_end=anchor_end,
        )
        return windows[0] if windows else None

    def _pattern_windows(
        self,
        role_name: str,
        date_value: datetime.date,
        block_label: str,
        *,
        anchor_start: Optional[datetime.datetime] = None,
        anchor_end: Optional[datetime.datetime] = None,
    ) -> List[Tuple[datetime.datetime, datetime.datetime]]:
        if block_label not in {"open", "mid", "pm"}:
            return []
        group_name = self._canonical_group(role_group(role_name))
        block_key = "am" if block_label in {"open", "mid"} else "pm"
        day_token = WEEKDAY_TOKENS[date_value.weekday()]
        override = self.shift_presets.get(group_name) or self.shift_presets.get(role_name) or {}
        templates = self.pattern_templates.get(group_name) or self.pattern_templates.get(role_name) or {}
        day_spec = templates.get(day_token) or templates.get("default") if isinstance(templates, dict) else {}
        if isinstance(override, dict) and block_key in override:
            windows = override.get(block_key)
        elif isinstance(day_spec, dict):
            windows = day_spec.get(block_key)
        else:
            windows = []
        parsed: List[Tuple[datetime.datetime, datetime.datetime]] = []
        if not isinstance(windows, list):
            return parsed
        for window in windows:
            parsed_window = self._parse_pattern_window(date_value, window)
            if not parsed_window:
                continue
            start_dt, end_dt = parsed_window
            if anchor_start:
                duration = end_dt - start_dt
                if duration.total_seconds() <= 0:
                    continue
                anchored_start = anchor_start
                anchored_end = anchor_start + duration
                if anchor_end and anchored_end > anchor_end:
                    anchored_end = anchor_end
                if anchored_end <= anchored_start:
                    continue
                parsed_window = (anchored_start, anchored_end)
            parsed.append(parsed_window)
        return parsed

    def _window_subset_for_slots(
        self, windows: List[Tuple[datetime.datetime, datetime.datetime]], slots: int, day_index: int
    ) -> List[Tuple[datetime.datetime, datetime.datetime]]:
        """Pick a deterministic subset of pattern windows to spread starts instead of front-loading."""
        if not windows or slots <= 0:
            return []
        count = len(windows)
        if slots >= count:
            return list(windows)
        # Rotate by day index so different weekdays do not always consume the earliest windows.
        rotated = list(windows[day_index % count :]) + list(windows[: day_index % count])
        if slots == 1:
            return [rotated[0]]
        indices: List[int] = []
        for i in range(slots):
            idx = math.floor(i * (len(rotated) - 1) / (slots - 1))
            if idx not in indices:
                indices.append(idx)
        subset = [rotated[idx] for idx in indices]
        return subset[:slots]

    def _constrain_windows_to_block(
        self,
        windows: List[Tuple[datetime.datetime, datetime.datetime]],
        block_start: datetime.datetime,
        block_end: datetime.datetime,
        block_label: str,
        role_name: str,
    ) -> List[Tuple[datetime.datetime, datetime.datetime]]:
        """Clamp pattern windows to the resolved block window so non-openers don't start before open/close."""
        adjusted: List[Tuple[datetime.datetime, datetime.datetime]] = []
        for start_dt, end_dt in windows:
            duration = end_dt - start_dt
            if duration.total_seconds() <= 0:
                continue
            new_start = start_dt
            new_end = end_dt
            if block_label not in {"open", "close"}:
                # Prevent mids/PMs from starting before the block start (e.g., before open).
                if new_start < block_start:
                    new_start = block_start
                    new_end = new_start + duration
            if new_end > block_end:
                new_end = block_end
            if new_end <= new_start:
                continue
            adjusted.append((new_start, new_end))
        return adjusted

    def _parse_pattern_window(
        self, date_value: datetime.date, window: Dict[str, Any]
    ) -> Optional[Tuple[datetime.datetime, datetime.datetime]]:
        if not isinstance(window, dict):
            return None
        start_label = window.get("start")
        end_label = window.get("end")
        start_minutes = parse_time_label(str(start_label)) if start_label is not None else None
        end_minutes = parse_time_label(str(end_label)) if end_label is not None else None
        if start_minutes is None or end_minutes is None:
            return None
        base = datetime.datetime.combine(date_value, datetime.time.min, tzinfo=UTC)
        start_dt = base + datetime.timedelta(minutes=int(start_minutes))
        end_dt = base + datetime.timedelta(minutes=int(end_minutes))
        return start_dt, end_dt

    def _calculate_block_need(
        self,
        role_name: str,
        role_cfg: Dict[str, Any],
        block_cfg: Dict[str, Any],
        block_name: str,
        day_index: int,
    ) -> Tuple[int, int, int]:
        block_label = (block_name or "").strip().lower()
        always_on = bool(role_cfg.get("always_on", False))
        base = int(block_cfg.get("base", block_cfg.get("min", 0)))
        min_staff = int(block_cfg.get("min", base))
        if min_staff <= 0 and base > 0:
            min_staff = base
        max_staff = max(int(block_cfg.get("max", max(base, min_staff))), min_staff)
        per_sales = float(block_cfg.get("per_1000_sales", 0.0))
        per_modifier = float(block_cfg.get("per_modifier", 0.0))
        sales = self._day_sales_value(day_index)
        sales_component = int(math.floor((sales / 1000.0) * per_sales))
        modifier_component = int(round(len(self.modifiers_by_day.get(day_index, [])) * per_modifier))
        boosts = role_cfg.get("daily_boost", {}) or {}
        day_token = WEEKDAY_TOKENS[day_index]
        daily_boost = int(boosts.get(day_token, 0))
        need = base + sales_component + modifier_component + daily_boost
        need += self._threshold_adjustment(role_cfg, block_cfg, day_index)
        if block_label == "open":
            if not self._role_allows_open_shift(role_name):
                return 0, 0, 0
            need = 1 if need > 0 else 0
        demand_index = 1.0
        if 0 <= day_index < len(self.day_contexts):
            demand_index = self.day_contexts[day_index].get("indices", {}).get("demand_index", 1.0)
        if not role_cfg.get("critical") and not always_on and block_label in {"mid", "pm"}:
            if demand_index < 0.3:
                need = min(need, min_staff)
        floor_rules = block_cfg.get("floor_by_demand", []) if isinstance(block_cfg, dict) else []
        if isinstance(floor_rules, list):
            for rule in floor_rules:
                if not isinstance(rule, dict):
                    continue
                gte = rule.get("gte")
                minimum_floor = rule.get("min")
                try:
                    gte_val = float(gte)
                    floor_val = int(minimum_floor)
                except (TypeError, ValueError):
                    continue
                if demand_index >= gte_val:
                    need = max(need, floor_val)
        need = max(min_staff, need)
        if max_staff > 0:
            need = min(max_staff, need)
        else:
            max_staff = need
        minimum = max(0, min_staff)
        if always_on:
            minimum = max(1, minimum)
        need = max(minimum, need)
        if self._is_opener_block(role_name, block_name):
            max_staff = 1
            need = min(1, need) if need > 0 else 0
            minimum = min(minimum, need)
        if self._is_closer_block(role_name, block_name):
            max_staff = 1
            need = min(1, need) if need > 0 else 0
            minimum = min(minimum, need)
        if max_staff <= 0:
            max_staff = need
        minimum = min(minimum, max_staff)
        return max(0, need), max(0, minimum), max_staff

    def _role_group_name(self, role_name: str, role_cfg: Optional[Dict[str, Any]]) -> str:
        explicit = ""
        if isinstance(role_cfg, dict):
            explicit = (role_cfg.get("group") or "").strip()
        if explicit:
            return self._canonical_group(explicit)
        inferred = role_group(role_name)
        return self._canonical_group(inferred if inferred else "Other")

    def _canonical_group(self, name: str) -> str:
        label = (name or "").strip()
        normalized = label.lower()
        if normalized in self.group_aliases:
            return self.group_aliases[normalized]
        if label:
            return label
        return "Other"

    def _apply_labor_allocations(self, demands: List[BlockDemand]) -> None:
        if not demands or not self.role_group_settings or not self.group_budget_by_day:
            return
        buckets: Dict[Tuple[int, str], Dict[str, Any]] = {}
        for demand in demands:
            key = (demand.day_index, demand.role_group)
            bucket = buckets.setdefault(
                key,
                {"budget": self._group_budget_for_day(demand.day_index, demand.role_group), "demands": []},
            )
            bucket["demands"].append(demand)
        for (day_index, group_name), payload in buckets.items():
            budget = payload.get("budget")
            if budget is None or budget <= 0:
                continue
            total_cost = sum(self._slot_cost(demand) * demand.need for demand in payload["demands"])
            locked_cost = sum(self._locked_slot_cost(demand) for demand in payload["demands"])
            base_budget = max(budget, locked_cost)
            allowed_max = max(budget * (1 + self.labor_budget_tolerance), locked_cost)
            if total_cost <= allowed_max + 1e-6:
                continue
            soft_mode = (total_cost / base_budget) <= (1.0 + self.labor_budget_tolerance + 1e-6)
            demand_index = 1.0
            if 0 <= day_index < len(self.day_contexts):
                demand_index = self.day_contexts[day_index].get("indices", {}).get("demand_index", 1.0)
            for demand in payload["demands"]:
                if demand.allow_cuts and not self._is_anchor_demand(demand):
                    demand.minimum = min(demand.minimum, 0)
            adjustable_budget = max(0.0, allowed_max - locked_cost)
            if adjustable_budget <= 0:
                continue
            removable: List[Tuple[float, float, BlockDemand]] = []
            for demand in payload["demands"]:
                if not demand.allow_cuts or demand.need <= demand.minimum:
                    continue
                slots = demand.need - demand.minimum
                if slots <= 0:
                    continue
                if self._is_anchor_demand(demand):
                    continue
                if soft_mode and demand.priority >= 1.0:
                    continue
                slot_cost = self._slot_cost(demand)
                removable.extend([(slot_cost, demand.priority, demand)] * slots)
            removable.sort(key=lambda entry: (entry[1], -entry[0]))
            idx = 0
            while total_cost > allowed_max + 0.01 and idx < len(removable):
                slot_cost, _priority, demand = removable[idx]
                if demand.need <= demand.minimum:
                    idx += 1
                    continue
                demand.need -= 1
                total_cost -= slot_cost
                if "trimmed by budget" not in demand.labels:
                    demand.labels.append("trimmed by budget")
                idx += 1
            if total_cost > allowed_max + 0.01:
                overage = max(0.0, total_cost - allowed_max)
                self.warnings.append(
                    f"Budget shortfall for {group_name} on {self._day_label(day_index)} (${overage:.2f})"
                )

    def _boost_under_budget_groups(self, demands: List[BlockDemand]) -> None:
        """Add incremental coverage when a group/day is materially under budget."""
        if not demands or not self.group_budget_by_day:
            return
        tolerance = max(0.0, min(0.9, self.labor_budget_tolerance))
        buckets: Dict[Tuple[int, str], Dict[str, Any]] = {}
        for demand in demands:
            if demand.need <= 0:
                continue
            budget = self._group_budget_for_day(demand.day_index, demand.role_group)
            if not budget or budget <= 0:
                continue
            key = (demand.day_index, demand.role_group)
            bucket = buckets.setdefault(key, {"budget": budget, "demands": []})
            bucket["demands"].append(demand)
        for (day_index, group_name), payload in buckets.items():
            budget = payload["budget"]
            if budget <= 0:
                continue
            allowed_min = budget * max(0.0, 1.0 - tolerance)
            if allowed_min <= 0:
                continue
            current_cost = sum(self._slot_cost(demand) * demand.need for demand in payload["demands"])
            if current_cost >= allowed_min - 0.5:
                continue
            allowed_max = budget * (1.0 + tolerance)
            expandable = [
                demand
                for demand in payload["demands"]
                if demand.allow_cuts
                and getattr(demand, "max_capacity", demand.need) > demand.need
                and self._slot_cost(demand) > 0
            ]
            if not expandable:
                continue
            expandable.sort(key=self._budget_boost_rank)
            needed_extra = allowed_min - current_cost
            iterations = 0
            while needed_extra > 5.0 and expandable:
                progress = False
                for demand in list(expandable):
                    capacity = getattr(demand, "max_capacity", demand.need)
                    slack = max(0, capacity - demand.need)
                    if slack <= 0:
                        expandable.remove(demand)
                        continue
                    slot_cost = self._slot_cost(demand)
                    if slot_cost <= 0:
                        expandable.remove(demand)
                        continue
                    if current_cost + slot_cost > allowed_max + 0.5:
                        continue
                    demand.need += 1
                    demand.minimum = min(demand.need, demand.minimum + 1)
                    demand.allow_cuts = False
                    current_cost += slot_cost
                    needed_extra = max(0.0, allowed_min - current_cost)
                    if "budget boost" not in demand.labels:
                        demand.labels.append("budget boost")
                    progress = True
                    break
                if not progress:
                    break
                iterations += 1
                if iterations >= 500:
                    break

    def _budget_boost_rank(self, demand: BlockDemand) -> Tuple[int, float, float]:
        block_label = demand.block_name.strip().lower()
        block_order = {"pm": 0, "mid": 1, "open": 2, "close": 3}.get(block_label, 4)
        return (block_order, -demand.priority, -self._slot_cost(demand))

    def _locked_slot_cost(self, demand: BlockDemand) -> float:
        slot_cost = self._slot_cost(demand)
        if slot_cost <= 0:
            return 0.0
        locked_units = 0
        if not demand.allow_cuts or self._is_anchor_demand(demand) or demand.always_on:
            locked_units = max(demand.need, locked_units)
        return slot_cost * max(0, locked_units)

    def _rebalance_budget_targets(self, demands: List[BlockDemand]) -> None:
        """Nudge cut windows so total cost better matches the configured budget."""
        if not demands or not self.group_budget_by_day:
            return
        min_ratio = self.budget_target_ratio
        max_ratio = 1.0 + self.labor_budget_tolerance
        buckets: Dict[Tuple[int, str], Dict[str, Any]] = {}
        for demand in demands:
            if demand.need <= 0 or not demand.allow_cuts:
                continue
            budget = self._group_budget_for_day(demand.day_index, demand.role_group)
            if not budget or budget <= 0:
                continue
            key = (demand.day_index, demand.role_group)
            bucket = buckets.setdefault(key, {"budget": budget, "demands": []})
            bucket["demands"].append(demand)
            bucket["cost"] = bucket.get("cost", 0.0) + (self._effective_slot_cost(demand) * max(0, demand.need))
        if not buckets:
            return
        for (day_idx, group_name), payload in buckets.items():
            budget = payload.get("budget", 0.0)
            if budget <= 0:
                continue
            cost = payload.get("cost", 0.0)
            ratio = cost / budget if budget else 1.0
            if ratio < min_ratio - 0.01:
                shortfall = (min_ratio * budget) - cost
                self._extend_demands_for_budget(payload["demands"], shortfall)
            elif ratio > max_ratio + 0.01:
                excess = cost - (max_ratio * budget)
                self._shrink_demands_for_budget(payload["demands"], excess)

    def _effective_slot_cost(self, demand: BlockDemand) -> float:
        end_time = demand.recommended_cut or demand.end
        if not end_time or end_time <= demand.start:
            return 0.0
        hours = max(0.0, (end_time - demand.start).total_seconds() / 3600)
        return hours * max(0.0, demand.hourly_rate or 0.0)

    def _extend_demands_for_budget(self, bucket: List[BlockDemand], dollars_needed: float) -> None:
        if dollars_needed <= 1.0 or not bucket:
            return
        ordered = sorted(bucket, key=lambda d: (-d.priority, d.start))
        for demand in ordered:
            if dollars_needed <= 0.5:
                break
            if not demand.recommended_cut or demand.recommended_cut >= demand.end:
                continue
            slack_minutes = int((demand.end - demand.recommended_cut).total_seconds() / 60)
            if slack_minutes <= 0:
                continue
            per_minute = max(0.0, demand.hourly_rate or 0.0) * max(1, demand.need) / 60.0
            if per_minute <= 0:
                continue
            extend_minutes = min(slack_minutes, int(math.ceil(dollars_needed / per_minute)))
            if extend_minutes <= 0:
                continue
            demand.recommended_cut = demand.recommended_cut + datetime.timedelta(minutes=extend_minutes)
            dollars_needed -= extend_minutes * per_minute
            self._update_demand_cut_label(demand)
            self._record_budget_rebalance_insight(demand, "extend", extend_minutes)

    def _shrink_demands_for_budget(self, bucket: List[BlockDemand], dollars_to_trim: float) -> None:
        if dollars_to_trim <= 1.0 or not bucket:
            return
        ordered = sorted(bucket, key=lambda d: (d.start, d.priority))
        for demand in ordered:
            if dollars_to_trim <= 0.5:
                break
            if demand.recommended_cut is None:
                continue
            min_hours, _ = shift_length_limits(self.policy, demand.role, demand.role_group)
            min_duration = datetime.timedelta(minutes=int(min_hours * 60))
            min_end = demand.start + min_duration
            current_end = demand.recommended_cut
            slack_minutes = int((current_end - min_end).total_seconds() / 60)
            if slack_minutes <= 0:
                continue
            per_minute = max(0.0, demand.hourly_rate or 0.0) * max(1, demand.need) / 60.0
            if per_minute <= 0:
                continue
            trim_minutes = min(slack_minutes, int(math.ceil(dollars_to_trim / per_minute)))
            if trim_minutes <= 0:
                continue
            demand.recommended_cut = current_end - datetime.timedelta(minutes=trim_minutes)
            dollars_to_trim -= trim_minutes * per_minute
            self._update_demand_cut_label(demand)
            self._record_budget_rebalance_insight(demand, "trim", trim_minutes)

    def _update_demand_cut_label(self, demand: BlockDemand) -> None:
        if not demand.recommended_cut:
            return
        existing = [label for label in demand.labels if not label.lower().startswith("cut around")]
        label = f"cut around {demand.recommended_cut.strftime('%H:%M')}"
        existing.append(label)
        demand.labels = existing

    def _record_budget_rebalance_insight(self, demand: BlockDemand, action: str, minutes: int) -> None:
        if minutes <= 0:
            return
        self.cut_insights.append(
            {
                "day": self._day_label(demand.day_index),
                "day_index": demand.day_index,
                "role": demand.role,
                "group": demand.role_group,
                "block": demand.block_name,
                "cut_time": (demand.recommended_cut or demand.end).isoformat(),
                "score": demand.cut_score,
                "factors": {**demand.cut_factors, "budget_adjustment": action},
                "labels": list(demand.labels),
            }
        )

    def _entry_start_rank(
        self,
        employee: Optional[Dict[str, Any]],
        day_index: int,
        default_start: int,
    ) -> int:
        if not employee:
            return default_start
        assignments = employee["assignments"].get(day_index, [])
        if not assignments:
            return default_start
        return min(start for start, _ in assignments)

    def _fifo_violation_state(
        self,
        entries: List[Tuple[Dict[str, Any], Optional[Dict[str, Any]]]],
        planned_end_times: List[datetime.datetime],
        demand: BlockDemand,
        start_minutes: int,
    ) -> Tuple[List[Dict[str, Any]], bool]:
        ordering: List[Tuple[int, datetime.datetime, bool, int]] = []
        for idx, (payload, employee) in enumerate(entries):
            earliest = self._entry_start_rank(employee, demand.day_index, start_minutes)
            locked = bool(payload.get("_followup_locked"))
            ordering.append((earliest, planned_end_times[idx], locked, idx))
        ordering.sort(key=lambda item: item[0])
        tolerance = datetime.timedelta(minutes=self._fifo_tolerance_minutes())
        violations: List[Dict[str, Any]] = []
        locked_only = True
        for first, second in zip(ordering, ordering[1:]):
            if first[1] > second[1] + tolerance:
                locked_pair = first[2] or second[2]
                violations.append(
                    {
                        "prev_idx": first[3],
                        "next_idx": second[3],
                        "locked_pair": locked_pair,
                    }
                )
                if not locked_pair:
                    locked_only = False
        return violations, locked_only

    def _rebalance_fifo_entries(
        self,
        entries: List[Tuple[Dict[str, Any], Optional[Dict[str, Any]]]],
        planned_end_times: List[datetime.datetime],
        demand: BlockDemand,
        start_minutes: int,
        min_duration: datetime.timedelta,
        violations: List[Dict[str, Any]],
    ) -> bool:
        swappable: List[int] = [
            idx for idx, (payload, _employee) in enumerate(entries) if not payload.get("_followup_locked")
        ]
        changed = False
        if len(swappable) >= 2:
            ranked = [
                (self._entry_start_rank(entries[idx][1], demand.day_index, start_minutes), idx) for idx in swappable
            ]
            ranked.sort(key=lambda item: (item[0], planned_end_times[item[1]]))
            target_indices = [idx for _score, idx in ranked]
            sorted_endings = sorted([planned_end_times[idx] for idx in target_indices])
            for idx, new_end in zip(target_indices, sorted_endings):
                if planned_end_times[idx] != new_end:
                    planned_end_times[idx] = new_end
                    changed = True
        tolerance = datetime.timedelta(minutes=self._fifo_tolerance_minutes())
        if violations:
            if self._force_fifo_adjustments(
                entries,
                planned_end_times,
                demand.start,
                min_duration,
                demand.end,
                tolerance,
                violations,
            ):
                changed = True
        return changed

    def _fifo_tolerance_minutes(self) -> int:
        return max(5, self.round_to_minutes)

    def _force_fifo_adjustments(
        self,
        entries: List[Tuple[Dict[str, Any], Optional[Dict[str, Any]]]],
        planned_end_times: List[datetime.datetime],
        block_start: datetime.datetime,
        min_duration: datetime.timedelta,
        block_end: datetime.datetime,
        tolerance: datetime.timedelta,
        violations: List[Dict[str, Any]],
    ) -> bool:
        changed = False
        min_allowed_end = block_start + min_duration
        for violation in violations:
            prev_idx = violation["prev_idx"]
            next_idx = violation["next_idx"]
            prev_payload, _prev_employee = entries[prev_idx]
            next_payload, _next_employee = entries[next_idx]
            prev_locked = bool(prev_payload.get("_followup_locked"))
            next_locked = bool(next_payload.get("_followup_locked"))
            if not prev_locked:
                candidate = planned_end_times[next_idx] - tolerance
                if candidate > planned_end_times[prev_idx]:
                    candidate = planned_end_times[prev_idx]
                start_dt = prev_payload.get("start", block_start)
                floor_time = max(min_allowed_end, start_dt + min_duration)
                if candidate > floor_time and candidate < planned_end_times[prev_idx]:
                    planned_end_times[prev_idx] = candidate
                    changed = True
                    continue
        if prev_locked and not next_locked:
            candidate = planned_end_times[prev_idx] + tolerance
            if candidate < planned_end_times[next_idx]:
                candidate = planned_end_times[next_idx]
            if candidate > block_end:
                candidate = block_end
            if candidate > planned_end_times[next_idx]:
                planned_end_times[next_idx] = candidate
                changed = True
        return changed

    def _apply_cut_labels(
        self,
        planned_labels: List[List[str]],
        planned_end_times: List[datetime.datetime],
        base_labels: List[str],
        early_indices: List[int],
        final_indices: List[int],
    ) -> None:
        for idx in range(len(planned_labels)):
            planned_labels[idx] = list(base_labels)
        for idx in final_indices:
            cut_time = planned_end_times[idx]
            planned_labels[idx] = list(base_labels) + [f"final cut around {cut_time.strftime('%H:%M')}"]
        if early_indices:
            sorted_indices = sorted(early_indices, key=lambda i: planned_end_times[i])
            for ordinal, idx in enumerate(sorted_indices, start=1):
                cut_time = planned_end_times[idx]
                planned_labels[idx] = list(base_labels) + [
                    f"{self._ordinal_label(ordinal)} cut around {cut_time.strftime('%H:%M')}"
                ]

    def _record_group_pressure(self, demands: List[BlockDemand]) -> None:
        """Estimate group-level budget pressure for dynamic cut targeting."""
        pressure: Dict[int, Dict[str, float]] = {}
        if not demands:
            self.group_pressure = pressure
            return
        cost_totals: Dict[Tuple[int, str], float] = {}
        budget_totals: Dict[Tuple[int, str], float] = {}
        for demand in demands:
            if demand.need <= 0:
                continue
            key = (demand.day_index, demand.role_group)
            cost_totals[key] = cost_totals.get(key, 0.0) + (self._slot_cost(demand) * demand.need)
            budget = self._group_budget_for_day(demand.day_index, demand.role_group)
            if budget is not None:
                budget_totals[key] = budget
        for (day_idx, group_name), cost_value in cost_totals.items():
            budget = budget_totals.get((day_idx, group_name), 0.0)
            ratio = cost_value / budget if budget > 0 else 1.0
            day_map = pressure.setdefault(day_idx, {})
            day_map[group_name] = round(max(0.0, ratio), 3)
        self.group_pressure = pressure

    def _enforce_anchor_shift_caps(self, demands: List[BlockDemand]) -> None:
        """Ensure opener/closer counts match policy anchors."""
        if not demands:
            return
        opener_caps = {self._canonical_group(k): v for k, v in (self.anchors.get("openers") or {}).items()}
        closer_caps = {self._canonical_group(k): v for k, v in (self.anchors.get("closers") or {}).items()}
        opener_roles = {self._canonical_group(k): [normalize_role(r) for r in v] for k, v in (self.anchors.get("opener_roles") or {}).items()}
        closer_roles = {self._canonical_group(k): [normalize_role(r) for r in v] for k, v in (self.anchors.get("closer_roles") or {}).items()}
        allow_cashier_close = bool(self.anchors.get("allow_cashier_closer", False))

        by_day: Dict[int, List[BlockDemand]] = {}
        for demand in demands:
            by_day.setdefault(demand.day_index, []).append(demand)

        for day_demands in by_day.values():
            self._apply_anchor_caps(day_demands, opener_caps, opener_roles, block_label="open")
            blocked_groups = set()
            if not allow_cashier_close:
                blocked_groups.add("Cashier")
            self._apply_anchor_caps(day_demands, closer_caps, closer_roles, block_label="close", blocked_groups=blocked_groups)
            self._dedupe_anchor_assignments(day_demands, "open")

    def _apply_anchor_caps(
        self,
        day_demands: List[BlockDemand],
        caps: Dict[str, int],
        role_preferences: Dict[str, List[str]],
        *,
        block_label: str,
        blocked_groups: Optional[Set[str]] = None,
    ) -> None:
        targets = [d for d in day_demands if d.block_name.strip().lower() == block_label]
        if not targets:
            return
        blocked_groups = blocked_groups or set()
        for demand in list(targets):
            if demand.role_group in blocked_groups:
                demand.need = 0
                demand.minimum = 0
        for demand in list(targets):
            if demand.role_group not in caps:
                demand.need = 0
                demand.minimum = 0
        for group_name, count in caps.items():
            if count <= 0:
                for demand in [d for d in targets if d.role_group == group_name]:
                    demand.need = 0
                    demand.minimum = 0
                continue
            candidates = [d for d in targets if d.role_group == group_name and d.need >= 0]
            if not candidates:
                continue
            prefs = role_preferences.get(group_name, [])
            chosen_list = self._pick_preferred_candidates(candidates, prefs, count)
            chosen_set = {id(item) for item in chosen_list}
            for demand in candidates:
                if id(demand) in chosen_set:
                    demand.need = 1
                    demand.minimum = max(1, demand.minimum)
                    demand.allow_cuts = False
                else:
                    demand.need = 0
                    demand.minimum = 0

    @staticmethod
    def _pick_preferred_candidates(
        candidates: List[BlockDemand], preferred_roles: List[str], target_count: int
    ) -> List[BlockDemand]:
        if not candidates or target_count <= 0:
            return []
        normalized_preferences = [normalize_role(role) for role in preferred_roles]

        def rank(demand: BlockDemand) -> Tuple[int, float]:
            normalized = normalize_role(demand.role)
            try:
                preferred_index = normalized_preferences.index(normalized)
            except ValueError:
                preferred_index = len(normalized_preferences)
            return (preferred_index, -demand.priority)

        return sorted(candidates, key=rank)[:target_count]

    @staticmethod
    def _dedupe_anchor_assignments(day_demands: List[BlockDemand], block_label: str) -> None:
        seen: Set[Tuple[str, str]] = set()
        for demand in sorted(
            [d for d in day_demands if d.block_name.strip().lower() == block_label],
            key=lambda d: (-d.priority, d.start),
        ):
            key = (demand.role_group, demand.block_name.strip().lower())
            if demand.need > 0 and key not in seen:
                seen.add(key)
                continue
            if demand.need > 0:
                demand.need = 0
                demand.minimum = 0

    @staticmethod
    def _server_cut_preference() -> List[str]:
        """Server roles cut order: patio -> dining -> cocktail."""
        return ["server - patio", "server - dining", "server - cocktail"]

    def _role_cut_preferences(self, group: str) -> List[str]:
        role_order = self.cut_priority_settings.get("role_order", {}) if self.cut_priority_settings else {}
        custom = role_order.get(group)
        if custom and (self.cut_priority_settings.get("enabled") or group == "Servers"):
            return custom
        if group == "Servers":
            return self._server_cut_preference()
        return []

    @staticmethod
    def _role_section_label(role: str) -> str:
        label = normalize_role(role)
        if "patio" in label:
            return "Patio"
        if "cocktail" in label or "bar" in label:
            return "Cocktail"
        if "dining" in label:
            return "Dining"
        return "Dining"

    def _section_capacity_weight(self, demand: BlockDemand) -> float:
        group_weights = self.section_capacity.get(demand.role_group)
        if not isinstance(group_weights, dict):
            return 1.0
        section_label = self._role_section_label(demand.role)
        try:
            weight = float(group_weights.get(section_label, group_weights.get("default", 1.0)))
        except (TypeError, ValueError):
            weight = 1.0
        return max(0.2, min(3.0, weight or 1.0))

    def _role_preference_rank(self, demand: BlockDemand) -> int:
        preferences = self._role_cut_preferences(demand.role_group)
        normalized = normalize_role(demand.role)
        try:
            return preferences.index(normalized)
        except ValueError:
            return len(preferences)

    def _cut_sort_key(self, demand: BlockDemand) -> Tuple[Any, ...]:
        score = self._cut_pressure_score(demand)
        pref_rank = self._role_preference_rank(demand)
        return (
            -score,
            pref_rank,
            demand.start,
            -demand.priority,
            demand.recommended_cut or demand.end,
        )

    def _cut_pressure_score(self, demand: BlockDemand) -> float:
        day_idx = demand.day_index
        pressure_ratio = self.group_pressure.get(day_idx, {}).get(demand.role_group, 1.0)
        budget_component = max(0.0, pressure_ratio - 1.0)
        demand_index = 1.0
        if 0 <= day_idx < len(self.day_contexts):
            demand_index = self.day_contexts[day_idx].get("indices", {}).get("demand_index", 1.0)
        workload_component = max(0.0, demand_index - 0.5)
        peak_component = max(0.0, self._block_progress_fraction(demand) - 0.6)
        base_rank = self.cut_priority_rank.get(demand.role_group, 2)
        score = (budget_component * 2.0) + (peak_component * 1.4) + (workload_component * 0.6) - (base_rank * 0.05)
        capacity_weight = self._section_capacity_weight(demand)
        demand.cut_factors = {
            "budget_component": round(budget_component, 3),
            "budget_gap": round(pressure_ratio - self.budget_target_ratio, 3),
            "peak_component": round(peak_component, 3),
            "workload_component": round(workload_component, 3),
            "pressure_ratio": round(pressure_ratio, 3),
            "demand_index": round(demand_index, 3),
            "capacity_weight": round(capacity_weight, 3),
        }
        score = score / max(0.25, capacity_weight)
        demand.cut_score = round(score, 3)
        return score

    def _block_progress_fraction(self, demand: BlockDemand) -> float:
        if not demand.recommended_cut:
            return 1.0
        total = max(1.0, (demand.end - demand.start).total_seconds())
        progressed = max(0.0, (demand.recommended_cut - demand.start).total_seconds())
        return self._clamp(progressed / total, 0.0, 1.0)

    @staticmethod
    def _stagger_step_for_pressure(pressure: float) -> int:
        step = 10
        if pressure >= 1.1:
            step = 15
        if pressure >= 1.3:
            step = 20
        return step

    def _assign_cut_for_demand(
        self,
        demand: BlockDemand,
        pressure: float,
        stagger_step: int,
        offset: int,
    ) -> bool:
        slot_cut = demand.recommended_cut or demand.end
        if not slot_cut:
            return False
        if pressure > 1.0:
            extra_pull = int(min(120, 30 + (pressure - 1.0) * 90))
            slot_cut = slot_cut - datetime.timedelta(minutes=extra_pull)
        slot_cut = slot_cut - datetime.timedelta(minutes=stagger_step * offset)
        min_hours, _ = shift_length_limits(self.policy, demand.role, demand.role_group)
        min_duration = datetime.timedelta(minutes=int(min_hours * 60))
        if slot_cut < demand.start + min_duration:
            slot_cut = demand.start + min_duration
        if slot_cut >= demand.end:
            return False
        demand.recommended_cut = slot_cut
        label = f"cut around {slot_cut.strftime('%H:%M')}"
        if label not in demand.labels:
            demand.labels.append(label)
        return True

    def _annotate_cut_windows(self, demands: List[BlockDemand]) -> None:
        if not demands:
            return
        # First pass: generate base cut candidates.
        candidates: List[BlockDemand] = []
        for demand in demands:
            if demand.need <= 0 or not demand.allow_cuts:
                continue
            cut_time = self._recommend_cut_time(demand)
            if not cut_time:
                continue
            demand.recommended_cut = cut_time
            candidates.append(demand)

        if not candidates:
            return

        # Second pass: stagger cuts within each day and optionally rotate across groups.
        by_day: Dict[int, List[BlockDemand]] = {}
        for demand in candidates:
            by_day.setdefault(demand.day_index, []).append(demand)

        for day_idx, bucket in by_day.items():
            if not bucket:
                continue
            bucket.sort(key=lambda d: self._cut_sort_key(d))
            group_offsets: Dict[str, int] = defaultdict(int)
            for demand in bucket:
                group = demand.role_group
                pressure = self.group_pressure.get(day_idx, {}).get(group, 1.0)
                stagger_step = self._stagger_step_for_pressure(pressure)
                applied = self._assign_cut_for_demand(demand, pressure, stagger_step, group_offsets[group])
                if applied:
                    group_offsets[group] += 1
                    self.cut_insights.append(
                        {
                            "day": self._day_label(day_idx),
                            "day_index": day_idx,
                            "role": demand.role,
                            "group": demand.role_group,
                            "block": demand.block_name,
                            "cut_time": (demand.recommended_cut or demand.end).isoformat(),
                            "score": demand.cut_score,
                            "factors": demand.cut_factors.copy(),
                            "labels": list(demand.labels),
                        }
                    )

    def _recommend_cut_time(self, demand: BlockDemand) -> Optional[datetime.datetime]:
        if self._is_closer_block(demand.role, demand.block_name):
            return None
        if demand.block_name.strip().lower() == "open":
            return None

        role_cfg = self.roles_config.get(demand.role, {})
        group_cfg = self.role_group_settings.get(demand.role_group, {})
        group_name = demand.role_group
        base_buffer = role_cfg.get("cut_buffer_minutes", group_cfg.get("cut_buffer_minutes", 30))
        try:
            buffer_minutes = int(base_buffer)
        except (TypeError, ValueError):
            buffer_minutes = 30

        demand_index = 1.0
        if 0 <= demand.day_index < len(self.day_contexts):
            demand_index = self.day_contexts[demand.day_index].get("indices", {}).get("demand_index", 1.0)

        pressure_ratio = self.group_pressure.get(demand.day_index, {}).get(demand.role_group, 1.0)
        priority_rank = self.cut_priority_rank.get(demand.role_group, 2)
        min_hours, max_hours = shift_length_limits(self.policy, demand.role, demand.role_group)
        # Allow faster releases than the nominal minimum when trimming: 1.5h floor for non-closers.
        if not self._is_closer_block(demand.role, demand.block_name):
            min_hours = min(min_hours, 1.5)

        normalized_role = normalize_role(demand.role)
        demand_softness = max(0.0, 1.0 - demand_index)  # softer = closer to 1
        pressure_factor = max(0.0, pressure_ratio - 1.0)
        cashier_bias = 1.0 if ("cashier" in normalized_role or "takeout" in normalized_role or "to-go" in normalized_role) else 0.0

        group_bias = 0
        if group_name in {"Cashier"}:
            group_bias = 90
        elif group_name in {"Servers"}:
            group_bias = 70
        elif group_name in {"Kitchen"}:
            group_bias = 55
        else:
            group_bias = 35

        # Pull earlier when demand is soft or the group is over budget.
        early_pull_minutes = int(demand_softness * 150) + int(
            min(320, (pressure_factor * 260) + priority_rank * 24 + group_bias)
        )
        if pressure_ratio >= 1.2:
            early_pull_minutes += 30
        if pressure_ratio >= 1.4:
            early_pull_minutes += 45
        if pressure_ratio >= 1.7:
            early_pull_minutes += 55
        if pressure_ratio >= 2.0:
            early_pull_minutes += 70
        if cashier_bias:
            early_pull_minutes += 45

        relax_level = max(0, self.cut_relax_level)
        if relax_level >= 1:
            early_pull_minutes = max(0, early_pull_minutes - relax_level * 45)

        buffer_minutes = max(0, buffer_minutes + early_pull_minutes)
        candidate = demand.end - datetime.timedelta(minutes=buffer_minutes)

        block_lower = demand.block_name.strip().lower()
        block_len = demand.end - demand.start
        if block_len.total_seconds() > 0:
            softness = max(0.0, 1.0 - demand_index)
            if block_lower == "mid":
                target_frac = 0.5 - 0.25 * softness
            elif block_lower == "pm":
                target_frac = 0.6 - 0.22 * softness
            else:
                target_frac = 0.7 - 0.18 * softness
            target_frac = max(0.3, min(0.85, target_frac))
            if relax_level >= 1:
                target_frac = min(0.98, target_frac + 0.05 * relax_level)
            duration_target = block_len.total_seconds() * target_frac
            # Scale harder when the group is over budget to hit closer to 100% of allocation.
            if pressure_ratio > 1.0:
                budget_scale = max(0.25, 1.0 / min(2.5, pressure_ratio + 0.1))
                duration_target = min(duration_target, block_len.total_seconds() * budget_scale)
            target_time = demand.start + datetime.timedelta(seconds=duration_target)
            if target_time < candidate:
                candidate = target_time

        min_duration = datetime.timedelta(minutes=int(min_hours * 60))
        max_span = datetime.timedelta(minutes=int(max_hours * 60))
        if candidate < demand.start + min_duration:
            candidate = demand.start + min_duration
        if demand.end - demand.start > max_span:
            # Encourage earlier cuts to respect max shift length.
            max_candidate = demand.start + max_span
            if candidate > max_candidate:
                candidate = max_candidate
        if candidate >= demand.end:
            return None
        return candidate

    def _group_budget_for_day(self, day_index: int, group_name: str) -> Optional[float]:
        if not self.group_budget_by_day:
            return None
        if 0 <= day_index < len(self.group_budget_by_day):
            return self.group_budget_by_day[day_index].get(group_name)
        return None

    @staticmethod
    def _slot_cost(demand: BlockDemand) -> float:
        return max(0.0, demand.duration_hours * max(0.0, demand.hourly_rate or 0.0))

    @staticmethod
    def _compute_cost(start: datetime.datetime, end: datetime.datetime, rate: float) -> float:
        hours = max(0.0, (end - start).total_seconds() / 3600)
        return round(hours * max(0.0, rate or 0.0), 2)

    def _day_label(self, day_index: int) -> str:
        if 0 <= day_index < len(self.day_contexts):
            return self.day_contexts[day_index].get("weekday_token") or f"Day {day_index + 1}"
        return f"Day {day_index + 1}"

    def _is_opener_block(self, role_name: str, block_name: str) -> bool:
        normalized_role = normalize_role(role_name)
        block_label = (block_name or "").strip().lower()
        if block_label == "open":
            return True
        if not normalized_role:
            return False
        return "opener" in normalized_role

    def _role_allows_open_shift(self, role_name: str) -> bool:
        normalized_role = normalize_role(role_name)
        if not normalized_role:
            return False
        allowed = {
            "kitchen opener",
            "bartender",
            "bartender - opener",
            "server",
            "server - dining",
            "server - dining opener",
            "server - cocktail",
            "server - cocktail opener",
        }
        return normalized_role in allowed

    def _is_closer_block(self, role_name: str, block_name: str) -> bool:
        normalized_role = normalize_role(role_name)
        if not normalized_role:
            return False
        return "closer" in normalized_role and block_name.strip().lower() == "close"

    def _is_anchor_demand(self, demand: BlockDemand) -> bool:
        normalized_role = normalize_role(demand.role)
        role_cfg = role_definition(self.policy, demand.role)
        return (
            normalized_role in self.non_cuttable_roles
            or not demand.allow_cuts
            or self._is_closer_block(demand.role, demand.block_name)
            or self._is_opener_block(demand.role, demand.block_name)
            or demand.always_on
            or bool(role_cfg.get("critical"))
        )

    def _demand_window_minutes(self, demand: BlockDemand) -> Tuple[int, int]:
        start_delta_days = (demand.start.date() - demand.date).days
        end_delta_days = (demand.end.date() - demand.date).days
        start_minutes = demand.start.hour * 60 + demand.start.minute + start_delta_days * 24 * 60
        end_minutes = demand.end.hour * 60 + demand.end.minute + end_delta_days * 24 * 60
        if end_minutes < start_minutes:
            end_minutes = start_minutes
        return start_minutes, end_minutes

    def _demand_day_segments(self, demand: BlockDemand) -> List[Tuple[int, int, int]]:
        start_minutes, end_minutes = self._demand_window_minutes(demand)
        segments: List[Tuple[int, int, int]] = []
        cursor = start_minutes
        minutes_per_day = 24 * 60
        while cursor < end_minutes:
            day_offset = cursor // minutes_per_day
            day_start = day_offset * minutes_per_day
            day_end = day_start + minutes_per_day
            segment_end = min(end_minutes, day_end)
            segments.append(
                (
                    int(day_offset),
                    int(cursor - day_start),
                    int(segment_end - day_start),
                )
            )
            cursor = segment_end
        return segments or [(0, 0, 0)]

    def _threshold_adjustment(self, role_cfg: Dict[str, Any], block_cfg: Dict[str, Any], day_index: int) -> int:
        thresholds = []
        if isinstance(block_cfg, dict):
            thresholds = block_cfg.get("thresholds") or []
        if not thresholds and isinstance(role_cfg, dict):
            thresholds = role_cfg.get("thresholds") or []
        if not thresholds:
            return 0
        indices: Dict[str, float] = {}
        if 0 <= day_index < len(self.day_contexts):
            indices = self.day_contexts[day_index].get("indices", {})
        adjustment = 0
        for rule in thresholds:
            if not isinstance(rule, dict):
                continue
            metric = (rule.get("metric") or "demand_index").strip()
            if not metric:
                continue
            value = indices.get(metric)
            if value is None:
                continue
            gte = self._to_float(rule.get("gte"), default=0.0)
            lte = rule.get("lte")
            if value < gte:
                continue
            if lte is not None and value > self._to_float(lte, default=value):
                continue
            add = self._to_int(rule.get("add"), default=0)
            adjustment += add
        return adjustment

    def _assign(self, demands: Dict[Tuple[int, str], Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert slot-level demand into template-anchored shifts, then assign employees.
        Template starts are honored and only nudged by small steps to close coverage gaps.
        """
        self.unfilled_slots = []
        if not demands:
            return []
        shift_plans: List[Dict[str, Any]] = []
        roles_by_group = self._roles_by_group()
        for (day_index, group_name), payload in sorted(demands.items()):
            slots = payload.get("slots", [])
            if not slots:
                continue
            templates = self._templates_for_group(group_name, day_index, payload.get("open"), payload.get("close"))
            remaining = [slot["target"] for slot in slots]
            used_starts: List[datetime.datetime] = []
            while remaining and max(remaining) > 0:
                peak_idx = self._peak_slot_index(remaining)
                plan = self._build_shift_from_peak(slots, remaining, templates, peak_idx, group_name, used_starts)
                if not plan:
                    self.warnings.append(
                        f"Unable to place shift for {group_name} on {slots[0]['date'].isoformat()} near "
                        f"{slots[peak_idx]['start'].strftime('%H:%M')}"
                    )
                    break
                used_starts.append(plan["start"])
                self._apply_plan_to_remaining(plan, remaining, slots)
                shift_plans.append(plan)
            self._ensure_edge_shifts(
                shift_plans,
                slots,
                group_name,
                payload.get("open"),
                payload.get("close"),
            )
        role_plans = self._map_plans_to_roles(shift_plans, roles_by_group)
        role_plans = self._inject_required_role_plans(role_plans, demands)
        return self._assign_employees(role_plans)

    def _inject_required_role_plans(
        self, plans: List[Dict[str, Any]], demands: Dict[Tuple[int, str], Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Ensure required roles appear on every day even before employee assignment."""
        if not self.required_role_labels:
            return plans
        existing = {(plan.get("day_index"), normalize_role(plan.get("role", ""))) for plan in plans}
        augmented = list(plans)
        for day_index in range(7):
            for role_label in self.required_role_labels:
                normalized = normalize_role(role_label)
                if (day_index, normalized) in existing:
                    continue
                start_dt, end_dt, slot_indices, role_group_name = self._required_window_for_day(
                    day_index, role_label, demands
                )
                if not start_dt or not end_dt or start_dt >= end_dt:
                    self.errors.append(f"Unable to place required role {role_label} on day index {day_index}.")
                    continue
                augmented.append(
                    {
                        "day_index": day_index,
                        "date": start_dt.date(),
                        "role_group": role_group_name,
                        "role": role_label,
                        "style": "Required",
                        "start": start_dt,
                        "end": end_dt,
                        "template_start": start_dt,
                        "slot_indices": slot_indices,
                        "essential": True,
                        "_required": True,
                    }
                )
                existing.add((day_index, normalized))
        return augmented

    def _required_window_for_day(
        self, day_index: int, role_label: str, demands: Dict[Tuple[int, str], Dict[str, Any]]
    ) -> Tuple[Optional[datetime.datetime], Optional[datetime.datetime], List[int], str]:
        group_name = self._canonical_group(role_group(role_label))
        payload = demands.get((day_index, group_name), {})
        slots = payload.get("slots", [])
        open_dt = payload.get("open")
        close_dt = payload.get("close")
        ctx_date = None
        if not open_dt or not close_dt:
            if 0 <= day_index < len(self.day_contexts):
                ctx_date = self.day_contexts[day_index].get("date")
            if ctx_date:
                day_start = datetime.datetime.combine(ctx_date, datetime.time.min, tzinfo=UTC)
                open_dt = day_start + datetime.timedelta(minutes=open_minutes(self.policy, ctx_date))
                close_dt = day_start + datetime.timedelta(minutes=close_minutes(self.policy, ctx_date))
        start_dt, end_dt = self._window_for_required_role(role_label, open_dt, close_dt)
        slot_indices = self._slot_indices_for_range(slots, start_dt, end_dt) if slots else []
        return start_dt, end_dt, slot_indices, group_name

    def _window_for_required_role(
        self, role_label: str, open_dt: Optional[datetime.datetime], close_dt: Optional[datetime.datetime]
    ) -> Tuple[Optional[datetime.datetime], Optional[datetime.datetime]]:
        if not open_dt or not close_dt:
            return None, None
        role_lower = normalize_role(role_label)
        opener = "opener" in role_lower
        closer = "closer" in role_lower
        buffer_open = datetime.timedelta(minutes=max(0, self.open_buffer_minutes))
        buffer_close = datetime.timedelta(minutes=max(0, self.close_buffer_minutes))
        if opener:
            start_dt = open_dt - buffer_open
            end_dt = min(close_dt, start_dt + datetime.timedelta(hours=5))
        elif closer:
            end_dt = close_dt + buffer_close
            start_dt = max(open_dt, end_dt - datetime.timedelta(hours=5))
        elif "preclose" in role_lower:
            end_dt = close_dt + buffer_close
            start_dt = max(open_dt, end_dt - datetime.timedelta(hours=3))
        elif "expo" in role_lower:
            start_dt = open_dt - datetime.timedelta(minutes=15)
            end_dt = close_dt
        else:
            start_dt = open_dt
            end_dt = close_dt
        return self._snap_datetime(start_dt), self._snap_datetime(end_dt)

    def _plan_coverage(self, plans: List[Dict[str, Any]], slots: List[Dict[str, Any]], group_name: str) -> List[int]:
        coverage = [0 for _ in slots]
        for plan in plans:
            if plan.get("role_group") != group_name:
                continue
            indices = plan.get("slot_indices")
            if not indices:
                indices = self._slot_indices_for_range(slots, plan["start"], plan["end"])
            for idx in indices:
                if 0 <= idx < len(coverage):
                    coverage[idx] += 1
        return coverage

    def _ensure_edge_shifts(
        self,
        plans: List[Dict[str, Any]],
        slots: List[Dict[str, Any]],
        group_name: str,
        open_dt: datetime.datetime,
        close_dt: datetime.datetime,
    ) -> None:
        if not slots:
            return
        coverage = self._plan_coverage(plans, slots, group_name)
        minima = [slot["minimum"] for slot in slots]
        targets = [slot["target"] for slot in slots]
        essential_group = self._canonical_group(group_name) in {"Servers", "Server", "Bartenders", "Bartender", "Kitchen"}

        def add_open():
            start_dt = self._snap_datetime(open_dt - datetime.timedelta(minutes=45))
            end_dt = min(close_dt, start_dt + datetime.timedelta(hours=max(MIN_SHIFT_HOURS, 6.0)))
            plan = {
                "day_index": slots[0]["day_index"],
                "date": slots[0]["date"],
                "role_group": group_name,
                "style": "Open",
                "start": start_dt,
                "end": end_dt,
                "template_start": start_dt,
                "slot_indices": self._slot_indices_for_range(slots, start_dt, end_dt),
                "essential": essential_group,
            }
            plans.append(plan)
            return plan

        def add_close():
            end_dt = close_dt
            start_dt = max(open_dt, end_dt - datetime.timedelta(hours=max(MIN_SHIFT_HOURS, 6.0)))
            start_dt = self._snap_datetime(start_dt)
            plan = {
                "day_index": slots[0]["day_index"],
                "date": slots[0]["date"],
                "role_group": group_name,
                "style": "Close",
                "start": start_dt,
                "end": end_dt,
                "template_start": start_dt,
                "slot_indices": self._slot_indices_for_range(slots, start_dt, end_dt),
                "essential": essential_group,
            }
            plans.append(plan)
            return plan

        # Ensure opener coverage
        if coverage[0] < minima[0] or coverage[0] < targets[0]:
            added = add_open()
            coverage = self._plan_coverage(plans, slots, group_name)
            if (coverage[0] < minima[0] or coverage[0] < targets[0]) and added:
                self.warnings.append(
                    f"Opener coverage still low for {group_name} on {slots[0]['date'].isoformat()}; review manually."
                )

        # Ensure closer coverage
        if coverage[-1] < minima[-1] or coverage[-1] < targets[-1]:
            added = add_close()
            coverage = self._plan_coverage(plans, slots, group_name)
            if (coverage[-1] < minima[-1] or coverage[-1] < targets[-1]) and added:
                self.warnings.append(
                    f"Closer coverage still low for {group_name} on {slots[-1]['date'].isoformat()}; review manually."
                )

    @staticmethod
    def _peak_slot_index(remaining: List[int]) -> int:
        max_val = max(remaining) if remaining else 0
        for idx, value in enumerate(remaining):
            if value == max_val:
                return idx
        return 0

    def _build_shift_from_peak(
        self,
        slots: List[Dict[str, Any]],
        remaining: List[int],
        templates: List[Dict[str, Any]],
        peak_idx: int,
        group_name: str,
        used_starts: List[datetime.datetime],
    ) -> Optional[Dict[str, Any]]:
        if not slots or peak_idx >= len(slots) or peak_idx < 0:
            return None
        peak_start = slots[peak_idx]["start"]
        candidates = [t for t in templates if t["start"] <= peak_start] or templates
        if not candidates:
            return None
        template = max(candidates, key=lambda t: t["start"])
        start_dt = self._snap_datetime(template["start"])
        nudge_step = datetime.timedelta(minutes=max(15, self.round_to_minutes))
        if used_starts and start_dt in used_starts and max(remaining) <= 1:
            start_dt = self._snap_datetime(start_dt + nudge_step)
        slot_start = slots[0]["start"]
        slot_count = len(slots)
        start_idx = int(max(0, (start_dt - slot_start).total_seconds() // HALF_HOUR.total_seconds()))
        start_idx = min(start_idx, slot_count - 1)
        min_slots = int(math.ceil(MIN_SHIFT_HOURS * 60 / 30))
        max_slots = int(math.floor(MAX_SHIFT_HOURS * 60 / 30))
        base_slots = int(round(max(1.0, float(template.get("hours", MIN_SHIFT_HOURS))) * 60 / 30))
        window_start = start_idx
        window_end = min(slot_count, start_idx + base_slots)
        if peak_idx >= window_end:
            window_end = min(slot_count, peak_idx + 1)
        if peak_idx < window_start:
            window_start = peak_idx
        max_left_shift = 1  # limit to 30 minutes of movement away from the template anchor
        min_window_start = max(0, start_idx - max_left_shift)
        window_start = max(min_window_start, window_start)
        if peak_idx < window_start:
            window_start = peak_idx
        window_end = max(window_end, window_start + min_slots)
        if window_end > slot_count:
            window_start = max(0, slot_count - min_slots)
            window_end = slot_count
        window_end = min(window_end, window_start + max_slots)
        target_level = remaining[peak_idx]
        while window_end < slot_count and window_end - window_start < max_slots:
            if remaining[window_end] >= max(1, int(round(target_level * 0.6))):
                window_end += 1
            else:
                break
        while window_start > min_window_start and window_end - window_start < max_slots:
            if remaining[window_start - 1] >= max(1, int(round(target_level * 0.6))):
                window_start -= 1
            else:
                break
        if window_end - window_start < min_slots:
            deficit = min_slots - (window_end - window_start)
            window_end = min(slot_count, window_end + deficit)
            if window_end - window_start < min_slots and window_start > 0:
                window_start = max(0, window_start - (min_slots - (window_end - window_start)))
        window_end = min(slot_count, window_start + max(min_slots, window_end - window_start))
        start_dt_final = slots[window_start]["start"]
        end_dt_final = slots[window_end - 1]["end"] if window_end > window_start else slots[window_start]["end"]
        return {
            "day_index": slots[window_start]["day_index"],
            "date": slots[window_start]["date"],
            "role_group": group_name,
            "style": template.get("style", "Mid"),
            "start": start_dt_final,
            "end": end_dt_final,
            "template_start": template.get("start"),
            "slot_indices": list(range(window_start, window_end)),
        }

    @staticmethod
    def _apply_plan_to_remaining(plan: Dict[str, Any], remaining: List[int], slots: List[Dict[str, Any]]) -> None:
        for idx in plan.get("slot_indices", []):
            if 0 <= idx < len(remaining):
                remaining[idx] = max(0, remaining[idx] - 1)

    def _templates_for_group(
        self, group_name: str, day_index: int, open_dt: Optional[datetime.datetime], close_dt: Optional[datetime.datetime]
    ) -> List[Dict[str, Any]]:
        canonical = self._canonical_group(group_name)
        config = SHIFT_TEMPLATE_CONFIG.get(canonical, SHIFT_TEMPLATE_CONFIG.get(group_name, SHIFT_TEMPLATE_CONFIG.get("Servers", [])))
        templates: List[Dict[str, Any]] = []
        day_token = WEEKDAY_TOKENS[day_index]
        for spec in config:
            base_hours = float(spec.get("hours", MIN_SHIFT_HOURS) or MIN_SHIFT_HOURS)
            if day_token in {"Fri", "Sat"}:
                base_hours += 0.5
            elif day_token == "Sun":
                base_hours -= 0.25
            start_dt = self._resolve_template_start(spec.get("time"), open_dt, close_dt)
            if not start_dt or (close_dt and start_dt >= close_dt):
                continue
            templates.append({"style": spec.get("style", "Mid"), "start": start_dt, "hours": base_hours})
        return sorted(templates, key=lambda item: item["start"])

    def _resolve_template_start(
        self, time_spec: Any, open_dt: Optional[datetime.datetime], close_dt: Optional[datetime.datetime]
    ) -> Optional[datetime.datetime]:
        if not open_dt:
            return None
        if isinstance(time_spec, str) and time_spec.lower().startswith("open"):
            offset_minutes = 0
            if "-" in time_spec or "+" in time_spec:
                try:
                    sign = -1 if "-" in time_spec else 1
                    raw = time_spec.split("-", 1)[-1] if "-" in time_spec else time_spec.split("+", 1)[-1]
                    hours, minutes = raw.split(":")
                    offset_minutes = sign * ((int(hours) * 60) + int(minutes))
                except (ValueError, IndexError):
                    offset_minutes = 0
            start_dt = open_dt + datetime.timedelta(minutes=offset_minutes)
        elif isinstance(time_spec, str) and ":" in time_spec:
            try:
                hour, minute = [int(part) for part in time_spec.split(":", 1)]
                start_dt = datetime.datetime.combine(open_dt.date(), datetime.time(hour, minute), tzinfo=UTC)
                if close_dt and start_dt > close_dt and hour < 6:
                    start_dt = start_dt - datetime.timedelta(days=1)
            except ValueError:
                return None
        else:
            return None
        if close_dt and start_dt > close_dt:
            start_dt = close_dt - datetime.timedelta(hours=MIN_SHIFT_HOURS)
        day_start = datetime.datetime.combine(open_dt.date(), datetime.time.min, tzinfo=UTC)
        if start_dt < day_start:
            start_dt = day_start
        return self._snap_datetime(start_dt)

    def _snap_datetime(self, dt_value: datetime.datetime) -> datetime.datetime:
        """Snap to nearest configured minute step to avoid non-template odd starts."""
        base = datetime.datetime.combine(dt_value.date(), datetime.time.min, tzinfo=dt_value.tzinfo)
        minutes = int((dt_value - base).total_seconds() // 60)
        rounded = self._round_minutes(minutes)
        return base + datetime.timedelta(minutes=rounded)

    def _map_plans_to_roles(
        self, plans: List[Dict[str, Any]], roles_by_group: Dict[str, List[str]]
    ) -> List[Dict[str, Any]]:
        mapped: List[Dict[str, Any]] = []
        buckets: Dict[Tuple[int, str], List[Dict[str, Any]]] = defaultdict(list)
        for plan in plans:
            key = (plan["day_index"], plan["role_group"])
            buckets[key].append(plan)
        for (day_index, group_name), bucket in buckets.items():
            canonical_group = self._canonical_group(group_name)
            roles = roles_by_group.get(group_name, roles_by_group.get(canonical_group, []))
            bucket_sorted = sorted(bucket, key=lambda p: (p["start"], SHIFT_STYLE_ORDER.get(p["style"], 5)))
            essential_flags = {"server": False, "bartender": False, "expo": False}
            for idx, plan in enumerate(bucket_sorted):
                if canonical_group.startswith("Server"):
                    role, is_floor = self._role_for_server_plan(idx, len(bucket_sorted), roles, plan)
                    is_essential = is_floor and not essential_flags["server"]
                    essential_flags["server"] = essential_flags["server"] or is_essential
                elif canonical_group.startswith("Kitchen"):
                    role, is_expo = self._role_for_kitchen_plan(idx, roles, plan)
                    is_essential = is_expo and not essential_flags["expo"]
                    essential_flags["expo"] = essential_flags["expo"] or is_essential
                elif canonical_group.startswith("Bartender"):
                    role = roles[0] if roles else plan.get("role") or "Bartender"
                    is_essential = not essential_flags["bartender"]
                    essential_flags["bartender"] = True
                else:
                    role = self._role_for_group_default(group_name, roles)
                    is_essential = False
                mapped.append({**plan, "role": role, "essential": is_essential, "role_group": canonical_group})
        return sorted(mapped, key=lambda p: (p["day_index"], p["start"], p["end"]))

    def _role_for_server_plan(
        self, idx: int, total: int, roles: List[str], plan: Dict[str, Any]
    ) -> Tuple[str, bool]:
        dining_roles = [r for r in roles if "cocktail" not in r.lower()]
        cocktail_roles = [r for r in roles if "cocktail" in r.lower()]
        dining_roles = dining_roles or roles
        cocktail_roles = cocktail_roles or roles
        dining_target = max(1, int(round(total * 0.7)))
        if idx < dining_target or plan.get("style") in {"Open", "Lunch"}:
            return dining_roles[0], True
        return cocktail_roles[0], False

    def _role_for_kitchen_plan(self, idx: int, roles: List[str], plan: Dict[str, Any]) -> Tuple[str, bool]:
        expo_roles = [r for r in roles if "expo" in r.lower() or "expeditor" in r.lower()]
        day_index = plan.get("day_index", 0)
        demand_index = 1.0
        if 0 <= day_index < len(self.day_contexts):
            demand_index = self.day_contexts[day_index].get("indices", {}).get("demand_index", 1.0)
        hoh_cfg = self.pre_engine_staffing.get("hoh", {})
        thresholds = hoh_cfg.get("combo_thresholds", {})
        low_max = float(thresholds.get("low_max", 0.55) or 0.55)
        split_min = float(thresholds.get("split_min", 0.75) or 0.75)
        peak_min = float(thresholds.get("peak_min", 1.0) or 1.0)
        if demand_index <= low_max:
            stage = "low"
        elif demand_index >= peak_min:
            stage = "peak"
        else:
            stage = "moderate"
        combo_roles = hoh_cfg.get("combo_roles", {})
        stage_order = {
            "low": [
                "HOH - Expo",
                combo_roles.get("sw_grill", "HOH - Southwest & Grill"),
                combo_roles.get("chip_shake", "HOH - Chip & Shake"),
                "HOH - Grill",
                "HOH - Southwest",
            ],
            "moderate": [
                "HOH - Expo",
                "HOH - Grill",
                "HOH - Southwest",
                combo_roles.get("chip_shake", "HOH - Chip & Shake"),
                "HOH - Shake",
            ],
            "peak": ["HOH - Expo", "HOH - Grill", "HOH - Southwest", "HOH - Chip", "HOH - Shake"],
        }
        ordered_roles = stage_order.get(stage, stage_order["moderate"])
        available = [role for role in ordered_roles if role in roles]
        if idx < len(available):
            role_choice = available[idx]
        else:
            role_choice = roles[idx % len(roles)] if roles else plan.get("role") or "Kitchen"
        is_expo = "expo" in role_choice.lower()
        return role_choice, is_expo

    def _role_for_group_default(self, group_name: str, roles: List[str]) -> str:
        if roles:
            return roles[0]
        return group_name

    def _assign_employees(self, plans: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        assignments: List[Dict[str, Any]] = []
        for plan in plans:
            demand = BlockDemand(
                day_index=plan["day_index"],
                date=plan["date"],
                start=plan["start"],
                end=plan["end"],
                role=plan["role"],
                block_name=plan.get("style", "Mid"),
                labels=[plan.get("style", "Mid")],
                need=1,
                priority=1.0,
                minimum=1,
                allow_cuts=not plan.get("essential", False),
                always_on=plan.get("essential", False),
                role_group=plan["role_group"],
                hourly_rate=self._role_wage(plan["role"]),
            )
            demand.recommended_cut = plan["end"]
            candidate = self._select_employee(demand)
            if candidate:
                self._register_assignment(candidate, demand)
            payload = self._build_assignment_payload(candidate if candidate else None, demand, override_end=plan["end"])
            payload["location"] = plan.get("style", "Mid")
            payload["notes"] = self._append_note(payload.get("notes"), "Template start")
            if plan.get("essential"):
                payload["notes"] = self._append_note(payload.get("notes"), "Essential coverage")
            payload["_style"] = plan.get("style", "Mid")
            payload["_role_group"] = plan.get("role_group")
            payload["_slot_indices"] = plan.get("slot_indices", [])
            payload["_essential"] = plan.get("essential", False)
            assignments.append(payload)
            if not candidate:
                self.unfilled_slots.append({"payload": payload, "demand": demand})
        return assignments

    @staticmethod
    def _slot_indices_for_range(slots: List[Dict[str, Any]], start: datetime.datetime, end: datetime.datetime) -> List[int]:
        indices: List[int] = []
        for idx, slot in enumerate(slots):
            if start < slot["end"] and end > slot["start"]:
                indices.append(idx)
        return indices

    def _apply_budget_cuts(
        self, assignments: List[Dict[str, Any]], slot_matrix: Optional[Dict[Tuple[int, str], Dict[str, Any]]]
    ) -> None:
        """
        Trim shifts from the back in 15/30-minute steps when a day/group is over budget.
        Protects essential roles (one bartender, one expo, one floor server) and enforces
        slot minima so coverage never falls below target - tolerance.
        """
        if not assignments:
            return
        matrix = slot_matrix or getattr(self, "current_slot_matrix", {}) or {}
        coverage = self._build_coverage(assignments, matrix)
        cut_order: Dict[Tuple[int, str], int] = defaultdict(int)
        tolerance = 1
        for (day_index, group_name), payload in sorted(matrix.items()):
            slots = payload.get("slots", [])
            if not slots:
                continue
            budget = None
            if 0 <= day_index < len(self.group_budget_by_day):
                budget = self.group_budget_by_day[day_index].get(group_name)
            if budget is None or budget <= 0:
                continue
            open_dt = payload.get("open")
            close_dt = payload.get("close")
            current_cost = self._group_cost(assignments, group_name, open_dt, close_dt)
            if current_cost <= budget:
                continue
            candidates = self._cut_candidates(assignments, group_name, open_dt, close_dt)
            for shift in candidates:
                if current_cost <= budget:
                    break
                for trim_minutes in (60, 30):
                    if self._attempt_trim_shift(
                        shift, trim_minutes, slots, coverage[(day_index, group_name)], tolerance
                    ):
                        cut_order[(day_index, group_name)] += 1
                        ordinal = self._ordinal_label(cut_order[(day_index, group_name)])
                        shift["notes"] = self._append_note(shift.get("notes"), f"{ordinal} cut")
                        new_cost = self._compute_cost(shift["start"], shift["end"], shift.get("labor_rate", 0.0))
                        self.cut_insights.append(
                            {
                                "day": slots[0]["date"].isoformat(),
                                "role_group": group_name,
                                "shift_start": shift["start"].isoformat(),
                                "cut_time": shift["end"].isoformat(),
                                "minutes_trimmed": trim_minutes,
                            }
                        )
                        current_cost -= max(0.0, shift.get("labor_cost", 0.0) - new_cost)
                        shift["labor_cost"] = new_cost
                        break
            coverage[(day_index, group_name)] = coverage.get((day_index, group_name), [])
        self._log_coverage_tables(matrix, coverage)

    def _group_cost(
        self, assignments: List[Dict[str, Any]], group_name: str, open_dt: Optional[datetime.datetime], close_dt: Optional[datetime.datetime]
    ) -> float:
        total = 0.0
        for shift in assignments:
            shift_group = shift.get("_role_group") or self._canonical_group(role_group(shift.get("role")))
            if shift_group != group_name:
                continue
            if not self._overlaps_window(shift, open_dt, close_dt):
                continue
            total += float(shift.get("labor_cost", 0.0) or 0.0)
        return total

    def _cut_candidates(
        self, assignments: List[Dict[str, Any]], group_name: str, open_dt: Optional[datetime.datetime], close_dt: Optional[datetime.datetime]
    ) -> List[Dict[str, Any]]:
        candidates: List[Dict[str, Any]] = []
        for shift in assignments:
            shift_group = shift.get("_role_group") or self._canonical_group(role_group(shift.get("role")))
            if shift_group != group_name or shift.get("_essential"):
                continue
            if not self._overlaps_window(shift, open_dt, close_dt):
                continue
            candidates.append(shift)
        return sorted(candidates, key=self._cut_sort_key_simple)

    def _cut_sort_key_simple(self, shift: Dict[str, Any]) -> Tuple[Any, ...]:
        style = shift.get("_style", "Mid")
        role_label = (shift.get("role") or "").lower()
        group_label = shift.get("_role_group") or self._canonical_group(role_group(shift.get("role")))
        if group_label == "Kitchen":
            hoh_order = [normalize_role(name) for name in self.pre_engine_staffing.get("hoh", {}).get("cut_priority", [])]
            role_norm = normalize_role(shift.get("role"))
            role_rank = hoh_order.index(role_norm) if role_norm in hoh_order else len(hoh_order) + 1
        elif "cocktail" in role_label:
            role_rank = 0
        elif "server" in role_label:
            role_rank = 1
        elif "bartender" in role_label:
            role_rank = 3
        else:
            role_rank = 2
        return (
            SHIFT_STYLE_ORDER.get(style, 5),
            shift.get("start"),
            role_rank,
            -(shift.get("end") - shift.get("start")).total_seconds(),
        )

    def _attempt_trim_shift(
        self,
        shift: Dict[str, Any],
        trim_minutes: int,
        slots: List[Dict[str, Any]],
        coverage_row: List[int],
        tolerance: int,
    ) -> bool:
        if trim_minutes <= 0 or not slots:
            return False
        new_end = shift["end"] - datetime.timedelta(minutes=trim_minutes)
        if new_end <= shift["start"]:
            return False
        duration_hours = (new_end - shift["start"]).total_seconds() / 3600.0
        if duration_hours < MIN_SHIFT_HOURS:
            return False
        trimmed_indices = self._slot_indices_for_range(slots, new_end, shift["end"])
        if not trimmed_indices:
            return False
        for idx in trimmed_indices:
            if idx >= len(coverage_row):
                continue
            if coverage_row[idx] - 1 < slots[idx]["minimum"]:
                return False
            if coverage_row[idx] - 1 < max(slots[idx]["target"] - tolerance, slots[idx]["minimum"]):
                return False
        for idx in trimmed_indices:
            if idx < len(coverage_row):
                coverage_row[idx] = max(0, coverage_row[idx] - 1)
        shift["end"] = new_end
        return True

    def _build_coverage(
        self, assignments: List[Dict[str, Any]], matrix: Dict[Tuple[int, str], Dict[str, Any]]
    ) -> Dict[Tuple[int, str], List[int]]:
        coverage: Dict[Tuple[int, str], List[int]] = {}
        for key, payload in matrix.items():
            coverage[key] = [0 for _ in payload.get("slots", [])]
        for shift in assignments:
            shift_group = shift.get("_role_group") or self._canonical_group(role_group(shift.get("role")))
            for (day_index, group_name), payload in matrix.items():
                if group_name != shift_group:
                    continue
                slots = payload.get("slots", [])
                if not slots or not self._overlaps_window(shift, payload.get("open"), payload.get("close")):
                    continue
                for idx in self._slot_indices_for_range(slots, shift["start"], shift["end"]):
                    if idx < len(coverage[(day_index, group_name)]):
                        coverage[(day_index, group_name)][idx] += 1
        return coverage

    @staticmethod
    def _overlaps_window(
        shift: Dict[str, Any], start_dt: Optional[datetime.datetime], end_dt: Optional[datetime.datetime]
    ) -> bool:
        if not start_dt or not end_dt:
            return True
        return shift["start"] < end_dt and shift["end"] > start_dt

    def _log_coverage_tables(
        self, matrix: Dict[Tuple[int, str], Dict[str, Any]], coverage: Dict[Tuple[int, str], List[int]]
    ) -> None:
        """Print a simple time vs coverage table for visual validation."""
        for day_index in range(7):
            day_groups = [(key, payload) for key, payload in matrix.items() if key[0] == day_index]
            if not day_groups:
                continue
            date_label = day_groups[0][1]["slots"][0]["date"].isoformat() if day_groups[0][1].get("slots") else ""
            print(f"[coverage] {date_label}")
            for (idx, group_name), payload in sorted(
                day_groups, key=lambda item: item[1]["slots"][0]["start"] if item[1].get("slots") else datetime.datetime.min
            ):
                slots = payload.get("slots", [])
                if not slots:
                    continue
                cov_row = coverage.get((idx, group_name), [0 for _ in slots])
                hourly: List[str] = []
                current_hour = None
                bucket: List[int] = []
                for slot, cov in zip(slots, cov_row):
                    hour = slot["start"].replace(minute=0, second=0, microsecond=0)
                    if current_hour is None:
                        current_hour = hour
                    if hour != current_hour:
                        hourly.append(f"{current_hour.strftime('%H:%M')}:{max(bucket) if bucket else 0}")
                        bucket = []
                        current_hour = hour
                    bucket.append(cov)
                if bucket:
                    hourly.append(f"{current_hour.strftime('%H:%M')}:{max(bucket)}")
                print(f"  {group_name}: " + " ".join(hourly))

    def _retry_unfilled_assignments(self, assignments: List[Dict[str, Any]]) -> None:
        if not self.unfilled_slots:
            return
        remaining: List[Dict[str, Any]] = []
        for slot in self.unfilled_slots:
            payload = slot.get("payload")
            demand = slot.get("demand")
            if not payload or not isinstance(demand, BlockDemand):
                continue
            if payload.get("employee_id"):
                continue
            candidate = self._find_emergency_candidate(demand)
            if not candidate:
                if self._attempt_manager_fallback(demand, payload):
                    continue
                remaining.append(slot)
                continue
            self._apply_recovery_assignment(candidate, demand, payload)
        for slot in remaining:
            demand = slot.get("demand")
            if not isinstance(demand, BlockDemand):
                continue
            self.warnings.append(
                f"No coverage for {demand.role} on {demand.date.isoformat()} "
                f"{demand.start.strftime('%H:%M')} - {demand.end.strftime('%H:%M')} ({demand.block_name})"
            )
        self.unfilled_slots = []

    def _find_emergency_candidate(self, demand: BlockDemand) -> Optional[Dict[str, Any]]:
        ordered = sorted(self.employees, key=lambda record: record.get("total_hours", 0.0))
        for employee in ordered:
            if not self._employee_can_cover_role(employee, demand.role):
                continue
            if not self._employee_available(
                employee,
                demand,
                allow_desired_overflow=True,
                ignore_split=self.allow_split_shifts,
            ):
                continue
            return employee
        return None

    def _attempt_manager_fallback(self, demand: BlockDemand, payload: Dict[str, Any]) -> bool:
        """Reserve manager fallback for last-resort coverage with strict limits."""
        if not self.manager_fallback_allowed:
            return False
        role_norm = normalize_role(demand.role)
        if any(token in role_norm for token in self.manager_fallback_disallow):
            return False
        if "bartender" in role_norm or "expo" in role_norm or "opener" in role_norm or "closer" in role_norm:
            return False
        period = "am" if demand.start.hour < 15 else "pm"
        if self.manager_fallback_counts.get(demand.day_index, {}).get(period, 0) >= self.manager_fallback_limits.get(
            period, 0
        ):
            return False
        self.manager_fallback_counts[demand.day_index][period] += 1
        payload["employee_id"] = None
        payload["labor_rate"] = 0.0
        payload["labor_cost"] = 0.0
        payload["notes"] = self._append_note(payload.get("notes"), self.manager_fallback_tag)
        payload["_manager_fallback"] = True
        self.warnings.append(
            f"MGR–FOH fallback assigned to {demand.role} on {demand.date.isoformat()} "
            f"{demand.start.strftime('%H:%M')}–{demand.end.strftime('%H:%M')} (period {period.upper()})."
        )
        return True

    def _apply_recovery_assignment(
        self,
        employee: Dict[str, Any],
        demand: BlockDemand,
        payload: Dict[str, Any],
    ) -> None:
        start_dt = payload.get("start", demand.start)
        end_dt = payload.get("end", demand.end)
        adjusted = BlockDemand(
            day_index=demand.day_index,
            date=demand.date,
            start=start_dt,
            end=end_dt,
            role=demand.role,
            block_name=demand.block_name,
            labels=list(demand.labels),
            need=1,
            priority=demand.priority,
            minimum=1,
            allow_cuts=False,
            always_on=demand.always_on,
            role_group=demand.role_group,
            hourly_rate=demand.hourly_rate,
        )
        adjusted.recommended_cut = end_dt
        adjusted.max_capacity = 1
        self._register_assignment(employee, adjusted)
        rate = self._employee_role_wage(employee, demand.role)
        payload["employee_id"] = employee.get("id")
        payload["labor_rate"] = rate
        payload["labor_cost"] = self._compute_cost(start_dt, end_dt, rate)
        payload["notes"] = self._append_note(payload.get("notes"), "Recovered coverage")

    def _apply_staggered_cuts_for_demand(
        self,
        demand: BlockDemand,
        entries: List[Tuple[Dict[str, Any], Optional[Dict[str, Any]]]],
    ) -> None:
        """Stagger cut times within a block, prioritizing longest-worked staff first."""
        if not entries:
            return

        base_labels = [
            label
            for label in demand.labels
            if not label.lower().startswith("cut around") and label.lower() != "pattern"
        ]
        if not base_labels:
            base_labels = [demand.block_name]

        block_is_close = demand.block_name.strip().lower() == "close"
        is_closer = self._is_closer_block(demand.role, demand.block_name) or block_is_close

        min_hours, _ = shift_length_limits(self.policy, demand.role, demand.role_group)
        if not is_closer:
            # Allow shorter-than-policy shifts *a little* for non-closers,
            # but don't go all the way to zero.
            min_hours = min(min_hours, 1.5)
        min_duration = datetime.timedelta(minutes=int(min_hours * 60)) if not is_closer else datetime.timedelta()
        block_len = demand.end - demand.start
        if min_duration > block_len:
            min_duration = block_len

        latest_cut = demand.recommended_cut
        if not demand.allow_cuts or not latest_cut:
            # No cuts allowed or no recommended cut -> everyone works full block.
            for payload, employee in entries:
                end_time = demand.end
                if end_time < demand.start + min_duration:
                    end_time = demand.start + min_duration
                if end_time > demand.end:
                    end_time = demand.end
                self._finalize_assignment_payload(payload, employee, end_time, base_labels, demand)
            return

        total_slots = len(entries)
        if total_slots <= 0:
            return

        core_slots = min(total_slots, max(0, demand.minimum))
        cuttable_slots = max(0, total_slots - core_slots)

        if latest_cut > demand.end:
            latest_cut = demand.end
        if latest_cut < demand.start + min_duration:
            latest_cut = demand.start + min_duration

        start_minutes, _ = self._demand_window_minutes(demand)
        pressure = self.group_pressure.get(demand.day_index, {}).get(demand.role_group, 1.0)
        demand_index = 1.0
        if 0 <= demand.day_index < len(self.day_contexts):
            demand_index = self.day_contexts[demand.day_index].get("indices", {}).get("demand_index", 1.0)

        stagger_step = 12
        if pressure >= 1.15:
            stagger_step = 10
        if pressure >= 1.35:
            stagger_step = 8
        if pressure >= 1.6:
            stagger_step = 6
        if demand_index < 0.9:
            stagger_step += 3
        stagger_step = max(4, min(20, stagger_step))

        softness = max(0.0, 1.0 - demand_index)
        jitter_range = int(round(softness * 4 + max(0.0, pressure - 1.0) * 3))

        # Rank potential cuts.
        scored: List[Tuple[float, int]] = []
        fifo_mode = self.open_close_order_mode
        for idx, (_payload, employee) in enumerate(entries):
            base_score = self._cut_priority_score(employee, demand, start_minutes, pressure)
            fifo_bonus = 0.0
            if fifo_mode in {"prefer", "enforce"}:
                fifo_bonus = self._fifo_order_weight(employee, demand, start_minutes)
            if fifo_mode == "enforce":
                # Strong FIFO: base score secondary to start order.
                score = (fifo_bonus * 5.0) + (base_score * 0.25)
            elif fifo_mode == "prefer":
                score = base_score + fifo_bonus
            else:
                score = base_score
            scored.append((score, idx))
        scored.sort(key=lambda item: item[0], reverse=True)

        early_indices = [idx for _score, idx in scored[:cuttable_slots]]
        planned_end_times: List[datetime.datetime] = [latest_cut for _ in entries]
        planned_labels: List[List[str]] = [list(base_labels) for _ in entries]

        span = max(datetime.timedelta(), latest_cut - (demand.start + min_duration))
        span_minutes = max(0, int(span.total_seconds() // 60))
        linear_step = span_minutes / max(1, len(early_indices) - 1) if len(early_indices) > 1 else span_minutes
        linear_step = max(4, min(stagger_step, int(linear_step))) if linear_step > 0 else stagger_step

        for order, idx in enumerate(early_indices):
            payload, employee = entries[idx]
            jitter = self.random.randint(-jitter_range, jitter_range) if jitter_range else 0
            cut_time = latest_cut - datetime.timedelta(minutes=(linear_step * order) + jitter)
            floor_time = demand.start + min_duration
            if cut_time < floor_time:
                cut_time = floor_time
            if cut_time > latest_cut:
                cut_time = latest_cut
            planned_end_times[idx] = cut_time
            # NOTE: do not label here; labels are assigned after chronological sort.

        remaining_indices = [idx for _score, idx in scored if idx not in early_indices]
        trailing_step = max(3, min(10, (stagger_step // 2) + 2))
        for order, idx in enumerate(remaining_indices):
            payload, employee = entries[idx]
            back_offset = max(len(remaining_indices) - order - 1, 0)
            cut_time = latest_cut - datetime.timedelta(minutes=trailing_step * back_offset)
            if jitter_range and len(remaining_indices) > 1:
                jitter = self.random.randint(0, jitter_range)
                cut_time = cut_time - datetime.timedelta(minutes=jitter)
            if cut_time < demand.start + min_duration:
                cut_time = demand.start + min_duration
            if cut_time > latest_cut:
                cut_time = latest_cut
            planned_end_times[idx] = cut_time

        if self.open_close_order_mode != "off" and len(entries) > 1:
            violations, locked_only = self._fifo_violation_state(entries, planned_end_times, demand, start_minutes)
            if violations and not locked_only and self.open_close_order_mode in {"prefer", "enforce"}:
                if self._rebalance_fifo_entries(
                    entries, planned_end_times, demand, start_minutes, min_duration, violations
                ):
                    violations, locked_only = self._fifo_violation_state(
                        entries, planned_end_times, demand, start_minutes
                    )
            if violations and not locked_only:
                self.warnings.append(
                    f"Could not fully honor opener/closer order for {demand.role_group} on {self._day_label(demand.day_index)}; review cuts."
                )

        self._apply_cut_labels(planned_labels, planned_end_times, base_labels, early_indices, remaining_indices)

        for idx, (payload, employee) in enumerate(entries):
            self._finalize_assignment_payload(payload, employee, planned_end_times[idx], planned_labels[idx], demand)

    def _finalize_assignment_payload(
        self,
        payload: Dict[str, Any],
        employee: Optional[Dict[str, Any]],
        end_time: datetime.datetime,
        labels: List[str],
        demand: BlockDemand,
    ) -> None:
        payload["end"] = end_time
        rate = self._employee_role_wage(employee, demand.role)
        payload["labor_rate"] = rate
        payload["labor_cost"] = self._compute_cost(payload["start"], end_time, rate)
        payload["notes"] = ", ".join(labels)
        if employee and end_time < demand.end:
            delta_hours = max(0.0, (demand.end - end_time).total_seconds() / 3600)
            employee["total_hours"] = max(0.0, employee.get("total_hours", 0.0) - delta_hours)

    def _cut_priority_score(
        self,
        employee: Optional[Dict[str, Any]],
        demand: BlockDemand,
        start_minutes: int,
        pressure: float,
    ) -> float:
        """Higher scores mean the employee should be released earlier."""
        if not employee:
            return float("-inf")
        day_assignments = employee["assignments"].get(demand.day_index, [])
        day_minutes = sum(end - start for start, end in day_assignments)
        day_hours = day_minutes / 60.0
        earliest_start = min((start for start, _ in day_assignments), default=start_minutes)
        started_early = earliest_start < start_minutes
        long_day_bonus = 1.5 if day_hours >= 7 else (0.75 if day_hours >= 5 else 0.0)
        weekly_hours = employee.get("total_hours", 0.0)
        pressure_bonus = max(0.0, pressure - 1.0) * 2.0
        return (weekly_hours * 0.7) + (day_hours * 1.6) + (2.0 if started_early else 0.0) + long_day_bonus + pressure_bonus

    def _fifo_order_weight(self, employee: Optional[Dict[str, Any]], demand: BlockDemand, start_minutes: int) -> float:
        """Higher weight means this person should leave earlier based on earliest start (FIFO)."""
        if not employee:
            return 0.0
        day_assignments = employee["assignments"].get(demand.day_index, [])
        if not day_assignments:
            return 0.0
        earliest_start = min(start for start, _ in day_assignments)
        age_minutes = max(0, start_minutes - earliest_start)
        # 30 minute chunks, capped to avoid overpowering budget logic.
        return min(6.0, age_minutes / 30.0)

    @staticmethod
    def _ordinal_label(value: int) -> str:
        if 10 <= value % 100 <= 20:
            suffix = "th"
        else:
            suffix = {1: "st", 2: "nd", 3: "rd"}.get(value % 10, "th")
        return f"{value}{suffix}"

    def _select_employee(self, demand: BlockDemand) -> Optional[Dict[str, Any]]:
        candidate_sets: List[List[Dict[str, Any]]] = []
        pending_openers = self._pending_opener_candidates(demand)
        if pending_openers:
            candidate_sets.append(pending_openers)
        candidate_sets.append(self.employees)

        for allow_overflow in (False, True):
            for pool in candidate_sets:
                if not pool:
                    continue
                exact_pool = [candidate for candidate in pool if demand.role in candidate.get("roles", set())]
                candidates = exact_pool if exact_pool else pool
                best_candidate = None
                best_score = float("-inf")
                for employee in candidates:
                    if not self._employee_can_cover_role(employee, demand.role):
                        continue
                    if not self._employee_available(employee, demand, allow_desired_overflow=allow_overflow):
                        continue
                    score = self._score_candidate(employee, demand, allow_overflow=allow_overflow)
                    if score > best_score:
                        best_score = score
                        best_candidate = employee
                if best_candidate:
                    return best_candidate
        return None

    def _pending_opener_candidates(self, demand: BlockDemand) -> List[Dict[str, Any]]:
        start_minutes, _ = self._demand_window_minutes(demand)
        tolerance = max(5, self.round_to_minutes)
        normalized_role = normalize_role(demand.role)
        matches: Dict[int, Tuple[int, Dict[str, Any]]] = {}
        for employee in self.employees:
            queue = employee.get("pending_open_links", {}).get(demand.day_index, [])
            if not queue:
                continue
            for link in queue:
                if link.get("fulfilled"):
                    continue
                target_start = link.get("target_start")
                deadline = link.get("deadline", target_start)
                if target_start is None:
                    continue
                if start_minutes < target_start - tolerance or start_minutes > deadline + tolerance:
                    continue
                covers = link.get("covers") or set()
                role_group = link.get("role_group")
                if normalized_role not in covers and demand.role_group != role_group:
                    continue
                existing = matches.get(employee["id"])
                if existing is None or target_start < existing[0]:
                    matches[employee["id"]] = (target_start, employee)
                break
        ordered = sorted(matches.values(), key=lambda entry: (entry[0], entry[1]["id"]))
        return [entry[1] for entry in ordered]

    def _employee_can_cover_role(self, employee: Dict[str, Any], role_name: str) -> bool:
        if not role_name:
            return False
        candidate_roles = employee.get("roles") or set()
        target_group = role_group(role_name)
        disallowed_groups = {"Kitchen"} if target_group == "Cashier" else set()
        for candidate in candidate_roles:
            candidate_group = role_group(candidate)
            if candidate_group in disallowed_groups:
                continue
            if candidate_group == "Management" and target_group in {"Servers", "Bartenders"}:
                continue
            if target_group == "Kitchen" and candidate_group not in {"Kitchen"}:
                # Allow explicit cross-train only if the employee has the exact role.
                if not role_matches(candidate, role_name):
                    continue
            if target_group in {"Servers", "Bartenders"} and candidate_group == "Kitchen":
                if not role_matches(candidate, role_name):
                    continue
            if role_matches(candidate, role_name):
                return True
            candidate_cfg = role_definition(self.policy, candidate)
            covers = candidate_cfg.get("covers") if isinstance(candidate_cfg, dict) else []
            if covers and role_name in covers:
                return True
        if target_group == "Cashier":
            # Cashier can be covered by Servers/Bartenders/Managers when explicitly assigned.
            for candidate in candidate_roles:
                candidate_group = role_group(candidate)
                if candidate_group in {"Servers", "Bartenders", "Management"}:
                    return True
        if target_group and target_group in self.interchangeable_groups:
            for candidate in candidate_roles:
                if role_group(candidate) == target_group:
                    return True
        role_cfg = role_definition(self.policy, role_name)
        qualifiers = role_cfg.get("qualifiers") if isinstance(role_cfg, dict) else None
        if qualifiers:
            for qualifier in qualifiers:
                for candidate in candidate_roles:
                    if role_matches(candidate, qualifier):
                        return True
        return False

    def _employee_available(
        self,
        employee: Dict[str, Any],
        demand: BlockDemand,
        *,
        allow_desired_overflow: bool = False,
        ignore_split: bool = False,
    ) -> bool:
        assignments = employee["assignments"][demand.day_index]
        demand_start_minutes, demand_end_minutes = self._demand_window_minutes(demand)
        for start_minute, end_minute in assignments:
            if demand_start_minutes < end_minute and demand_end_minutes > start_minute:
                return False
        if not self.allow_split_shifts and assignments and not ignore_split:
            return False
        for offset, seg_start, seg_end in self._demand_day_segments(demand):
            day_idx = (demand.day_index + offset) % 7
            for window_start, window_end in employee["unavailability"].get(day_idx, []):
                if seg_start < window_end and seg_end > window_start:
                    return False
        same_day_assignment = demand.day_index in employee["days_with_assignments"]
        last_end = employee["last_assignment_end"]
        if last_end and not same_day_assignment:
            pass
        block_hours = demand.duration_hours
        projected_hours = employee["total_hours"] + block_hours
        if projected_hours > self.max_hours_per_week + 1e-6:
            return False
        role_cap = role_definition(self.policy, demand.role).get("max_weekly_hours")
        try:
            if role_cap and projected_hours > float(role_cap):
                return False
        except (TypeError, ValueError):
            pass
        desired_ceiling = employee.get("desired_ceiling")
        if (
            not allow_desired_overflow
            and desired_ceiling is not None
            and desired_ceiling > 0
            and projected_hours > desired_ceiling + 1e-6
        ):
            return False
        if self._would_violate_consecutive(employee, demand.day_index):
            return False
        return True

    def _would_violate_consecutive(self, employee: Dict[str, Any], day_index: int) -> bool:
        if self.max_consecutive_days <= 0:
            return False
        if day_index in employee["days_with_assignments"]:
            return False
        last_day = employee.get("last_day_index")
        consecutive = employee.get("consecutive_days", 0)
        if last_day is None:
            projected = 1
        elif day_index == last_day:
            projected = consecutive
        elif last_day is not None and day_index == last_day + 1:
            projected = consecutive + 1
        else:
            projected = 1
        return projected > self.max_consecutive_days

    def _score_candidate(
        self,
        employee: Dict[str, Any],
        demand: BlockDemand,
        *,
        allow_overflow: bool = False,
    ) -> float:
        role_cfg = role_definition(self.policy, demand.role)
        priority = 0.5
        try:
            priority = float(role_cfg.get("priority", 0.5))
        except (TypeError, ValueError):
            priority = 0.5
        block_hours = demand.duration_hours
        projected_hours = employee["total_hours"] + block_hours
        desired = employee["desired_hours"] or employee.get("desired_ceiling") or self.max_hours_per_week
        desired = desired or self.max_hours_per_week
        floor = max(0.0, employee.get("desired_floor", 0.0))
        ceiling = max(floor, employee.get("desired_ceiling", self.max_hours_per_week) or self.max_hours_per_week)
        window_span = max(1.0, ceiling - floor)
        if projected_hours < floor:
            coverage_focus = 1.0 + (floor - projected_hours) / max(1.0, desired)
        elif projected_hours <= ceiling:
            coverage_focus = 0.6 * (ceiling - projected_hours) / window_span
        else:
            overflow = projected_hours - ceiling
            coverage_focus = -overflow / max(1.0, desired)
            if not allow_overflow:
                coverage_focus -= 0.5
        day_minutes = sum(end - start for start, end in employee["assignments"][demand.day_index])
        day_hours = day_minutes / 60.0
        continuity = 0.2 if self._continues_assignment(employee, demand) else 0.0
        if role_group(demand.role) == "Servers":
            continuity *= 0.5
        is_closer = self._is_closer_block(demand.role, demand.block_name) or demand.block_name.strip().lower() == "close"
        if is_closer:
            if employee["assignments"][demand.day_index]:
                continuity += 0.4
            else:
                continuity -= 0.3
        day_load = len(employee["assignments"][demand.day_index])
        availability_bonus = max(-0.15, 0.3 - 0.1 * day_load)
        day_fairness = max(-0.4, 0.15 * (1 - (day_hours / 6.0)))  # prefer those working less today
        if day_hours >= 7.0:
            day_fairness -= 0.25
        wage_penalty = self._role_wage(demand.role) * 0.02
        overtime_penalty = self.overtime_penalty if projected_hours > self.max_hours_per_week else 0.0
        consecutive_penalty = 0.05 * max(0, employee.get("consecutive_days", 0) - 3)
        distribution_bonus = max(-0.2, 0.2 * (1 - (employee["total_hours"] / max(1.0, ceiling))))
        return (
            priority
            + coverage_focus
            + continuity
            + availability_bonus
            + day_fairness
            + distribution_bonus
            - wage_penalty
            - overtime_penalty
            - consecutive_penalty
            + self.random.uniform(-0.05, 0.05)
        )

    def _continues_assignment(self, employee: Dict[str, Any], demand: BlockDemand) -> bool:
        last_end = employee["day_last_block_end"][demand.day_index]
        start_minutes, _ = self._demand_window_minutes(demand)
        if last_end is None:
            return False
        tolerance = max(1, self.round_to_minutes)
        return abs(start_minutes - last_end) <= tolerance

    def _closer_has_continuity(self, assignments: List[Tuple[int, int]], demand_start_minutes: int) -> bool:
        if not assignments:
            return False
        tolerance = max(5, self.round_to_minutes)
        latest_end = max(end for _start, end in assignments)
        earliest_start = min(start for start, _end in assignments)
        return latest_end >= demand_start_minutes - tolerance and earliest_start < demand_start_minutes

    def _register_assignment(self, employee: Dict[str, Any], demand: BlockDemand) -> None:
        start_minutes, end_minutes = self._demand_window_minutes(demand)
        employee["assignments"][demand.day_index].append((start_minutes, end_minutes))
        employee["day_last_block_end"][demand.day_index] = end_minutes
        employee["total_hours"] += demand.duration_hours
        employee["last_assignment_end"] = demand.end
        if demand.day_index not in employee["days_with_assignments"]:
            last_day = employee.get("last_day_index")
            if last_day is not None and demand.day_index == last_day + 1:
                employee["consecutive_days"] = employee.get("consecutive_days", 0) + 1
            else:
                employee["consecutive_days"] = 1
            employee["last_day_index"] = demand.day_index
            employee["days_with_assignments"].add(demand.day_index)
        self._track_opener_continuity(employee, demand, start_minutes, end_minutes)

    def _track_opener_continuity(
        self,
        employee: Optional[Dict[str, Any]],
        demand: BlockDemand,
        start_minutes: int,
        end_minutes: int,
    ) -> None:
        if not employee:
            return
        if self._is_opener_block(demand.role, demand.block_name):
            self._create_open_link_requirement(employee, demand, end_minutes)
        else:
            self._fulfill_open_link_requirement(employee, demand, start_minutes)

    def _create_open_link_requirement(self, employee: Dict[str, Any], demand: BlockDemand, end_minutes: int) -> None:
        queue = employee["pending_open_links"].setdefault(demand.day_index, [])
        role_cfg = role_definition(self.policy, demand.role)
        covers = {normalize_role(demand.role)}
        for cover in role_cfg.get("covers", []) or []:
            normalized = normalize_role(cover)
            if normalized:
                covers.add(normalized)
        tolerance = max(5, self.round_to_minutes)
        queue.append(
            {
                "target_start": end_minutes,
                "deadline": end_minutes + tolerance,
                "covers": covers,
                "role_group": demand.role_group,
                "role": demand.role,
                "fulfilled": False,
            }
        )

    def _fulfill_open_link_requirement(self, employee: Dict[str, Any], demand: BlockDemand, start_minutes: int) -> None:
        queue = employee["pending_open_links"].get(demand.day_index)
        if not queue:
            return
        normalized_role = normalize_role(demand.role)
        tolerance = max(5, self.round_to_minutes)
        for idx, link in enumerate(list(queue)):
            if link.get("fulfilled"):
                continue
            target_start = link.get("target_start")
            if target_start is None:
                continue
            deadline = link.get("deadline", target_start)
            if start_minutes < target_start - tolerance or start_minutes > deadline + tolerance:
                continue
            covers = link.get("covers") or set()
            role_group = link.get("role_group")
            if normalized_role not in covers and demand.role_group != role_group:
                continue
            queue.pop(idx)
            break

    def _enforce_shift_continuity(self, assignments: List[Dict[str, Any]], week_start: datetime.date) -> None:
        """
        Merge back-to-back or overlapping shifts for the same employee/role on the same day.
        Removes micro-fragments created by trimming and keeps coverage visually clean.
        """
        if not assignments:
            return
        tolerance = datetime.timedelta(minutes=max(5, self.round_to_minutes))
        grouped: Dict[Tuple[Any, str, datetime.date], List[Dict[str, Any]]] = defaultdict(list)
        for payload in assignments:
            emp_id = payload.get("employee_id")
            role_name = payload.get("role")
            key = (emp_id, role_name, payload["start"].date())
            grouped[key].append(payload)
        merged: List[Dict[str, Any]] = []
        for key, shifts in grouped.items():
            shifts.sort(key=lambda s: s["start"])
            current = shifts[0]
            for nxt in shifts[1:]:
                if nxt["start"] <= current["end"] + tolerance:
                    current["end"] = max(current["end"], nxt["end"])
                    current["labor_cost"] = self._compute_cost(
                        current["start"], current["end"], current.get("labor_rate", 0.0)
                    )
                    if current.get("_slot_indices") and nxt.get("_slot_indices"):
                        current["_slot_indices"] = sorted(
                            set(current["_slot_indices"]).union(set(nxt["_slot_indices"]))
                        )
                    if current.get("notes") and nxt.get("notes"):
                        current["notes"] = self._append_note(current["notes"], nxt["notes"])
                    continue
                merged.append(current)
                current = nxt
            merged.append(current)
        assignments.clear()
        assignments.extend(merged)

    def _ensure_opener_followups(
        self,
        day_map: Dict[int, List[Dict[str, Any]]],
        assignments: List[Dict[str, Any]],
        week_start: datetime.date,
    ) -> None:
        tolerance = datetime.timedelta(minutes=max(5, self.round_to_minutes))
        for employee in self.employees:
            links_by_day = employee.get("pending_open_links") or {}
            for day_index, links in list(links_by_day.items()):
                if not links:
                    continue
                active_shifts = [
                    shift
                    for shift in day_map.get(day_index, [])
                    if (shift.get("location") or "").strip().lower() not in {"open", "close"}
                ]
                if not active_shifts:
                    links_by_day[day_index] = []
                    continue
                for link in list(links):
                    if link.get("fulfilled"):
                        continue
                    target_dt = self._day_datetime(week_start, day_index, link.get("target_start", 0))
                    if self._assign_existing_followup(active_shifts, employee, target_dt, link, tolerance):
                        links.remove(link)
                        continue
                    if self._create_open_followup_shift(
                        assignments, day_map, day_index, employee, target_dt, link, week_start, active_shifts
                    ):
                        links.remove(link)
                if not links:
                    links_by_day[day_index] = []

    def _assign_existing_followup(
        self,
        day_shifts: List[Dict[str, Any]],
        employee: Dict[str, Any],
        target_dt: datetime.datetime,
        link: Dict[str, Any],
        tolerance: datetime.timedelta,
    ) -> bool:
        if not day_shifts:
            return False
        emp_id = employee.get("id")
        if not emp_id:
            return True
        normalized_role = normalize_role(link.get("role") or "")
        candidates: List[Dict[str, Any]] = []
        for shift in day_shifts:
            loc = (shift.get("location") or "").strip().lower()
            if loc in {"open", "close"}:
                continue
            if shift.get("employee_id") == emp_id:
                if shift["start"] >= target_dt - tolerance:
                    return True
                continue
            if shift.get("_followup_locked"):
                continue
            if shift["start"] < target_dt - tolerance:
                continue
            if normalized_role and normalize_role(shift["role"]) != normalized_role:
                covers = link.get("covers") or set()
                candidate_group = role_group(shift["role"])
                if normalize_role(shift["role"]) not in covers and link.get("role_group") != candidate_group:
                    continue
            candidates.append(shift)
        if not candidates:
            return False
        chosen = sorted(candidates, key=lambda payload: (payload["start"], payload["end"]))[0]
        self._transfer_shift_employee(chosen, emp_id, tag="Opener follow-up")
        chosen["_followup_locked"] = True
        return True

    def _create_open_followup_shift(
        self,
        assignments: List[Dict[str, Any]],
        day_map: Dict[int, List[Dict[str, Any]]],
        day_index: int,
        employee: Dict[str, Any],
        target_dt: datetime.datetime,
        link: Dict[str, Any],
        week_start: datetime.date,
        day_shifts: List[Dict[str, Any]],
    ) -> bool:
        if not day_shifts or not self.allow_split_shifts:
            return False
        role_name = link.get("role") or next(iter(employee["roles"]), None)
        if not role_name:
            return False
        min_hours, _ = shift_length_limits(self.policy, role_name, link.get("role_group") or role_group(role_name))
        duration = datetime.timedelta(hours=max(2.5, min_hours))
        start_dt = target_dt
        end_dt = target_dt + duration
        payload = self._create_manual_shift(
            employee.get("id"),
            role_name,
            start_dt,
            end_dt,
            location="Follow-up",
            note="Auto follow-up for opener",
        )
        assignments.append(payload)
        day_map[day_index].append(payload)
        return True

    def _ensure_closer_continuity(
        self,
        day_map: Dict[int, List[Dict[str, Any]]],
        assignments: List[Dict[str, Any]],
        week_start: datetime.date,
    ) -> None:
        tolerance = datetime.timedelta(minutes=max(5, self.round_to_minutes))
        for day_index, shifts in list(day_map.items()):
            for shift in list(shifts):
                location = (shift.get("location") or "").strip().lower()
                if location != "close":
                    continue
                emp_id = shift.get("employee_id")
                if not emp_id:
                    continue
                op_day = self._closer_operational_day_index(week_start, shift["start"])
                closer_group = role_group(shift.get("role"))
                if self._closer_has_prior_assignment(
                    emp_id, shift["start"], day_map.get(op_day, []), tolerance, closer_group
                ):
                    continue
                if self._pair_closer_with_existing(emp_id, shift, day_map.get(op_day, []), tolerance):
                    continue
                self._create_closer_lead_in(assignments, day_map, op_day, shift, emp_id)

    def _closer_operational_day_index(self, week_start: datetime.date, start_dt: datetime.datetime) -> int:
        op_date = start_dt.date()
        if start_dt.time() < datetime.time(6, 0):
            op_date = start_dt.date() - datetime.timedelta(days=1)
        return (op_date - week_start).days

    def _closer_has_prior_assignment(
        self,
        employee_id: int,
        close_start: datetime.datetime,
        day_shifts: List[Dict[str, Any]],
        tolerance: datetime.timedelta,
        closer_group: Optional[str],
    ) -> bool:
        if not day_shifts:
            return False
        for shift in day_shifts:
            if shift.get("employee_id") != employee_id:
                continue
            loc = (shift.get("location") or "").strip().lower()
            if loc in {"open", "close"}:
                continue
            if closer_group and role_group(shift.get("role")) != closer_group:
                continue
            if shift["end"] > close_start - tolerance:
                return True
        return False

    def _pair_closer_with_existing(
        self,
        employee_id: int,
        close_shift: Dict[str, Any],
        day_shifts: List[Dict[str, Any]],
        tolerance: datetime.timedelta,
    ) -> bool:
        if not day_shifts:
            return False
        closer_group = role_group(close_shift.get("role"))
        candidates: List[Dict[str, Any]] = []
        for shift in day_shifts:
            loc = (shift.get("location") or "").strip().lower()
            if loc in {"open", "close"}:
                continue
            if closer_group and role_group(shift.get("role")) != closer_group:
                continue
            if shift.get("employee_id") == employee_id:
                if shift["end"] > close_shift["start"] - tolerance:
                    return True
                continue
            if shift.get("_followup_locked"):
                continue
            if shift["end"] <= close_shift["start"]:
                candidates.append(shift)
        if not candidates:
            return False
        chosen = sorted(candidates, key=lambda payload: payload["end"], reverse=True)[0]
        self._transfer_shift_employee(chosen, employee_id, tag="Closer lead-in")
        chosen["_followup_locked"] = True
        return True

    def _create_closer_lead_in(
        self,
        assignments: List[Dict[str, Any]],
        day_map: Dict[int, List[Dict[str, Any]]],
        op_day_index: int,
        close_shift: Dict[str, Any],
        employee_id: int,
    ) -> None:
        role_name = close_shift.get("role")
        if not role_name:
            return
        min_hours, _ = shift_length_limits(self.policy, role_name, role_group(role_name))
        duration = datetime.timedelta(hours=max(2.5, min_hours))
        start_dt = close_shift["start"] - duration
        payload = self._create_manual_shift(
            employee_id,
            role_name,
            start_dt,
            close_shift["start"],
            location="Close lead",
            note="Auto lead-in for closer",
        )
        payload["_followup_locked"] = True
        assignments.append(payload)
        day_map[op_day_index].append(payload)

    def _transfer_shift_employee(self, shift: Dict[str, Any], new_employee_id: Optional[int], *, tag: str) -> None:
        old_employee_id = shift.get("employee_id")
        if old_employee_id == new_employee_id or new_employee_id is None:
            return
        shift["employee_id"] = new_employee_id
        shift["notes"] = self._append_note(shift.get("notes"), tag)

    def _create_manual_shift(
        self,
        employee_id: Optional[int],
        role_name: str,
        start_dt: datetime.datetime,
        end_dt: datetime.datetime,
        *,
        location: str,
        note: str,
    ) -> Dict[str, Any]:
        employee = self.employee_lookup.get(employee_id) if employee_id else None
        rate = self._employee_role_wage(employee, role_name)
        payload = {
            "employee_id": employee_id,
            "role": role_name,
            "start": start_dt,
            "end": end_dt,
            "labor_rate": rate,
            "labor_cost": self._compute_cost(start_dt, end_dt, rate),
            "location": location,
            "notes": note,
        }
        return payload

    def _append_note(self, existing: Optional[str], addition: str) -> str:
        if not existing:
            return addition
        if addition.lower() in existing.lower():
            return existing
        return f"{existing}, {addition}"

    def _day_datetime(self, week_start: datetime.date, day_index: int, minutes: int) -> datetime.datetime:
        day = week_start + datetime.timedelta(days=day_index)
        return datetime.datetime.combine(day, datetime.time.min, tzinfo=UTC) + datetime.timedelta(minutes=minutes)

    def _day_index_from_datetime(self, week_start: datetime.date, dt: datetime.datetime) -> int:
        return (dt.date() - week_start).days

    def _warn_unpaired_openers(self) -> None:
        for employee in self.employees:
            for day_index, links in (employee.get("pending_open_links") or {}).items():
                if not links:
                    continue
                for _link in links:
                    self.warnings.append(
                        f"Opener continuity missing for {employee['name']} on {self._day_label(day_index)}; add a follow-up shift."
                    )
                links.clear()

    def _check_section_thresholds(self, assignments: List[Dict[str, Any]]) -> None:
        server_cfg = self.pre_engine_staffing.get("servers", {})
        dining_cfg = server_cfg.get("dining", {})
        cocktail_cfg = server_cfg.get("cocktail", {})
        cashier_cfg = self.pre_engine_staffing.get("cashier", {})
        for day_index in range(7):
            demand_index = 1.0
            ctx_date = None
            if 0 <= day_index < len(self.day_contexts):
                demand_index = self.day_contexts[day_index].get("indices", {}).get("demand_index", 1.0)
                ctx_date = self.day_contexts[day_index].get("date")
            tier = self._volume_tier(demand_index)
            dining_count = self._count_roles_for_day(
                assignments, day_index, lambda name: "server" in name and "cocktail" not in name and "patio" not in name
            )
            cocktail_count = self._count_roles_for_day(assignments, day_index, lambda name: "cocktail" in name)
            cashier_count = self._count_roles_for_day(assignments, day_index, lambda name: "cashier" in name or "host" in name)
            if tier == "slow" and dining_count > int(dining_cfg.get("slow_max", 4)):
                self.warnings.append(
                    f"Dining staffing {dining_count} exceeds slow cap ({dining_cfg.get('slow_max', 4)}) on {ctx_date or self._day_label(day_index)}."
                )
            if tier in {"moderate", "peak"} and dining_count < int(dining_cfg.get("slow_min", 1)):
                self.errors.append(f"Dining staffing below minimum on {ctx_date or self._day_label(day_index)}.")
            manual_dining = int(dining_cfg.get("manual_max", 7))
            if manual_dining and dining_count > manual_dining:
                self.warnings.append(
                    f"Dining staffing {dining_count} exceeds manual override threshold ({manual_dining}) on {ctx_date or self._day_label(day_index)}."
                )
            if tier == "peak" and dining_count < int(dining_cfg.get("peak", 6)):
                self.warnings.append(
                    f"Dining staffing below peak target ({dining_cfg.get('peak', 6)}) on {ctx_date or self._day_label(day_index)}."
                )
            if tier == "peak" and cocktail_count > int(cocktail_cfg.get("manual_max", 4)):
                self.warnings.append(
                    f"Cocktail staffing {cocktail_count} requires manual override (>{cocktail_cfg.get('manual_max', 4)}) on {ctx_date or self._day_label(day_index)}."
                )
            if tier == "slow" and cocktail_count < int(cocktail_cfg.get("normal", 2)):
                self.errors.append(f"Cocktail staffing below normal ({cocktail_cfg.get('normal', 2)}) on {ctx_date or self._day_label(day_index)}.")
            if cashier_count > int(cashier_cfg.get("manual_max", 4)):
                self.warnings.append(
                    f"Cashier staffing {cashier_count} exceeds manual override threshold ({cashier_cfg.get('manual_max', 4)}) on {ctx_date or self._day_label(day_index)}."
                )
            if cashier_count < int(cashier_cfg.get("am_default", 1)):
                self.errors.append(f"Cashier coverage missing on {ctx_date or self._day_label(day_index)}.")

    def _record_required_role_gaps(self, assignments: List[Dict[str, Any]]) -> None:
        if not self.required_role_labels:
            return
        for day_index in range(7):
            for role_label in self.required_role_labels:
                if any(
                    role_matches(shift.get("role", ""), role_label) and self._day_index_for_shift(shift) == day_index
                    for shift in assignments
                ):
                    continue
                self.errors.append(f"Missing required role {role_label} on {self._day_label(day_index)}.")

    def _count_roles_for_day(
        self, assignments: List[Dict[str, Any]], day_index: int, predicate: Callable[[str], bool]
    ) -> int:
        count = 0
        for shift in assignments:
            if self._day_index_for_shift(shift) != day_index:
                continue
            role_name = (shift.get("role") or "").lower()
            if "training" in role_name:
                continue
            if predicate(role_name):
                count += 1
        return count

    def _day_index_for_shift(self, shift: Dict[str, Any]) -> int:
        start_date = shift.get("start").date() if isinstance(shift.get("start"), datetime.datetime) else None
        for ctx in self.day_contexts:
            if ctx.get("date") == start_date:
                return ctx.get("day_index", start_date.weekday() if start_date else 0)
        return start_date.weekday() if start_date else 0

    def _volume_tier(self, demand_index: float) -> str:
        try:
            slow_max = float(self.volume_thresholds.get("slow_max", 0.45))
            moderate_max = float(self.volume_thresholds.get("moderate_max", 0.75))
        except Exception:  # noqa: BLE001
            slow_max, moderate_max = 0.45, 0.75
        if demand_index <= slow_max:
            return "slow"
        if demand_index <= moderate_max:
            return "moderate"
        return "peak"

    def _build_assignment_payload(
        self,
        employee: Optional[Dict[str, Any]],
        demand: BlockDemand,
        *,
        override_end: Optional[datetime.datetime] = None,
    ) -> Dict[str, Any]:
        rate = self._employee_role_wage(employee, demand.role) if employee else 0.0
        start_time = demand.start
        end_time = override_end or demand.recommended_cut or demand.end
        if end_time <= start_time:
            end_time = demand.end
        hours = max(0.0, (end_time - start_time).total_seconds() / 3600)
        cost = round(hours * rate, 2)
        notes = ", ".join(demand.labels)
        return {
            "employee_id": employee["id"] if employee else None,
            "role": demand.role,
            "start": start_time,
            "end": end_time,
            "labor_rate": rate,
            "labor_cost": cost,
            "location": demand.block_name,
            "notes": notes,
        }

    def _build_summary(self, week: WeekSchedule, shifts: List[Dict[str, Any]]) -> Dict[str, Any]:
        totals = []
        total_cost = 0.0
        total_shifts = 0
        for day_offset in range(7):
            date_value = week.week_start_date + datetime.timedelta(days=day_offset)
            day_shifts = [shift for shift in shifts if shift["start"].date() == date_value]
            day_cost = sum(shift["labor_cost"] for shift in day_shifts if shift["labor_cost"])
            totals.append(
                {
                    "date": date_value.isoformat(),
                    "shifts_created": len(day_shifts),
                    "cost": round(day_cost, 2),
                }
            )
            total_cost += day_cost
            total_shifts += len(day_shifts)
        total_budget = 0.0
        if self.group_budget_by_day:
            for day_budget in self.group_budget_by_day:
                if not isinstance(day_budget, dict):
                    continue
                total_budget += sum(float(value) for value in day_budget.values())
        budget_ratio = total_cost / total_budget if total_budget > 1e-6 else None
        return {
            "week_id": week.id,
            "days": totals,
            "total_cost": round(total_cost, 2),
            "total_shifts": total_shifts,
            "projected_budget_total": round(total_budget, 2),
            "policy_budget_ratio": round(budget_ratio, 4) if budget_ratio is not None else None,
            "warnings": [],
            "cut_insights": list(self.cut_insights),
        }

    @staticmethod
    def _parse_projection_notes(raw: Optional[str]) -> Dict[str, Any]:
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except (json.JSONDecodeError, TypeError):
            return {}
