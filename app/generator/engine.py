from __future__ import annotations

import datetime
import json
import math
import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

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
    upsert_shift,
)
from policy import (
    build_default_policy,
    hourly_wage,
    open_minutes,
    resolve_policy_block,
    role_definition,
    shift_length_rule,
)
from roles import is_manager_role, normalize_role, role_matches, role_group

UTC = datetime.timezone.utc
WEEKDAY_TOKENS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


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

    @property
    def duration_hours(self) -> float:
        return max(0.0, (self.end - self.start).total_seconds() / 3600)


class ScheduleGenerator:
    def __init__(self, session, policy: Dict, actor: str = "system", wage_overrides: Optional[Dict[str, float]] = None) -> None:
        self.session = session
        self.policy = policy or {}
        self.actor = actor or "system"
        self.wage_overrides = wage_overrides or {}
        raw_roles = self.policy.get("roles") if isinstance(self.policy.get("roles"), dict) else {}
        self.roles_config: Dict[str, Dict] = {
            role: config for role, config in raw_roles.items() if not is_manager_role(role)
        }
        global_cfg = self.policy.get("global") or {}
        self.max_hours_per_week: float = float(global_cfg.get("max_hours_week", 40) or 40)
        self.max_consecutive_days: int = int(global_cfg.get("max_consecutive_days", 6) or 6)
        self.round_to_minutes: int = int(global_cfg.get("round_to_minutes", 15) or 15)
        self.allow_split_shifts: bool = bool(global_cfg.get("allow_split_shifts", True))
        self.overtime_penalty: float = float(global_cfg.get("overtime_penalty", 1.5) or 1.5)
        desired_floor = float(global_cfg.get("desired_hours_floor_pct", 0.85) or 0.0)
        desired_ceiling = float(global_cfg.get("desired_hours_ceiling_pct", 1.15) or 0.0)
        self.desired_hours_floor_pct: float = self._clamp(desired_floor, 0.0, 1.0)
        min_ceiling = max(self.desired_hours_floor_pct + 0.05, 0.1)
        self.desired_hours_ceiling_pct: float = self._clamp(max(desired_ceiling, min_ceiling), min_ceiling, 2.0)
        self.open_buffer_minutes: int = int(global_cfg.get("open_buffer_minutes", 30) or 0)
        self.close_buffer_minutes: int = int(global_cfg.get("close_buffer_minutes", 35) or 0)
        labor_pct = float(global_cfg.get("labor_budget_pct", 0.27) or 0.0)
        if labor_pct > 1.0:
            labor_pct /= 100.0
        self.labor_budget_pct = self._clamp(labor_pct, 0.05, 0.9)
        labor_tol = float(global_cfg.get("labor_budget_tolerance_pct", 0.08) or 0.0)
        if labor_tol > 1.0:
            labor_tol /= 100.0
        self.labor_budget_tolerance = self._clamp(labor_tol, 0.0, 0.5)

        self.employees: List[Dict[str, Any]] = []
        self.modifiers_by_day: Dict[int, List[Dict[str, Any]]] = {}
        self.day_contexts: List[Dict[str, Any]] = []
        self.role_group_settings: Dict[str, Dict[str, Any]] = self._load_role_group_settings()
        self.group_budget_by_day: List[Dict[str, float]] = []
        self.warnings: List[str] = []
        self.interchangeable_groups: Set[str] = {"Cashier & Takeout"}
        self.random = random.Random()
        self.group_pressure: Dict[int, Dict[str, float]] = {}
        self.cut_priority_rank: Dict[str, int] = {
            "Cashier": 0,
            "Cashier & Takeout": 0,
            "Servers": 1,
            "Heart of House": 2,
            "Kitchen": 2,
            "Bartenders": 3,
            "Other": 2,
        }

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
        demands = self._compute_block_demands(week.week_start_date)
        assignments = self._assign(demands)

        created_ids: List[int] = []
        for payload in assignments:
            payload.update({"week_id": week.id, "status": "draft", "week_start": week.week_start_date})
            created_ids.append(upsert_shift(self.session, payload))

        summary = self._build_summary(week, assignments)
        summary["warnings"].extend(self.warnings)
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

    def _load_employee_profiles(self) -> List[Dict[str, Any]]:
        stmt = (
            select(Employee)
            .where(Employee.status == "active")
            .options(selectinload(Employee.unavailability))
            .order_by(Employee.full_name.asc())
        )
        employees: List[Dict[str, Any]] = []
        for employee in self.session.scalars(stmt):
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
            employees.append(
                {
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
                }
            )
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
            return mapping
        for name, raw in specs.items():
            if not isinstance(raw, dict):
                continue
            label = (name or "Group").strip() or "Group"
            allocation = self._parse_allocation_pct(raw.get("allocation_pct"))
            mapping[label] = {
                "allocation_pct": allocation,
                "allow_cuts": bool(raw.get("allow_cuts", True)),
                "always_on": bool(raw.get("always_on", False)),
                "cut_buffer_minutes": int(raw.get("cut_buffer_minutes", 30) or 0),
            }
        if not mapping:
            mapping = build_default_policy().get("role_groups", {})
        return mapping

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

    def _compute_block_demands(self, week_start: datetime.date) -> List[BlockDemand]:
        demands: List[BlockDemand] = []
        for day_index in range(7):
            date_value = week_start + datetime.timedelta(days=day_index)
            for role_name, role_cfg in self.roles_config.items():
                if not role_cfg.get("enabled", True):
                    continue
                block_targets = role_cfg.get("blocks") or {}
                group_name = self._role_group_name(role_name, role_cfg)
                group_defaults = self.role_group_settings.get(group_name, {})
                allow_cuts = bool(role_cfg.get("allow_cuts", group_defaults.get("allow_cuts", True)))
                always_on = bool(role_cfg.get("always_on", group_defaults.get("always_on", False)))
                for block_name, block_cfg in block_targets.items():
                    block_label = (block_name or "").strip().lower()
                    if block_label == "open" and not self._role_allows_open_shift(role_name):
                        continue
                    overrides: Dict[str, str] = {}
                    if isinstance(block_cfg, dict):
                        custom_start = block_cfg.get("start")
                        custom_end = block_cfg.get("end")
                        if custom_start:
                            overrides["start"] = str(custom_start)
                        if custom_end:
                            overrides["end"] = str(custom_end)
                    resolved = resolve_policy_block(
                        self.policy,
                        block_name,
                        date_value,
                        overrides=overrides or None,
                    )
                    if not resolved:
                        continue
                    _, start_dt, end_dt = resolved
                    start_dt, end_dt = self._adjust_block_window(role_name, block_name, date_value, start_dt, end_dt)
                    need = self._calculate_block_need(role_name, role_cfg, block_cfg, block_name, day_index)
                    if need <= 0:
                        continue
                    min_staff = 0
                    if isinstance(block_cfg, dict):
                        min_staff = int(block_cfg.get("min", block_cfg.get("base", 0)))
                        if min_staff <= 0:
                            try:
                                base_value = int(block_cfg.get("base", 0))
                            except (TypeError, ValueError):
                                base_value = 0
                            if base_value > 0:
                                min_staff = base_value
                    minimum = max(0, min_staff)
                    if always_on:
                        minimum = max(1, minimum)
                    need = max(minimum, need)
                    rate = self._role_wage(role_name)
                    labels = [block_name]
                    demands.append(
                        BlockDemand(
                            day_index=day_index,
                            date=date_value,
                            start=start_dt,
                            end=end_dt,
                            role=role_name,
                            block_name=block_name,
                            labels=labels,
                            need=need,
                            priority=float(role_cfg.get("priority", 1.0)),
                            minimum=minimum,
                            allow_cuts=allow_cuts,
                            always_on=always_on,
                            role_group=group_name,
                            hourly_rate=rate,
                        )
                    )
        self._enforce_anchor_shift_caps(demands)
        self._apply_labor_allocations(demands)
        self._record_group_pressure(demands)
        self._annotate_cut_windows(demands)
        return demands

    def _role_wage(self, role_name: str) -> float:
        if role_name in self.wage_overrides:
            try:
                override = float(self.wage_overrides[role_name])
                if override > 0:
                    return override
            except (TypeError, ValueError):
                pass
        return hourly_wage(self.policy, role_name, 0.0)

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
        if self.close_buffer_minutes and not is_cashier and any(keyword in normalized_role for keyword in closer_keywords):
            buffer_minutes = self._round_minutes(max(0, self.close_buffer_minutes))
            end_dt += datetime.timedelta(minutes=buffer_minutes)
        return start_dt, end_dt


    def _day_sales_value(self, day_index: int) -> float:
        if 0 <= day_index < len(self.day_contexts):
            ctx = self.day_contexts[day_index]
            sales = float(ctx.get("sales", 0.0))
            modifier_multiplier = float(ctx.get("modifier_multiplier", 1.0))
            return sales * modifier_multiplier
        return 0.0

    def _calculate_block_need(
        self,
        role_name: str,
        role_cfg: Dict[str, Any],
        block_cfg: Dict[str, Any],
        block_name: str,
        day_index: int,
    ) -> int:
        block_label = (block_name or "").strip().lower()
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
                return 0
            need = 1 if need > 0 else 0

        group_name = role_group(role_name)
        demand_index = 1.0
        if 0 <= day_index < len(self.day_contexts):
            demand_index = self.day_contexts[day_index].get("indices", {}).get("demand_index", 1.0)
        if group_name == "Servers" and block_label in {"mid", "pm"}:
            if "dining" in normalize_role(role_name):
                floor = 3
                if demand_index >= 0.7:
                    floor = 4
                if demand_index >= 1.0:
                    floor = 5
                need = max(need, floor)
            elif "cocktail" in normalize_role(role_name):
                floor = 1
                if demand_index >= 0.6:
                    floor = 2
                if demand_index >= 0.9:
                    floor = 3
                need = max(need, floor)
            elif "patio" in normalize_role(role_name):
                if demand_index >= 0.8:
                    need = max(need, 1)
        need = max(min_staff, need)
        if max_staff > 0:
            need = min(max_staff, need)
        if self._is_opener_block(role_name, block_name):
            need = min(1, need) if need > 0 else 0
        if self._is_closer_block(role_name, block_name):
            need = min(1, need) if need > 0 else 0
        return max(0, need)

    def _role_group_name(self, role_name: str, role_cfg: Optional[Dict[str, Any]]) -> str:
        explicit = ""
        if isinstance(role_cfg, dict):
            explicit = (role_cfg.get("group") or "").strip()
        if explicit:
            return explicit
        inferred = role_group(role_name)
        return inferred if inferred else "Other"

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
            allowed_max = budget * (1 + self.labor_budget_tolerance)
            total_cost = sum(self._slot_cost(demand) * demand.need for demand in payload["demands"])
            if total_cost <= allowed_max + 0.01:
                continue
            removable: List[Tuple[float, float, BlockDemand]] = []
            for demand in payload["demands"]:
                if not demand.allow_cuts or demand.need <= demand.minimum:
                    continue
                slots = demand.need - demand.minimum
                if slots <= 0:
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
        """Ensure opener/closer counts match invariants (1 per group, no cashier closers)."""
        if not demands:
            return
        open_caps: Dict[str, List[str]] = {
            "Kitchen": ["kitchen opener"],
            "Heart of House": ["kitchen opener"],
            "Bartenders": ["bartender - opener", "bartender"],
            "Servers": [
                "server - dining opener",
                "server - cocktail opener",
                "server - dining",
                "server - cocktail",
                "server",
            ],
        }
        close_slots: Dict[str, List[List[str]]] = {
            "Bartenders": [["bartender - closer", "bartender"]],
            "Kitchen": [["kitchen closer"]],
            "Heart of House": [["kitchen closer"]],
            "Servers": [
                ["server - dining closer", "server - dining", "server"],
                ["server - cocktail closer", "server - cocktail", "server"],
            ],
        }
        blocked_close_groups = {"Cashier", "Cashier & Takeout"}

        by_day: Dict[int, List[BlockDemand]] = {}
        for demand in demands:
            by_day.setdefault(demand.day_index, []).append(demand)

        for day_demands in by_day.values():
            self._apply_opener_caps(day_demands, open_caps)
            self._apply_closer_caps(day_demands, close_slots, blocked_close_groups)

    def _apply_opener_caps(self, day_demands: List[BlockDemand], caps: Dict[str, List[str]]) -> None:
        open_demands = [
            demand for demand in day_demands if demand.block_name.strip().lower() == "open"
        ]
        if not open_demands:
            return
        for demand in list(open_demands):
            if demand.role_group not in caps:
                demand.need = 0
                demand.minimum = 0
        for group_name, preferred_roles in caps.items():
            candidates = [d for d in open_demands if d.role_group == group_name and d.need >= 0]
            if not candidates:
                continue
            chosen = self._pick_preferred_candidate(candidates, preferred_roles)
            if not chosen:
                continue
            for demand in candidates:
                if demand is chosen:
                    demand.need = 1
                    demand.minimum = max(1, demand.minimum)
                    demand.allow_cuts = False
                else:
                    demand.need = 0
                    demand.minimum = 0

    def _apply_closer_caps(
        self,
        day_demands: List[BlockDemand],
        slots: Dict[str, List[List[str]]],
        blocked_groups: Set[str],
    ) -> None:
        close_demands = [
            demand for demand in day_demands if demand.block_name.strip().lower() == "close"
        ]
        if not close_demands:
            return
        for demand in close_demands:
            if demand.role_group in blocked_groups:
                demand.need = 0
                demand.minimum = 0
        for demand in list(close_demands):
            if demand.need <= 0:
                continue
            if demand.role_group not in slots:
                demand.need = 0
                demand.minimum = 0
        for group_name, role_slots in slots.items():
            group_demands = [d for d in close_demands if d.role_group == group_name and d.need > 0]
            if not group_demands:
                continue
            available = list(group_demands)
            for role_preferences in role_slots:
                candidates = [d for d in available if normalize_role(d.role) in role_preferences]
                chosen = self._pick_preferred_candidate(candidates or available, role_preferences)
                if not chosen:
                    continue
                chosen.need = 1
                chosen.minimum = max(1, chosen.minimum)
                chosen.allow_cuts = False
                if chosen in available:
                    available.remove(chosen)
            for demand in available:
                demand.need = 0
                demand.minimum = 0

    @staticmethod
    def _pick_preferred_candidate(candidates: List[BlockDemand], preferred_roles: List[str]) -> Optional[BlockDemand]:
        if not candidates:
            return None
        normalized_preferences = [normalize_role(role) for role in preferred_roles]

        def rank(demand: BlockDemand) -> Tuple[int, float]:
            normalized = normalize_role(demand.role)
            try:
                preferred_index = normalized_preferences.index(normalized)
            except ValueError:
                preferred_index = len(normalized_preferences)
            return (preferred_index, -demand.priority)

        return sorted(candidates, key=rank)[0]

    def _annotate_cut_windows(self, demands: List[BlockDemand]) -> None:
        if not demands:
            return
        for demand in demands:
            if demand.need <= demand.minimum:
                continue
            if not demand.allow_cuts:
                continue
            cut_time = self._recommend_cut_time(demand)
            if cut_time:
                demand.recommended_cut = cut_time
                label = f"cut around {cut_time.strftime('%H:%M')}"
                if label not in demand.labels:
                    demand.labels.append(label)

    def _recommend_cut_time(self, demand: BlockDemand) -> Optional[datetime.datetime]:
        if self._is_closer_block(demand.role, demand.block_name):
            return None
        if demand.block_name.strip().lower() == "open":
            return None

        role_cfg = self.roles_config.get(demand.role, {})
        group_cfg = self.role_group_settings.get(demand.role_group, {})
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
        min_hours = shift_length_rule(role_cfg).get("minHrs", 3)
        try:
            min_hours = float(min_hours)
        except (TypeError, ValueError):
            min_hours = 3.0
        group_name = demand.role_group
        if group_name in {"Cashier", "Cashier & Takeout"}:
            min_hours = max(2.0, min_hours - 2.5)
        elif group_name == "Servers":
            min_hours = max(3.5, min_hours - 1.0)
        else:
            min_hours = max(3.0, min_hours - 1.5)

        normalized_role = normalize_role(demand.role)
        demand_softness = max(0.0, 0.9 - demand_index)
        pressure_factor = max(0.0, pressure_ratio - 1.0)
        cashier_bias = 0.0
        if "cashier" in normalized_role or "takeout" in normalized_role or "to-go" in normalized_role:
            cashier_bias = 0.35

        group_bias = 0
        if group_name in {"Cashier", "Cashier & Takeout"}:
            group_bias = 60
        elif group_name in {"Heart of House", "Kitchen"}:
            group_bias = 45
        elif group_name == "Servers":
            group_bias = 15
        else:
            group_bias = 10

        # Pull earlier when demand is soft or the group is over budget.
        early_pull_minutes = int(demand_softness * 90) + int(
            min(240, (pressure_factor * 140) + priority_rank * 12 + group_bias)
        )
        if pressure_ratio >= 1.5:
            early_pull_minutes += 40
        if pressure_ratio >= 2.0:
            early_pull_minutes += 40
        if cashier_bias:
            early_pull_minutes += int(30 * cashier_bias)

        buffer_minutes = max(0, buffer_minutes + early_pull_minutes)
        candidate = demand.end - datetime.timedelta(minutes=buffer_minutes)

        min_duration = datetime.timedelta(minutes=int(min_hours * 60))
        if candidate < demand.start + min_duration:
            candidate = demand.start + min_duration
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

    def _day_label(self, day_index: int) -> str:
        if 0 <= day_index < len(self.day_contexts):
            return self.day_contexts[day_index].get("weekday_token") or f"Day {day_index + 1}"
        return f"Day {day_index + 1}"

    def _is_opener_block(self, role_name: str, block_name: str) -> bool:
        normalized_role = normalize_role(role_name)
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
        }
        return normalized_role in allowed

    def _is_closer_block(self, role_name: str, block_name: str) -> bool:
        normalized_role = normalize_role(role_name)
        if not normalized_role:
            return False
        return "closer" in normalized_role and block_name.strip().lower() == "close"

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

    def _assign(self, demands: List[BlockDemand]) -> List[Dict[str, Any]]:
        assignments: List[Dict[str, Any]] = []
        essential_batch: List[Tuple[BlockDemand, int]] = []
        extra_batch: List[Tuple[BlockDemand, int]] = []
        for demand in demands:
            required = min(max(0, demand.minimum), demand.need)
            remaining = max(0, demand.need - required)
            if required > 0:
                essential_batch.append((demand, required))
            if remaining > 0:
                extra_batch.append((demand, remaining))
        essential_key: Callable[[Tuple[BlockDemand, int]], Tuple[Any, ...]] = lambda entry: (
            entry[0].day_index,
            entry[0].start,
            -entry[0].priority,
            entry[0].role,
        )
        extra_key: Callable[[Tuple[BlockDemand, int]], Tuple[Any, ...]] = lambda entry: (
            -entry[0].priority,
            entry[0].day_index,
            entry[0].start,
            entry[0].role,
        )
        assignments.extend(self._process_demand_batch(essential_batch, order_key=essential_key))
        assignments.extend(self._process_demand_batch(extra_batch, order_key=extra_key))
        return assignments

    def _process_demand_batch(
        self,
        batch: List[Tuple[BlockDemand, int]],
        *,
        order_key: Callable[[Tuple[BlockDemand, int]], Tuple[Any, ...]],
    ) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        if not batch:
            return results
        for demand, count in sorted(batch, key=order_key):
            for _ in range(count):
                candidate = self._select_employee(demand)
                if not candidate:
                    results.append(self._build_assignment_payload(None, demand))
                    self.warnings.append(
                        f"No coverage for {demand.role} on {demand.date.isoformat()} "
                        f"{demand.start.strftime('%H:%M')} - {demand.end.strftime('%H:%M')} ({demand.block_name})"
                    )
                    continue
                results.append(self._build_assignment_payload(candidate, demand))
                self._register_assignment(candidate, demand)
        return results

    def _select_employee(self, demand: BlockDemand) -> Optional[Dict[str, Any]]:
        for allow_overflow in (False, True):
            best_candidate = None
            best_score = float("-inf")
            for employee in self.employees:
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

    def _employee_can_cover_role(self, employee: Dict[str, Any], role_name: str) -> bool:
        if not role_name:
            return False
        candidate_roles = employee.get("roles") or set()
        for candidate in candidate_roles:
            if role_matches(candidate, role_name):
                return True
            candidate_cfg = role_definition(self.policy, candidate)
            covers = candidate_cfg.get("covers") if isinstance(candidate_cfg, dict) else []
            if covers and role_name in covers:
                return True
        target_group = role_group(role_name)
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
    ) -> bool:
        assignments = employee["assignments"][demand.day_index]
        demand_start_minutes, demand_end_minutes = self._demand_window_minutes(demand)
        for start_minute, end_minute in assignments:
            if demand_start_minutes < end_minute and demand_end_minutes > start_minute:
                return False
        if not self.allow_split_shifts and assignments:
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

    def _build_assignment_payload(
        self,
        employee: Optional[Dict[str, Any]],
        demand: BlockDemand,
    ) -> Dict[str, Any]:
        rate = self._role_wage(demand.role) if employee else 0.0
        start_time = demand.start
        end_time = demand.recommended_cut or demand.end
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
        return {
            "week_id": week.id,
            "days": totals,
            "total_cost": round(total_cost, 2),
            "total_shifts": total_shifts,
            "warnings": [],
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
