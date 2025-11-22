from __future__ import annotations

import copy
import datetime
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from database import get_active_policy, upsert_policy


UTC = datetime.timezone.utc
WEEKDAY_TOKENS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def load_active_policy(conn) -> Dict:
    """Return the active policy payload as a dict."""
    if conn is None:
        return {}
    if callable(conn):
        with conn() as session:
            policy = get_active_policy(session)
            return policy.params_dict() if policy else {}
    policy = get_active_policy(conn)
    return policy.params_dict() if policy else {}


def role_catalog(policy: Dict) -> Set[str]:
    roles = policy.get("roles") if isinstance(policy, dict) else {}
    if not isinstance(roles, dict):
        return set()
    return {name for name in roles.keys()}


def role_definition(policy: Dict, role: str) -> Dict:
    roles = policy.get("roles") if isinstance(policy, dict) else {}
    if not isinstance(roles, dict):
        return {}
    details = roles.get(role) or {}
    return details if isinstance(details, dict) else {}


def anchor_rules(policy: Dict) -> Dict[str, Any]:
    anchors = policy.get("anchors") if isinstance(policy, dict) else {}
    if isinstance(anchors, dict) and anchors:
        return anchors
    return copy.deepcopy(ANCHOR_RULES)


SHIFT_LENGTH_DEFAULTS: Dict[str, Dict[str, float]] = {
    "Kitchen": {"min": 3.0, "max": 8.0},
    "Servers": {"min": 4.0, "max": 8.0},
    "Bartenders": {"min": 5.0, "max": 9.0},
    "Cashier": {"min": 2.5, "max": 6.0},
    "Other": {"min": 3.0, "max": 8.0},
}


def shift_length_limits(policy: Dict, role: str, group: str) -> Tuple[float, float]:
    cfg = role_definition(policy, role)
    group_defaults = SHIFT_LENGTH_DEFAULTS.get(group or "Other", SHIFT_LENGTH_DEFAULTS["Other"])
    min_val = cfg.get("min_shift_hours", cfg.get("shift_length_rule", {}).get("minHrs", group_defaults["min"]))
    max_val = cfg.get("max_shift_hours", cfg.get("shift_length_rule", {}).get("maxHrs", group_defaults["max"]))
    try:
        min_val = float(min_val)
    except (TypeError, ValueError):
        min_val = group_defaults["min"]
    try:
        max_val = float(max_val)
    except (TypeError, ValueError):
        max_val = group_defaults["max"]
    return max(0.5, min_val), max(1.0, max_val)


def hourly_wage(policy: Dict, role: str, default: float = 0.0) -> float:
    details = role_definition(policy, role)
    value = details.get("hourly_wage") if isinstance(details, dict) else None
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def weekday_token(date_: datetime.date) -> str:
    return WEEKDAY_TOKENS[date_.weekday()]


def parse_time_label(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    label = value.strip()
    if not label:
        return None
    if label.startswith("@"):
        return None
    if ":" not in label:
        return None
    hour_str, minute_str = label.split(":", 1)
    try:
        hours = int(hour_str)
        minutes = int(minute_str)
    except ValueError:
        return None
    total_minutes = max(0, hours) * 60 + max(0, minutes)
    return total_minutes


def minutes_to_datetime(date_: datetime.date, minutes: int) -> datetime.datetime:
    day_offset = minutes // (24 * 60)
    minute_offset = minutes % (24 * 60)
    base = datetime.datetime.combine(date_, datetime.time.min, tzinfo=UTC)
    return base + datetime.timedelta(days=day_offset, minutes=minute_offset)


def _hours_entry(policy: Dict, date_: datetime.date) -> Dict[str, str]:
    hours_cfg = policy.get("business_hours") or {}
    weekday = weekday_token(date_)
    entry = hours_cfg.get(weekday) or hours_cfg.get(weekday.lower()) or hours_cfg.get(weekday.capitalize())
    return entry if isinstance(entry, dict) else {}


def open_minutes(policy: Dict, date_: datetime.date) -> int:
    entry = _hours_entry(policy, date_)
    label = entry.get("open") if entry else None
    parsed = parse_time_label(label)
    if parsed is not None:
        return parsed
    return parse_time_label("10:00") or 10 * 60


def close_minutes(policy: Dict, date_: datetime.date) -> int:
    entry = _hours_entry(policy, date_)
    label = entry.get("close") if entry else None
    parsed = parse_time_label(label)
    if parsed is not None:
        return parsed
    timeblocks = policy.get("timeblocks") or {}
    close_spec = timeblocks.get("close", {})
    weekday = weekday_token(date_)
    by_weekday = close_spec.get("byWeekdayEnd") or {}
    value = (
        by_weekday.get(weekday)
        or by_weekday.get(weekday.lower())
        or by_weekday.get(weekday.capitalize())
        or close_spec.get("end")
        or "24:00"
    )
    parsed = parse_time_label(value)
    return parsed if parsed is not None else 24 * 60


def mid_minutes(policy: Dict, date_: datetime.date) -> int:
    entry = _hours_entry(policy, date_)
    label = entry.get("mid") if entry else None
    parsed = parse_time_label(label)
    if parsed is not None:
        return parsed
    open_value = open_minutes(policy, date_)
    close_value = close_minutes(policy, date_)
    return open_value + (close_value - open_value) // 2


ANCHOR_PATTERN = re.compile(r"^@(?P<anchor>open|close|mid)(?P<offset>[+-]\d+)?$", re.IGNORECASE)


def _parse_time_expression(
    policy: Dict,
    date_: datetime.date,
    label: Optional[str],
    *,
    close_min: int,
) -> Optional[int]:
    if label is None:
        return None
    raw = label.strip()
    if not raw:
        return None
    match = ANCHOR_PATTERN.match(raw)
    if match:
        anchor = match.group("anchor").lower()
        offset_raw = match.group("offset")
        offset = int(offset_raw) if offset_raw else 0
        if anchor == "open":
            base = open_minutes(policy, date_)
        elif anchor == "mid":
            base = mid_minutes(policy, date_)
        else:
            base = close_min
        return base + offset
    return parse_time_label(raw)


def _resolve_block_window(
    policy: Dict,
    date_: datetime.date,
    block_spec: Dict[str, Any],
    *,
    close_min: int,
) -> Optional[Tuple[datetime.datetime, datetime.datetime]]:
    if not isinstance(block_spec, dict):
        return None
    weekday = weekday_token(date_)
    start_label = block_spec.get("start")
    end_label = block_spec.get("end")
    if not end_label and isinstance(block_spec.get("byWeekdayEnd"), dict):
        options = block_spec["byWeekdayEnd"]
        end_label = options.get(weekday) or options.get(weekday.lower()) or options.get(weekday.capitalize())
    start_minutes = _parse_time_expression(policy, date_, start_label, close_min=close_min)
    if start_minutes is None:
        start_minutes = parse_time_label("09:00") or 0
    end_minutes = _parse_time_expression(policy, date_, end_label, close_min=close_min)
    if end_minutes is None:
        end_minutes = close_min
    if end_minutes <= start_minutes:
        end_minutes = max(start_minutes + 60, end_minutes)
    start_dt = minutes_to_datetime(date_, start_minutes)
    end_dt = minutes_to_datetime(date_, end_minutes)
    return (start_dt, end_dt)


def resolve_policy_block(
    policy: Dict,
    block_name: str,
    date_: datetime.date,
    *,
    close_min: Optional[int] = None,
    overrides: Optional[Dict[str, str]] = None,
) -> Optional[Tuple[str, datetime.datetime, datetime.datetime]]:
    timeblocks = policy.get("timeblocks") or {}
    block_spec = timeblocks.get(block_name)
    if not isinstance(block_spec, dict):
        return None
    if overrides:
        merged = block_spec.copy()
        merged.update({key: value for key, value in overrides.items() if value})
        block_spec = merged
    close_value = close_min if close_min is not None else close_minutes(policy, date_)
    window = _resolve_block_window(policy, date_, block_spec, close_min=close_value)
    if not window:
        return None
    return (block_name, window[0], window[1])


def resolve_role_blocks(policy: Dict, role_cfg: Dict, date_: datetime.date) -> List[Dict[str, Any]]:
    blocks: List[Dict[str, Any]] = []
    if not isinstance(role_cfg, dict):
        return blocks
    block_specs = role_cfg.get("blocks") or []
    if not isinstance(block_specs, list):
        return blocks
    close_min = close_minutes(policy, date_)
    for entry in block_specs:
        if isinstance(entry, str):
            resolved = resolve_policy_block(policy, entry, date_, close_min=close_min)
            if not resolved:
                continue
            name, start_dt, end_dt = resolved
            blocks.append({"name": name, "start": start_dt, "end": end_dt})
            continue
        if isinstance(entry, dict):
            name = entry.get("name") or entry.get("label") or entry.get("id") or "block"
            window = _resolve_block_window(date_, entry, close_min=close_min)
            if not window:
                continue
            blocks.append({"name": name, "start": window[0], "end": window[1]})
    return blocks


def shift_length_rule(role_cfg: Dict) -> Dict[str, Any]:
    if not isinstance(role_cfg, dict):
        return {}
    rule = role_cfg.get("shift_length_rule") or {}
    return rule if isinstance(rule, dict) else {}

def _block_config(
    base: int,
    *,
    min_staff: int | None = None,
    max_staff: int | None = None,
    per_sales: float = 0.0,
    per_modifier: float = 0.0,
    start: Optional[str] = None,
    end: Optional[str] = None,
    floor_by_demand: Optional[List[Dict[str, float | int]]] = None,
) -> Dict[str, float | int | str]:
    minimum = base if min_staff is None else min_staff
    maximum = max(base, minimum) if max_staff is None else max_staff
    normalized_per_sales = max(0.0, min(per_sales, 0.05))
    normalized_per_modifier = max(0.0, min(per_modifier, 0.5))
    payload: Dict[str, float | int | str] = {
        "base": max(0, base),
        "min": max(0, minimum),
        "max": max(0, maximum),
        "per_1000_sales": normalized_per_sales,
        "per_modifier": normalized_per_modifier,
    }
    if start:
        payload["start"] = start
    if end:
        payload["end"] = end
    if floor_by_demand:
        payload["floor_by_demand"] = floor_by_demand
    return payload


def _role_config(
    *,
    wage: float,
    priority: float,
    max_weekly: int,
    blocks: Dict[str, Dict[str, float | int]],
    daily_boost: Dict[str, int] | None = None,
    enabled: bool = True,
    thresholds: List[Dict[str, float | int | str]] | None = None,
    group: str = "Other",
    allow_cuts: bool = True,
    always_on: bool = False,
    cut_buffer_minutes: int = 30,
    covers: Optional[List[str]] = None,
    critical: bool = False,
) -> Dict[str, Any]:
    return {
        "enabled": enabled,
        "hourly_wage": wage,
        "priority": priority,
        "max_weekly_hours": max_weekly,
        "daily_boost": daily_boost or {},
        "shift_length_rule": {"minHrs": 5, "maxHrs": 8, "preferBlocks": True},
        "thresholds": thresholds or [],
        "blocks": blocks,
        "group": group,
        "allow_cuts": allow_cuts,
        "always_on": always_on,
        "cut_buffer_minutes": cut_buffer_minutes,
        "covers": covers or [],
        "critical": critical,
    }


DEFAULT_TIMEBLOCKS: Dict[str, Dict[str, str]] = {
    "Open": {"start": "@open-30", "end": "@open"},
    "Mid": {"start": "@open", "end": "@mid"},
    "PM": {"start": "@mid", "end": "@close"},
    "Close": {"start": "@close", "end": "@close+35"},
}

BUSINESS_HOURS: Dict[str, Dict[str, str]] = {
    "Mon": {"open": "11:00", "mid": "16:00", "close": "24:00"},
    "Tue": {"open": "11:00", "mid": "16:00", "close": "24:00"},
    "Wed": {"open": "11:00", "mid": "16:00", "close": "24:00"},
    "Thu": {"open": "11:00", "mid": "16:00", "close": "24:00"},
    "Fri": {"open": "11:00", "mid": "16:00", "close": "25:00"},
    "Sat": {"open": "11:00", "mid": "16:00", "close": "25:00"},
    "Sun": {"open": "11:00", "mid": "16:00", "close": "23:00"},
}

PATTERN_TEMPLATES: Dict[str, Dict[str, Any]] = {
    # Cashier + To-Go. AM/PM windows mirror the shared reference schedule.
    "Cashier": {
        "Mon": {
            "am": [{"start": "11:30", "end": "14:00"}],
            "pm": [{"start": "16:30", "end": "20:45"}],
        },
        "Tue": {
            "am": [{"start": "11:00", "end": "14:00"}],
            "pm": [{"start": "16:00", "end": "21:00"}, {"start": "17:30", "end": "20:45"}],
        },
        "Wed": {
            "am": [{"start": "11:30", "end": "13:45"}],
            "pm": [{"start": "16:30", "end": "20:45"}],
        },
        "Thu": {
            "am": [{"start": "11:00", "end": "14:00"}],
            "pm": [
                {"start": "16:00", "end": "22:00"},
                {"start": "16:30", "end": "21:00"},
                {"start": "17:30", "end": "21:00"},
            ],
        },
        "Fri": {
            "am": [{"start": "11:00", "end": "14:00"}],
            "pm": [
                {"start": "16:30", "end": "21:00"},
                {"start": "17:30", "end": "21:00"},
            ],
        },
        "Sat": {
            "am": [{"start": "11:00", "end": "15:00"}],
            "pm": [
                {"start": "16:30", "end": "21:00"},
                {"start": "17:30", "end": "21:00"},
            ],
        },
        "Sun": {
            "am": [{"start": "11:00", "end": "15:00"}],
            "pm": [{"start": "16:30", "end": "20:45"}],
        },
    },
    # Bartenders follow a consistent AM bar prep window with day-specific PM closes.
    "Bartenders": {
        "default": {"am": [{"start": "10:30", "end": "16:30"}]},
        "Mon": {"pm": [{"start": "16:00", "end": "24:30"}]},
        "Tue": {"pm": [{"start": "16:00", "end": "24:30"}]},
        "Wed": {"pm": [{"start": "16:00", "end": "24:30"}]},
        "Thu": {"pm": [{"start": "16:00", "end": "25:00"}]},
        "Fri": {"pm": [{"start": "16:00", "end": "25:30"}]},
        "Sat": {"pm": [{"start": "16:00", "end": "25:30"}]},
        "Sun": {"pm": [{"start": "16:00", "end": "23:30"}]},
    },
    # Kitchen / HOH pattern options (used by Cook/Grill/Prep/Expo/etc.).
    "Kitchen": {
        "Mon": {
            "am": [
                {"start": "10:30", "end": "16:15"},
                {"start": "11:30", "end": "14:30"},
            ],
            "pm": [
                {"start": "16:00", "end": "20:45"},
                {"start": "17:00", "end": "21:00"},
                {"start": "17:00", "end": "23:30"},
                {"start": "17:30", "end": "24:30"},
            ],
        },
        "Tue": {
            "am": [
                {"start": "10:30", "end": "16:15"},
                {"start": "11:30", "end": "14:30"},
                {"start": "11:30", "end": "13:30"},
            ],
            "pm": [
                {"start": "16:00", "end": "21:00"},
                {"start": "16:30", "end": "21:00"},
                {"start": "17:00", "end": "21:00"},
                {"start": "17:30", "end": "24:30"},
                {"start": "17:30", "end": "23:30"},
            ],
        },
        "Wed": {
            "am": [
                {"start": "10:30", "end": "16:15"},
                {"start": "11:30", "end": "14:00"},
            ],
            "pm": [
                {"start": "16:00", "end": "20:45"},
                {"start": "17:00", "end": "21:00"},
                {"start": "17:00", "end": "23:30"},
                {"start": "17:30", "end": "24:30"},
            ],
        },
        "Thu": {
            "am": [
                {"start": "10:30", "end": "16:15"},
                {"start": "11:30", "end": "14:30"},
                {"start": "11:30", "end": "13:30"},
            ],
            "pm": [
                {"start": "16:00", "end": "21:00"},
                {"start": "16:00", "end": "21:45"},
                {"start": "16:30", "end": "21:45"},
                {"start": "17:00", "end": "21:30"},
                {"start": "17:00", "end": "24:30"},
                {"start": "17:00", "end": "24:00"},
            ],
        },
        "Fri": {
            "am": [
                {"start": "10:30", "end": "16:15"},
                {"start": "11:30", "end": "14:30"},
                {"start": "11:30", "end": "13:30"},
            ],
            "pm": [
                {"start": "16:00", "end": "21:00"},
                {"start": "16:30", "end": "21:00"},
                {"start": "17:00", "end": "21:00"},
                {"start": "17:30", "end": "24:30"},
                {"start": "17:30", "end": "23:30"},
                {"start": "18:00", "end": "20:45"},
            ],
        },
        "Sat": {
            "am": [
                {"start": "10:30", "end": "16:15"},
                {"start": "11:00", "end": "14:30"},
                {"start": "11:30", "end": "16:00"},
                {"start": "11:45", "end": "14:30"},
            ],
            "pm": [
                {"start": "16:00", "end": "21:00"},
                {"start": "16:30", "end": "21:00"},
                {"start": "17:00", "end": "21:00"},
                {"start": "17:30", "end": "24:30"},
                {"start": "17:30", "end": "23:30"},
                {"start": "18:00", "end": "20:45"},
            ],
        },
        "Sun": {
            "am": [
                {"start": "10:30", "end": "16:15"},
                {"start": "11:00", "end": "14:30"},
                {"start": "11:30", "end": "16:00"},
                {"start": "11:45", "end": "14:30"},
            ],
            "pm": [
                {"start": "16:00", "end": "20:45"},
                {"start": "16:00", "end": "21:00"},
                {"start": "16:45", "end": "21:00"},
                {"start": "17:00", "end": "23:30"},
                {"start": "17:30", "end": "24:00"},
            ],
        },
    },
}


ROLES: Dict[str, Dict[str, Any]] = {
    "Server - Dining": _role_config(
        wage=6.25,
        priority=1.0,
        max_weekly=38,
        daily_boost={"Fri": 1, "Sat": 1, "Sun": -3},
        thresholds=[
            {"metric": "demand_index", "gte": 0.65, "add": 1},
            {"metric": "demand_index", "gte": 1.0, "add": 1},
        ],
        blocks={
            "Open": _block_config(2, max_staff=3, per_sales=0.2, per_modifier=0.4),
            "Mid": _block_config(
                2,
                max_staff=5,
                per_sales=0.25,
                per_modifier=0.4,
                floor_by_demand=[{"gte": 0.4, "min": 3}, {"gte": 0.7, "min": 4}, {"gte": 1.0, "min": 5}],
            ),
            "PM": _block_config(
                3,
                max_staff=5,
                per_sales=0.3,
                per_modifier=0.5,
                floor_by_demand=[{"gte": 0.4, "min": 3}, {"gte": 0.7, "min": 4}, {"gte": 1.0, "min": 5}],
            ),
            "Close": _block_config(1, max_staff=3, per_sales=0.15, per_modifier=0.3),
        },
        group="Servers",
        cut_buffer_minutes=35,
        covers=["Server - Cocktail", "Server - Patio"],
    ),
    "Server - Cocktail": _role_config(
        wage=6.75,
        priority=0.95,
        max_weekly=36,
        daily_boost={"Thu": 1, "Fri": 1, "Sat": 1, "Sun": -3},
        thresholds=[
            {"metric": "demand_index", "gte": 0.6, "add": 1},
            {"metric": "demand_index", "gte": 0.95, "add": 1},
        ],
        blocks={
            "Open": _block_config(1, max_staff=2, per_sales=0.1, per_modifier=0.2),
            "Mid": _block_config(
                1,
                max_staff=3,
                per_sales=0.15,
                per_modifier=0.3,
                floor_by_demand=[{"gte": 0.3, "min": 1}, {"gte": 0.6, "min": 2}, {"gte": 0.9, "min": 3}],
            ),
            "PM": _block_config(
                2,
                max_staff=3,
                per_sales=0.2,
                per_modifier=0.4,
                floor_by_demand=[{"gte": 0.3, "min": 1}, {"gte": 0.6, "min": 2}, {"gte": 0.9, "min": 3}],
            ),
            "Close": _block_config(1, max_staff=2, per_sales=0.15, per_modifier=0.3),
        },
        group="Servers",
        cut_buffer_minutes=35,
        covers=["Server - Dining", "Server - Patio"],
    ),
    "Server - Patio": _role_config(
        wage=6.0,
        priority=0.8,
        max_weekly=32,
        daily_boost={"Fri": 1, "Sat": 1, "Sun": -2},
        thresholds=[{"metric": "demand_index", "gte": 0.7, "add": 1}],
        blocks={
            "Mid": _block_config(1, max_staff=2, per_sales=0.1, per_modifier=0.2),
            "PM": _block_config(1, max_staff=2, per_sales=0.15, per_modifier=0.3),
        },
        group="Servers",
        cut_buffer_minutes=30,
        covers=["Server - Dining", "Server - Cocktail"],
    ),
    "Server - Dining Closer": _role_config(
        wage=6.75,
        priority=1.05,
        max_weekly=35,
        daily_boost={"Thu": 1, "Fri": 1, "Sat": 1},
        blocks={"Close": _block_config(1, min_staff=1, max_staff=1)},
        group="Servers",
        allow_cuts=False,
        cut_buffer_minutes=5,
        covers=["Server - Dining"],
    ),
    "Server - Cocktail Closer": _role_config(
        wage=7.0,
        priority=1.0,
        max_weekly=35,
        daily_boost={"Thu": 1, "Fri": 1, "Sat": 1},
        blocks={"Close": _block_config(1, min_staff=1, max_staff=1)},
        group="Servers",
        allow_cuts=False,
        cut_buffer_minutes=5,
        covers=["Server - Cocktail"],
    ),
    "Bartender": _role_config(
        wage=10.0,
        priority=0.98,
        max_weekly=40,
        daily_boost={"Sun": -2},
        thresholds=[{"metric": "demand_index", "gte": 0.75, "add": 1}],
        blocks={
            "Open": _block_config(1, max_staff=2, per_sales=0.05, per_modifier=0.1),
            "Mid": _block_config(1, max_staff=2, per_sales=0.1, per_modifier=0.2),
            "PM": _block_config(1, max_staff=2, per_sales=0.15, per_modifier=0.3),
            "Close": _block_config(1, max_staff=2, per_sales=0.1, per_modifier=0.2),
        },
        group="Bartenders",
        allow_cuts=False,
        always_on=True,
        cut_buffer_minutes=0,
    ),
    "Bartender - Closer": _role_config(
        wage=11.0,
        priority=1.05,
        max_weekly=40,
        blocks={"Close": _block_config(1, min_staff=1, max_staff=1)},
        group="Bartenders",
        allow_cuts=False,
        always_on=True,
        cut_buffer_minutes=0,
        covers=["Bartender"],
    ),
    "Cashier": _role_config(
        wage=15.0,
        priority=0.9,
        max_weekly=35,
        daily_boost={"Sun": -2},
        thresholds=[],
        blocks={
            "Open": _block_config(0, min_staff=0, max_staff=1, per_sales=0.01),
            "Mid": _block_config(0, min_staff=0, max_staff=1, per_sales=0.02),
            "PM": _block_config(0, min_staff=0, max_staff=1, per_sales=0.03),
            "Close": _block_config(0, min_staff=0, max_staff=0, per_sales=0.0),
        },
        group="Cashier",
        cut_buffer_minutes=20,
        covers=["Cashier - To-Go Specialist", "Host"],
    ),
    "Cashier - To-Go Specialist": _role_config(
        wage=15.0,
        priority=0.88,
        max_weekly=35,
        daily_boost={"Sun": -2},
        thresholds=[],
        blocks={
            "Mid": _block_config(0, min_staff=0, max_staff=1, per_sales=0.02),
            "PM": _block_config(0, min_staff=0, max_staff=1, per_sales=0.03),
            "Close": _block_config(0, min_staff=0, max_staff=0, per_sales=0.0),
        },
        group="Cashier",
        cut_buffer_minutes=25,
        covers=["Cashier", "Host"],
    ),
    "Host": _role_config(
        wage=14.0,
        priority=0.75,
        max_weekly=32,
        daily_boost={"Sun": -2},
        thresholds=[],
        blocks={
            "Open": _block_config(0, min_staff=0, max_staff=1, per_sales=0.01),
            "Mid": _block_config(0, min_staff=0, max_staff=1, per_sales=0.01),
        },
        group="Cashier",
        cut_buffer_minutes=25,
        covers=["Cashier", "Cashier - To-Go Specialist"],
    ),
    "Expo": _role_config(
        wage=17.5,
        priority=0.92,
        max_weekly=40,
        daily_boost={"Sun": -2},
        thresholds=[{"metric": "demand_index", "gte": 0.8, "add": 1}],
        blocks={
            "Open": _block_config(1, max_staff=1),
            "Mid": _block_config(1, max_staff=1),
            "PM": _block_config(1, max_staff=1, per_sales=0.0),
            "Close": _block_config(0, max_staff=0),
        },
        group="Kitchen",
        cut_buffer_minutes=25,
        covers=["Prep", "Chip", "Shake"],
        critical=True,
    ),
    "Grill": _role_config(
        wage=18.0,
        priority=0.94,
        max_weekly=40,
        daily_boost={"Sun": -2},
        thresholds=[],
        blocks={
            "Open": _block_config(0, min_staff=0, max_staff=1, per_sales=0.0),
            "Mid": _block_config(0, min_staff=0, max_staff=1, per_sales=0.02),
            "PM": _block_config(0, min_staff=0, max_staff=1, per_sales=0.03),
            "Close": _block_config(0, min_staff=0, max_staff=0),
        },
        group="Kitchen",
        cut_buffer_minutes=30,
        covers=["Cook", "Prep"],
    ),
    "Cook": _role_config(
        wage=17.0,
        priority=0.9,
        max_weekly=40,
        daily_boost={"Sun": -2},
        thresholds=[],
        blocks={
            "Open": _block_config(0, min_staff=0, max_staff=1, per_sales=0.0),
            "Mid": _block_config(0, min_staff=0, max_staff=1, per_sales=0.02),
            "PM": _block_config(0, min_staff=0, max_staff=1, per_sales=0.03),
            "Close": _block_config(0, min_staff=0, max_staff=0),
        },
        group="Kitchen",
        cut_buffer_minutes=30,
        covers=["Prep", "Chip"],
    ),
    "Prep": _role_config(
        wage=16.0,
        priority=0.8,
        max_weekly=34,
        daily_boost={"Sun": -2},
        thresholds=[],
        blocks={
            "Open": _block_config(0, min_staff=0, max_staff=1, per_sales=0.0),
            "Mid": _block_config(0, min_staff=0, max_staff=1, per_sales=0.02),
        },
        group="Kitchen",
        cut_buffer_minutes=25,
        covers=["Chip", "Shake"],
    ),
    "Chip": _role_config(
        wage=15.5,
        priority=0.78,
        max_weekly=34,
        daily_boost={"Sun": -2},
        thresholds=[],
        blocks={
            "Mid": _block_config(0, max_staff=1, per_sales=0.01),
            "PM": _block_config(0, max_staff=1, per_sales=0.02),
        },
        group="Kitchen",
        cut_buffer_minutes=25,
        covers=["Prep", "Shake"],
    ),
    "Shake": _role_config(
        wage=15.5,
        priority=0.78,
        max_weekly=34,
        daily_boost={"Sun": -2},
        thresholds=[],
        blocks={
            "Mid": _block_config(0, max_staff=1, per_sales=0.01),
            "PM": _block_config(0, max_staff=1, per_sales=0.02),
        },
        group="Kitchen",
        cut_buffer_minutes=25,
        covers=["Prep", "Chip"],
    ),
    "Kitchen Opener": _role_config(
        wage=18.5,
        priority=0.85,
        max_weekly=38,
        daily_boost={"Sun": -2},
        blocks={"Open": _block_config(1, max_staff=2)},
        group="Kitchen",
        cut_buffer_minutes=20,
        covers=["Prep", "Chip", "Shake"],
    ),
    "Kitchen Closer": _role_config(
        wage=19.0,
        priority=0.87,
        max_weekly=38,
        daily_boost={"Sun": -2},
        blocks={"Close": _block_config(1, max_staff=2)},
        group="Kitchen",
        allow_cuts=False,
        cut_buffer_minutes=0,
        covers=["Prep", "Chip", "Shake"],
    ),
}


ROLE_GROUP_ALLOCATIONS: Dict[str, Dict[str, Any]] = {
    "Kitchen": {"allocation_pct": 0.34, "allow_cuts": True, "cut_buffer_minutes": 25},
    "Servers": {"allocation_pct": 0.4, "allow_cuts": True, "cut_buffer_minutes": 35},
    "Bartenders": {"allocation_pct": 0.12, "allow_cuts": False, "always_on": True, "cut_buffer_minutes": 0},
    "Cashier": {"allocation_pct": 0.14, "allow_cuts": True, "cut_buffer_minutes": 25},
}

ANCHOR_RULES: Dict[str, Any] = {
    "openers": {"Kitchen": 1, "Servers": 1, "Bartenders": 1, "Cashier": 0},
    "closers": {"Kitchen": 1, "Servers": 2, "Bartenders": 1, "Cashier": 0},
    "opener_roles": {
        "Kitchen": ["Kitchen Opener"],
        "Servers": ["Server - Dining Opener", "Server - Cocktail Opener", "Server - Dining", "Server - Cocktail"],
        "Bartenders": ["Bartender - Opener", "Bartender"],
        "Cashier": ["Cashier", "Cashier - To-Go Specialist"],
    },
    "closer_roles": {
        "Kitchen": ["Kitchen Closer"],
        "Servers": ["Server - Dining Closer", "Server - Cocktail Closer", "Server - Dining", "Server - Cocktail"],
        "Bartenders": ["Bartender - Closer", "Bartender"],
        "Cashier": ["Cashier - To-Go Specialist", "Cashier"],
    },
    "non_cuttable_roles": [
        "Bartender",
        "Bartender - Closer",
        "Kitchen Closer",
        "Server - Dining Closer",
        "Server - Cocktail Closer",
    ],
    "allow_cashier_closer": False,
    # Controls FIFO/LILO bias for open/close patterns: "off", "prefer", "enforce".
    "open_close_order": "prefer",
}


BASELINE_POLICY: Dict[str, Any] = {
    "name": "Baseline Coverage",
    "description": "Seeded policy that balances FOH/BOH coverage for the automation workflow.",
    "global": {
        "max_hours_week": 48,
        "max_consecutive_days": 7,
        "round_to_minutes": 15,
        "allow_split_shifts": True,
        "overtime_penalty": 1.5,
        "desired_hours_floor_pct": 0.85,
        "desired_hours_ceiling_pct": 1.15,
        "open_buffer_minutes": 30,
        "close_buffer_minutes": 35,
        "labor_budget_pct": 0.27,
        "labor_budget_tolerance_pct": 0.08,
        "trim_aggressive_ratio": 0.7,
    },
    "timeblocks": DEFAULT_TIMEBLOCKS,
    "business_hours": BUSINESS_HOURS,
    "anchors": ANCHOR_RULES,
    "role_groups": ROLE_GROUP_ALLOCATIONS,
    "pattern_templates": PATTERN_TEMPLATES,
    "roles": ROLES,
}


def build_default_policy() -> Dict[str, Any]:
    """Return a deepcopy so callers can mutate the policy safely."""
    return copy.deepcopy(BASELINE_POLICY)


def ensure_default_policy(session_factory) -> None:
    """Seed the baseline policy exactly once so the generator can run end-to-end."""

    with session_factory() as session:
        if get_active_policy(session):
            return
        spec = build_default_policy()
        name = spec.get("name", "Baseline Coverage")
        params = {key: value for key, value in spec.items() if key != "name"}
        upsert_policy(session, name, params, edited_by="system")
