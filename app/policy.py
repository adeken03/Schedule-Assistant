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
            return _normalize_policy(policy.params_dict() if policy else {})
    policy = get_active_policy(conn)
    return _normalize_policy(policy.params_dict() if policy else {})


def _normalize_policy(policy: Dict) -> Dict:
    """Apply lightweight defaults/upgrades so runtime matches code expectations."""
    if not isinstance(policy, dict):
        return {}
    normalized = copy.deepcopy(policy)
    # Drop legacy HOH roles we no longer schedule.
    roles_cfg = normalized.get("roles")
    if isinstance(roles_cfg, dict):
        for legacy_role in ("HOH - Cook", "HOH - Prep"):
            roles_cfg.pop(legacy_role, None)
    normalized.setdefault("section_priority", "normal")
    normalized.setdefault("hoh_mode", "auto")
    normalized.setdefault("allow_mgr_fallback", normalized.get("pre_engine", {}).get("fallback", {}).get("allow_mgr_fallback", True))
    global_defaults = BASELINE_POLICY.get("global", {})
    global_cfg = normalized.setdefault("global", {})
    try:
        trim_ratio = float(global_cfg.get("trim_aggressive_ratio", global_defaults.get("trim_aggressive_ratio", 1.0)))
    except (TypeError, ValueError):
        trim_ratio = global_defaults.get("trim_aggressive_ratio", 1.0)
    default_trim = global_defaults.get("trim_aggressive_ratio", 1.0)
    # Ensure trim_aggressive_ratio is at least the code default so budgets are not silently capped.
    global_cfg["trim_aggressive_ratio"] = max(default_trim, trim_ratio)
    # Seasonal patio toggle: if not present, derive from the patio role's enabled state to keep UI toggles sticky.
    seasonal_defaults = BASELINE_POLICY.get("seasonal_settings", {})
    seasonal_cfg = normalized.setdefault("seasonal_settings", {})
    if "server_patio_enabled" not in seasonal_cfg:
        patio_role = role_definition(normalized, "Server - Patio")
        patio_enabled = bool(patio_role.get("enabled", seasonal_defaults.get("server_patio_enabled", True)))
        seasonal_cfg["server_patio_enabled"] = patio_enabled
    normalized["pre_engine"] = pre_engine_settings(normalized)
    anchors = normalized.setdefault("anchors", copy.deepcopy(ANCHOR_RULES))
    # Strip legacy cook/prep from anchors if user stored them.
    for key in ("opener_roles", "closer_roles"):
        if isinstance(anchors.get(key), dict):
            for group, roles in list(anchors[key].items()):
                if isinstance(roles, list):
                    anchors[key][group] = [r for r in roles if r not in {"HOH - Cook", "HOH - Prep"}]
    non_cuttable = set(anchors.get("non_cuttable_roles", []))
    for role in required_roles(normalized):
        non_cuttable.add(role)
    anchors["non_cuttable_roles"] = sorted(non_cuttable)
    return normalized


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


def _deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_update(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def pre_engine_settings(policy: Dict) -> Dict[str, Any]:
    payload = policy.get("pre_engine") if isinstance(policy, dict) else {}
    if isinstance(payload, dict):
        merged = _deep_update(PRE_ENGINE_DEFAULTS, payload)
        if not isinstance(merged.get("budget"), dict):
            merged["budget"] = copy.deepcopy(PRE_ENGINE_DEFAULTS["budget"])
        else:
            budget_cfg = merged["budget"]
            mode = budget_cfg.get("mode", PRE_ENGINE_DEFAULTS["budget"]["mode"])
            budget_cfg["mode"] = mode if mode in {"adaptive", "strict"} else PRE_ENGINE_DEFAULTS["budget"]["mode"]
            try:
                tolerance_pct = float(budget_cfg.get("tolerance_pct", PRE_ENGINE_DEFAULTS["budget"]["tolerance_pct"]))
            except (TypeError, ValueError):
                tolerance_pct = PRE_ENGINE_DEFAULTS["budget"]["tolerance_pct"]
            if tolerance_pct <= 1:
                tolerance_pct *= 100
            budget_cfg["tolerance_pct"] = max(0.0, tolerance_pct)
        fallback_cfg = merged.get("fallback")
        payload_fallback = payload.get("fallback") if isinstance(payload.get("fallback"), dict) else {}
        if isinstance(fallback_cfg, dict) and isinstance(policy, dict) and "allow_mgr_fallback" in policy:
            # Legacy compatibility: older payloads stored allow_mgr_fallback at the top level.
            if "allow_mgr_fallback" not in payload_fallback:
                fallback_cfg["allow_mgr_fallback"] = bool(policy.get("allow_mgr_fallback"))
        return merged
    merged = copy.deepcopy(PRE_ENGINE_DEFAULTS)
    fallback_cfg = merged.get("fallback")
    if isinstance(fallback_cfg, dict) and isinstance(policy, dict) and "allow_mgr_fallback" in policy:
        fallback_cfg["allow_mgr_fallback"] = bool(policy.get("allow_mgr_fallback"))
    return merged


def required_roles(policy: Dict) -> List[str]:
    settings = pre_engine_settings(policy)
    roles = settings.get("required_roles", [])
    return [role for role in roles if isinstance(role, str) and role.strip()]


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
    if parsed is None:
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
    close_min = parsed if parsed is not None else 24 * 60
    open_min = open_minutes(policy, date_)
    if close_min < open_min:
        close_min += 24 * 60
    return close_min


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
    normalized_per_sales = max(0.0, float(per_sales))
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

SERVER_TEMPLATE = {
    "am": [
        {"start": "10:30", "end": "15:30"},
        {"start": "10:45", "end": "15:45"},
        {"start": "11:00", "end": "16:00"},
        {"start": "11:15", "end": "16:15"},
        {"start": "11:30", "end": "16:30"},
        {"start": "11:45", "end": "16:45"},
        {"start": "12:00", "end": "17:00"},
        {"start": "12:30", "end": "17:30"},
    ],
    "pm": [
        {"start": "15:30", "end": "21:30"},
        {"start": "16:00", "end": "22:30"},
        {"start": "16:15", "end": "22:45"},
        {"start": "16:30", "end": "23:00"},
        {"start": "17:00", "end": "23:30"},
        {"start": "17:15", "end": "23:45"},
        {"start": "17:30", "end": "24:30"},
        {"start": "18:00", "end": "25:00"},
    ],
}

SHIFT_PRESET_DEFAULTS: Dict[str, Dict[str, List[Dict[str, str]]]] = {
    "Servers": copy.deepcopy(SERVER_TEMPLATE),
    "Kitchen": copy.deepcopy(SERVER_TEMPLATE),
    "Cashier": copy.deepcopy(SERVER_TEMPLATE),
}

SEASONAL_SETTINGS_DEFAULT: Dict[str, Any] = {"server_patio_enabled": True}

SECTION_CAPACITY_DEFAULTS: Dict[str, Dict[str, float]] = {
    "Servers": {"Dining": 1.0, "Patio": 0.6, "Cocktail": 0.8},
}


ROLES: Dict[str, Dict[str, Any]] = {
    "Server - Dining": _role_config(
        wage=6.25,
        priority=1.0,
        max_weekly=38,
        daily_boost={"Tue": 1, "Thu": 1, "Fri": 1, "Sat": 1, "Sun": -3},
        thresholds=[
            {"metric": "demand_index", "gte": 0.65, "add": 1},
            {"metric": "demand_index", "gte": 1.0, "add": 1},
        ],
        blocks={
            "Open": _block_config(2, max_staff=3, per_sales=0.2, per_modifier=0.4),
            "Mid": _block_config(
                2,
                max_staff=6,
                per_sales=0.32,
                per_modifier=0.4,
                floor_by_demand=[
                    {"gte": 0.3, "min": 3},
                    {"gte": 0.6, "min": 4},
                    {"gte": 0.9, "min": 5},
                    {"gte": 1.1, "min": 6},
                ],
            ),
            "PM": _block_config(
                3,
                max_staff=6,
                per_sales=0.35,
                per_modifier=0.5,
                floor_by_demand=[
                    {"gte": 0.3, "min": 4},
                    {"gte": 0.6, "min": 5},
                    {"gte": 0.9, "min": 6},
                    {"gte": 1.1, "min": 6},
                ],
            ),
            "Close": _block_config(1, max_staff=3, per_sales=0.15, per_modifier=0.3),
        },
        group="Servers",
        cut_buffer_minutes=35,
        covers=["Server - Cocktail", "Server - Patio"],
        always_on=True,
    ),
    "Server - Cocktail": _role_config(
        wage=6.75,
        priority=0.95,
        max_weekly=36,
        daily_boost={"Tue": 1, "Thu": 1, "Fri": 1, "Sat": 1, "Sun": -3},
        thresholds=[
            {"metric": "demand_index", "gte": 0.6, "add": 1},
            {"metric": "demand_index", "gte": 0.95, "add": 1},
        ],
        blocks={
            "Open": _block_config(1, max_staff=2, per_sales=0.1, per_modifier=0.2),
            "Mid": _block_config(
                1,
                max_staff=4,
                per_sales=0.15,
                per_modifier=0.3,
                floor_by_demand=[
                    {"gte": 0.1, "min": 1},
                    {"gte": 0.4, "min": 2},
                    {"gte": 0.7, "min": 3},
                    {"gte": 1.1, "min": 4},
                ],
            ),
            "PM": _block_config(
                2,
                max_staff=4,
                per_sales=0.25,
                per_modifier=0.4,
                floor_by_demand=[
                    {"gte": 0.1, "min": 1},
                    {"gte": 0.6, "min": 2},
                    {"gte": 0.85, "min": 3},
                    {"gte": 1.05, "min": 4},
                ],
            ),
            "Close": _block_config(1, max_staff=2, per_sales=0.15, per_modifier=0.3),
        },
        group="Servers",
        cut_buffer_minutes=35,
        covers=["Server - Dining", "Server - Patio"],
        always_on=True,
    ),
    "Server - Patio": _role_config(
        wage=6.0,
        priority=0.8,
        max_weekly=32,
        daily_boost={"Fri": 1, "Sat": 1, "Sun": -2},
        thresholds=[{"metric": "demand_index", "gte": 0.7, "add": 1}],
        blocks={
            "Mid": _block_config(1, max_staff=2, per_sales=0.1, per_modifier=0.2),
            "PM": _block_config(
                1,
                max_staff=3,
                per_sales=0.22,
                per_modifier=0.3,
                floor_by_demand=[{"gte": 0.7, "min": 2}],
            ),
        },
        group="Servers",
        cut_buffer_minutes=30,
        covers=["Server - Dining", "Server - Cocktail"],
    ),
    "Server - Training": _role_config(
        wage=5.5,
        priority=0.4,
        max_weekly=25,
        blocks={
            "Mid": _block_config(0, max_staff=1),
            "PM": _block_config(0, max_staff=1),
        },
        group="Servers",
        cut_buffer_minutes=20,
        covers=["Server - Dining", "Server - Cocktail"],
        enabled=False,
    ),
    "Server - Opener": _role_config(
        wage=6.5,
        priority=1.02,
        max_weekly=35,
        blocks={"Open": _block_config(1, min_staff=1, max_staff=1, start="@open-30", end="@open+240")},
        group="Servers",
        allow_cuts=False,
        cut_buffer_minutes=10,
        covers=["Server - Dining", "Server - Cocktail"],
    ),
    "Server - Dining Preclose": _role_config(
        wage=6.5,
        priority=0.97,
        max_weekly=34,
        blocks={"PM": _block_config(1, min_staff=1, max_staff=1, start="@close-300", end="@close-45")},
        group="Servers",
        allow_cuts=True,
        cut_buffer_minutes=15,
        covers=["Server - Dining"],
    ),
    "Server - Cocktail Preclose": _role_config(
        wage=6.7,
        priority=0.96,
        max_weekly=34,
        blocks={"PM": _block_config(1, min_staff=1, max_staff=1, start="@close-300", end="@close-60")},
        group="Servers",
        allow_cuts=True,
        cut_buffer_minutes=15,
        covers=["Server - Cocktail"],
    ),
    "Server - All Roles": _role_config(
        wage=6.75,
        priority=0.85,
        max_weekly=38,
        blocks={"Mid": _block_config(0, max_staff=0)},
        group="Servers",
        allow_cuts=True,
        enabled=False,
        covers=[
            "Server - Dining",
            "Server - Cocktail",
            "Server - Dining Preclose",
            "Server - Cocktail Preclose",
            "Server - Dining Closer",
            "Server - Cocktail Closer",
        ],
    ),
    "Server - Dining Closer": _role_config(
        wage=6.75,
        priority=1.05,
        max_weekly=35,
        daily_boost={"Thu": 1, "Fri": 1, "Sat": 1},
        blocks={"Close": _block_config(1, min_staff=1, max_staff=1, start="@close-240", end="@close+45")},
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
        blocks={"Close": _block_config(1, min_staff=1, max_staff=1, start="@close-240", end="@close+45")},
        group="Servers",
        allow_cuts=False,
        cut_buffer_minutes=5,
        covers=["Server - Cocktail"],
    ),
    "Bartender": _role_config(
        wage=10.0,
        priority=0.98,
        max_weekly=40,
        daily_boost={"Fri": 1, "Sat": 1, "Sun": -2},
        thresholds=[{"metric": "demand_index", "gte": 0.75, "add": 1}],
        blocks={
            "Open": _block_config(1, max_staff=2, per_sales=0.05, per_modifier=0.1),
            "Mid": _block_config(1, max_staff=2, per_sales=0.1, per_modifier=0.2),
            "PM": _block_config(
                1,
                max_staff=2,
                per_sales=0.15,
                per_modifier=0.3,
                floor_by_demand=[{"gte": 0.85, "min": 2}],
            ),
            "Close": _block_config(1, max_staff=2, per_sales=0.1, per_modifier=0.2),
        },
        group="Bartenders",
        allow_cuts=False,
        always_on=True,
        cut_buffer_minutes=0,
    ),
    "Bartender - Opener": _role_config(
        wage=10.5,
        priority=1.0,
        max_weekly=40,
        blocks={
            "Open": _block_config(1, min_staff=1, max_staff=1, start="@open-30", end="@open+240"),
            "Mid": _block_config(0, min_staff=0, max_staff=1),
        },
        group="Bartenders",
        allow_cuts=False,
        always_on=True,
        cut_buffer_minutes=0,
        covers=["Bartender"],
    ),
    "Bartender - Closer": _role_config(
        wage=11.0,
        priority=1.05,
        max_weekly=40,
        blocks={"Close": _block_config(1, min_staff=1, max_staff=1, start="@close-240", end="@close+60")},
        group="Bartenders",
        allow_cuts=False,
        always_on=True,
        cut_buffer_minutes=0,
        covers=["Bartender"],
    ),
    "Bartender - Training": _role_config(
        wage=9.0,
        priority=0.45,
        max_weekly=24,
        blocks={
            "Mid": _block_config(0, max_staff=1),
            "PM": _block_config(0, max_staff=1),
        },
        group="Bartenders",
        cut_buffer_minutes=15,
        covers=["Bartender"],
        enabled=False,
    ),
    "Cashier": _role_config(
        wage=15.0,
        priority=0.9,
        max_weekly=35,
        daily_boost={"Fri": 1, "Sat": 1, "Sun": -2},
        thresholds=[],
        blocks={
            "Open": _block_config(0, min_staff=0, max_staff=1, per_sales=0.02),
            "Mid": _block_config(
                1,
                min_staff=1,
                max_staff=3,
                per_sales=0.12,
                floor_by_demand=[{"gte": 0.4, "min": 1}, {"gte": 0.85, "min": 2}, {"gte": 1.05, "min": 3}],
            ),
            "PM": _block_config(
                1,
                min_staff=1,
                max_staff=3,
                per_sales=0.18,
                floor_by_demand=[{"gte": 0.4, "min": 1}, {"gte": 0.8, "min": 2}, {"gte": 1.0, "min": 3}],
            ),
            "Close": _block_config(0, min_staff=0, max_staff=0, per_sales=0.0),
        },
        group="Cashier",
        cut_buffer_minutes=20,
        covers=["Cashier - To-Go Specialist", "Host"],
    ),
    "Cashier - To-Go": _role_config(
        wage=15.0,
        priority=0.9,
        max_weekly=35,
        daily_boost={"Sun": -2},
        blocks={
            "Mid": _block_config(
                1,
                min_staff=1,
                max_staff=2,
                per_sales=0.1,
                floor_by_demand=[{"gte": 0.5, "min": 1}],
            ),
            "PM": _block_config(
                1,
                min_staff=1,
                max_staff=2,
                per_sales=0.15,
                floor_by_demand=[{"gte": 0.5, "min": 1}],
            ),
            "Close": _block_config(0, min_staff=0, max_staff=0, per_sales=0.0),
        },
        group="Cashier",
        cut_buffer_minutes=25,
        covers=["Cashier", "Cashier - Host"],
    ),
    "Cashier - Host": _role_config(
        wage=14.0,
        priority=0.82,
        max_weekly=32,
        daily_boost={"Sun": -2},
        thresholds=[],
        blocks={
            "Open": _block_config(0, min_staff=0, max_staff=1, per_sales=0.01),
            "Mid": _block_config(0, min_staff=0, max_staff=1, per_sales=0.05),
        },
        group="Cashier",
        cut_buffer_minutes=25,
        covers=["Cashier", "Cashier - To-Go"],
    ),
    "Cashier - To-Go Specialist": _role_config(
        wage=15.0,
        priority=0.88,
        max_weekly=35,
        daily_boost={"Sun": -2},
        thresholds=[],
        blocks={
            "Mid": _block_config(
                1,
                min_staff=1,
                max_staff=2,
                per_sales=0.1,
                floor_by_demand=[{"gte": 0.5, "min": 1}],
            ),
            "PM": _block_config(
                1,
                min_staff=1,
                max_staff=2,
                per_sales=0.15,
                floor_by_demand=[{"gte": 0.5, "min": 1}],
            ),
            "Close": _block_config(0, min_staff=0, max_staff=0, per_sales=0.0),
        },
        group="Cashier",
        cut_buffer_minutes=25,
        covers=["Cashier", "Host"],
    ),
    "Cashier - Training": _role_config(
        wage=13.0,
        priority=0.35,
        max_weekly=24,
        blocks={
            "Mid": _block_config(0, max_staff=1),
            "PM": _block_config(0, max_staff=1),
        },
        group="Cashier",
        cut_buffer_minutes=15,
        covers=["Cashier", "Cashier - To-Go Specialist"],
        enabled=False,
    ),
    "Cashier - All Roles": _role_config(
        wage=14.5,
        priority=0.8,
        max_weekly=38,
        blocks={"Mid": _block_config(0, max_staff=0)},
        group="Cashier",
        allow_cuts=True,
        enabled=False,
        covers=["Cashier", "Cashier - To-Go", "Cashier - Host"],
    ),
    "Host": _role_config(
        wage=14.0,
        priority=0.75,
        max_weekly=32,
        daily_boost={"Sun": -2},
        thresholds=[],
        blocks={
            "Open": _block_config(0, min_staff=0, max_staff=1, per_sales=0.01),
            "Mid": _block_config(0, min_staff=0, max_staff=1, per_sales=0.05),
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
        allow_cuts=False,
    ),
    "Grill": _role_config(
        wage=18.0,
        priority=0.94,
        max_weekly=40,
        daily_boost={"Sun": -2},
        thresholds=[],
        blocks={
            "Open": _block_config(0, min_staff=0, max_staff=1, per_sales=0.0),
            "Mid": _block_config(
                0,
                min_staff=0,
                max_staff=3,
                per_sales=0.06,
                floor_by_demand=[{"gte": 0.5, "min": 1}, {"gte": 0.9, "min": 2}],
            ),
            "PM": _block_config(
                0,
                min_staff=0,
                max_staff=3,
                per_sales=0.1,
                floor_by_demand=[{"gte": 0.5, "min": 1}, {"gte": 0.9, "min": 2}],
            ),
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
            "Mid": _block_config(
                0,
                min_staff=0,
                max_staff=3,
                per_sales=0.06,
                floor_by_demand=[{"gte": 0.5, "min": 1}, {"gte": 0.9, "min": 2}],
            ),
            "PM": _block_config(
                0,
                min_staff=0,
                max_staff=3,
                per_sales=0.1,
                floor_by_demand=[{"gte": 0.5, "min": 1}, {"gte": 0.9, "min": 2}],
            ),
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
            "Mid": _block_config(
                0,
                min_staff=0,
                max_staff=2,
                per_sales=0.04,
                floor_by_demand=[{"gte": 0.6, "min": 1}],
            ),
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
            "Mid": _block_config(
                0,
                max_staff=2,
                per_sales=0.04,
                floor_by_demand=[{"gte": 0.6, "min": 1}],
            ),
            "PM": _block_config(
                0,
                max_staff=2,
                per_sales=0.06,
                floor_by_demand=[{"gte": 0.6, "min": 1}],
            ),
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
            "Mid": _block_config(
                0,
                max_staff=2,
                per_sales=0.04,
                floor_by_demand=[{"gte": 0.6, "min": 1}],
            ),
            "PM": _block_config(
                0,
                max_staff=2,
                per_sales=0.06,
                floor_by_demand=[{"gte": 0.6, "min": 1}],
            ),
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
        blocks={"Close": _block_config(1, max_staff=2, start="@close-300", end="@close+60")},
        group="Kitchen",
        allow_cuts=False,
        cut_buffer_minutes=0,
        covers=["Prep", "Chip", "Shake"],
    ),
    "Kitchen - Training": _role_config(
        wage=17.0,
        priority=0.5,
        max_weekly=28,
        blocks={
            "Mid": _block_config(0, max_staff=1),
            "PM": _block_config(0, max_staff=1),
        },
        group="Kitchen",
        cut_buffer_minutes=20,
        covers=["Expo", "Grill", "Chip", "Shake"],
    ),
    "HOH - Expo": _role_config(
        wage=17.5,
        priority=0.95,
        max_weekly=40,
        thresholds=[{"metric": "demand_index", "gte": 0.8, "add": 1}],
        blocks={
            "Open": _block_config(1, max_staff=1, start="@open-15", end="@open+180"),
            "Mid": _block_config(1, max_staff=1),
            "PM": _block_config(1, max_staff=1),
            "Close": _block_config(0, max_staff=0),
        },
        group="Kitchen",
        cut_buffer_minutes=25,
        covers=["HOH - Chip", "HOH - Shake"],
        critical=True,
        allow_cuts=False,
    ),
    "HOH - Southwest & Grill": _role_config(
        wage=18.5,
        priority=0.9,
        max_weekly=40,
        blocks={
            "Mid": _block_config(1, min_staff=0, max_staff=2, per_sales=0.08),
            "PM": _block_config(1, min_staff=0, max_staff=2, per_sales=0.12),
        },
        group="Kitchen",
        cut_buffer_minutes=30,
        covers=["HOH - Grill", "HOH - Southwest"],
    ),
    "HOH - Grill": _role_config(
        wage=18.0,
        priority=0.94,
        max_weekly=40,
        blocks={
            "Mid": _block_config(0, min_staff=0, max_staff=2, per_sales=0.06),
            "PM": _block_config(0, min_staff=0, max_staff=2, per_sales=0.1),
        },
        group="Kitchen",
        cut_buffer_minutes=30,
        covers=["HOH - Southwest"],
    ),
    "HOH - Southwest": _role_config(
        wage=17.0,
        priority=0.9,
        max_weekly=40,
        blocks={
            "Mid": _block_config(0, min_staff=0, max_staff=2, per_sales=0.06),
            "PM": _block_config(0, min_staff=0, max_staff=2, per_sales=0.1),
        },
        group="Kitchen",
        cut_buffer_minutes=30,
        covers=["HOH - Grill"],
    ),
    "HOH - Chip & Shake": _role_config(
        wage=15.5,
        priority=0.8,
        max_weekly=34,
        blocks={
            "Mid": _block_config(0, min_staff=0, max_staff=2, per_sales=0.05),
            "PM": _block_config(0, min_staff=0, max_staff=2, per_sales=0.08),
        },
        group="Kitchen",
        cut_buffer_minutes=25,
        covers=["HOH - Chip", "HOH - Shake"],
    ),
    "HOH - Chip": _role_config(
        wage=15.5,
        priority=0.78,
        max_weekly=34,
        blocks={
            "Mid": _block_config(0, max_staff=2, per_sales=0.04),
            "PM": _block_config(0, max_staff=2, per_sales=0.06),
        },
        group="Kitchen",
        cut_buffer_minutes=25,
        covers=["HOH - Shake"],
    ),
    "HOH - Shake": _role_config(
        wage=15.5,
        priority=0.78,
        max_weekly=34,
        blocks={
            "Mid": _block_config(0, max_staff=2, per_sales=0.04),
            "PM": _block_config(0, max_staff=2, per_sales=0.06),
        },
        group="Kitchen",
        cut_buffer_minutes=25,
        covers=["HOH - Chip"],
    ),
    "HOH - Opener": _role_config(
        wage=18.5,
        priority=0.88,
        max_weekly=38,
        blocks={"Open": _block_config(1, max_staff=1, start="@open-30", end="@open+240")},
        group="Kitchen",
        cut_buffer_minutes=20,
        covers=["HOH - Chip", "HOH - Shake"],
    ),
    "HOH - Closer": _role_config(
        wage=19.0,
        priority=0.9,
        max_weekly=38,
        blocks={"Close": _block_config(1, max_staff=1, start="@close-300", end="@close+60")},
        group="Kitchen",
        allow_cuts=False,
        cut_buffer_minutes=0,
        covers=["HOH - Chip", "HOH - Shake"],
    ),
    "HOH - Training": _role_config(
        wage=17.0,
        priority=0.5,
        max_weekly=28,
        blocks={"Mid": _block_config(0, max_staff=1), "PM": _block_config(0, max_staff=1)},
        group="Kitchen",
        cut_buffer_minutes=20,
        covers=["HOH - Expo", "HOH - Grill", "HOH - Chip", "HOH - Shake"],
        enabled=False,
    ),
    "HOH - All Roles": _role_config(
        wage=18.0,
        priority=0.85,
        max_weekly=40,
        blocks={"Mid": _block_config(0, max_staff=0)},
        group="Kitchen",
        allow_cuts=True,
        enabled=False,
        covers=[
            "HOH - Expo",
            "HOH - Grill",
            "HOH - Southwest",
            "HOH - Chip",
            "HOH - Shake",
            "HOH - Southwest & Grill",
            "HOH - Chip & Shake",
        ],
    ),
    "MGR - FOH": _role_config(
        wage=0.0,
        priority=0.6,
        max_weekly=50,
        blocks={"Mid": _block_config(0, max_staff=0)},
        group="Management",
        allow_cuts=True,
        enabled=False,
    ),
    "Shift Lead": _role_config(
        wage=22.0,
        priority=1.1,
        max_weekly=45,
        blocks={"Mid": _block_config(0, max_staff=0)},
        group="Management",
        allow_cuts=True,
        enabled=False,
    ),
}


ROLE_GROUP_ALLOCATIONS: Dict[str, Dict[str, Any]] = {
    "Kitchen": {"allocation_pct": 0.34, "allow_cuts": True, "cut_buffer_minutes": 25},
    "Servers": {"allocation_pct": 0.39, "allow_cuts": True, "cut_buffer_minutes": 35},
    "Bartenders": {"allocation_pct": 0.12, "allow_cuts": False, "always_on": True, "cut_buffer_minutes": 0},
    "Cashier": {"allocation_pct": 0.15, "allow_cuts": True, "cut_buffer_minutes": 25},
    "Management": {"allocation_pct": 0.0, "allow_cuts": True, "cut_buffer_minutes": 30},
}

VOLUME_THRESHOLDS_DEFAULT: Dict[str, float] = {
    "slow_max": 0.45,
    "moderate_max": 0.75,
    "peak_min": 0.9,
    # Optional absolute mode (sales-based) for dev/internal use; keep default off.
    "use_absolute": False,
    "slow_sales_max": 4500.0,
    "moderate_sales_max": 8500.0,
}
PRE_ENGINE_DEFAULTS: Dict[str, Any] = {
    "required_roles": [
        "Bartender - Opener",
        "Bartender - Closer",
        "HOH - Opener",
        "HOH - Closer",
        "HOH - Expo",
        "Server - Opener",
        "Server - Dining Preclose",
        "Server - Dining Closer",
        "Server - Cocktail Preclose",
        "Server - Cocktail Closer",
        "Cashier",
    ],
    "fallback": {
        "allow_mgr_fallback": True,
        "am_limit": 1,
        "pm_limit": 1,
        "tag": "MANAGER COVERING — REVIEW REQUIRED",
        "disallow_roles": ["bartender", "server", "expo", "opener", "closer"],
    },
    "staffing": {
        "volume_thresholds": VOLUME_THRESHOLDS_DEFAULT,
        "servers": {
            "dining": {"slow_min": 1, "slow_max": 4, "moderate": 5, "peak": 6, "manual_max": 7},
            "cocktail": {"normal": 2, "busy": 3, "peak": 4, "manual_max": 4},
            "opener_count": 1,
            "preclose": {"dining": 1, "cocktail": 1},
            "closers": {"dining": 1, "cocktail": 1},
            "cut_order": {
                "dining": ["Server - Dining", "Server - Dining Preclose", "Server - Dining Closer"],
                "cocktail": ["Server - Cocktail", "Server - Cocktail Preclose", "Server - Cocktail Closer"],
                "fifo": True,
            },
        },
        "cashier": {
            "am_default": 1,
            "pm_default": 1,
            "busy_split": 2,
            "peak": 3,
            "manual_max": 4,
            "coverable_by": ["Servers", "Bartenders", "MGR - FOH"],
            "cut_after_minutes": 90,
        },
        "hoh": {
            "combo_thresholds": {"low_max": 0.55, "split_min": 0.75, "peak_min": 1.0},
            "open_sequence": {"southwest": "11:00", "chip": "11:15", "expo": "@open"},
            "combo_roles": {"sw_grill": "HOH - Southwest & Grill", "chip_shake": "HOH - Chip & Shake"},
            "cut_priority": [
                "HOH - Southwest",
                "HOH - Chip",
                "HOH - Shake",
                "HOH - Grill",
                "HOH - Chip & Shake",
                "HOH - Southwest & Grill",
                "HOH - Expo",
            ],
            "non_combinable": ["HOH - Expo"],
        },
    },
    "training": {"auto_generate": False, "preserve_manual": True},
    "budget": {"mode": "adaptive", "tolerance_pct": 8.0},
}

CUT_PRIORITY_DEFAULT: Dict[str, Any] = {
    "enabled": False,
    "include_unlisted": True,
    "sequence": [
        {
            "group": "Servers",
            "roles": [
                "Server - Dining",
                "Server - Dining Opener",
                "Server - Dining Preclose",
                "Server - Dining Closer",
            ],
        },
        {"group": "Kitchen", "roles": []},
        {
            "group": "Cashier",
            "roles": ["Cashier", "Cashier - To-Go Specialist", "Host"],
        },
        {
            "group": "Servers",
            "roles": [
                "Server - Cocktail",
                "Server - Cocktail Opener",
                "Server - Cocktail Preclose",
                "Server - Cocktail Closer",
            ],
        },
    ],
    "role_order": {
        "Servers": [
            "Server - Patio",
            "Server - Dining",
            "Server - Cocktail",
            "Server - Dining Opener",
            "Server - Cocktail Opener",
            "Server - Dining Closer",
            "Server - Cocktail Closer",
        ],
        "Kitchen": ["Expo", "Grill", "Chip", "Shake", "Prep"],
        "Cashier": ["Cashier", "Cashier - To-Go Specialist", "Host"],
    },
}


ANCHOR_RULES: Dict[str, Any] = {
    "openers": {"Kitchen": 1, "Servers": 1, "Bartenders": 1, "Cashier": 0},
    "closers": {"Kitchen": 1, "Servers": 2, "Bartenders": 1, "Cashier": 0},
    "opener_roles": {
        "Kitchen": ["HOH - Opener"],
        "Servers": ["Server - Opener", "Server - Dining", "Server - Cocktail"],
        "Bartenders": ["Bartender - Opener", "Bartender"],
        "Cashier": ["Cashier", "Cashier - To-Go", "Cashier - Host"],
    },
    "closer_roles": {
        "Kitchen": ["HOH - Closer"],
        "Servers": ["Server - Dining Closer", "Server - Cocktail Closer", "Server - Dining", "Server - Cocktail"],
        "Bartenders": ["Bartender - Closer", "Bartender"],
        "Cashier": ["Cashier - To-Go", "Cashier - Host", "Cashier"],
    },
    "non_cuttable_roles": [
        "Bartender - Opener",
        "Bartender - Closer",
        "HOH - Expo",
        "HOH - Closer",
        "Server - Dining Closer",
        "Server - Cocktail Closer",
    ],
    "allow_cashier_closer": False,
    # Controls FIFO/LILO bias for open/close patterns: static enforcement.
    "open_close_order": "enforce",
    # Cut rotation configurable via UI removed; ordering handled in code.
    "cut_priority": {},
}


BASELINE_POLICY: Dict[str, Any] = {
    "name": "Baseline Coverage",
    "description": "Seeded policy that balances FOH/BOH coverage for the automation workflow.",
    "section_priority": "normal",
    "hoh_mode": "auto",
    "allow_mgr_fallback": True,
    "global": {
        "max_hours_week": 48,
        "max_consecutive_days": 7,
        "overtime_penalty": 1.5,
        "desired_hours_floor_pct": 0.85,
        "desired_hours_ceiling_pct": 1.15,
        "close_buffer_minutes": 35,
        "labor_budget_pct": 0.27,
        "labor_budget_tolerance_pct": 0.08,
        "trim_aggressive_ratio": 1.0,
    },
    "timeblocks": DEFAULT_TIMEBLOCKS,
    "business_hours": BUSINESS_HOURS,
    "anchors": ANCHOR_RULES,
    "pattern_templates": PATTERN_TEMPLATES,
    "seasonal_settings": SEASONAL_SETTINGS_DEFAULT,
    "shift_presets": SHIFT_PRESET_DEFAULTS,
    "section_capacity": SECTION_CAPACITY_DEFAULTS,
    "roles": ROLES,
    "pre_engine": PRE_ENGINE_DEFAULTS,
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


def resolve_section_weights(policy: Dict) -> Dict[str, float]:
    """Return section weights based on priority selection or stored custom values."""
    priority = (policy.get("section_priority") or "normal").lower()
    weights = DEFAULT_ENGINE_TUNING["section_weights"].copy()
    if priority == "patio_light":
        weights["patio"] = 0.3
    elif priority == "cocktail_light":
        weights["cocktail"] = 0.5
    elif priority == "custom":
        section_capacity = policy.get("section_capacity", {}).get("Servers", {})
        weights["dining"] = float(section_capacity.get("Dining", weights["dining"]))
        weights["patio"] = float(section_capacity.get("Patio", weights["patio"]))
        weights["cocktail"] = float(section_capacity.get("Cocktail", weights["cocktail"]))
    return weights


def resolve_hoh_thresholds(policy: Dict) -> Dict[str, float]:
    """Return HOH thresholds based on selected mode."""
    mode = (policy.get("hoh_mode") or "auto").lower()
    base = DEFAULT_ENGINE_TUNING["hoh_thresholds"]
    if mode == "combo":
        return {"combo_enabled_max": 0.65, "split_threshold": 0.85, "peak_threshold": base["peak_threshold"]}
    if mode == "split":
        return {"combo_enabled_max": 0.45, "split_threshold": 0.65, "peak_threshold": base["peak_threshold"]}
    if mode == "peak":
        return {"combo_enabled_max": 0.0, "split_threshold": 0.0, "peak_threshold": base["peak_threshold"]}
    return base.copy()


def resolve_fallback_limits(policy: Dict) -> Dict[str, int]:
    """Return fallback limits derived from allow_mgr_fallback flag."""
    allow = bool(policy.get("allow_mgr_fallback", True))
    if not allow:
        return {"am": 0, "pm": 0}
    return DEFAULT_ENGINE_TUNING["fallback_limits"].copy()
DEFAULT_ENGINE_TUNING: Dict[str, Dict[str, float | int]] = {
    "section_weights": {
        "dining": 1.0,
        "patio": 0.5,
        "cocktail": 0.7,
    },
    "hoh_thresholds": {
        "combo_enabled_max": 0.55,
        "split_threshold": 0.75,
        "peak_threshold": 1.00,
    },
    "fallback_limits": {
        "am": 1,
        "pm": 1,
    },
}
