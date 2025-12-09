from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable

from policy import build_default_policy
from roles import ROLE_GROUPS, normalize_role


DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
WAGES_FILE = DATA_DIR / "role_wages.json"
ALLOW_ZERO_ROLES = {"mgr - foh"}


def _defined_roles() -> Iterable[str]:
    for names in ROLE_GROUPS.values():
        for role in names:
            yield role


def baseline_wages() -> Dict[str, Dict[str, Any]]:
    policy = build_default_policy()
    roles = policy.get("roles", {})
    payload: Dict[str, Dict[str, Any]] = {}
    for role in _defined_roles():
        spec = roles.get(role) if isinstance(roles, dict) else {}
        wage = float(spec.get("hourly_wage", 0.0) if isinstance(spec, dict) else 0.0)
        normalized = normalize_role(role)
        default_confirmed = normalized in ALLOW_ZERO_ROLES or wage > 0
        payload[role] = {
            "wage": round(max(0.0, wage), 2),
            "confirmed": default_confirmed,
        }
    return payload


def load_wages() -> Dict[str, Dict[str, Any]]:
    if WAGES_FILE.exists():
        try:
            data = json.loads(WAGES_FILE.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError
        except Exception:  # noqa: BLE001
            data = baseline_wages()
    else:
        data = baseline_wages()

    baseline = baseline_wages()
    for role, default in baseline.items():
        entry = data.get(role)
        if not isinstance(entry, dict):
            data[role] = default
            continue
        if "wage" not in entry:
            entry["wage"] = default["wage"]
        if "confirmed" not in entry:
            entry["confirmed"] = default["confirmed"]
        else:
            entry["confirmed"] = bool(entry["confirmed"] or default["confirmed"])
    return data


def save_wages(data: Dict[str, Dict[str, Any]]) -> None:
    WAGES_FILE.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def wage_amounts() -> Dict[str, float]:
    data = load_wages()
    amounts: Dict[str, float] = {}
    for role, entry in data.items():
        try:
            amounts[role] = float(entry.get("wage", 0.0) or 0.0)
        except (TypeError, ValueError):
            amounts[role] = 0.0
    return amounts


def validate_wages(roles: Iterable[str]) -> Dict[str, str]:
    """Return dict of roles missing confirmed wages -> reason."""
    allow_zero = ALLOW_ZERO_ROLES
    data = load_wages()
    problems: Dict[str, str] = {}
    for role in roles:
        entry = data.get(role)
        if not isinstance(entry, dict):
            problems[role] = "not configured"
            continue
        try:
            wage = float(entry.get("wage", 0.0) or 0.0)
        except (TypeError, ValueError):
            wage = 0.0
        normalized = normalize_role(role)
        if wage <= 0.0 and normalized not in allow_zero:
            problems[role] = "wage is zero"
            continue
        if not entry.get("confirmed", False):
            if normalized not in allow_zero:
                problems[role] = "not confirmed"
            else:
                problems[role] = "confirm salary role"
    return problems


def export_wages(target: Path) -> Path:
    target.write_text(json.dumps(load_wages(), indent=2, sort_keys=True), encoding="utf-8")
    return target


def import_wages(source: Path) -> int:
    data = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Wages file must be a JSON object.")
    normalized: Dict[str, Dict[str, Any]] = load_wages()
    count = 0
    for role, entry in data.items():
        if role not in normalized:
            continue
        if not isinstance(entry, dict):
            continue
        record = normalized.setdefault(role, {"wage": 0.0, "confirmed": False})
        if "wage" in entry:
            try:
                record["wage"] = round(float(entry["wage"]), 2)
            except (TypeError, ValueError):
                pass
        if "confirmed" in entry:
            record["confirmed"] = (
                bool(entry["confirmed"])
                or normalize_role(role) in ALLOW_ZERO_ROLES
                or record.get("wage", 0) > 0
            )
        count += 1
    save_wages(normalized)
    return count


def reset_wages_to_defaults() -> None:
    save_wages(baseline_wages())
