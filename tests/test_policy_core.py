from __future__ import annotations

import sys
from pathlib import Path

import pytest

APP_DIR = Path(__file__).resolve().parents[1] / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from policy import (  # noqa: E402
    SHIFT_LENGTH_DEFAULTS,
    PRE_ENGINE_DEFAULTS,
    _normalize_policy,
    build_default_policy,
    pre_engine_settings,
    role_catalog,
    shift_length_limits,
)


def test_build_default_policy_returns_isolated_copy() -> None:
    first = build_default_policy()
    second = build_default_policy()

    first["global"]["max_hours_week"] = 99

    assert second["global"]["max_hours_week"] != 99
    assert build_default_policy()["global"]["max_hours_week"] != 99


def test_pre_engine_budget_normalization_handles_mode_and_percent() -> None:
    cfg = pre_engine_settings({"pre_engine": {"budget": {"mode": "strict", "tolerance_pct": 0.5}}})

    assert cfg["budget"]["mode"] == "strict"
    assert cfg["budget"]["tolerance_pct"] == pytest.approx(50.0)


def test_pre_engine_budget_defaults_on_invalid_values() -> None:
    cfg = pre_engine_settings({"pre_engine": {"budget": {"mode": "??", "tolerance_pct": "abc"}}})

    assert cfg["budget"]["mode"] == PRE_ENGINE_DEFAULTS["budget"]["mode"]
    assert cfg["budget"]["tolerance_pct"] == PRE_ENGINE_DEFAULTS["budget"]["tolerance_pct"]


def test_shift_length_limits_fall_back_to_group_defaults() -> None:
    policy = {"roles": {"Server": {"shift_length_rule": {"minHrs": "bad", "maxHrs": None}}}}
    min_hours, max_hours = shift_length_limits(policy, "Server", "Servers")

    assert min_hours == SHIFT_LENGTH_DEFAULTS["Servers"]["min"]
    assert max_hours == SHIFT_LENGTH_DEFAULTS["Servers"]["max"]


def test_normalize_policy_adds_required_roles_to_non_cuttable() -> None:
    policy = {
        "anchors": {"non_cuttable_roles": ["Custom Role"]},
        "pre_engine": {"required_roles": ["Server - Dining Closer", "Server - Cocktail Closer"]},
        "roles": {"Server - Dining Closer": {}, "Server - Cocktail Closer": {}},
        "global": {"trim_aggressive_ratio": 0.1},
    }

    normalized = _normalize_policy(policy)

    assert {"Custom Role", "Server - Dining Closer", "Server - Cocktail Closer"}.issubset(
        set(normalized["anchors"]["non_cuttable_roles"])
    )
    assert normalized["global"]["trim_aggressive_ratio"] >= 1.0


def test_role_catalog_returns_role_names() -> None:
    policy = {"roles": {"Server": {}, "Bartender": {}}}

    assert role_catalog(policy) == {"Server", "Bartender"}
