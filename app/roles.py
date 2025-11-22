from __future__ import annotations

from typing import Dict, Iterable, List, Set, Tuple


ROLE_GROUPS: Dict[str, List[str]] = {
    "Kitchen": [
        "Kitchen Opener",
        "Kitchen Closer",
        "Expo",
        "Grill",
        "Chip",
        "Shake",
        "Prep",
        "Cook",
    ],
    "Servers": [
        "Server - Dining",
        "Server - Dining Opener",
        "Server - Dining Preclose",
        "Server - Dining Closer",
        "Server - Patio",
        "Server - Cocktail",
        "Server - Cocktail Opener",
        "Server - Cocktail Preclose",
        "Server - Cocktail Closer",
    ],
    "Bartenders": [
        "Bartender",
        "Bartender - Opener",
        "Bartender - Closer",
    ],
    "Cashier": [
        "Cashier",
        "Cashier - To-Go Specialist",
        "Host",
    ],
}

ROLE_COLORS: Dict[str, str] = {
    "Kitchen": "#2f3a4f",
    "Servers": "#1c4641",
    "Bartenders": "#4a1f43",
    "Cashier": "#4a3a1f",
    "Other": "#2f2f2f",
}

_KEYWORD_RULES: List[Tuple[str, str]] = [
    ("server", "Servers"),
    ("bartend", "Bartenders"),
    ("bar", "Bartenders"),
    ("cashier", "Cashier"),
    ("to-go", "Cashier"),
    ("host", "Cashier"),
    ("expo", "Kitchen"),
    ("kitchen", "Kitchen"),
    ("cook", "Kitchen"),
    ("prep", "Kitchen"),
    ("grill", "Kitchen"),
    ("chip", "Kitchen"),
    ("shake", "Kitchen"),
]


def normalize_role(role: str) -> str:
    return (role or "").strip().lower()


def is_manager_role(role: str) -> bool:
    label = normalize_role(role)
    if not label:
        return False
    return "manager" in label or "mgr" in label


def role_group(role: str) -> str:
    label = normalize_role(role)
    if not label:
        return "Other"
    if "heart of house" in label:
        return "Kitchen"
    if "cashier & takeout" in label:
        return "Cashier"
    for group, names in ROLE_GROUPS.items():
        for name in names:
            if label == normalize_role(name):
                return group
    for keyword, target in _KEYWORD_RULES:
        if keyword in label:
            return target
    return "Other"


def palette_for_role(role: str) -> str:
    group = role_group(role)
    return ROLE_COLORS.get(group, ROLE_COLORS["Other"])


def grouped_roles(roles: Iterable[str]) -> Dict[str, List[str]]:
    mapping: Dict[str, List[str]] = {group: [] for group in ROLE_GROUPS}
    mapping["Other"] = []
    for role in roles:
        if not role or is_manager_role(role):
            continue
        group = role_group(role)
        if role not in mapping.setdefault(group, []):
            mapping[group].append(role)
    for group in mapping:
        mapping[group].sort()
    return {group: entries for group, entries in mapping.items() if entries}


def defined_roles() -> List[str]:
    """Return a sorted list of roles explicitly supported by the app."""
    roles: List[str] = []
    for names in ROLE_GROUPS.values():
        roles.extend(names)
    return sorted(set(roles))


def role_aliases(role: str) -> List[str]:
    """Return simplified aliases for a role label."""
    label = (role or "").strip()
    aliases: List[str] = []
    if not label:
        return aliases
    parts = [part.strip() for part in label.split(" - ") if part.strip()]
    if len(parts) >= 2:
        alias = " - ".join(parts[:2])
        if alias and alias != label:
            aliases.append(alias)
    if parts:
        base = parts[0]
        if base and base != label:
            aliases.append(base)
    return aliases


def _normalized_variants(role: str) -> Set[str]:
    variants: Set[str] = set()
    normalized = normalize_role(role)
    if normalized:
        variants.add(normalized)
    for alias in role_aliases(role):
        alias_norm = normalize_role(alias)
        if alias_norm:
            variants.add(alias_norm)
    return variants


def role_matches(candidate_role: str, target_role: str) -> bool:
    """Return True if a candidate role label should satisfy the requested role."""
    target_variants = _normalized_variants(target_role)
    candidate_variants = _normalized_variants(candidate_role)
    if not target_variants or not candidate_variants:
        return False
    for candidate in candidate_variants:
        for target in target_variants:
            if candidate == target:
                return True
            if candidate in target or target in candidate:
                return True
    return False
