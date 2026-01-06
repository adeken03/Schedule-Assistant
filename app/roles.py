from __future__ import annotations

from typing import Dict, Iterable, List, Set, Tuple


ROLE_GROUPS: Dict[str, List[str]] = {
    "Kitchen": [
        "HOH - Opener",
        "HOH - Closer",
        "HOH - Training",
        "HOH - Expo",
        "HOH - Southwest & Grill",
        "HOH - Grill",
        "HOH - Southwest",
        "HOH - Chip & Shake",
        "HOH - Chip",
        "HOH - Shake",
        "HOH - All Roles",  # simply means all other roles can be fulfilled/considered for this employee
    ],
    "Servers": [
        "Server - Dining",
        "Server - Opener",
        "Server - Dining Preclose",
        "Server - Dining Closer",
        "Server - Training",
        "Server - Patio",
        "Server - Cocktail",
        "Server - Cocktail Preclose",
        "Server - Cocktail Closer",
        "Server - All Roles", # not a real role, but rather easy to select that employee can fulfill all
    ],
    "Bartenders": [
        "Bartender",
        "Bartender - Opener",
        "Bartender - Closer",
        "Bartender - Training",
    ],
    "Cashier": [
        "Cashier",
        "Cashier - To-Go",
        "Cashier - Host",
        "Cashier - Training",
        "Cashier - All Roles", # all role flag, not a real working role
    ],
    "Management": [
        "Shift Lead",
        "MGR - FOH",
    ],
}

ROLE_COLORS: Dict[str, str] = {
    "Kitchen": "#2f2f33",      # HOH steel grey
    "Servers": "#1f3b70",      # FOH royal blue
    "Bartenders": "#8b4a2b",   # Aged bourbon amber
    "Cashier": "#2e5939",      # Money green
    "Management": "#7d0000",   # Escalation red
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
    ("hoh", "Kitchen"),
    ("southwest", "Kitchen"),
    ("lead", "Management"),
    ("mgr", "Management"),
]


def normalize_role(role: str) -> str:
    return (role or "").strip().lower()


def is_manager_role(role: str) -> bool:
    label = normalize_role(role)
    if not label:
        return False
    if "shift lead" in label:
        return True
    return "mgr" in label or "manager" in label


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
