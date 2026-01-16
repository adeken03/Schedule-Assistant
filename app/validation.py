from __future__ import annotations

import datetime
from collections import defaultdict
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from sqlalchemy import select
from sqlalchemy.orm import selectinload

from database import (
    Employee,
    Shift,
    WeekSchedule,
    get_active_policy,
    get_week_daily_projections,
    shift_display_date,
)
from policy import anchor_rules, build_default_policy, close_minutes, mid_minutes, open_minutes, pre_engine_settings, required_roles
from roles import normalize_role, role_group, role_matches

UTC = datetime.timezone.utc
WEEKDAY_TOKENS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
ROLE_CONCURRENCY_LIMITS = {
    "Server - Dining": 6,
    "Server - Cocktail": 4,
}
NORMALIZED_ROLE_LIMITS = {
    normalize_role(name): (name, limit) for name, limit in ROLE_CONCURRENCY_LIMITS.items()
}
ARRIVAL_WINDOW_NORMAL_MINUTES = 90
ARRIVAL_WINDOW_MAX_MINUTES = 120


def validate_week_schedule(session, week_start: datetime.date, *, employee_session=None) -> Dict[str, Any]:
    """Return validation findings for the requested week."""
    normalized_start = _normalize_week_start(week_start)
    week = session.execute(
        select(WeekSchedule).where(WeekSchedule.week_start_date == normalized_start)
    ).scalar_one_or_none()
    if not week:
        return {
            "week_start": normalized_start.isoformat(),
            "week_id": None,
            "checks": [
                {
                    "label": "Schedule exists?",
                    "status": "fail",
                    "details": "No schedule exists for the requested week.",
                }
            ],
            "issues": [
                {
                    "type": "missing_schedule",
                    "severity": "error",
                    "message": "No schedule exists for the requested week.",
                }
            ],
            "warnings": [],
        }

    shifts = list(session.scalars(select(Shift).where(Shift.week_id == week.id)))
    policy_model = get_active_policy(session)
    if policy_model:
        policy_payload = policy_model.params_dict()
    else:
        policy_payload = build_default_policy()

    employee_ids = {shift.employee_id for shift in shifts if shift.employee_id}
    employee_map: Dict[int, Employee] = {}
    if employee_ids and employee_session:
        employees = employee_session.scalars(
            select(Employee)
            .options(selectinload(Employee.unavailability))
            .where(Employee.id.in_(employee_ids))
        )
        for employee in employees:
            employee_map[employee.id] = employee

    issues: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []
    issues.extend(_unassigned_shift_issues(shifts))
    issues.extend(_empty_day_issues(shifts, normalized_start))
    issues.extend(_availability_issues(shifts, employee_map))
    issues.extend(_role_match_issues(shifts, employee_map))
    issues.extend(_coverage_issues(shifts, policy_payload))
    issues.extend(_open_close_issues(shifts, policy_payload))
    issues.extend(_open_close_continuity_issues(shifts))
    pre_engine_cfg = pre_engine_settings(policy_payload)
    demand_indices = _demand_indices_for_week(session, week)
    issues.extend(_required_role_issues(shifts, policy_payload))
    fallback_errors, fallback_warnings = _fallback_issues(shifts, pre_engine_cfg)
    issues.extend(fallback_errors)
    warnings.extend(fallback_warnings)
    staffing_errors, staffing_warnings = _staffing_threshold_issues(shifts, pre_engine_cfg, demand_indices)
    issues.extend(staffing_errors)
    warnings.extend(staffing_warnings)
    hoh_errors, hoh_warnings = _hoh_combo_issues(shifts, pre_engine_cfg, demand_indices)
    issues.extend(hoh_errors)
    warnings.extend(hoh_warnings)
    warnings.extend(_concurrency_warnings(shifts))
    warnings.extend(_weekly_hours_warnings(shifts, employee_map, policy_payload))
    warnings.extend(_arrival_window_warnings(shifts, policy_payload))
    checks = _build_validation_checklist(shifts, issues=issues)
    return {
        "week_start": normalized_start.isoformat(),
        "week_id": week.id,
        "checks": checks,
        "issues": issues,
        "warnings": warnings,
    }


def _normalize_week_start(date_value: datetime.date | datetime.datetime) -> datetime.date:
    if isinstance(date_value, datetime.datetime):
        date_value = date_value.date()
    weekday = date_value.weekday()
    if weekday == 0:
        return date_value
    return date_value - datetime.timedelta(days=weekday)


def _availability_issues(shifts: List[Shift], employees: Dict[int, Employee]) -> List[Dict[str, Any]]:
    issues: List[Dict[str, Any]] = []
    unavailability = {emp_id: _unavailability_windows(employee) for emp_id, employee in employees.items()}
    for shift in shifts:
        if not shift.employee_id or shift.employee_id not in unavailability:
            continue
        segments = _shift_segments(shift)
        windows = unavailability[shift.employee_id]
        employee = employees[shift.employee_id]
        for day_index, day_start, seg_start, seg_end in segments:
            start_minutes = int((seg_start - day_start).total_seconds() // 60)
            end_minutes = int((seg_end - day_start).total_seconds() // 60)
            for window_start, window_end in windows.get(day_index, []):
                if start_minutes < window_end and end_minutes > window_start:
                    issues.append(
                        {
                            "type": "availability",
                            "severity": "error",
                            "shift_id": shift.id,
                            "employee_id": shift.employee_id,
                            "employee": employee.full_name,
                            "day": WEEKDAY_TOKENS[day_index],
                            "message": f"{employee.full_name} is unavailable between "
                            f"{_format_minutes(window_start)}-{_format_minutes(window_end)} "
                            f"on {WEEKDAY_TOKENS[day_index]}.",
                        }
                    )
                    break
    return issues


def _unassigned_shift_issues(shifts: List[Shift]) -> List[Dict[str, Any]]:
    issues: List[Dict[str, Any]] = []
    for shift in shifts:
        if shift.employee_id:
            continue
        op_date = shift_display_date(shift.start, shift.location)
        start = _as_local_datetime(shift.start)
        end = _as_local_datetime(shift.end)
        op_day_start = datetime.datetime.combine(op_date, datetime.time.min)
        start_minutes = int((start - op_day_start).total_seconds() // 60)
        end_minutes = int((end - op_day_start).total_seconds() // 60)
        day_token = WEEKDAY_TOKENS[op_date.weekday()]
        time_label = f"{day_token} {_format_minutes(start_minutes)}"
        if end_minutes > start_minutes:
            time_label = f"{time_label}-{_format_minutes(end_minutes)}"
        issues.append(
            {
                "type": "assignment",
                "severity": "error",
                "shift_id": shift.id,
                "role": shift.role,
                "day": WEEKDAY_TOKENS[_shift_day_index(shift)],
                "message": f"Unassigned shift: {shift.role} at {time_label}.",
            }
        )
    return issues


def _empty_day_issues(shifts: List[Shift], week_start: datetime.date) -> List[Dict[str, Any]]:
    issues: List[Dict[str, Any]] = []
    counts = {idx: 0 for idx in range(7)}
    for shift in shifts:
        counts[_shift_day_index(shift)] += 1
    for day_index in range(7):
        if counts.get(day_index, 0) > 0:
            continue
        date_value = week_start + datetime.timedelta(days=day_index)
        issues.append(
            {
                "type": "coverage",
                "severity": "error",
                "day": WEEKDAY_TOKENS[day_index],
                "message": f"No shifts scheduled on {date_value.strftime('%a %Y-%m-%d')}.",
            }
        )
    return issues


def _role_match_issues(shifts: List[Shift], employees: Dict[int, Employee]) -> List[Dict[str, Any]]:
    issues: List[Dict[str, Any]] = []
    if not employees:
        return issues
    for shift in shifts:
        if not shift.employee_id:
            continue
        employee = employees.get(shift.employee_id)
        if not employee:
            continue
        if any(role_matches(role, shift.role) for role in employee.role_list):
            continue
        start = _as_local_datetime(shift.start)
        issues.append(
            {
                "type": "role_match",
                "severity": "error",
                "shift_id": shift.id,
                "employee_id": shift.employee_id,
                "employee": employee.full_name,
                "role": shift.role,
                "day": WEEKDAY_TOKENS[_shift_day_index(shift)],
                "message": f"{employee.full_name} is assigned {shift.role} at {start.strftime('%a %H:%M')} but is not eligible for that role.",
            }
        )
    return issues


def _open_close_continuity_issues(shifts: List[Shift]) -> List[Dict[str, Any]]:
    """
    Open/close continuity checks:
    - Open buffer shifts (location=Open) must have an immediate same-group follow-up.
    - Close buffer shifts must have an immediate same-group lead-in.
    """
    issues: List[Dict[str, Any]] = []
    tolerance = datetime.timedelta(minutes=10)

    def group_for(shift: Shift) -> str:
        return role_group(shift.role or "")

    open_shifts = [shift for shift in shifts if (shift.location or "").strip().lower() == "open"]
    for opener in open_shifts:
        day = WEEKDAY_TOKENS[_shift_day_index(opener)]
        if opener.employee_id is None:
            issues.append(
                {
                    "type": "continuity",
                    "kind": "open_followup",
                    "severity": "error",
                    "shift_id": opener.id,
                    "role": opener.role,
                    "day": day,
                    "message": f"Opener buffer shift is unassigned: {opener.role}.",
                }
            )
            continue
        opener_end = _as_local_datetime(opener.end)
        opener_group = group_for(opener)
        has_followup = any(
            candidate is not opener
            and candidate.employee_id == opener.employee_id
            and group_for(candidate) == opener_group
            and (candidate.location or "").strip().lower() != "open"
            and abs(_as_local_datetime(candidate.start) - opener_end) <= tolerance
            for candidate in shifts
        )
        if not has_followup:
            issues.append(
                {
                    "type": "continuity",
                    "kind": "open_followup",
                    "severity": "error",
                    "shift_id": opener.id,
                    "role": opener.role,
                    "employee_id": opener.employee_id,
                    "day": day,
                    "message": f"{opener.role} opener buffer has no same-group follow-up shift.",
                }
            )

    def is_close_buffer(shift: Shift) -> bool:
        if (shift.location or "").strip().lower() != "close":
            return False
        note = (shift.notes or "").lower()
        if "close buffer" in note:
            return True
        role_norm = normalize_role(shift.role)
        duration = shift.end - shift.start
        if duration > datetime.timedelta(hours=2):
            return False
        if role_norm == "hoh - expo":
            return True
        return "closer" in role_norm

    close_buffers = [shift for shift in shifts if is_close_buffer(shift)]
    for buffer_shift in close_buffers:
        day = WEEKDAY_TOKENS[_shift_day_index(buffer_shift)]
        op_date = shift_display_date(buffer_shift.start, buffer_shift.location)
        op_day_start = datetime.datetime.combine(op_date, datetime.time.min)
        buffer_start_dt = _as_local_datetime(buffer_shift.start)
        buffer_end_dt = _as_local_datetime(buffer_shift.end)
        start_minutes = int((buffer_start_dt - op_day_start).total_seconds() // 60)
        end_minutes = int((buffer_end_dt - op_day_start).total_seconds() // 60)
        time_label = f"{WEEKDAY_TOKENS[op_date.weekday()]} {_format_minutes(start_minutes)}"
        if end_minutes > start_minutes:
            time_label = f"{time_label}-{_format_minutes(end_minutes)}"
        if buffer_shift.employee_id is None:
            issues.append(
                {
                    "type": "continuity",
                    "kind": "close_leadin",
                    "severity": "error",
                    "shift_id": buffer_shift.id,
                    "role": buffer_shift.role,
                    "day": day,
                    "message": f"Close buffer shift is unassigned: {buffer_shift.role} at {time_label}.",
                }
            )
            continue
        buffer_start = buffer_start_dt
        buffer_group = group_for(buffer_shift)
        has_leadin = any(
            candidate is not buffer_shift
            and candidate.employee_id == buffer_shift.employee_id
            and group_for(candidate) == buffer_group
            and abs(_as_local_datetime(candidate.end) - buffer_start) <= tolerance
            for candidate in shifts
        )
        if not has_leadin:
            issues.append(
                {
                    "type": "continuity",
                    "kind": "close_leadin",
                    "severity": "error",
                    "shift_id": buffer_shift.id,
                    "role": buffer_shift.role,
                    "employee_id": buffer_shift.employee_id,
                    "day": day,
                    "message": f"{buffer_shift.role} close buffer has no same-group lead-in shift.",
                }
            )

    return issues


def _coverage_issues(shifts: List[Shift], policy: Dict[str, Any]) -> List[Dict[str, Any]]:
    issues: List[Dict[str, Any]] = []
    anchors = anchor_rules(policy)
    opener_requirements = anchors.get("openers", {})
    closer_requirements = anchors.get("closers", {})
    opener_counts: Dict[int, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    closer_counts: Dict[int, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    def _is_close_buffer(shift: Shift) -> bool:
        if (shift.location or "").strip().lower() != "close":
            return False
        note = (shift.notes or "").lower()
        if "close buffer" in note:
            return True
        role_norm = normalize_role(shift.role)
        duration = shift.end - shift.start
        if duration > datetime.timedelta(hours=2):
            return False
        if role_norm == "hoh - expo":
            return True
        return "closer" in role_norm

    for shift in shifts:
        day_index = _shift_day_index(shift)
        group = role_group(shift.role or "")
        if (shift.location or "").strip().lower() == "open":
            opener_counts[day_index][group] += 1
        if _is_close_buffer(shift):
            closer_counts[day_index][group] += 1
    issues.extend(
        _compare_anchor_counts(
            opener_counts,
            opener_requirements,
            label="openers",
        )
    )
    issues.extend(
        _compare_anchor_counts(
            closer_counts,
            closer_requirements,
            label="closers",
        )
    )
    return issues


def _build_validation_checklist(
    shifts: List[Shift],
    *,
    issues: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Produce a concise, UI-friendly checklist:
    - `status`: ok|fail
    - `label`: human readable prompt
    - `details`: optional context for failures
    """
    checks: List[Dict[str, Any]] = []

    def summarize(items: List[Dict[str, Any]], *, limit: int = 5) -> str:
        parts: List[str] = []
        for entry in items[:limit]:
            message = str(entry.get("message") or "").strip()
            if message:
                parts.append(message)
                continue
            day = entry.get("day")
            group = entry.get("group") or entry.get("role")
            if day and group:
                parts.append(f"{day} {group}")
        if len(items) > limit:
            parts.append(f"+{len(items) - limit} more")
        return "; ".join(parts)

    def add_check(label: str, ok: bool, *, details: str = "") -> None:
        checks.append(
            {
                "label": label,
                "status": "ok" if ok else "fail",
                "details": details if not ok else "",
            }
        )

    def issues_of(type_name: str, *, kind: Optional[str] = None, anchor: Optional[str] = None) -> List[Dict[str, Any]]:
        matches: List[Dict[str, Any]] = []
        for issue in issues:
            if issue.get("type") != type_name:
                continue
            if kind and issue.get("kind") != kind:
                continue
            if anchor and issue.get("anchor") != anchor:
                continue
            matches.append(issue)
        return matches

    total_shifts = len(shifts)
    add_check("Shifts scheduled?", total_shifts > 0, details="No shifts found.")

    day_counts = {idx: 0 for idx in range(7)}
    for shift in shifts:
        day_counts[_shift_day_index(shift)] += 1
    empty_days = [WEEKDAY_TOKENS[idx] for idx in range(7) if day_counts.get(idx, 0) == 0]
    add_check(
        "Coverage every day?",
        not empty_days,
        details=f"Missing: {', '.join(empty_days)}",
    )

    unassigned = issues_of("assignment")
    add_check(
        "All shifts assigned?",
        not unassigned,
        details=summarize(unassigned),
    )

    hours_issues = issues_of("hours")
    add_check(
        "Time boundaries respected?",
        not hours_issues,
        details=summarize(hours_issues),
    )

    openers_missing = issues_of("coverage", anchor="openers")
    add_check(
        "Openers present?",
        not openers_missing,
        details=summarize(openers_missing),
    )

    closers_missing = issues_of("coverage", anchor="closers")
    add_check(
        "Closers present?",
        not closers_missing,
        details=summarize(closers_missing),
    )

    opener_continuity = issues_of("continuity", kind="open_followup")
    add_check(
        "Opener \u2192 AM continuity?",
        not opener_continuity,
        details=summarize(opener_continuity),
    )

    closer_continuity = issues_of("continuity", kind="close_leadin")
    add_check(
        "PM \u2192 closer continuity?",
        not closer_continuity,
        details=summarize(closer_continuity),
    )

    role_mismatch = issues_of("role_match")
    add_check(
        "Role matching correct?",
        not role_mismatch,
        details=summarize(role_mismatch),
    )

    availability = issues_of("availability")
    add_check(
        "Availability respected?",
        not availability,
        details=summarize(availability),
    )

    required_missing = issues_of("required_role")
    add_check(
        "Required roles fulfilled?",
        not required_missing,
        details=summarize(required_missing),
    )

    staffing = issues_of("staffing")
    add_check(
        "Staffing thresholds met?",
        not staffing,
        details=summarize(staffing),
    )

    hoh_combo = issues_of("hoh_combo")
    add_check(
        "Kitchen station logic ok?",
        not hoh_combo,
        details=summarize(hoh_combo),
    )

    fallback = issues_of("fallback")
    add_check(
        "Manager fallback ok?",
        not fallback,
        details=summarize(fallback),
    )

    return checks


def _weekly_hours_warnings(
    shifts: List[Shift], employees: Dict[int, Employee], policy: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Check if any employee exceeds the maximum weekly hours limit."""
    warnings: List[Dict[str, Any]] = []
    global_cfg = policy.get("global") or {}
    max_hours_per_week = float(global_cfg.get("max_hours_week", 40) or 40)
    
    # Calculate total hours per employee
    employee_hours: Dict[int, float] = defaultdict(float)
    for shift in shifts:
        if not shift.employee_id:
            continue
        hours = (shift.end - shift.start).total_seconds() / 3600
        employee_hours[shift.employee_id] += hours
    
    # Check each employee against the limit
    for employee_id, total_hours in employee_hours.items():
        if total_hours > max_hours_per_week + 1e-6:  # Small tolerance for floating point
            employee = employees.get(employee_id)
            employee_name = employee.full_name if employee else f"Employee {employee_id}"
            warnings.append(
                {
                    "type": "weekly_hours",
                    "severity": "warning",
                    "employee_id": employee_id,
                    "employee": employee_name,
                    "hours": round(total_hours, 2),
                    "limit": max_hours_per_week,
                    "message": f"{employee_name} is scheduled {round(total_hours, 2)} hours "
                    f"(exceeds {max_hours_per_week}-hour limit by {round(total_hours - max_hours_per_week, 2)} hours).",
                }
            )
    return warnings


def _concurrency_warnings(shifts: List[Shift]) -> List[Dict[str, Any]]:
    warnings: List[Dict[str, Any]] = []
    if not NORMALIZED_ROLE_LIMITS:
        return warnings
    day_windows: Dict[str, Dict[int, List[Tuple[datetime.datetime, datetime.datetime]]]] = {}
    for normalized_role, (label, _limit) in NORMALIZED_ROLE_LIMITS.items():
        day_windows[label] = defaultdict(list)
    for shift in shifts:
        normalized = normalize_role(shift.role)
        if normalized not in NORMALIZED_ROLE_LIMITS:
            continue
        label, _limit = NORMALIZED_ROLE_LIMITS[normalized]
        segments = _shift_segments(shift)
        for day_index, _day_start, seg_start, seg_end in segments:
            day_windows[label][day_index].append((seg_start, seg_end))
    for label, by_day in day_windows.items():
        limit = ROLE_CONCURRENCY_LIMITS[label]
        for day_index in range(7):
            windows = by_day.get(day_index, [])
            if not windows:
                continue
            overlap = _max_overlap(windows)
            if overlap > limit:
                warnings.append(
                    {
                        "type": "concurrency",
                        "severity": "warning",
                        "role": label,
                        "day": WEEKDAY_TOKENS[day_index],
                        "allowed": limit,
                        "actual": overlap,
                        "message": f"{label} scheduled {overlap} concurrent shifts on "
                        f"{WEEKDAY_TOKENS[day_index]} (limit {limit}).",
                    }
                )
    return warnings


def _open_close_issues(shifts: List[Shift], policy: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Reject shifts that start before open (unless opener) or end after close (unless closer)."""
    issues: List[Dict[str, Any]] = []
    for shift in shifts:
        loc = (shift.location or "").strip().lower()
        segments = _shift_segments(shift)
        if loc == "close":
            continue
        for day_index, day_start, seg_start, seg_end in segments:
            date_value = day_start.date()
            op_day_start = day_start
            op_date = date_value
            if seg_start.time() < datetime.time(6, 0):
                op_day_start = day_start - datetime.timedelta(days=1)
                op_date = op_day_start.date()
            open_min = open_minutes(policy, op_date)
            close_min = close_minutes(policy, op_date)
            open_dt = op_day_start + datetime.timedelta(minutes=open_min)
            close_dt = op_day_start + datetime.timedelta(minutes=close_min)
            if loc not in {"open", "close"} and seg_start < open_dt:
                issues.append(
                    {
                        "type": "hours",
                        "severity": "error",
                        "shift_id": shift.id,
                        "role": shift.role,
                        "day": WEEKDAY_TOKENS[day_index],
                        "message": f"{shift.role} shift starts before open ({seg_start.strftime('%H:%M')} < {open_dt.strftime('%H:%M')})",
                    }
                )
                break
            role_norm = normalize_role(shift.role)
            buffer_minutes = 0
            try:
                buffer_minutes = int((policy.get("global") or {}).get("close_buffer_minutes", 35) or 0)
            except Exception:  # noqa: BLE001
                buffer_minutes = 0
            close_limit = close_dt
            if "expo" in role_norm or "closer" in role_norm:
                close_limit = close_dt + datetime.timedelta(minutes=buffer_minutes)
            if loc != "close" and seg_end > close_limit:
                issues.append(
                    {
                        "type": "hours",
                        "severity": "error",
                        "shift_id": shift.id,
                        "role": shift.role,
                        "day": WEEKDAY_TOKENS[day_index],
                        "message": f"{shift.role} shift ends after close ({seg_end.strftime('%H:%M')} > {close_dt.strftime('%H:%M')})",
                    }
                )
                break
    return issues


def _arrival_window_warnings(shifts: List[Shift], policy: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Warn when a manual shift starts outside the observed AM/PM arrival windows.

    - Normal latest arrivals: 1.5 hours after period start
    - Hard latest arrivals: 2 hours after period start

    This is informational only; it does not block saves/publishing.
    """
    warnings: List[Dict[str, Any]] = []
    normal_delta = datetime.timedelta(minutes=ARRIVAL_WINDOW_NORMAL_MINUTES)
    hard_delta = datetime.timedelta(minutes=ARRIVAL_WINDOW_MAX_MINUTES)

    for shift in shifts:
        role_norm = normalize_role(shift.role)
        if not role_norm:
            continue
        if "bartender" in role_norm:
            continue
        if "close buffer" in (shift.notes or "").lower():
            continue
        loc = (shift.location or "").strip().lower()
        if loc == "open":
            continue

        start = _as_local_datetime(shift.start)
        op_day_start = datetime.datetime.combine(start.date(), datetime.time.min)
        op_date = op_day_start.date()
        # Close segments that start after midnight belong to the previous operational day.
        if loc == "close" and start.time() < datetime.time(6, 0):
            op_day_start = op_day_start - datetime.timedelta(days=1)
            op_date = op_day_start.date()

        open_dt = op_day_start + datetime.timedelta(minutes=open_minutes(policy, op_date))
        mid_dt = op_day_start + datetime.timedelta(minutes=mid_minutes(policy, op_date))
        if mid_dt <= open_dt:
            mid_dt = open_dt + datetime.timedelta(hours=5)
        close_dt = op_day_start + datetime.timedelta(minutes=close_minutes(policy, op_date))
        if close_dt <= open_dt:
            close_dt = close_dt + datetime.timedelta(days=1)

        period = None
        period_start = None
        if open_dt <= start < mid_dt:
            period = "AM"
            period_start = open_dt
        elif mid_dt <= start < close_dt:
            period = "PM"
            period_start = mid_dt
        else:
            continue

        normal_latest = period_start + normal_delta
        hard_latest = period_start + hard_delta
        if start <= normal_latest:
            continue

        day_label = WEEKDAY_TOKENS[op_date.weekday()]
        level = "normal"
        if start > hard_latest:
            level = "max"
            message = (
                f"{shift.role} starts at {start.strftime('%H:%M')} on {day_label}, which is after the "
                f"2h arrival cap ({hard_latest.strftime('%H:%M')}) for {period}."
            )
        else:
            message = (
                f"{shift.role} starts at {start.strftime('%H:%M')} on {day_label}, which is after the "
                f"normal 1.5h arrival window ({normal_latest.strftime('%H:%M')}) for {period}."
            )

        warnings.append(
            {
                "type": "arrival_window",
                "severity": "warning",
                "shift_id": shift.id,
                "role": shift.role,
                "day": day_label,
                "period": period,
                "level": level,
                "message": message,
            }
        )

    return warnings


def _demand_indices_for_week(session, week: WeekSchedule) -> Dict[int, float]:
    indices: Dict[int, float] = {}
    try:
        projections = get_week_daily_projections(session, week.context_id or week.id)
    except Exception:  # noqa: BLE001
        return indices
    sales_values = [float(proj.projected_sales_amount or 0.0) for proj in projections]
    # Absolute Rolla tiers to avoid misclassifying slow weeks as peak.
    def _tier_value(sales: float) -> float:
        if sales <= 5000:
            return 0.3
        if sales <= 9000:
            return 0.6
        if sales <= 12500:
            return 0.9
        return 1.1
    for proj in projections:
        sales = float(proj.projected_sales_amount or 0.0)
        indices[int(proj.day_of_week)] = _tier_value(sales)
    return indices


def _required_role_issues(shifts: List[Shift], policy: Dict[str, Any]) -> List[Dict[str, Any]]:
    issues: List[Dict[str, Any]] = []
    # Mirror the generator's "must exist" role labels for baseline validation.
    bww_required = {
        "Bartender - Opener",
        "Bartender - Closer",
        "Server - Opener",
        "Server - Dining Preclose",
        "Server - Dining Closer",
        "Server - Cocktail Preclose",
        "Server - Cocktail Closer",
        "HOH - Opener",
        "HOH - Expo",
        "HOH - Southwest",
        "HOH - Chip",
        "HOH - Shake",
        "HOH - Grill",
        "Cashier",
    }
    req_roles = {role for role in set(required_roles(policy)).union(bww_required) if normalize_role(role) != "hoh - closer"}
    if not req_roles:
        return issues

    def role_present(day_index: int, required_role: str) -> bool:
        required_norm = normalize_role(required_role)
        for shift in shifts:
            if _shift_day_index(shift) != day_index:
                continue
            role_norm = normalize_role(shift.role)
            if required_norm.startswith("hoh - "):
                if required_norm.endswith("southwest") and ("southwest" in role_norm or "sw/" in role_norm):
                    return True
                if required_norm.endswith("grill") and "grill" in role_norm:
                    return True
                if required_norm.endswith("chip") and "chip" in role_norm:
                    return True
                if required_norm.endswith("shake") and "shake" in role_norm:
                    return True
                if required_norm.endswith("expo") and required_norm == role_norm:
                    return True
                if required_norm.endswith("opener") and required_norm == role_norm:
                    return True
                continue
            if required_norm == role_norm:
                return True
        return False

    for day_index in range(7):
        for role in req_roles:
            if role_present(day_index, role):
                continue
            issues.append(
                {
                    "type": "required_role",
                    "severity": "error",
                    "day": WEEKDAY_TOKENS[day_index],
                    "role": role,
                    "message": f"Required role {role} missing on {WEEKDAY_TOKENS[day_index]}.",
                }
            )
    return issues


def _fallback_issues(shifts: List[Shift], pre_engine_cfg: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    errors: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []
    fallback_cfg = pre_engine_cfg.get("fallback", {})
    tag = str(fallback_cfg.get("tag", "MANAGER COVERING ΓÇö REVIEW REQUIRED")).lower()
    am_limit = int(fallback_cfg.get("am_limit", 1) or 0)
    pm_limit = int(fallback_cfg.get("pm_limit", 1) or 0)
    disallow_terms = [str(entry).lower() for entry in fallback_cfg.get("disallow_roles", [])]
    usage: Dict[Tuple[int, str], int] = defaultdict(int)
    for shift in shifts:
        note = (shift.notes or "").lower()
        if tag not in note:
            continue
        day_index = _shift_day_index(shift)
        period = "am" if shift.start.hour < 15 else "pm"
        usage[(day_index, period)] += 1
        role_norm = normalize_role(shift.role)
        if any(term in role_norm for term in disallow_terms) or "bartender" in role_norm or "expo" in role_norm:
            errors.append(
                {
                    "type": "fallback",
                    "severity": "error",
                    "day": WEEKDAY_TOKENS[day_index],
                    "role": shift.role,
                    "message": f"Manager fallback used for disallowed role {shift.role} on {WEEKDAY_TOKENS[day_index]}.",
                }
            )
        if "opener" in role_norm or "closer" in role_norm:
            errors.append(
                {
                    "type": "fallback",
                    "severity": "error",
                    "day": WEEKDAY_TOKENS[day_index],
                    "role": shift.role,
                    "message": f"Manager fallback cannot cover opener/closer roles ({shift.role}).",
                }
            )
    for (day_index, period), count in usage.items():
        limit = am_limit if period == "am" else pm_limit
        if count > limit:
            errors.append(
                {
                    "type": "fallback",
                    "severity": "error",
                    "day": WEEKDAY_TOKENS[day_index],
                    "message": f"Manager fallback used {count} times in {period.upper()} on {WEEKDAY_TOKENS[day_index]} (limit {limit}).",
                }
            )
    return errors, warnings


def _staffing_threshold_issues(
    shifts: List[Shift], pre_engine_cfg: Dict[str, Any], demand_indices: Dict[int, float]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    errors: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []
    staffing = pre_engine_cfg.get("staffing", {})
    server_cfg = staffing.get("servers", {})
    dining_cfg = server_cfg.get("dining", {})
    cocktail_cfg = server_cfg.get("cocktail", {})
    cashier_cfg = staffing.get("cashier", {})
    volume_thresholds = staffing.get("volume_thresholds", {})
    for day_index in range(7):
        demand_index = demand_indices.get(day_index, 1.0)
        tier = _volume_tier(demand_index, volume_thresholds)
        dining_count = _count_roles(shifts, day_index, lambda role: "server" in role and "cocktail" not in role and "patio" not in role)
        cocktail_count = _count_roles(shifts, day_index, lambda role: "cocktail" in role)
        cashier_count = _count_roles(shifts, day_index, lambda role: "cashier" in role or "host" in role)
        if tier == "slow" and dining_count > int(dining_cfg.get("slow_max", 4)):
            warnings.append(
                {
                    "type": "staffing",
                    "severity": "warning",
                    "day": WEEKDAY_TOKENS[day_index],
                    "message": f"Dining staffing {dining_count} exceeds slow cap ({dining_cfg.get('slow_max', 4)}).",
                }
            )
        if tier in {"moderate", "peak"} and dining_count < int(dining_cfg.get("slow_min", 1)):
            errors.append(
                {
                    "type": "staffing",
                    "severity": "error",
                    "day": WEEKDAY_TOKENS[day_index],
                    "message": "Dining staffing below minimum.",
                }
            )
        manual_dining = int(dining_cfg.get("manual_max", 7))
        if manual_dining and dining_count > manual_dining:
            warnings.append(
                {
                    "type": "staffing",
                    "severity": "warning",
                    "day": WEEKDAY_TOKENS[day_index],
                    "message": f"Dining staffing {dining_count} exceeds manual override threshold ({manual_dining}).",
                }
            )
        if tier == "peak" and dining_count < int(dining_cfg.get("peak", 6)):
            warnings.append(
                {
                    "type": "staffing",
                    "severity": "warning",
                    "day": WEEKDAY_TOKENS[day_index],
                    "message": f"Dining staffing below peak target ({dining_cfg.get('peak', 6)}).",
                }
            )
        if tier == "peak" and cocktail_count > int(cocktail_cfg.get("manual_max", 4)):
            warnings.append(
                {
                    "type": "staffing",
                    "severity": "warning",
                    "day": WEEKDAY_TOKENS[day_index],
                    "message": f"Cocktail staffing {cocktail_count} requires manual override (>{cocktail_cfg.get('manual_max', 4)}).",
                }
            )
        if tier == "slow" and cocktail_count < int(cocktail_cfg.get("normal", 2)):
            errors.append(
                {
                    "type": "staffing",
                    "severity": "error",
                    "day": WEEKDAY_TOKENS[day_index],
                    "message": f"Cocktail staffing below normal ({cocktail_cfg.get('normal', 2)}).",
                }
            )
        if cashier_count > int(cashier_cfg.get("manual_max", 4)):
            warnings.append(
                {
                    "type": "staffing",
                    "severity": "warning",
                    "day": WEEKDAY_TOKENS[day_index],
                    "message": f"Cashier staffing {cashier_count} exceeds manual override threshold ({cashier_cfg.get('manual_max', 4)}).",
                }
            )
        if cashier_count < int(cashier_cfg.get("am_default", 1)):
            errors.append(
                {
                    "type": "staffing",
                    "severity": "error",
                    "day": WEEKDAY_TOKENS[day_index],
                    "message": "Cashier coverage missing.",
                }
            )
    return errors, warnings


def _hoh_combo_issues(
    shifts: List[Shift], pre_engine_cfg: Dict[str, Any], demand_indices: Dict[int, float]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    errors: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []
    hoh_cfg = pre_engine_cfg.get("staffing", {}).get("hoh", {})
    thresholds = hoh_cfg.get("combo_thresholds", {})
    low_max = float(thresholds.get("low_max", 0.55) or 0.55)
    peak_min = float(thresholds.get("peak_min", 1.0) or 1.0)
    combo_roles = {normalize_role("HOH - Southwest & Grill"), normalize_role("HOH - Chip & Shake")}
    for day_index in range(7):
        demand_index = demand_indices.get(day_index, 1.0)
        combos_present = [
            shift for shift in shifts if normalize_role(shift.role) in combo_roles and _shift_day_index(shift) == day_index
        ]
        if demand_index >= peak_min and combos_present:
            errors.append(
                {
                    "type": "hoh_combo",
                    "severity": "error",
                    "day": WEEKDAY_TOKENS[day_index],
                    "message": "HOH combos scheduled during peak volume; split stations required.",
                }
            )
        if demand_index <= low_max and not combos_present:
            bo_roles = [
                shift
                for shift in shifts
                if _shift_day_index(shift) == day_index and any(term in normalize_role(shift.role) for term in ["grill", "southwest", "chip", "shake"])
            ]
            if len(bo_roles) >= 2:
                warnings.append(
                    {
                        "type": "hoh_combo",
                        "severity": "warning",
                        "day": WEEKDAY_TOKENS[day_index],
                        "message": "Low volume without HOH combo; consider merging Southwest/Grill or Chip/Shake.",
                    }
                )
    return errors, warnings


def _volume_tier(demand_index: float, thresholds: Dict[str, Any]) -> str:
    try:
        slow_max = float(thresholds.get("slow_max", 0.45))
        moderate_max = float(thresholds.get("moderate_max", 0.75))
    except Exception:  # noqa: BLE001
        slow_max, moderate_max = 0.45, 0.75
    if demand_index <= slow_max:
        return "slow"
    if demand_index <= moderate_max:
        return "moderate"
    return "peak"


def _compare_anchor_counts(
    counts: Dict[int, Dict[str, int]],
    requirements: Dict[str, int],
    *,
    label: str,
) -> List[Dict[str, Any]]:
    issues: List[Dict[str, Any]] = []
    for day_index in range(7):
        day_counts = counts.get(day_index, {})
        for group, required in requirements.items():
            actual = day_counts.get(group, 0)
            if actual < required:
                issues.append(
                    {
                        "type": "coverage",
                        "anchor": label,
                        "severity": "error",
                        "day": WEEKDAY_TOKENS[day_index],
                        "group": group,
                        "required": required,
                        "actual": actual,
                        "message": f"{group} requires {required} {label} on {WEEKDAY_TOKENS[day_index]}, "
                        f"but only {actual} were scheduled.",
                    }
                )
    return issues


def _shift_segments(shift: Shift) -> List[Tuple[int, datetime.datetime, datetime.datetime, datetime.datetime]]:
    start = _as_local_datetime(shift.start)
    end = _as_local_datetime(shift.end)
    if end < start:
        end = start
    segments: List[Tuple[int, datetime.datetime, datetime.datetime, datetime.datetime]] = []
    cursor = start
    while cursor < end:
        day_start = datetime.datetime.combine(cursor.date(), datetime.time.min)
        day_end = day_start + datetime.timedelta(days=1)
        segment_end = min(end, day_end)
        segments.append((cursor.weekday(), day_start, cursor, segment_end))
        cursor = segment_end
    return segments


def _shift_day_index(shift: Shift) -> int:
    try:
        return shift_display_date(shift.start, shift.location).weekday()
    except Exception:  # noqa: BLE001
        return 0


def _max_overlap(windows: List[Tuple[datetime.datetime, datetime.datetime]]) -> int:
    events: List[Tuple[datetime.datetime, int]] = []
    for start, end in windows:
        events.append((start, 1))
        events.append((end, -1))
    events.sort()
    running = 0
    best = 0
    for _, delta in events:
        running += delta
        if running > best:
            best = running
    return best


def _count_roles(shifts: List[Shift], day_index: int, predicate: Callable[[str], bool]) -> int:
    count = 0
    for shift in shifts:
        if _shift_day_index(shift) != day_index:
            continue
        role_label = (shift.role or "").lower()
        if "training" in role_label:
            continue
        if predicate(role_label):
            count += 1
    return count


def _as_local_datetime(value: datetime.datetime) -> datetime.datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=None)
    return value.astimezone(UTC).replace(tzinfo=None)


def _unavailability_windows(employee: Employee) -> Dict[int, List[Tuple[int, int]]]:
    mapping: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    for entry in employee.unavailability:
        window = (
            entry.start_time.hour * 60 + entry.start_time.minute,
            entry.end_time.hour * 60 + entry.end_time.minute,
        )
        mapping[entry.day_of_week].append(window)
    for day in mapping:
        mapping[day].sort()
    return mapping


def _matches_role(role_name: str, targets: Iterable[str]) -> bool:
    for target in targets:
        if role_matches(role_name, target):
            return True
    return False


def _format_minutes(value: int) -> str:
    hours = value // 60
    minutes = value % 60
    return f"{hours:02d}:{minutes:02d}"
