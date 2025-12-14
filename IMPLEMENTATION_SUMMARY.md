# 40-Hour Weekly Limit Warning Implementation

## Overview
Added a warning/alert system that flags workers scheduled over the 40-hour per week limit during schedule validation.

## Changes Made

### 1. Modified `app/validation.py`

#### Added new validation function: `_weekly_hours_warnings()`
- **Location**: Lines 153-186
- **Purpose**: Checks if any employee exceeds the maximum weekly hours limit
- **Logic**:
  - Reads the `max_hours_week` setting from policy (defaults to 40)
  - Calculates total hours per employee from all shifts in the week
  - Flags employees exceeding the limit with a tolerance of 1e-6 for floating point precision
  - Returns warnings in the standard format

#### Warning Structure
Each warning includes:
- `type`: "weekly_hours" (for filtering)
- `severity`: "warning" (non-blocking)
- `employee_id`: ID of the affected employee
- `employee`: Employee name
- `hours`: Total scheduled hours (rounded to 2 decimals)
- `limit`: The policy's maximum weekly hours (default 40)
- `message`: Human-readable description with overage amount

#### Integration into validation flow
- Added call to `_weekly_hours_warnings()` in `validate_week_schedule()` (line 69)
- Warnings are appended to the existing warnings list
- Works alongside existing validation checks (availability, coverage, concurrency, etc.)

### 2. Added Test Coverage in `tests/test_validation.py`

Three new test cases:

1. **`test_warns_when_employee_exceeds_40_hours`** (lines 106-125)
   - Creates shifts totaling 42 hours
   - Verifies a warning is generated
   - Confirms warning has correct employee ID and hour values

2. **`test_no_warning_for_employee_at_40_hours`** (lines 127-144)
   - Creates shifts totaling exactly 40 hours
   - Verifies NO warning is generated (boundary case)

3. **`test_no_warning_for_employee_under_40_hours`** (lines 146-163)
   - Creates shifts totaling 30 hours
   - Verifies NO warning is generated

## How to Use

### Via API
When calling the validation endpoint, the response will now include warnings:
```
GET /api/v1/schedules/{week_start}/validate
```

Response includes:
```json
{
  "week_start": "2024-04-01",
  "week_id": 123,
  "issues": [...],
  "warnings": [
    {
      "type": "weekly_hours",
      "severity": "warning",
      "employee_id": 5,
      "employee": "John Doe",
      "hours": 42.5,
      "limit": 40,
      "message": "John Doe is scheduled 42.5 hours (exceeds 40-hour limit by 2.5 hours)."
    }
  ]
}
```

### Manual Schedule Validation
If validation is called directly in code:
```python
from validation import validate_week_schedule

report = validate_week_schedule(session, week_start, employee_session=employee_session)
hours_warnings = [w for w in report["warnings"] if w["type"] == "weekly_hours"]
```

## Policy Configuration
The maximum weekly hours limit is configured via the policy's global settings:
```json
{
  "global": {
    "max_hours_week": 48,
    ...
  }
}
```

Note: The default baseline policy uses 48 hours per week, not 40. This can be customized in the policy settings.
The validation function will use whatever limit is configured in the active policy.

## Key Features
- ✅ Respects policy-configured hour limits
- ✅ Floating-point safe comparison
- ✅ Clear, actionable warning messages
- ✅ Includes overage amount
- ✅ Integrates seamlessly with existing validation
- ✅ Non-blocking (warnings, not errors)
- ✅ Covers boundary cases (exactly at limit, over limit, under limit)
