# Schedule Assistant - CS 4090 Capstone

Schedule Assistant is a Buffalo Wild Wings planning tool built with PySide6. It helps the GM/SM team project sales, manage modifiers, define scheduling policies, and generate a week of coverage without touching raw JSON.

Latest feature work (2025-11-18): warning-gated SM edits in Policy Manager, policy-driven labor budget percentages (global + per group), richer seed data, stochastic scheduling, and live budget gauges on Week Schedule.

## Team
- Aaron Deken
- Jacob Lee

## What it does
- Role-aware login with session timeouts and audit logging.
- Week Preparation pulls projections, modifiers, and policy presets into one place before building the schedule.
- Week Schedule gives drag-friendly coverage editing plus swap/grant helpers, plus live labor budget gauges to keep spending in check and highlight projected % of sales consumed by labor.
- Validate / Import / Export moves week assets (employees, projections, modifiers, shifts) between files while tracking each transfer.
- Policy Composer lets GMs adjust global guardrails, time blocks, and per-role coverage without editing JSON.
- Policy settings UI captures per-day open/mid/close hours and clean role-group allocations, automatically driving AM/PM/Close blocks and labor budgets.
- Role Wage Manager keeps hourly rates confirmed per role; the scheduler won't run until wages, projected sales, and active employees are all in place.
- Account Manager and Employee Directory dialogs cover user onboarding and roster maintenance.
- Backup & Restore system automatically protects data on startup and allows manual backup/restore operations (IT/GM only).

## Tech stack
- Python 3.10+
- PySide6 (Qt for Python)
- SQLite + SQLAlchemy ORM
- FastAPI scaffold in place for future remote API work
- Designed around Windows 11, but runs anywhere Qt has GUI support

## Project layout
```
Schedule-Assistant/
  app/
    main.py           PySide6 application shell
    database.py       SQLAlchemy models and persistence helpers
    data/             Local SQLite DB, logs, cached week state
    requirements.txt  Runtime dependencies
    ...
  launch_app.py       Cross-platform launcher (creates .venv, installs deps)
  launch.bat          Windows shortcut to the launcher
  history.txt         Release notes
  TODO.txt            Working task list
```

## Getting started
### Quick start
1. Clone the repo and `cd Schedule-Assistant`.
2. Run `python launch_app.py` (or double-click `launch.bat` on Windows).

The launcher creates `.venv`, installs `app/requirements.txt`, and starts the UI.

### Manual setup
```
python -m venv .venv
.\.venv\Scripts\activate          # bash/zsh: source .venv/bin/activate
pip install -r app/requirements.txt
python -m app.main
```

## Default login
The first boot seeds an IT Assistant account so you can sign in and add real users:
- Username: `it_assistant`
- Password: `letmein`

Change that password immediately from the Account Manager dialog once you are in.

## Daily workflow
- **Week Preparation** - pick the ISO week, enter projections, and layer modifiers/events.
- **Week Schedule** - build a draft schedule, tweak coverage, and run swap/grant actions.
- **Validate / Import / Export** - run coverage checks, move data between weeks, or export PDFs/CSVs.
- **Policies** - GM/IT roles can edit the active policy, and SMs can edit after acknowledging a system-wide warning. Global labor % defaults to a realistic 27% of projected sales, with per-role group allocations layered on top.
- **Employee Directory / Account Manager** - maintain roster data and application accounts.

## Policy + automation
The Policy Composer exposes:
- Global guardrails (min rest, max hours, split shift toggle, overtime penalty).
- A global labor budget % (default 27% of projected sales) with a tolerance band managers can tighten or loosen.
- Anchor-aware time blocks that understand `@open` / `@close` plus buffer minutes.
- Role-group labor allocations, cut buffers, and cross-coverage rules so backup roles can fill last-resort gaps.
- Inline policy editor with import/export so IT/GM can adjust guardrails without touching raw JSON.
- Sequential workflow checks (wages, employees, projections) before generation so drafts stay grounded in real budgets.

The generator consumes active policy, projections, modifiers, and availability to build a week plan. It honors rest windows, desired-hour ranges, labor budgets, and role group distributions while falling back to configured cross-coverage roles when no dedicated staff remain. Each run randomizes within the constraints to surface multiple valid drafts.

## Sample data
Populate a starter roster and realistic availability:
```
python -m app.scripts.seed_employees
```
Run it once on a fresh database to get 65+ FOH/BOH records for demos and tests, with balanced coverage for Prep/Chip/Shake and Cashier/To-Go.

## Backup & Restore
The application includes automatic and manual backup capabilities:
- **Automatic**: Creates a backup on every startup (keeps 5 most recent)
- **Manual**: IT/GM users can create, restore, and manage backups via the Backup & Restore dialog
- **Location**: All backups stored in `app/backups/`
- See `docs/BACKUP_RESTORE.md` for detailed documentation

## Development notes & troubleshooting
- Data lives in `app/data/`. Delete `schedule.db` if you need a fresh environment (dev only).
- Backups live in `app/backups/`. The app creates automatic backups on startup.
- `python launch_app.py` reuses the last dependency hash; remove `.venv` if you want a clean reinstall.
- Smoke test the workflow:
  ```
  python app/scripts/workflow_smoke.py --week-start YYYY-MM-DD   # week arg optional
  ```
- Test backup functionality:
  ```
  python3 test_backup.py
  ```
- If UI assets fail to load, confirm the app is running inside the repo root so paths resolve correctly.

### Assignment note: low-cohesion schema (conceptual)
- The live app uses a high-cohesion, FK-backed schema (`app/database.py`).
- For the Lecture 19 low-cohesion assignment, see `docs/low_cohesion_schema.md` (concepts + ERD) and `docs/low_cohesion_schema.sql` (DDL sketch without foreign keys). These files are documentation-only and not wired into the running app.

### REST API (additive, optional)
- A lightweight FastAPI wrapper is available at `app/api.py`; it sits on the current schema (high cohesion) and exposes:
  - `GET /health`
  - `POST /api/v1/auth/login` (stubbed to the default demo user)
  - `GET /api/v1/weeks/{week}/summary`
  - `GET /api/v1/weeks/{week}/modifiers`
  - `GET /api/v1/weeks/{week}/shifts`
  - `POST /api/v1/weeks/{week}/projection`
  - `POST /api/v1/modifiers/apply-template`
  - `POST /api/v1/schedules/generate`
  - `POST /api/v1/schedules/{week}/publish`
  - `GET/PUT /api/v1/policy/active`
  - `POST /api/v1/employees/{id}/roles-wages`
- Run with: `uvicorn app.api:app --reload`

