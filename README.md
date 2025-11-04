# Schedule Assistant — CS 4090 Capstone

Buffalo Wild Wings Scheduling Assistant built as a hybrid Python project combining FastAPI (backend-ready) and PySide6 (desktop UI). The app supports role-based access for General Managers (GM), Scheduling Managers (SM), and IT Assistants, with a focus on demand planning, modifiers, and policy management.

## Team

- Me
- Aaron Deken
- Jacob Lee

## Technologies

- Python 3.10+
- PySide6 (Qt for Python) for desktop UI
- SQLite + SQLAlchemy ORM for local storage
- FastAPI scaffolding (future API integration)
- Windows/macOS/Linux compatible

## Project Structure

```
Schedule-Assistant/
├─ app/
│  ├─ main.py                # PySide6 application and dialogs
│  ├─ database.py            # SQLAlchemy models and data helpers
│  ├─ data/                  # Local SQLite DB, logs, and state
│  └─ requirements.txt       # App Python dependencies
├─ launch.py                 # Cross‑platform launcher (creates venv, installs deps)
├─ launch.bat                # Windows helper to run the launcher
├─ README.md                 # Project documentation (this file)
└─ history.txt               # Project history and notable changes
```

Key Modules:

- `app/main.py`
  - Login and session management with timeouts
  - Demand Planning workspace: weekly projections + sales modifiers
  - Role-based dialogs: Account Manager (GM/IT), Employee Directory
  - New Policy Management dialog (CRUD for GM/IT; SM read-only when enabled)
- `app/database.py`
  - Models: Employee, EmployeeUnavailability, WeekContext, WeekDailyProjection, Modifier, Policy
  - Policy fields: `name`, `paramsJSON`, `lastEditedBy`, `lastEditedAt`
  - Helpers: `get_policies`, `upsert_policy`, `delete_policy`, `get_active_policy`

## Setup

Prerequisites:

- Python 3.10+ (3.11 recommended)
- Windows/macOS/Linux with GUI support

Steps:

1) Clone the repository

```bash
git clone <repo-url>
cd Schedule-Assistant
```

2) Launch using the helper (creates a venv, installs requirements, and starts the app)

```bash
python launch.py
```

On Windows you can also double-click `launch.bat`.

If running manually:

```bash
python -m venv .venv
. .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r app/requirements.txt
python -m app.main
```

## Default Login

On first launch (no existing users), an IT Assistant account is available to bootstrap:

- Username: `it_assistant`
- Password: `letmein`

Use the Account Manager dialog to add GM/SM accounts.

## Using the App

- Select a week via the week selector.
- Enter projected weekly sales per day; add Modifiers (e.g., events) to adjust demand windows.
- Manage Policies (GM/IT full access; SM optionally read-only) to define schedule rules.
- Employee Directory supports viewing and editing employee data and availability.

Role Access:

- GM/IT: Full access to all management dialogs, including Policies (create, edit, delete).
- SM: Read-only Policies when enabled; otherwise hidden/disabled.

## Development Notes

- Data and logs are stored in `app/data/`.
- The launcher will skip reinstalling dependencies if they haven’t changed.
- To reset the local database, delete `app/data/schedule.db` (dev only).

## Troubleshooting

- If dependencies change, rerun `python launch.py` or delete `.venv` to force a full reinstall.
- If GUI doesn’t start, ensure your OS has a display server and `PySide6` installed in the venv.

