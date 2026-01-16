from __future__ import annotations

import base64
import calendar
import copy
import datetime
import hashlib
import json
import secrets
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from sqlalchemy import select

from PySide6.QtCore import Qt, QDate, QTime, QEvent, QTimer
from PySide6.QtGui import QCloseEvent, QColor, QIcon, QIntValidator, QFont
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QAbstractItemView,
    QApplication,
    QCalendarWidget,
    QCheckBox,
    QComboBox,
    QDateEdit,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QScrollArea,
    QTabWidget,
    QToolButton,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTimeEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QHeaderView,
)

from database import (
    Employee,
    EmployeeUnavailability,
    Modifier,
    Policy,
    SavedModifier,
    SessionLocal,
    EmployeeSessionLocal,
    WeekContext,
    WeekDailyProjection,
    apply_saved_modifier_to_week,
    delete_saved_modifier,
    get_active_policy,
    get_all_employees,
    get_all_weeks,
    get_or_create_week_context,
    get_shifts_for_week,
    get_week_daily_projections,
    get_week_modifiers,
    get_week_summary,
    init_database,
    list_saved_modifiers,
    get_employee_role_wages,
    save_week_daily_projection_values,
    save_modifier_template,
    save_employee_role_wages,
    set_week_status,
    upsert_policy,
)

from exporter import DATA_DIR as EXPORT_DIR, export_week
from data_exchange import (
    copy_week_dataset,
    export_employees,
    export_week_modifiers,
    export_week_projections,
    export_week_schedule,
    get_weeks_summary,
    export_role_wages_dataset,
    import_role_wages_dataset,
    import_employees,
    import_week_modifiers,
    import_week_projections,
    import_week_schedule,
)
from policy import (
    CUT_PRIORITY_DEFAULT,
    DEFAULT_ENGINE_TUNING,
    build_default_policy,
    ensure_default_policy,
    pre_engine_settings,
    resolve_fallback_limits,
    resolve_hoh_thresholds,
    role_catalog,
)
from wages import (
    baseline_wages,
    export_wages as export_wages_file,
    import_wages as import_wages_file,
    load_wages,
    reset_wages_to_defaults,
    save_wages,
    validate_wages,
    ALLOW_ZERO_ROLES,
)
from roles import ROLE_GROUPS, role_group, normalize_role
from validation import validate_week_schedule
from ui.week_view import WeekSchedulePage
from ui.backup_dialog import BackupManagerDialog


DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
ICON_FILE = Path(__file__).resolve().parents[1] / "BWW_Scheduler.ico"
ACCOUNTS_FILE = DATA_DIR / "accounts.json"
AUDIT_FILE = DATA_DIR / "audit.log"
WEEK_STATE_FILE = DATA_DIR / "week_state.json"
LEGACY_PASSWORD_SALT = "schedule-assistant-salt"
PBKDF2_ITERATIONS = 200_000
MIN_PASSWORD_LENGTH = 8
DEFAULT_IT_PASSWORD = "letmein"
MAX_FAILED_ATTEMPTS = 5
LOCKOUT_MINUTES = 15
SESSION_WARNING_SECONDS = 9 * 60
SESSION_TIMEOUT_SECONDS = 10 * 60
SHOW_POLICY_TO_SM = True
ACCENT_COLOR = "#f5b942"
SUCCESS_COLOR = "#66d9a6"
WARNING_COLOR = "#f5b942"
INFO_COLOR = "#a8aec6"
ERROR_COLOR = "#ff7a7a"

WEEKDAY_LABELS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
ROLE_CATALOG = sorted({role for group in ROLE_GROUPS.values() for role in group})

def week_start_date(iso_year: int, iso_week: int) -> datetime.date:
    return datetime.date.fromisocalendar(iso_year, iso_week, 1)


def week_label(iso_year: int, iso_week: int) -> str:
    start = week_start_date(iso_year, iso_week)
    end = start + datetime.timedelta(days=6)
    start_str = start.strftime("%b %d")
    end_str = end.strftime("%b %d")
    if start.year != end.year:
        start_str = start.strftime("%b %d %Y")
        end_str = end.strftime("%b %d %Y")
    return f"{iso_year} W{iso_week:02d} ({start_str} - {end_str})"


def load_active_policy_spec(session_factory) -> Dict[str, Any]:
    # Helper for generator/validator integration points
    with session_factory() as session:
        policy = get_active_policy(session)
        return policy.params_dict() if policy else {}


def load_active_week(session_factory) -> Dict[str, Any]:
    iso_year = iso_week = None
    if WEEK_STATE_FILE.exists():
        try:
            data = json.loads(WEEK_STATE_FILE.read_text(encoding="utf-8"))
            iso_year = int(data.get("iso_year"))
            iso_week = int(data.get("iso_week"))
        except (ValueError, TypeError, json.JSONDecodeError):
            iso_year = iso_week = None
    if not iso_year or not iso_week:
        today = datetime.date.today()
        iso_year, iso_week, _ = today.isocalendar()
    label = week_label(iso_year, iso_week)
    with session_factory() as session:
        week = get_or_create_week_context(session, iso_year, iso_week, label)
        label = week.label
    save_active_week(iso_year, iso_week)
    return {"iso_year": iso_year, "iso_week": iso_week, "label": label}


def save_active_week(iso_year: int, iso_week: int) -> None:
    WEEK_STATE_FILE.write_text(
        json.dumps({"iso_year": iso_year, "iso_week": iso_week}),
        encoding="utf-8",
    )


def format_time_label(value: datetime.time) -> str:
    hour = value.hour % 12 or 12
    suffix = "AM" if value.hour < 12 else "PM"
    return f"{hour}:{value.minute:02d} {suffix}"
EMPLOYEE_ROLE_GROUPS = {
    # Note in open/close separation: open/close are paid differently than general servers so its essential the open/close roles are use separately for the time period they are applied to.
    "Bartenders": [
        "Bartender", # bartender logic is simple, usually there is two per day. one opener there until shift change at 4, and then a closer thats there until close. Only on extremer outliers like training or very busy nights are one or two added.
        "Bartender - Opener", # comes in 30 minutes before open, which is always at 11am
        "Bartender - Closer", # always works until close and is expected to take the policy set adjustable 30 minutes close time.
        "Bartender - Training", #set manually like all trainee shifts
    ],
    "Cashier": [
        # Cashiers responsibilities can be fulfilled when business is slow enough to allow other roles to assist. Primarily by remaining servers and a bartender.
        "Cashier", # Normally on slower days someone is assigned a general cashier role, expected to do To-Go, Host, 
        "Cashier - To-Go", # Cashier part 1, handles takeout orders/phone/cashier POS
        "Cashier - Host", # Cashier part 2, handles seating and cashier POS/phone
        "Cashier - Training", # training for cashier roles, trained in all roles assumedly
        "Cashier - All Roles", # trained and able to fulfill all sub roles as needed
    ],
    "Servers": [
        "Server - Training", # training for server roles, ideally all but we separate to allow for flexibility
        "Server - All Roles", # able to fulfill all subroles as needed
        "Server - Opener", # there is only one server opener, not specific to dining or cocktail
        "Server - Dining", # general server working in dining area can be 1st, 2nd, 3rd cut etc. prefer FIFO/LILO for cuts. Dining normally includes 4, but can include 5 on busier or 6 max busiest days
        "Server - Dining Preclose", # dining precloser, must be last to leave before dining closer, expected to come in second to last (before dining closer)
        "Server - Dining Closer", # to adhere to LILO, dining closer should come in last out of the servers in dining, and must work until resteraunt close. Is expected to take est 30 min to close (set in policy)
        "Server - Patio", # outlier, not dining or cocktail, seasonally scheduled
        "Server - Cocktail", # general server in cocktail section, on normal days theres 2 cut down to 1. On busier days 3, on busiest 4.
        "Server - Cocktail Preclose", # LILO, leaves before closer/close but comes in after all other cocktail employees (saving closer)
        "Server - Cocktail Closer", #LILO, in time is latest in cocktail and works until close and leaves after the expected policy set 30 min of closing time.
    ],
    "Kitchen": [
        "HOH - Opener", # comes in 30 minutes before open and works after opentime which is consistently 11am   
        "HOH - Closer", # works until close and is expected to take the policy set time to close which is currently 30 min i think
        "HOH - Training", # HOH training for role(s), hopefully listed in notes when trainee scheduled manually
        "HOH - All Roles", # can perform all duties required of each subrole in HOH/kitchen category set
        "HOH - Expo", # Works the window/always required and can fulfill other roles when they're absent/cut
        "HOH - Southwest & Grill", # Combination roles, when not as busy
        "HOH - Grill", # Often paired with southwest, unless busy.
        "HOH - Southwest", # Often paired with grill, unless busy.
        "HOH - Chip & Shake", # Combination role, when not as busy
        "HOH - Chip", # Often paired with shake, unless busy.
        "HOH - Shake", # Often paired with chip, unless busy.
        #"HOH - Prep", # Potential mistake, leaving out because its not always schedule and could just be a note
        #"HOH - Cook", # Mightve been a mistake, not an actual scheduled role
    ],
    "Management": [
        "Shift Lead", # absolute last resort applied manually, this is not a salaried MGR - FOH, but rather a normally high wage meant to cover a person working and fulfilling multiple roles/in an emergency at the expense of the budget. 
        "MGR - FOH", # Wage is $0, they are already factored into the budget via salary and do have a wage considered by the scheduler. Meant to be fallback to replace roles (usually in kitchen) as last resort or when not possible to be covered.
    ],
}
EMPLOYEE_ROLE_OPTIONS = [role for group in EMPLOYEE_ROLE_GROUPS.values() for role in group]
DAYS_OF_WEEK = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]
THEME_STYLESHEET = """
QWidget {
    background-color: #090a0e;
    color: #f5f6fa;
    font-family: 'Segoe UI', sans-serif;
    font-size: 15px;
}

QLabel {
    color: #f5f6fa;
}

QFrame, QGroupBox, QDialog, QMenu, QToolTip {
    background-color: #111217;
    border: 1px solid #1c1d23;
    border-radius: 12px;
}

QGroupBox {
    margin-top: 20px;
    padding: 20px;
}

QGroupBox::title {
    color: #f9d24a;
    font-weight: 600;
    subcontrol-origin: margin;
    subcontrol-position: top left;
    margin-left: 14px;
    padding: 2px 10px;
    background-color: #111217;
    border-radius: 8px;
}

QPushButton {
    background-color: #f5b942;
    color: #0b0b0f;
    border-radius: 10px;
    padding: 10px 22px;
    font-weight: 600;
    border: none;
    min-height: 34px;
}

QPushButton:hover {
    background-color: #ffd36a;
}

QPushButton:pressed {
    background-color: #e0a027;
}

QPushButton:disabled {
    background-color: #262730;
    color: #7d7f8f;
}

QLineEdit,
QComboBox,
QSpinBox,
QDateEdit,
QTimeEdit,
QPlainTextEdit {
    background-color: #15161c;
    border: 1px solid #25262d;
    border-radius: 10px;
    padding: 8px 14px;
    color: #f5f6fa;
    selection-background-color: #f5b942;
    selection-color: #0b0b0f;
}

QLineEdit::placeholder {
    color: #9ea2b2;
}

QComboBox QAbstractItemView::item {
    color: #d7d9e4;
}

QLineEdit:focus,
QComboBox:focus,
QSpinBox:focus,
QDateEdit:focus,
QTimeEdit:focus,
QPlainTextEdit:focus {
    border: 1px solid #f5b942;
}

QComboBox QAbstractItemView {
    background-color: #0e0f13;
    border: 1px solid #25262d;
    selection-background-color: #f5b942;
    selection-color: #0b0b0f;
    color: #f5f6fa;
}

QPlainTextEdit {
    padding: 12px;
}

QTabWidget::pane {
    border: 1px solid #1b1c22;
    border-radius: 10px;
    background: #070708;
}

QTabWidget::tab-bar {
    alignment: left;
}

QTabBar::tab {
    background: #0d0d11;
    color: #f5f6fa;
    padding: 8px 18px;
    margin-right: 2px;
    border: 1px solid #1b1c22;
    border-bottom: none;
    border-top-left-radius: 10px;
    border-top-right-radius: 10px;
}

QTabBar::tab:selected {
    background: #16171d;
    color: #f9d24a;
    border-color: #2b2c33;
}

QTabBar::tab:hover {
    background: #1a1b21;
}

QTableWidget,
QTableView {
    background-color: #14151c;
    alternate-background-color: #1b1c24;
    border: 1px solid #1c1d23;
    border-radius: 12px;
    gridline-color: #26272f;
    selection-background-color: #f5b942;
    selection-color: #0b0b0f;
}

QTableWidget::item,
QTableView::item {
    padding: 6px;
    border: none;
}

QTableWidget::item:selected,
QTableView::item:selected {
    background-color: #f5b942;
    color: #0b0b0f;
    border: 1px solid #fadd6b;
}

QHeaderView::section {
    background-color: #0d0e13;
    color: #f5f6fa;
    padding: 9px 14px;
    border: none;
    font-weight: 600;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
}

QTableCornerButton::section {
    background-color: #0d0e13;
    border: none;
}

QListWidget,
QListView,
QTreeView {
    background-color: #14151c;
    border: 1px solid #1c1d23;
    border-radius: 12px;
    selection-background-color: #f5b942;
    selection-color: #0b0b0f;
}

QListWidget::item,
QListView::item,
QTreeView::item {
    padding: 6px 10px;
    border-radius: 6px;
}

QListWidget::item:selected,
QListView::item:selected,
QTreeView::item:selected {
    background-color: #f5b942;
    color: #0b0b0f;
    border: 1px solid #fadd6b;
}

QScrollArea {
    border: none;
    background-color: transparent;
}

QScrollArea > QWidget > QWidget {
    background: transparent;
}

QScrollBar:vertical,
QScrollBar:horizontal {
    background: transparent;
    border: none;
    margin: 4px;
    width: 12px;
    height: 12px;
}

QScrollBar::handle {
    background: #2e2f37;
    border-radius: 6px;
}

QScrollBar::handle:hover {
    background: #3a3b45;
}

QScrollBar::add-line,
QScrollBar::sub-line,
QScrollBar::add-page,
QScrollBar::sub-page {
    background: transparent;
    border: none;
}
"""


def secure_hash_password(password: str, *, enforce_length: bool = True) -> tuple[str, str]:
    if enforce_length and len(password) < MIN_PASSWORD_LENGTH:
        raise ValueError(f"Password must be at least {MIN_PASSWORD_LENGTH} characters long.")
    salt = secrets.token_bytes(16)
    derived = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS)
    return (
        base64.b64encode(salt).decode("ascii"),
        base64.b64encode(derived).decode("ascii"),
    )


def verify_secure_password(password: str, salt_b64: str, hash_b64: str) -> bool:
    try:
        salt = base64.b64decode(salt_b64.encode("ascii"))
        stored = base64.b64decode(hash_b64.encode("ascii"))
    except (base64.binascii.Error, ValueError):
        return False
    derived = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS)
    return secrets.compare_digest(derived, stored)


def legacy_hash_password(password: str) -> str:
    salted = f"{LEGACY_PASSWORD_SALT}:{password}"
    return hashlib.sha256(salted.encode("utf-8")).hexdigest()


class AccountLockedError(Exception):
    """Raised when an account is locked and cannot authenticate."""

    def __init__(self, until: datetime.datetime) -> None:
        super().__init__("Account locked")
        self.until = until


class AuditLogger:
    """Append-only JSON line logger for security-relevant events."""

    def __init__(self, file_path: Path) -> None:
        self.file_path = file_path
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.file_path.exists():
            self.file_path.touch()

    def log(
        self,
        event: str,
        username: Optional[str],
        *,
        role: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        entry = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "event": event,
            "username": username,
        }
        if role:
            entry["role"] = role
        if details:
            entry["details"] = details

        with self.file_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry))
            handle.write("\n")


audit_logger = AuditLogger(AUDIT_FILE)


class AccountStore:
    """Simple JSON-backed account store with role-aware operations."""

    def __init__(self, file_path: Path) -> None:
        self.file_path = file_path
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_seed()

    def _ensure_seed(self) -> None:
        if self.file_path.exists():
            try:
                with self.file_path.open("r", encoding="utf-8") as handle:
                    data = json.load(handle)
            except json.JSONDecodeError:
                data = {}
        else:
            data = {}

        users = data.get("users") or []
        if users:
            return

        salt, password_hash = secure_hash_password(DEFAULT_IT_PASSWORD, enforce_length=False)
        data["users"] = [
            {
                "username": "it_assistant",
                "display_name": "IT Assistant",
                "role": "IT",
                "password_salt": salt,
                "password_hash": password_hash,
            }
        ]
        self._write(data)

    def _read(self) -> Dict[str, List[Dict[str, str]]]:
        if not self.file_path.exists():
            return {"users": []}
        with self.file_path.open("r", encoding="utf-8") as handle:
            try:
                data = json.load(handle)
            except json.JSONDecodeError:
                data = {}
        data.setdefault("users", [])
        return data

    def _write(self, data: Dict[str, List[Dict[str, str]]]) -> None:
        with self.file_path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)

    def _normalize_record(self, record: Dict[str, Any], *, persist: bool = False) -> Dict[str, Any]:
        updated = False
        if "failed_attempts" not in record:
            record["failed_attempts"] = 0
            updated = True
        if "locked_until" not in record:
            record["locked_until"] = None
            updated = True
        if persist and updated:
            data = self._read()
            index, _, data = self._find_user(record["username"], data)
            if index is not None:
                data["users"][index] = record
                self._write(data)
        return record

    def _locked_until(self, record: Dict[str, Any]) -> Optional[datetime.datetime]:
        locked_until = record.get("locked_until")
        if not locked_until:
            return None
        try:
            value = datetime.datetime.fromisoformat(locked_until)
            if value.tzinfo is None:
                value = value.replace(tzinfo=datetime.timezone.utc)
            return value
        except ValueError:
            return None

    def _find_user(
        self,
        username: str,
        data: Optional[Dict[str, List[Dict[str, str]]]] = None,
    ) -> tuple[Optional[int], Optional[Dict[str, str]], Dict[str, List[Dict[str, str]]]]:
        if data is None:
            data = self._read()
        username_lower = username.lower()
        for index, user in enumerate(data["users"]):
            if user["username"].lower() == username_lower:
                record = self._normalize_record(user)
                data["users"][index] = record
                return index, record, data
        return None, None, data

    def _password_matches(self, record: Dict[str, str], password: str) -> bool:
        password_hash = record.get("password_hash")
        if not password_hash:
            return False

        salt = record.get("password_salt")
        if salt:
            return verify_secure_password(password, salt, password_hash)
        return secrets.compare_digest(legacy_hash_password(password), password_hash)

    def get_user(self, username: str) -> Optional[Dict[str, str]]:
        _, record, _ = self._find_user(username)
        if not record:
            return None
        return {
            "username": record["username"],
            "role": record["role"],
            "display_name": record.get("display_name", record["username"]),
        }

    def list_users(self) -> List[Dict[str, Any]]:
        data = self._read()
        sanitized: List[Dict[str, Any]] = []
        for index, user in enumerate(data["users"]):
            record = self._normalize_record(user)
            data["users"][index] = record
            locked_until = self._locked_until(record)
            sanitized.append(
                {
                    "username": record["username"],
                    "role": record["role"],
                    "display_name": record.get("display_name", record["username"]),
                    "failed_attempts": record.get("failed_attempts", 0),
                    "locked_until": locked_until.isoformat() if locked_until else None,
                }
            )
        self._write(data)
        return sorted(sanitized, key=lambda entry: (entry["role"], entry["username"]))

    def create_user(
        self,
        creator_username: str,
        creator_role: str,
        username: str,
        password: str,
        role: str,
    ) -> None:
        username = username.strip()
        if not username or not password:
            raise ValueError("Username and password are required.")
        if role not in {"GM", "SM"}:
            raise ValueError("Only GM or SM accounts can be created.")
        if creator_role not in {"IT", "GM"}:
            raise PermissionError("You are not allowed to create accounts.")

        _, existing, data = self._find_user(username)
        if existing:
            raise ValueError("Username already exists.")

        salt, password_hash = secure_hash_password(password)
        new_user = {
            "username": username,
            "display_name": username.title(),
            "role": role,
            "password_salt": salt,
            "password_hash": password_hash,
            "failed_attempts": 0,
            "locked_until": None,
        }
        data["users"].append(new_user)
        self._write(data)
        audit_logger.log(
            "account_create",
            username,
            role=role,
            details={
                "created_by": creator_username,
                "created_by_role": creator_role,
            },
        )

    def delete_user(self, deleter_username: str, deleter_role: str, target_username: str) -> None:
        if deleter_role == "SM":
            raise PermissionError("Scheduling Managers cannot delete accounts.")
        index, target, data = self._find_user(target_username)

        if not target:
            raise ValueError("Account not found.")
        if target_username == deleter_username:
            raise PermissionError("You cannot delete your own account.")
        if target["role"] == "IT":
            if deleter_role != "IT":
                raise PermissionError("Only IT can delete IT accounts.")
        if target["role"] == "GM" and deleter_role == "GM":
            raise PermissionError("GMs cannot delete other GMs.")

        if index is not None:
            del data["users"][index]
            self._write(data)
            audit_logger.log(
                "account_delete",
                target_username,
                role=target["role"],
                details={
                    "deleted_by": deleter_username,
                    "deleted_by_role": deleter_role,
                },
            )

    def verify_credentials(self, username: str, password: str) -> Optional[Dict[str, str]]:
        data = self._read()
        index, record, data = self._find_user(username, data)
        if not record:
            return None

        now = datetime.datetime.now(datetime.timezone.utc)
        locked_until = self._locked_until(record)
        if locked_until and locked_until > now:
            raise AccountLockedError(locked_until)
        if locked_until and locked_until <= now:
            record["locked_until"] = None
            record["failed_attempts"] = 0

        if not self._password_matches(record, password):
            record["failed_attempts"] = record.get("failed_attempts", 0) + 1
            locked_time: Optional[datetime.datetime] = None
            if record["failed_attempts"] >= MAX_FAILED_ATTEMPTS:
                locked_time = now + datetime.timedelta(minutes=LOCKOUT_MINUTES)
                record["locked_until"] = locked_time.isoformat()
            data["users"][index] = record
            self._write(data)
            if locked_time:
                raise AccountLockedError(locked_time)
            return None

        if record.get("failed_attempts"):
            record["failed_attempts"] = 0
            record["locked_until"] = None
        if not record.get("password_salt"):
            salt, password_hash = secure_hash_password(password, enforce_length=False)
            record["password_salt"] = salt
            record["password_hash"] = password_hash
        data["users"][index] = record
        self._write(data)

        return {
            "username": record["username"],
            "role": record["role"],
            "display_name": record.get("display_name", record["username"]),
        }

    def change_password(self, username: str, current_password: str, new_password: str) -> None:
        data = self._read()
        index, record, data = self._find_user(username, data)
        if not record:
            raise ValueError("Account not found.")
        if not self._password_matches(record, current_password):
            raise PermissionError("Current password is incorrect.")

        salt, password_hash = secure_hash_password(new_password)
        record["password_salt"] = salt
        record["password_hash"] = password_hash
        record["failed_attempts"] = 0
        record["locked_until"] = None
        data["users"][index] = record
        self._write(data)
        audit_logger.log(
            "password_change",
            username,
            role=record.get("role"),
        )


class LoginDialog(QDialog):
    def __init__(self, store: AccountStore) -> None:
        super().__init__()
        self.store = store
        self.authenticated_user: Optional[Dict[str, str]] = None
        self.setWindowTitle("Schedule Assistant - Sign in")
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        heading = QLabel(f"<h2 style='color:{ACCENT_COLOR};'>Sign in to Schedule Assistant</h2>")
        subheading = QLabel("Access is limited to authorized staff.")
        subheading.setStyleSheet("color:#c9cede;")

        self.username_input = QLineEdit()
        self.username_input.setPlaceholderText("Username")
        self.password_input = QLineEdit()
        self.password_input.setPlaceholderText("Password")
        self.password_input.setEchoMode(QLineEdit.Password)

        form = QFormLayout()
        form.addRow("Username", self.username_input)
        form.addRow("Password", self.password_input)

        self.error_label = QLabel()
        self.error_label.setStyleSheet(f"color:{ERROR_COLOR};")
        self.error_label.setWordWrap(True)
        self.error_label.setVisible(False)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.attempt_login)
        button_box.rejected.connect(self.reject)

        layout.addWidget(heading)
        layout.addWidget(subheading)
        layout.addSpacing(10)
        layout.addLayout(form)
        layout.addWidget(self.error_label)
        layout.addWidget(button_box)

    def _set_error(self, message: str = "") -> None:
        self.error_label.setText(message)
        self.error_label.setVisible(bool(message.strip()))

    def attempt_login(self) -> None:
        username = self.username_input.text().strip()
        password = self.password_input.text()
        try:
            account = self.store.verify_credentials(username, password)
        except AccountLockedError as exc:
            self.password_input.clear()
            locked_local = exc.until.astimezone()
            message = locked_local.strftime("Account locked until %Y-%m-%d %H:%M %Z.")
            self._set_error(message)
            audit_logger.log(
                "login_failure",
                username,
                details={
                    "reason": "account_locked",
                    "locked_until": exc.until.isoformat(),
                },
            )
            return

        self.password_input.clear()

        if not account:
            self._set_error("Invalid username or password.")
            audit_logger.log(
                "login_failure",
                username,
                details={"reason": "invalid_credentials"},
            )
            return

        self._set_error("")
        self.username_input.setText(account["username"])
        audit_logger.log(
            "login_success",
            account["username"],
            role=account.get("role"),
        )
        self.authenticated_user = account
        self.accept()


class AccountManagerDialog(QDialog):
    def __init__(self, store: AccountStore, active_user: Dict[str, str]) -> None:
        super().__init__()
        self.store = store
        self.active_user = active_user
        self.setWindowTitle("Manage Accounts")
        self.resize(520, 360)
        self._build_ui()
        self.refresh_table()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        intro = QLabel(
            "<b>Provision new accounts or remove access.</b><br>"
            "IT and General Managers can create new GM or SM accounts."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Username", "Role", "Display name", "Status"])
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        layout.addWidget(self.table)

        form_container = QWidget()
        form_layout = QGridLayout(form_container)
        form_layout.addWidget(QLabel("Username"), 0, 0)
        self.new_username = QLineEdit()
        form_layout.addWidget(self.new_username, 0, 1)

        form_layout.addWidget(QLabel("Temporary password"), 1, 0)
        self.new_password = QLineEdit()
        self.new_password.setPlaceholderText(f"Provide starter password (min {MIN_PASSWORD_LENGTH} chars)")
        form_layout.addWidget(self.new_password, 1, 1)

        form_layout.addWidget(QLabel("Role"), 2, 0)
        self.new_role = QComboBox()
        self.new_role.addItems(["GM", "SM"])
        form_layout.addWidget(self.new_role, 2, 1)

        self.feedback_label = QLabel()
        self.feedback_label.setWordWrap(True)
        form_layout.addWidget(self.feedback_label, 3, 0, 1, 2)

        buttons_row = QHBoxLayout()
        self.create_button = QPushButton("Create account")
        self.create_button.clicked.connect(self.handle_create)
        buttons_row.addWidget(self.create_button)

        self.delete_button = QPushButton("Delete selected")
        self.delete_button.clicked.connect(self.handle_delete)
        buttons_row.addWidget(self.delete_button)
        buttons_row.addStretch()

        layout.addWidget(form_container)
        layout.addLayout(buttons_row)

        if self.active_user["role"] == "SM":
            self.create_button.setDisabled(True)
            self.delete_button.setDisabled(True)
            self.feedback_label.setText("Scheduling Managers cannot modify accounts.")

    def refresh_table(self) -> None:
        users = self.store.list_users()
        self.table.setRowCount(len(users))
        for row, user in enumerate(users):
            self.table.setItem(row, 0, QTableWidgetItem(user["username"]))
            self.table.setItem(row, 1, QTableWidgetItem(user["role"]))
            self.table.setItem(row, 2, QTableWidgetItem(user.get("display_name", "")))
            status_text = "Active"
            locked_until_str = user.get("locked_until")
            if locked_until_str:
                try:
                    lock_dt = datetime.datetime.fromisoformat(locked_until_str)
                    if lock_dt.tzinfo is None:
                        lock_dt = lock_dt.replace(tzinfo=datetime.timezone.utc)
                    local_time = lock_dt.astimezone()
                    status_text = f"Locked until {local_time.strftime('%Y-%m-%d %H:%M %Z')}"
                except ValueError:
                    status_text = "Locked"
            elif user.get("failed_attempts"):
                status_text = f"{user['failed_attempts']} failed attempts"
            self.table.setItem(row, 3, QTableWidgetItem(status_text))
        self.table.resizeColumnsToContents()
    def handle_create(self) -> None:
        if self.active_user["role"] not in {"IT", "GM"}:
            self.feedback_label.setText("You are not permitted to create accounts.")
            return

        username = self.new_username.text().strip()
        password = self.new_password.text()
        role = self.new_role.currentText()
        try:
            self.store.create_user(
                self.active_user["username"],
                self.active_user["role"],
                username,
                password,
                role,
            )
        except (ValueError, PermissionError) as exc:
            self.new_password.clear()
            self.feedback_label.setText(f"<span style='color:{ERROR_COLOR};'>{exc}</span>")
            return

        self.feedback_label.setText(f"<span style='color:{SUCCESS_COLOR};'>Created {role} account for {username}.</span>")
        self.new_username.clear()
        self.new_password.clear()
        self.refresh_table()

    def handle_delete(self) -> None:
        selected = self.table.selectedItems()
        if not selected:
            self.feedback_label.setText(f"<span style='color:{ERROR_COLOR};'>Select an account to delete.</span>")
            return
        username = selected[0].text()
        target_role = selected[1].text()

        confirm = QMessageBox.question(
            self,
            "Confirm deletion",
            f"Remove account '{username}' ({target_role})?",
        )
        if confirm != QMessageBox.Yes:
            return

        try:
            self.store.delete_user(
                deleter_username=self.active_user["username"],
                deleter_role=self.active_user["role"],
                target_username=username,
            )
        except (ValueError, PermissionError) as exc:
            self.feedback_label.setText(f"<span style='color:{ERROR_COLOR};'>{exc}</span>")
            return

        self.feedback_label.setText(f"<span style='color:{SUCCESS_COLOR};'>Deleted account '{username}'.</span>")
        self.refresh_table()


class WeekSelectorWidget(QWidget):
    def __init__(self, session_factory, active_week: Dict[str, Any], on_change) -> None:
        super().__init__()
        self.session_factory = session_factory
        self.on_change = on_change
        self.active_week = active_week
        self._updating = False
        self.week_start = week_start_date(active_week["iso_year"], active_week["iso_week"])
        self._build_ui()
        self.set_active_week(active_week)

    def _build_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        self.prev_button = QPushButton("◀")
        self.prev_button.setFixedSize(30, 30)
        self.prev_button.setStyleSheet("border-radius: 15px; padding: 0;")
        self.prev_button.clicked.connect(lambda: self._navigate(-7))
        layout.addWidget(self.prev_button)

        self.week_picker = QDateEdit()
        self.week_picker.setCalendarPopup(True)
        self.week_picker.setDisplayFormat("yyyy-MM-dd")
        self.week_picker.setMinimumWidth(120)
        self.week_picker.dateChanged.connect(self._handle_date_change)
        layout.addWidget(self.week_picker)

        self.next_button = QPushButton("▶")
        self.next_button.setFixedSize(30, 30)
        self.next_button.setStyleSheet("border-radius: 15px; padding: 0;")
        self.next_button.clicked.connect(lambda: self._navigate(7))
        layout.addWidget(self.next_button)

        self.week_label = QLabel("Week of --")
        layout.addWidget(self.week_label)
        layout.addStretch(1)

    def set_active_week(self, active_week: Dict[str, Any]) -> None:
        self.active_week = active_week
        self.week_start = week_start_date(active_week["iso_year"], active_week["iso_week"])
        self._updating = True
        qdate = QDate(self.week_start.year, self.week_start.month, self.week_start.day)
        self.week_picker.setDate(qdate)
        self.week_label.setText(f"Week of {self.week_start.isoformat()}")
        self._updating = False

    def _notify_change(self) -> None:
        iso_year, iso_week, _ = self.week_start.isocalendar()
        label = week_label(iso_year, iso_week)
        if self.on_change:
            self.on_change(iso_year, iso_week, label)

    def _navigate(self, delta_days: int) -> None:
        self.week_start = self.week_start + datetime.timedelta(days=delta_days)
        monday = self.week_start - datetime.timedelta(days=self.week_start.weekday())
        self.week_start = monday
        self.set_active_week(
            {"iso_year": self.week_start.isocalendar()[0], "iso_week": self.week_start.isocalendar()[1], "label": week_label(*self.week_start.isocalendar()[:2])}
        )
        self._notify_change()

    def _handle_date_change(self) -> None:
        if self._updating:
            return
        qdate = self.week_picker.date()
        new_date = datetime.date(qdate.year(), qdate.month(), qdate.day())
        monday = new_date - datetime.timedelta(days=new_date.weekday())
        self.week_start = monday
        self.set_active_week(
            {"iso_year": monday.isocalendar()[0], "iso_week": monday.isocalendar()[1], "label": week_label(*monday.isocalendar()[:2])}
        )
        self._notify_change()


class WeekPickerDialog(QDialog):
    def __init__(self, default_date: datetime.date, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Select week")
        self.resize(360, 360)
        layout = QVBoxLayout(self)
        hint = QLabel("Pick any date within the target week.")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        self.calendar = QCalendarWidget()
        self.calendar.setGridVisible(True)
        if default_date:
            self.calendar.setSelectedDate(QDate(default_date.year, default_date.month, default_date.day))
        layout.addWidget(self.calendar)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def selected_date(self) -> datetime.date:
        qdate = self.calendar.selectedDate()
        return datetime.date(qdate.year(), qdate.month(), qdate.day())


class ValidationImportExportPage(QWidget):
    def __init__(
        self,
        session_factory,
        employee_session_factory,
        user: Dict[str, Any],
        active_week: Dict[str, Any],
        *,
        on_week_changed: Optional[Callable[[int, int, str], None]] = None,
        on_status_updated: Optional[Callable[[str], None]] = None,
        on_week_data_changed: Optional[Callable[[], None]] = None,
    ) -> None:
        super().__init__()
        self.session_factory = session_factory
        self.employee_session_factory = employee_session_factory
        self.user = user
        self.active_week = active_week
        self.on_week_changed = on_week_changed
        self.on_status_updated = on_status_updated
        self.on_week_data_changed = on_week_data_changed
        self.summary_data: Dict[str, Any] = {}
        self.last_validation_report: Optional[Dict[str, Any]] = None
        self.last_validation_week_id: Optional[int] = None
        self.week_selector: Optional[WeekSelectorWidget] = None
        self._build_ui()
        self.set_active_week(active_week)

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        intro = QLabel("Validate coverage, confirm readiness, and export approved schedules.")
        intro.setWordWrap(True)
        layout.addWidget(intro)

        self.week_selector = WeekSelectorWidget(self.session_factory, self.active_week, self._handle_week_change)
        layout.addWidget(self.week_selector)

        summary_box = QGroupBox("Week snapshot")
        summary_layout = QFormLayout(summary_box)
        self.status_badge = QLabel("--")
        self.status_badge.setStyleSheet("font-weight:600;")
        self.shift_label = QLabel("0 shifts")
        self.cost_label = QLabel("$0.00")
        summary_layout.addRow("Status", self.status_badge)
        summary_layout.addRow("Scheduled shifts", self.shift_label)
        summary_layout.addRow("Projected labor", self.cost_label)
        layout.addWidget(summary_box)

        validation_box = QGroupBox("Validation & warnings")
        validation_layout = QVBoxLayout(validation_box)
        self.validation_list = QListWidget()
        self.validation_list.setAlternatingRowColors(True)
        validation_layout.addWidget(self.validation_list)
        layout.addWidget(validation_box)

        self.feedback_label = QLabel()
        self.feedback_label.setWordWrap(True)
        layout.addWidget(self.feedback_label)

        self.results_list = QListWidget()
        self.results_list.setAlternatingRowColors(True)
        layout.addWidget(self.results_list)

        exchange_box = QGroupBox("Import / Export")
        exchange_layout = QVBoxLayout(exchange_box)

        self.dataset_combo = QComboBox()
        self.dataset_combo.addItem("Employee directory", "employees")
        self.dataset_combo.addItem("Week projections", "projections")
        self.dataset_combo.addItem("Week modifiers", "modifiers")
        self.dataset_combo.addItem("Week schedule", "shifts")
        self.dataset_combo.addItem("Role wages", "wages")
        self.dataset_combo.currentIndexChanged.connect(self._update_button_states)

        dataset_form = QFormLayout()
        dataset_form.addRow("Dataset", self.dataset_combo)
        exchange_layout.addLayout(dataset_form)

        self.employee_import_mode_label = QLabel("Employee import mode")
        self.employee_import_mode = QComboBox()
        self.employee_import_mode.addItem("Replace (default)", "replace")
        self.employee_import_mode.addItem("Merge / append", "merge")
        self.employee_import_mode.setCurrentIndex(0)
        dataset_form.addRow(self.employee_import_mode_label, self.employee_import_mode)

        dataset_buttons = QHBoxLayout()
        dataset_buttons.setSpacing(10)
        self.export_dataset_button = QPushButton("Export file")
        self.export_dataset_button.clicked.connect(self._handle_dataset_export)
        dataset_buttons.addWidget(self.export_dataset_button)
        self.import_dataset_button = QPushButton("Import file")
        self.import_dataset_button.clicked.connect(self._handle_dataset_import)
        dataset_buttons.addWidget(self.import_dataset_button)
        dataset_buttons.addStretch()
        exchange_layout.addLayout(dataset_buttons)

        self.copy_week_button = QPushButton("Copy dataset from another week")
        self.copy_week_button.clicked.connect(self._handle_copy_from_week)
        exchange_layout.addWidget(self.copy_week_button)

        layout.addWidget(exchange_box)

        controls = QHBoxLayout()
        controls.setSpacing(10)
        self.validate_button = QPushButton("Run validation")
        self.validate_button.clicked.connect(self._run_validation)
        controls.addWidget(self.validate_button)

        self.export_format = QComboBox()
        self.export_format.addItem("PDF", "pdf")
        self.export_format.addItem("CSV", "csv")
        controls.addWidget(self.export_format)

        self.export_button = QPushButton("Generate export")
        self.export_button.clicked.connect(self._handle_export)
        controls.addWidget(self.export_button)

        self.mark_exported_button = QPushButton("Mark exported")
        self.mark_exported_button.clicked.connect(lambda: self._finalize_export(manual=True))
        controls.addWidget(self.mark_exported_button)

        controls.addStretch()
        self.backup_button = QPushButton("Backup/Restore")
        self.backup_button.clicked.connect(self._open_backup_manager)
        controls.addWidget(self.backup_button)
        layout.addLayout(controls)
        layout.addStretch(1)
        self._update_button_states()

    def set_active_week(self, active_week: Dict[str, Any]) -> None:
        if not active_week:
            return
        self.active_week = active_week
        if self.week_selector:
            self.week_selector.set_active_week(active_week)
        self._refresh_summary()
        self._refresh_validation_notes()

    def _current_week_start(self) -> Optional[datetime.date]:
        if not self.active_week:
            return None
        return week_start_date(self.active_week["iso_year"], self.active_week["iso_week"])

    def _refresh_summary(self) -> None:
        week_start = self._current_week_start()
        if not week_start:
            return
        with self.session_factory() as session:
            self.summary_data = get_week_summary(session, week_start)
        status = (self.summary_data.get("status") or "draft").strip().lower()
        self.status_badge.setText(status.title())
        badge_color = {
            "draft": WARNING_COLOR,
            "validated": SUCCESS_COLOR,
            "exported": ACCENT_COLOR,
        }.get(status, INFO_COLOR)
        self.status_badge.setStyleSheet(f"font-weight:600; color:{badge_color};")
        self.shift_label.setText(f"{self.summary_data.get('total_shifts', 0)} shifts")
        self.cost_label.setText(f"${self.summary_data.get('total_cost', 0.0):.2f}")
        self._update_button_states()

    def _update_button_states(self) -> None:
        status = (self.summary_data.get("status") or "draft").lower()
        has_week = bool(self.summary_data.get("week_id"))
        can_export = status in {"validated", "exported"} and has_week
        self.export_button.setEnabled(can_export)
        self.mark_exported_button.setEnabled(status == "validated" and has_week)
        dataset = self.dataset_combo.currentData() if hasattr(self, "dataset_combo") else None
        requires_week = self._dataset_requires_week(dataset)
        allow_dataset = has_week or not requires_week
        if hasattr(self, "export_dataset_button"):
            self.export_dataset_button.setEnabled(bool(dataset) and allow_dataset)
        if hasattr(self, "import_dataset_button"):
            self.import_dataset_button.setEnabled(bool(dataset) and allow_dataset)
        if hasattr(self, "copy_week_button"):
            self.copy_week_button.setEnabled(bool(dataset) and dataset != "employees" and has_week)
        if hasattr(self, "employee_import_mode"):
            is_employee = dataset == "employees"
            self.employee_import_mode_label.setVisible(bool(is_employee))
            self.employee_import_mode.setVisible(bool(is_employee))
            self.employee_import_mode.setEnabled(bool(is_employee))

    def _refresh_validation_notes(self) -> None:
        self.validation_list.clear()
        self.results_list.clear()
        if not self._current_week_start():
            return
        report = self._collect_validation_report()
        if report:
            self._render_validation_report(report)
        else:
            self.validation_list.addItem("Run validation to see checklist results.")

    def _active_policy_roles(self, policy: Dict[str, Any]) -> List[str]:
        roles: List[str] = []
        roles_cfg = policy.get("roles", {}) if isinstance(policy, dict) else {}
        if isinstance(roles_cfg, dict):
            for role, cfg in roles_cfg.items():
                if isinstance(cfg, dict) and cfg.get("enabled", True) is False:
                    continue
                roles.append(role)
        if not roles and isinstance(policy, dict):
            roles = list(role_catalog(policy))
        return sorted({role for role in roles if role})

    def _collect_validation_report(self) -> Optional[Dict[str, Any]]:
        week_start = self._current_week_start()
        if not week_start:
            return None
        with self.session_factory() as session:
            with self.employee_session_factory() as employee_session:
                report = validate_week_schedule(session, week_start, employee_session=employee_session)
        policy = load_active_policy_spec(self.session_factory)
        roles = self._active_policy_roles(policy)
        warnings = list(report.get("warnings") or [])
        issues = list(report.get("issues") or [])
        if roles:
            missing_wages = validate_wages(roles)
            if missing_wages:
                roles_list = ", ".join(sorted(missing_wages.keys()))
                warnings.append(
                    {
                        "type": "wages",
                        "severity": "warning",
                        "message": f"Wages missing/unchecked for: {roles_list}",
                    }
                )
        report = dict(report)
        report["warnings"] = warnings
        report["issues"] = issues
        self.last_validation_report = report
        self.last_validation_week_id = report.get("week_id")
        return report

    def _format_finding(self, entry: Dict[str, Any]) -> str:
        label = str(entry.get("message") or "").strip()
        if label:
            return label
        kind = entry.get("type") or "issue"
        day = entry.get("day")
        role = entry.get("role")
        if day and role:
            return f"{kind}: {role} ({day})"
        return str(entry)

    def _render_validation_report(self, report: Dict[str, Any]) -> None:
        self.validation_list.clear()
        self.results_list.clear()
        checks = list(report.get("checks") or [])
        if checks:
            for entry in checks:
                status = str(entry.get("status") or "fail").strip().lower()
                label = str(entry.get("label") or "Check").strip()
                details = str(entry.get("details") or "").strip()
                prefix = "PASS" if status == "ok" else "FAIL"
                text = f"{prefix}: {label}"
                if status != "ok" and details:
                    text = f"{text} — {details}"
                item = QListWidgetItem(text)
                item.setForeground(QColor(SUCCESS_COLOR if status == "ok" else ERROR_COLOR))
                if details:
                    item.setToolTip(details)
                self.validation_list.addItem(item)
        else:
            self.validation_list.addItem("Checklist not available for this week.")

        issues = list(report.get("issues") or [])
        warnings = list(report.get("warnings") or [])
        if not issues and not warnings:
            self.results_list.addItem("No issues or warnings found.")
            return
        for entry in issues:
            item = QListWidgetItem(f"FAIL [{entry.get('type', 'issue')}] {self._format_finding(entry)}")
            item.setForeground(QColor(ERROR_COLOR))
            self.results_list.addItem(item)
        for entry in warnings:
            item = QListWidgetItem(f"WARN [{entry.get('type', 'warning')}] {self._format_finding(entry)}")
            item.setForeground(QColor(WARNING_COLOR))
            self.results_list.addItem(item)

    def _current_validation_report(self) -> Optional[Dict[str, Any]]:
        week_id = self.summary_data.get("week_id")
        if self.last_validation_report and self.last_validation_week_id == week_id:
            return self.last_validation_report
        return self._collect_validation_report()

    def _handle_week_change(self, iso_year: int, iso_week: int, label: str) -> None:
        self.active_week = {"iso_year": iso_year, "iso_week": iso_week, "label": label}
        self._refresh_summary()
        self._refresh_validation_notes()
        if self.on_week_changed:
            self.on_week_changed(iso_year, iso_week, label)

    def _run_validation(self) -> None:
        self.results_list.clear()
        week_start = self._current_week_start()
        if not week_start:
            self._set_feedback("Select a week to validate.", WARNING_COLOR)
            return
        report = self._collect_validation_report()
        if not report:
            self._set_feedback("Validation failed to run.", ERROR_COLOR)
            return
        self._render_validation_report(report)
        issues = report.get("issues") or []
        warnings = report.get("warnings") or []
        if issues:
            self._set_feedback(f"Validation failed ({len(issues)} issue(s)).", ERROR_COLOR)
            self._apply_week_status("draft")
            return
        self._apply_week_status("validated")
        if warnings:
            self._set_feedback(f"Week validated with {len(warnings)} warning(s).", WARNING_COLOR)
        else:
            self._set_feedback("All checks passed. Week marked as validated.", SUCCESS_COLOR)
        audit_logger.log(
            "week_validated",
            self.user.get("username", "unknown"),
            role=self.user.get("role"),
            details={
                "iso_year": self.active_week.get("iso_year"),
                "iso_week": self.active_week.get("iso_week"),
            },
        )

    def _handle_export(self) -> None:
        status = (self.summary_data.get("status") or "draft").lower()
        if status not in {"validated", "exported"}:
            self._set_feedback("Run validation before exporting the schedule.", WARNING_COLOR)
            return
        week_id = self.summary_data.get("week_id")
        if not week_id:
            self._set_feedback("No schedule data found for the selected week.", ERROR_COLOR)
            return
        report = self._current_validation_report()
        if report:
            issues = report.get("issues") or []
            warnings = report.get("warnings") or []
            if issues:
                self._set_feedback("Validation errors found. Resolve them before exporting.", ERROR_COLOR)
                self._apply_week_status("draft")
                return
            if warnings:
                warning_lines = "\n".join(f"- {self._format_finding(entry)}" for entry in warnings[:8])
                confirm = QMessageBox.question(
                    self,
                    "Export with warnings",
                    "Warnings were found during validation:\n\n"
                    f"{warning_lines}\n\nExport anyway?",
                )
                if confirm != QMessageBox.Yes:
                    return
        fmt = self.export_format.currentData() or "pdf"
        export_path = export_week(week_id, fmt)
        self._finalize_export(manual=False)
        audit_logger.log(
            "week_export",
            self.user.get("username", "unknown"),
            role=self.user.get("role"),
            details={
                "week_id": week_id,
                "format": fmt,
                "path": str(export_path),
            },
        )
        QMessageBox.information(
            self,
            "Export complete",
            f"Schedule exported to {export_path}",
        )
        self.results_list.addItem(f"Exported validated week to {export_path}")

    def _handle_dataset_export(self) -> None:
        dataset = self.dataset_combo.currentData()
        if not dataset:
            return
        try:
            path = self._export_dataset(dataset)
        except Exception as exc:
            self._set_feedback(f"Export failed: {exc}", ERROR_COLOR)
            return
        if not path:
            return
        self._set_feedback(f"Saved {dataset} export to {path}", SUCCESS_COLOR)
        self.results_list.addItem(f"Exported {dataset} -> {path}")
        audit_logger.log(
            "data_export",
            self.user.get("username", "unknown"),
            role=self.user.get("role"),
            details={"dataset": dataset, "path": str(path)},
        )

    def _handle_dataset_import(self) -> None:
        dataset = self.dataset_combo.currentData()
        if not dataset:
            return
        if self._dataset_requires_week(dataset) and not self._current_week_start():
            self._set_feedback("Select a week before importing this dataset.", WARNING_COLOR)
            return
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select data file",
            str(EXPORT_DIR),
            "JSON Files (*.json);;All Files (*)",
        )
        if not file_path:
            return
        path = Path(file_path)
        try:
            summary = self._import_dataset(dataset, path)
        except Exception as exc:
            self._set_feedback(f"Import failed: {exc}", ERROR_COLOR)
            return
        recap = ", ".join(f"{k}={v}" for k, v in summary.items())
        self._set_feedback(f"Imported {dataset} ({recap}).", SUCCESS_COLOR)
        self.results_list.addItem(f"Imported {dataset} from {path} ({recap})")
        if dataset != "employees":
            self._notify_week_mutation()
        audit_logger.log(
            "data_import",
            self.user.get("username", "unknown"),
            role=self.user.get("role"),
            details={"dataset": dataset, "path": str(path), "summary": summary},
        )

    def _open_backup_manager(self) -> None:
        dialog = BackupManagerDialog(self)
        dialog.setStyleSheet(THEME_STYLESHEET)
        dialog.exec()

    def _handle_copy_from_week(self) -> None:
        dataset = self.dataset_combo.currentData()
        if not dataset or dataset == "employees":
            self._set_feedback("Copying is available for week-based datasets only.", INFO_COLOR)
            return
        if not self._current_week_start():
            self._set_feedback("Select a destination week first.", WARNING_COLOR)
            return
        with self.session_factory() as session:
            weeks = [entry for entry in get_weeks_summary(session)]
            current_label = self.active_week.get("label") if self.active_week else None
            options = [entry["label"] for entry in weeks if entry["label"] != current_label]
            if not options:
                self._set_feedback("No other weeks are available to copy from.", INFO_COLOR)
                return
            selection, ok = QInputDialog.getItem(
                self,
                "Copy data",
                "Copy from week",
                options,
                editable=False,
            )
            if not ok or not selection:
                return
            source_meta = next((entry for entry in weeks if entry["label"] == selection), None)
            if not source_meta:
                return
            source_week = session.get(WeekContext, source_meta["id"])
            target_week = self._get_week_context(session)
            if not source_week or not target_week:
                self._set_feedback("Unable to resolve the requested weeks.", ERROR_COLOR)
                return
            employee_session = None
            try:
                if dataset == "shifts":
                    employee_session = self.employee_session_factory()
                summary = copy_week_dataset(
                    session,
                    source_week,
                    target_week,
                    dataset,
                    actor=self.user.get("username", "unknown"),
                    employee_session=employee_session,
                )
            finally:
                if employee_session:
                    employee_session.close()
        recap = ", ".join(f"{k}={v}" for k, v in summary.items())
        self._set_feedback(f"Copied {dataset} from {selection} ({recap}).", SUCCESS_COLOR)
        self.results_list.addItem(f"Copied {dataset} from {selection} -> {self.active_week.get('label')}")
        self._notify_week_mutation()
        audit_logger.log(
            "week_copy",
            self.user.get("username", "unknown"),
            role=self.user.get("role"),
            details={
                "dataset": dataset,
                "source_week": selection,
                "target_week": self.active_week.get("label") if self.active_week else None,
                "summary": summary,
            },
        )

    def _finalize_export(self, manual: bool) -> None:
        self._apply_week_status("exported")
        if manual:
            audit_logger.log(
                "week_mark_exported",
                self.user.get("username", "unknown"),
                role=self.user.get("role"),
                details={
                    "iso_year": self.active_week.get("iso_year"),
                    "iso_week": self.active_week.get("iso_week"),
                },
            )
        self._set_feedback("Week marked as exported.", ACCENT_COLOR)

    def _apply_week_status(self, status: str) -> None:
        week_start = self._current_week_start()
        if not week_start:
            return
        with self.session_factory() as session:
            set_week_status(session, week_start, status)
        self._refresh_summary()
        if self.on_status_updated:
            self.on_status_updated(status)

    def _format_shift_label(self, start: Optional[datetime.datetime]) -> str:
        if not isinstance(start, datetime.datetime):
            return "an unknown time"
        localized = start.astimezone()
        return localized.strftime("%a %m/%d %I:%M %p")

    def _set_feedback(self, message: str, color: str = INFO_COLOR) -> None:
        self.feedback_label.setStyleSheet(f"color:{color};")
        self.feedback_label.setText(message)
        self.feedback_label.repaint()

    def _export_dataset(self, dataset: str) -> Optional[Path]:
        if dataset == "wages":
            return export_role_wages_dataset()
        if dataset == "employees":
            with self.employee_session_factory() as employee_session:
                return export_employees(employee_session)
        with self.session_factory() as session:
            week = self._get_week_context(session)
            if not week:
                raise ValueError("Select a week first.")
            if dataset == "projections":
                return export_week_projections(session, week)
            if dataset == "modifiers":
                return export_week_modifiers(session, week)
            if dataset == "shifts":
                week_start = self._current_week_start()
                if not week_start:
                    raise ValueError("Select a week first.")
                with self.employee_session_factory() as employee_session:
                    return export_week_schedule(session, week_start, employee_session=employee_session)
        return None

    def _import_dataset(self, dataset: str, path: Path) -> Dict[str, int]:
        if dataset == "wages":
            count = import_role_wages_dataset(path)
            return {"roles": count}
        if dataset == "employees":
            with self.employee_session_factory() as employee_session:
                mode = "replace"
                if hasattr(self, "employee_import_mode"):
                    mode = self.employee_import_mode.currentData() or "replace"
                replace_existing = mode != "merge"
                created, updated = import_employees(employee_session, path, replace_existing=replace_existing)
                return {"created": created, "updated": updated}
        with self.session_factory() as session:
            week = self._get_week_context(session)
            if not week:
                raise ValueError("Select a week first.")
            if dataset == "projections":
                count = import_week_projections(session, week, path)
                return {"projections": count}
            if dataset == "modifiers":
                count = import_week_modifiers(session, week, path, created_by=self.user.get("username", "unknown"))
                return {"modifiers": count}
            if dataset == "shifts":
                week_start = self._current_week_start()
                if not week_start:
                    raise ValueError("Select a week first.")
                with self.employee_session_factory() as employee_session:
                    count = import_week_schedule(session, week_start, path, employee_session=employee_session)
                return {"shifts": count}
        raise ValueError(f"Unsupported dataset '{dataset}'")

    def _dataset_requires_week(self, dataset: Optional[str]) -> bool:
        return dataset in {"projections", "modifiers", "shifts"}

    def _get_week_context(self, session):
        if not self.active_week:
            return None
        iso_year = self.active_week.get("iso_year")
        iso_week = self.active_week.get("iso_week")
        label = self.active_week.get("label", f"{iso_year} W{iso_week}")
        if iso_year is None or iso_week is None:
            return None
        return get_or_create_week_context(session, iso_year, iso_week, label)

    def _notify_week_mutation(self) -> None:
        self._refresh_summary()
        self._refresh_validation_notes()
        if self.on_status_updated:
            self.on_status_updated("draft")
        if self.on_week_data_changed:
            self.on_week_data_changed()


class ModifierDialog(QDialog):
    TIME_CHOICES = [datetime.time(hour % 24, 0) for hour in list(range(2, 24)) + [0, 1]]

    def __init__(
        self,
        existing_modifiers: List[Modifier],
        *,
        modifier: Optional[Modifier] = None,
    ) -> None:
        super().__init__()
        self.existing_modifiers = existing_modifiers
        self.edit_modifier = modifier
        self.result_data: Optional[Dict[str, Any]] = None
        self.setWindowTitle("Edit modifier" if modifier else "Add modifier")
        self._build_ui()
        if modifier:
            self._load_modifier(modifier)

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        intro = QLabel(
            "Adjust projected sales for a specific day and window. "
            "Modifiers apply on top of the base projection."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        form = QFormLayout()

        self.title_input = QLineEdit()
        self.title_input.setPlaceholderText("e.g. Game Day Surge")
        form.addRow("Title", self.title_input)

        self.day_combo = QComboBox()
        self.day_combo.addItems(DAYS_OF_WEEK)
        form.addRow("Day of week", self.day_combo)

        time_row = QHBoxLayout()
        self.start_time_combo = self._build_time_combo()
        self._set_combo_to_time(self.start_time_combo, datetime.time(16, 0))
        time_row.addWidget(self.start_time_combo)
        time_row.addWidget(QLabel("to"))
        self.end_time_combo = self._build_time_combo()
        self._set_combo_to_time(self.end_time_combo, datetime.time(21, 0))
        time_row.addWidget(self.end_time_combo)
        form.addRow("Time window", time_row)

        impact_row = QHBoxLayout()
        self.sign_combo = QComboBox()
        self.sign_combo.addItem("Increase (+)", 1)
        self.sign_combo.addItem("Decrease (-)", -1)
        impact_row.addWidget(self.sign_combo)
        self.pct_input = QSpinBox()
        self.pct_input.setRange(1, 400)
        self.pct_input.setSuffix(" %")
        self.pct_input.setValue(10)
        impact_row.addWidget(self.pct_input)
        impact_row.addStretch()
        form.addRow("Percent change", impact_row)

        self.notes_input = QLineEdit()
        self.notes_input.setPlaceholderText("Optional note")
        form.addRow("Notes", self.notes_input)

        layout.addLayout(form)

        self.save_checkbox = QCheckBox("Save this modifier for future weeks")
        layout.addWidget(self.save_checkbox)

        self.feedback_label = QLabel()
        self.feedback_label.setStyleSheet(f"color:{ERROR_COLOR};")
        layout.addWidget(self.feedback_label)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _load_modifier(self, modifier: Modifier) -> None:
        self.title_input.setText(modifier.title)
        self.day_combo.setCurrentIndex(modifier.day_of_week)
        self._set_combo_to_time(self.start_time_combo, modifier.start_time)
        self._set_combo_to_time(self.end_time_combo, modifier.end_time)
        self.sign_combo.setCurrentIndex(0 if modifier.pct_change >= 0 else 1)
        self.pct_input.setValue(abs(modifier.pct_change))
        self.notes_input.setText(modifier.notes or "")
        self.save_checkbox.setChecked(False)

    def _build_time_combo(self) -> QComboBox:
        combo = QComboBox()
        for value in self.TIME_CHOICES:
            combo.addItem(format_time_label(value), value)
        return combo

    @staticmethod
    def _set_combo_to_time(combo: QComboBox, value: datetime.time) -> None:
        index = combo.findData(value)
        if index != -1:
            combo.setCurrentIndex(index)

    def _windows_overlap(
        self,
        day: int,
        start: datetime.time,
        end: datetime.time,
        ignore_id: Optional[int],
    ) -> Optional[Modifier]:
        def normalize(window_start: datetime.time, window_end: datetime.time) -> tuple[int, int]:
            start_minutes = window_start.hour * 60 + window_start.minute
            end_minutes = window_end.hour * 60 + window_end.minute
            if end_minutes <= start_minutes:
                end_minutes += 24 * 60
            return start_minutes, end_minutes

        start_minutes, end_minutes = normalize(start, end)
        for existing in self.existing_modifiers:
            if ignore_id and existing.id == ignore_id:
                continue
            if existing.day_of_week != day:
                continue
            existing_start, existing_end = normalize(existing.start_time, existing.end_time)
            for offset in (0, 24 * 60, -24 * 60):
                shifted_start = existing_start + offset
                shifted_end = existing_end + offset
                if start_minutes < shifted_end and end_minutes > shifted_start:
                    return existing
        return None

    def accept(self) -> None:  # type: ignore[override]
        title = self.title_input.text().strip()
        if not title:
            self.feedback_label.setText("Provide a title so the team knows why this modifier exists.")
            self.title_input.setFocus()
            return

        start_value: Optional[datetime.time] = self.start_time_combo.currentData()
        end_value: Optional[datetime.time] = self.end_time_combo.currentData()
        if start_value is None or end_value is None:
            self.feedback_label.setText("Select both start and end times.")
            return

        sign = self.sign_combo.currentData()
        if sign not in (-1, 1):
            sign = 1
        pct_change = int(self.pct_input.value() * sign)

        day_index = self.day_combo.currentIndex()
        ignore_id = self.edit_modifier.id if self.edit_modifier else None
        overlap = self._windows_overlap(day_index, start_value, end_value, ignore_id)
        if overlap:
            self.feedback_label.setText(
                f"Overlaps with '{overlap.title}' ({DAYS_OF_WEEK[overlap.day_of_week]} "
                f"{overlap.start_time.strftime('%H:%M')}–{overlap.end_time.strftime('%H:%M')})."
            )
            return

        self.result_data = {
            "title": title,
            "day_of_week": day_index,
            "start_time": start_value,
            "end_time": end_value,
            "pct_change": pct_change,
            "notes": self.notes_input.text().strip(),
            "save_for_later": self.save_checkbox.isChecked(),
        }
        super().accept()


class DemandPlanningWidget(QWidget):
    def __init__(self, session_factory, actor: Dict[str, Any], active_week: Dict[str, Any]) -> None:
        super().__init__()
        self.session_factory = session_factory
        self.actor = actor
        self.active_week = active_week
        self.week_id: Optional[int] = None
        self.week_label: str = active_week.get("label", "")
        self.projections: List[WeekDailyProjection] = []
        self.modifiers: List[Modifier] = []
        self.saved_modifiers: List[SavedModifier] = []
        self.day_inputs: Dict[int, QLineEdit] = {}
        self.day_note_inputs: Dict[int, QLineEdit] = {}
        self.heat_labels: Dict[int, QLabel] = {}
        self._pending_changes = False
        self._modifier_column_ratios: Dict[int, float] = {
            0: 0.28,  # Title
            1: 0.12,  # Impact
            2: 0.12,  # Day
            3: 0.18,  # Window
            4: 0.10,  # % Change
            5: 0.14,  # Applied by
            6: 0.14,  # Notes
        }
        self._build_ui()
        self.refresh()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(18)

        self.projection_group = QGroupBox("Projected sales by day")
        projection_layout = QVBoxLayout(self.projection_group)

        grid = QGridLayout()
        grid.addWidget(QLabel("<b>Day</b>"), 0, 0)
        grid.addWidget(QLabel("<b>Projected sales ($)</b>"), 0, 1)
        grid.addWidget(QLabel("<b>Notes</b>"), 0, 2)

        for day_index, day_name in enumerate(DAYS_OF_WEEK):
            label = QLabel(day_name)
            grid.addWidget(label, day_index + 1, 0)

            amount_input = QLineEdit()
            amount_input.setPlaceholderText("Projected sales")
            amount_input.setAlignment(Qt.AlignRight)
            amount_input.setValidator(QIntValidator(0, 9_999_999, amount_input))
            amount_input.setMaxLength(9)
            amount_input.setClearButtonEnabled(True)
            amount_input.textEdited.connect(self._handle_projection_field_edited)
            grid.addWidget(amount_input, day_index + 1, 1)
            self.day_inputs[day_index] = amount_input

            note_input = QLineEdit()
            note_input.setPlaceholderText("Optional context")
            note_input.textEdited.connect(self._handle_projection_field_edited)
            grid.addWidget(note_input, day_index + 1, 2)
            self.day_note_inputs[day_index] = note_input

        projection_layout.addLayout(grid)

        status_row = QHBoxLayout()
        self.completion_badge = QLabel()
        self.completion_badge.setObjectName("completionBadge")
        self.completion_badge.setStyleSheet(
            "padding:4px 10px; border-radius:8px; background-color:#141722; font-weight:600;"
        )
        status_row.addWidget(self.completion_badge)
        status_row.addStretch()
        self.save_status_label = QLabel()
        status_row.addWidget(self.save_status_label)
        projection_layout.addLayout(status_row)

        buttons_row = QHBoxLayout()
        self.save_button = QPushButton("Save daily projections")
        self.save_button.clicked.connect(self.handle_save_projections)
        buttons_row.addWidget(self.save_button)
        buttons_row.addStretch()
        projection_layout.addLayout(buttons_row)
        self._set_saved_state(True)

        layout.addWidget(self.projection_group)

        self.heat_group = QGroupBox("Sales heat map")
        heat_layout = QVBoxLayout(self.heat_group)
        heat_hint = QLabel(
            "Derived from projected sales and modifiers. Cool blues indicate slower days, red indicates higher sales volume expectations."
        )
        heat_hint.setWordWrap(True)
        heat_layout.addWidget(heat_hint)

        heat_row = QHBoxLayout()
        heat_row.setSpacing(8)
        for day_index, day_name in enumerate(DAYS_OF_WEEK):
            label = QLabel(day_name)
            label.setAlignment(Qt.AlignCenter)
            label.setMinimumWidth(110)
            label.setFixedHeight(96)
            label.setWordWrap(True)
            label.setStyleSheet(self._heat_label_style())
            self.heat_labels[day_index] = label
            heat_row.addWidget(label)
        heat_row.addStretch()
        heat_layout.addLayout(heat_row)
        self.sales_total_label = QLabel("Projected weekly sales: --")
        self.sales_total_label.setStyleSheet("font-weight:600;")
        heat_layout.addWidget(self.sales_total_label)

        layout.addWidget(self.heat_group)

        self.modifier_group = QGroupBox("Modifiers")
        modifier_layout = QVBoxLayout(self.modifier_group)

        self.modifier_table = QTableWidget(0, 7)
        self.modifier_table.setHorizontalHeaderLabels(
            ["Title", "Impact", "Day", "Window", "% Change", "Applied by", "Notes"]
        )
        self.modifier_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.modifier_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.modifier_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.modifier_table.setWordWrap(True)
        self.modifier_table.setTextElideMode(Qt.ElideNone)
        header = self.modifier_table.horizontalHeader()
        header.setMinimumSectionSize(80)
        header.setStretchLastSection(False)
        header.setSectionsMovable(False)
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(6, QHeaderView.Stretch)
        for column in range(1, 6):
            header.setSectionResizeMode(column, QHeaderView.ResizeToContents)
        self.modifier_table.verticalHeader().setDefaultSectionSize(40)
        self.modifier_table.itemSelectionChanged.connect(self._update_modifier_buttons)
        modifier_layout.addWidget(self.modifier_table)
        self._apply_modifier_column_layout()
        self._update_completion_status()

        self.modifier_feedback = QLabel()
        self.modifier_feedback.setStyleSheet(f"color:{INFO_COLOR};")
        modifier_layout.addWidget(self.modifier_feedback)

        modifier_buttons = QHBoxLayout()
        self.add_modifier_button = QPushButton("Add modifier")
        self.add_modifier_button.clicked.connect(self.handle_add_modifier)
        modifier_buttons.addWidget(self.add_modifier_button)

        self.edit_modifier_button = QPushButton("Edit")
        self.edit_modifier_button.clicked.connect(self.handle_edit_modifier)
        modifier_buttons.addWidget(self.edit_modifier_button)

        self.delete_modifier_button = QPushButton("Delete")
        self.delete_modifier_button.clicked.connect(self.handle_delete_modifier)
        modifier_buttons.addWidget(self.delete_modifier_button)

        self.save_modifier_button = QPushButton("Save for later")
        self.save_modifier_button.clicked.connect(self.handle_save_modifier_template)
        modifier_buttons.addWidget(self.save_modifier_button)
        modifier_buttons.addStretch()
        modifier_layout.addLayout(modifier_buttons)

        layout.addWidget(self.modifier_group)
        layout.addWidget(self._build_saved_modifier_library())
        layout.addStretch(1)

    def _build_saved_modifier_library(self) -> QGroupBox:
        self.saved_modifier_group = QGroupBox("Saved modifiers")
        library_layout = QVBoxLayout(self.saved_modifier_group)
        hint = QLabel("Double-click a saved modifier to drop it into the active week.")
        hint.setWordWrap(True)
        library_layout.addWidget(hint)

        self.saved_modifier_list = QListWidget()
        self.saved_modifier_list.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.saved_modifier_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.saved_modifier_list.itemSelectionChanged.connect(self._update_saved_modifier_buttons)
        self.saved_modifier_list.itemDoubleClicked.connect(lambda _: self.handle_apply_saved_modifier())
        library_layout.addWidget(self.saved_modifier_list)

        saved_buttons = QHBoxLayout()
        self.apply_saved_button = QPushButton("Add to week")
        self.apply_saved_button.clicked.connect(self.handle_apply_saved_modifier)
        saved_buttons.addWidget(self.apply_saved_button)

        self.delete_saved_button = QPushButton("Delete saved")
        self.delete_saved_button.clicked.connect(self.handle_delete_saved_modifier)
        saved_buttons.addWidget(self.delete_saved_button)
        saved_buttons.addStretch()
        library_layout.addLayout(saved_buttons)

        self.saved_feedback_label = QLabel()
        self.saved_feedback_label.setStyleSheet(f"color:{INFO_COLOR};")
        library_layout.addWidget(self.saved_feedback_label)
        self._update_saved_modifier_buttons()
        return self.saved_modifier_group

    def _heat_label_style(self, background: Optional[str] = None) -> str:
        base_bg = background or "#10131b"
        return (
            f"border:1px solid #1f2331; border-radius:10px; padding:10px; "
            f"background-color:{base_bg}; color:#f5f6fa; font-weight:600;"
        )

    def set_active_week(self, active_week: Dict[str, Any]) -> None:
        self.active_week = active_week
        self.week_label = active_week.get("label", "")
        self.refresh()

    def refresh(self) -> None:
        iso_year = self.active_week.get("iso_year")
        iso_week = self.active_week.get("iso_week")
        label = self.active_week.get("label") or ""
        with self.session_factory() as session:
            week = get_or_create_week_context(session, iso_year, iso_week, label)
            self.week_id = week.id
            self.week_label = week.label
            self.projections = get_week_daily_projections(session, week.id)
            self.modifiers = get_week_modifiers(session, week.id)
            self.saved_modifiers = list_saved_modifiers(session)
        self._populate_projection_inputs()
        self._refresh_modifiers_table()
        self._refresh_saved_modifier_panel()
        self._apply_heatmap()
        self._update_group_titles()
        self._update_modifier_buttons()
        self._update_completion_status()

    def _update_group_titles(self) -> None:
        suffix = f" - {self.week_label}" if self.week_label else ""
        self.projection_group.setTitle(f"Projected sales by day{suffix}")
        self.modifier_group.setTitle(f"Modifiers{suffix}")
        self.heat_group.setTitle(f"Sales heat map{suffix}")

    def _update_completion_status(self) -> None:
        if not hasattr(self, "completion_badge"):
            return
        all_days_filled = all(field.text().strip() for field in self.day_inputs.values())
        if not all_days_filled:
            status = "Incomplete"
            color = ERROR_COLOR
        elif self.modifiers:
            status = "Complete with Modifiers"
            color = SUCCESS_COLOR
        else:
            status = "Complete"
            color = WARNING_COLOR
        self.completion_badge.setText(f"Status: {status}")
        self.completion_badge.setStyleSheet(
            f"padding:4px 10px; border-radius:8px; background-color:#1b1f2d; color:{color}; font-weight:600;"
        )

    def _apply_modifier_column_layout(self) -> None:
        if not hasattr(self, "modifier_table"):
            return
        header = self.modifier_table.horizontalHeader()
        total_width = self.modifier_table.viewport().width()
        if total_width <= 0:
            total_width = header.length()
        if total_width <= 0:
            total_width = self.modifier_table.width()
        if total_width <= 0:
            return
        columns = sorted(self._modifier_column_ratios.keys())
        widths: Dict[int, int] = {}
        total_assigned = 0
        for column in columns:
            ratio = self._modifier_column_ratios.get(column, 0.0)
            min_width = 140 if column == 0 else 80
            width = max(int(total_width * ratio), min_width)
            widths[column] = width
            total_assigned += width
        if columns:
            last_column = columns[-1]
            min_width = 140 if last_column == 0 else 80
            widths[last_column] = max(widths[last_column] + (total_width - total_assigned), min_width)
        for column in columns:
            header.resizeSection(column, widths[column])

    def _populate_projection_inputs(self) -> None:
        mapping = {projection.day_of_week: projection for projection in self.projections}
        for day, field in self.day_inputs.items():
            projection = mapping.get(day)
            value = projection.projected_sales_amount if projection else 0.0
            field.blockSignals(True)
            field.setText(f"{int(round(value))}" if value else "")
            field.blockSignals(False)
            note = projection.projected_notes if projection else ""
            note_field = self.day_note_inputs[day]
            note_field.blockSignals(True)
            note_field.setText(note)
            note_field.blockSignals(False)
        self._set_saved_state(True)

    def _handle_projection_field_edited(self, _text: str) -> None:
        self._mark_unsaved()
        self._update_completion_status()

    def _set_saved_state(self, saved: bool) -> None:
        if saved:
            self.save_status_label.setText("All changes saved")
            self.save_status_label.setStyleSheet(f"color:{SUCCESS_COLOR};")
            self._pending_changes = False
        else:
            self.save_status_label.setText("Unsaved changes")
            self.save_status_label.setStyleSheet(f"color:{ACCENT_COLOR};")
            self._pending_changes = True

    def _mark_unsaved(self) -> None:
        if not self._pending_changes:
            self._set_saved_state(False)

    def handle_save_projections(self) -> None:
        if self.week_id is None:
            return
        payload: Dict[int, Dict[str, float | str]] = {}
        for day, field in self.day_inputs.items():
            text_value = field.text().strip()
            amount = float(int(text_value)) if text_value else 0.0
            payload[day] = {
                "projected_sales_amount": amount,
                "projected_notes": self.day_note_inputs[day].text().strip(),
            }
        with self.session_factory() as session:
            save_week_daily_projection_values(session, self.week_id, payload)
        audit_logger.log(
            "sales_projection_update",
            self.actor.get("username"),
            role=self.actor.get("role"),
            details={
                "iso_year": self.active_week.get("iso_year"),
                "iso_week": self.active_week.get("iso_week"),
                "values": {day: payload[day]["projected_sales_amount"] for day in payload},
            },
        )
        self._set_saved_state(True)
        self.refresh()

    def _apply_heatmap(self) -> None:
        summaries = []
        projection_map = {projection.day_of_week: projection for projection in self.projections}
        for day in range(7):
            projection = projection_map.get(day)
            base_sales = float(projection.projected_sales_amount) if projection else 0.0
            day_modifiers = [modifier for modifier in self.modifiers if modifier.day_of_week == day]
            # Weight modifiers by the portion of their window that falls within this day.
            net_pct = 0.0
            modifier_descriptions: List[str] = []
            for modifier in day_modifiers:
                frac = self._modifier_fraction_within_day(modifier.start_time, modifier.end_time)
                net_pct += modifier.pct_change * frac
                descriptor = (
                    f"{modifier.title} "
                    f"{modifier.pct_change:+d}% "
                    f"{format_time_label(modifier.start_time)}–{format_time_label(modifier.end_time)}"
                )
                modifier_descriptions.append(descriptor)

            # Add carryover from previous day's wrapping modifiers (past-midnight portion)
            prev_day = (day - 1) % 7
            for m in self.modifiers:
                if m.day_of_week != prev_day:
                    continue
                if m.end_time <= m.start_time:
                    carry_frac = self._modifier_fraction_carryover_from_previous(m.start_time, m.end_time)
                    if carry_frac > 0:
                        net_pct += m.pct_change * carry_frac
            adjusted_sales = max(base_sales * (1 + net_pct / 100.0), 0.0)
            summaries.append(
                {
                    "day": day,
                    "base": base_sales,
                    "net_pct": net_pct,
                    "adjusted": adjusted_sales,
                    "count": len(day_modifiers),
                    "descriptions": modifier_descriptions,
                }
            )

        adjusted_values = [summary["adjusted"] for summary in summaries]
        max_value = max(adjusted_values) if adjusted_values else 0.0
        min_value = min(adjusted_values) if adjusted_values else 0.0

        for summary in summaries:
            day = summary["day"]
            label = self.heat_labels[day]
            base = summary["base"]
            adjusted = summary["adjusted"]
            net_pct = summary["net_pct"]
            modifier_text = ", ".join(summary["descriptions"]) or "No modifiers"
            label.setToolTip(
                f"Base: {self._format_currency(base)}\n"
                f"Adjusted: {self._format_currency(adjusted)}\n"
                f"Net change: {net_pct:+.1f}%\n"
                f"Modifiers: {modifier_text}"
            )
            if max_value == min_value:
                ratio = 0.0 if adjusted <= 0 else 0.6
            else:
                ratio = (adjusted - min_value) / (max_value - min_value)
            palette = self._heat_color(ratio, adjusted > 0 or base > 0)
            label.setStyleSheet(self._heat_label_style(palette))
            delta = adjusted - base
            label.setText(
                f"{DAYS_OF_WEEK[day]}\n{self._format_currency(adjusted)}\n{self._format_delta(delta)}"
            )
        projected_total = sum(adjusted_values)
        self.sales_total_label.setText(
            f"Projected weekly sales (after modifiers): {self._format_currency(projected_total)}"
        )

    @staticmethod
    def _modifier_fraction_within_day(start: datetime.time, end: datetime.time) -> float:
        """Fraction of this day (0:00–24:00) covered by the window.
        If the window wraps past midnight (end <= start), only count the portion
        up to midnight for the same day.
        """
        start_minutes = start.hour * 60 + start.minute
        end_minutes = end.hour * 60 + end.minute
        if end_minutes > start_minutes:
            duration = end_minutes - start_minutes
        else:
            duration = (24 * 60) - start_minutes
        return max(0.0, min(1.0, duration / (24 * 60)))

    @staticmethod
    def _modifier_fraction_carryover_from_previous(start: datetime.time, end: datetime.time) -> float:
        """For a previous-day window that wraps past midnight, fraction that spills into the current day.
        Non-wrapping windows contribute zero to the next day.
        """
        start_minutes = start.hour * 60 + start.minute
        end_minutes = end.hour * 60 + end.minute
        if end_minutes <= start_minutes:
            duration = end_minutes  # 0:00 to end on the next day
            return max(0.0, min(1.0, duration / (24 * 60)))
        return 0.0

    @staticmethod
    def _format_currency(value: float) -> str:
        rounded = round(value)
        sign = "-" if rounded < 0 else ""
        return f"{sign}${abs(rounded):,}"

    @staticmethod
    def _format_delta(value: float) -> str:
        rounded = round(value)
        if rounded == 0:
            return "±$0"
        sign = "+" if rounded > 0 else "-"
        return f"{sign}${abs(rounded):,}"

    def _heat_color(self, ratio: float, has_value: bool) -> str:
        if not has_value:
            return "#1e2937"
        ratio = max(0.0, min(1.0, ratio))
        start_rgb = (29, 78, 216)  # calm blue
        end_rgb = (220, 38, 38)  # bold red
        r = int(start_rgb[0] + (end_rgb[0] - start_rgb[0]) * ratio)
        g = int(start_rgb[1] + (end_rgb[1] - start_rgb[1]) * ratio)
        b = int(start_rgb[2] + (end_rgb[2] - start_rgb[2]) * ratio)
        return f"rgb({r}, {g}, {b})"

    def _refresh_modifiers_table(self) -> None:
        self.modifier_table.setRowCount(len(self.modifiers))
        for row, modifier in enumerate(self.modifiers):
            type_label = "Increase" if modifier.pct_change >= 0 else "Decrease"
            change_text = f"{modifier.pct_change:+d}%"
            day_text = DAYS_OF_WEEK[modifier.day_of_week]
            window_text = f"{format_time_label(modifier.start_time)} - {format_time_label(modifier.end_time)}"
            notes_text = modifier.notes or ""

            title_item = QTableWidgetItem(modifier.title)
            title_item.setData(Qt.UserRole, modifier.id)
            self.modifier_table.setItem(row, 0, title_item)
            self.modifier_table.setItem(row, 1, QTableWidgetItem(type_label))
            self.modifier_table.setItem(row, 2, QTableWidgetItem(day_text))
            self.modifier_table.setItem(row, 3, QTableWidgetItem(window_text))
            self.modifier_table.setItem(row, 4, QTableWidgetItem(change_text))
            self.modifier_table.setItem(row, 5, QTableWidgetItem(modifier.created_by))
            self.modifier_table.setItem(row, 6, QTableWidgetItem(notes_text))

        self._apply_modifier_column_layout()
        self._update_completion_status()

    def _selected_modifier(self) -> Optional[Modifier]:
        selection = self.modifier_table.selectionModel()
        if not selection or not selection.hasSelection():
            return None
        row = selection.currentIndex().row()
        if row < 0 or row >= len(self.modifiers):
            return None
        return self.modifiers[row]

    def _update_modifier_buttons(self) -> None:
        has_selection = self._selected_modifier() is not None
        self.edit_modifier_button.setEnabled(has_selection)
        self.delete_modifier_button.setEnabled(has_selection)
        self.save_modifier_button.setEnabled(has_selection)

    def _selected_saved_modifier(self) -> Optional[SavedModifier]:
        if not hasattr(self, "saved_modifier_list"):
            return None
        row = self.saved_modifier_list.currentRow()
        if row < 0 or row >= len(self.saved_modifiers):
            return None
        return self.saved_modifiers[row]

    def _update_saved_modifier_buttons(self) -> None:
        has_selection = self._selected_saved_modifier() is not None
        for button in (self.apply_saved_button, self.delete_saved_button):
            button.setEnabled(has_selection)

    def _refresh_saved_modifier_panel(self) -> None:
        if not hasattr(self, "saved_modifier_list"):
            return
        self.saved_modifier_list.clear()
        for template in self.saved_modifiers:
            day_text = DAYS_OF_WEEK[template.day_of_week]
            window_text = f"{format_time_label(template.start_time)}–{format_time_label(template.end_time)}"
            change_text = f"{template.pct_change:+d}%"
            label = f"{template.title}  ({day_text} {window_text}, {change_text})"
            item = QListWidgetItem(label)
            item.setData(Qt.UserRole, template.id)
            self.saved_modifier_list.addItem(item)
        self._update_saved_modifier_buttons()

    def handle_save_modifier_template(self) -> None:
        current = self._selected_modifier()
        if not current:
            return
        self._create_saved_template(
            title=current.title,
            impact_type=current.modifier_type,
            day_of_week=current.day_of_week,
            start_time=current.start_time,
            end_time=current.end_time,
            pct_change=current.pct_change,
            notes=current.notes or "",
        )
        self.modifier_feedback.setStyleSheet(f"color:{SUCCESS_COLOR};")
        self.modifier_feedback.setText(f"Saved '{current.title}' for future weeks.")
        self.refresh()

    def handle_apply_saved_modifier(self) -> None:
        template = self._selected_saved_modifier()
        if not template or self.week_id is None:
            return
        with self.session_factory() as session:
            try:
                modifier = apply_saved_modifier_to_week(
                    session,
                    template.id,
                    self.week_id,
                    created_by=self.actor.get("username", "unknown"),
                )
            except ValueError:
                QMessageBox.warning(self, "Saved modifier missing", "That saved modifier no longer exists.")
                self.refresh()
                return
        audit_logger.log(
            "modifier_create_from_saved",
            self.actor.get("username"),
            role=self.actor.get("role"),
            details={"template_id": template.id, "modifier_id": modifier.id},
        )
        self.modifier_feedback.setStyleSheet(f"color:{SUCCESS_COLOR};")
        self.modifier_feedback.setText(f"Added '{template.title}' to {self.week_label}.")
        self.refresh()

    def handle_delete_saved_modifier(self) -> None:
        template = self._selected_saved_modifier()
        if not template:
            return
        confirm = QMessageBox.question(
            self,
            "Delete saved modifier",
            f"Remove '{template.title}' from the saved list?",
        )
        if confirm != QMessageBox.Yes:
            return
        with self.session_factory() as session:
            delete_saved_modifier(session, template.id)
        audit_logger.log(
            "modifier_template_delete",
            self.actor.get("username"),
            role=self.actor.get("role"),
            details={"template_id": template.id, "title": template.title},
        )
        self.saved_feedback_label.setText(f"Deleted saved modifier '{template.title}'.")
        self.refresh()

    def _create_saved_template(
        self,
        *,
        title: str,
        impact_type: str,
        day_of_week: int,
        start_time: datetime.time,
        end_time: datetime.time,
        pct_change: int,
        notes: str,
    ) -> None:
        with self.session_factory() as session:
            template = save_modifier_template(
                session,
                title=title,
                modifier_type=impact_type,
                day_of_week=day_of_week,
                start_time=start_time,
                end_time=end_time,
                pct_change=pct_change,
                notes=notes,
                created_by=self.actor.get("username", "unknown"),
            )
        audit_logger.log(
            "modifier_template_create",
            self.actor.get("username"),
            role=self.actor.get("role"),
            details={"template_id": template.id, "title": template.title},
        )
        if hasattr(self, "saved_feedback_label"):
            self.saved_feedback_label.setText(f"Saved '{title}' for future weeks.")

    def handle_add_modifier(self) -> None:
        dialog = ModifierDialog(self.modifiers)
        dialog.setStyleSheet(THEME_STYLESHEET)
        if dialog.exec() != QDialog.Accepted or not dialog.result_data or self.week_id is None:
            return
        data = dialog.result_data
        pct_change = int(data["pct_change"])
        impact_type = "increase" if pct_change >= 0 else "decrease"
        day_value = int(data["day_of_week"])
        notes_value = data.get("notes", "")
        with self.session_factory() as session:
            modifier = Modifier(
                week_id=self.week_id,
                title=data["title"],
                modifier_type=impact_type,
                day_of_week=day_value,
                start_time=data["start_time"],
                end_time=data["end_time"],
                pct_change=pct_change,
                notes=notes_value,
                created_by=self.actor.get("username", "unknown"),
            )
            session.add(modifier)
            session.commit()
            session.refresh(modifier)
        audit_logger.log(
            "modifier_create",
            self.actor.get("username"),
            role=self.actor.get("role"),
            details={
                "modifier_id": modifier.id,
                "week_id": self.week_id,
                "title": modifier.title,
                "modifier_type": modifier.modifier_type,
                "day_of_week": modifier.day_of_week,
                "start_time": modifier.start_time.isoformat(),
                "end_time": modifier.end_time.isoformat(),
                "pct_change": modifier.pct_change,
            },
        )
        self.modifier_feedback.setStyleSheet(f"color:{SUCCESS_COLOR};")
        self.modifier_feedback.setText(f"Added modifier '{modifier.title}'.")
        if data.get("save_for_later"):
            self._create_saved_template(
                title=data["title"],
                impact_type=impact_type,
                day_of_week=day_value,
                start_time=data["start_time"],
                end_time=data["end_time"],
                pct_change=pct_change,
                notes=notes_value,
            )
        self.refresh()

    def handle_edit_modifier(self) -> None:
        current = self._selected_modifier()
        if not current or self.week_id is None:
            return
        dialog = ModifierDialog(self.modifiers, modifier=current)
        dialog.setStyleSheet(THEME_STYLESHEET)
        if dialog.exec() != QDialog.Accepted or not dialog.result_data:
            return
        data = dialog.result_data
        pct_change = int(data["pct_change"])
        impact_type = "increase" if pct_change >= 0 else "decrease"
        day_value = int(data["day_of_week"])
        notes_value = data.get("notes", "")
        with self.session_factory() as session:
            modifier = session.get(Modifier, current.id)
            if not modifier:
                QMessageBox.warning(self, "Modifier missing", "The selected modifier no longer exists.")
                self.refresh()
                return
            modifier.title = data["title"]
            modifier.modifier_type = impact_type
            modifier.day_of_week = day_value
            modifier.start_time = data["start_time"]
            modifier.end_time = data["end_time"]
            modifier.pct_change = pct_change
            modifier.notes = notes_value
            session.commit()
        audit_logger.log(
            "modifier_update",
            self.actor.get("username"),
            role=self.actor.get("role"),
            details={
                "modifier_id": current.id,
                "week_id": self.week_id,
                "title": data["title"],
                "modifier_type": impact_type,
                "day_of_week": day_value,
                "start_time": data["start_time"].isoformat(),
                "end_time": data["end_time"].isoformat(),
                "pct_change": pct_change,
            },
        )
        self.modifier_feedback.setStyleSheet(f"color:{SUCCESS_COLOR};")
        self.modifier_feedback.setText(f"Updated modifier '{data['title']}'.")
        if data.get("save_for_later"):
            self._create_saved_template(
                title=data["title"],
                impact_type=impact_type,
                day_of_week=day_value,
                start_time=data["start_time"],
                end_time=data["end_time"],
                pct_change=pct_change,
                notes=notes_value,
            )
        self.refresh()

    def handle_delete_modifier(self) -> None:
        current = self._selected_modifier()
        if not current or self.week_id is None:
            return
        confirm = QMessageBox.question(
            self,
            "Delete modifier",
            f"Remove modifier '{current.title}'?",
        )
        if confirm != QMessageBox.Yes:
            return
        with self.session_factory() as session:
            modifier = session.get(Modifier, current.id)
            if modifier:
                session.delete(modifier)
                session.commit()
        audit_logger.log(
            "modifier_delete",
            self.actor.get("username"),
            role=self.actor.get("role"),
            details={
                "modifier_id": current.id,
                "week_id": self.week_id,
                "title": current.title,
            },
        )
        self.modifier_feedback.setStyleSheet(f"color:{ACCENT_COLOR};")
        self.modifier_feedback.setText(f"Deleted modifier '{current.title}'.")
        self.refresh()

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._apply_modifier_column_layout()


def _default_timeblocks() -> List[Dict[str, str]]:
    return [
        {"name": "Open", "start": "10:00", "end": "14:00"},
        {"name": "Mid", "start": "12:00", "end": "17:00"},
        {"name": "PM", "start": "16:00", "end": "21:00"},
        {"name": "Close", "start": "20:00", "end": "24:00"},
    ]


def _timeblocks_from_params(params: Dict[str, Any]) -> List[Dict[str, str]]:
    blocks = params.get("timeblocks") if isinstance(params, dict) else None
    if not isinstance(blocks, dict) or not blocks:
        return _default_timeblocks()
    rows: List[Dict[str, str]] = []
    for name, spec in blocks.items():
        if not isinstance(spec, dict):
            continue
        rows.append(
            {
                "name": name,
                "start": spec.get("start", "10:00"),
                "end": spec.get("end", "14:00"),
            }
        )
    return rows or _default_timeblocks()


def _default_role_payload(block_names: List[str]) -> Dict[str, Any]:
    return {
        "enabled": False,
        "priority": 1.0,
        "max_weekly_hours": 35,
        "daily_boost": {},
        "thresholds": [],
        "covers": [],
        "blocks": {
            block: {"base": 0, "min": 0, "max": 0, "per_1000_sales": 0.0, "per_modifier": 0.0}
            for block in block_names
        },
    }


def _default_business_hours() -> Dict[str, Dict[str, str]]:
    return {
        "Mon": {"open": "11:00", "mid": "16:00", "close": "24:00"},
        "Tue": {"open": "11:00", "mid": "16:00", "close": "24:00"},
        "Wed": {"open": "11:00", "mid": "16:00", "close": "24:00"},
        "Thu": {"open": "11:00", "mid": "16:00", "close": "24:00"},
        "Fri": {"open": "11:00", "mid": "16:00", "close": "25:00"},
        "Sat": {"open": "11:00", "mid": "16:00", "close": "25:00"},
        "Sun": {"open": "11:00", "mid": "16:00", "close": "23:00"},
    }


class RoleSelectionDialog(QDialog):
    def __init__(self, group: str, roles: List[str], selected: List[str]) -> None:
        super().__init__()
        self.setWindowTitle(f"Select roles for {group}")
        self.resize(360, 420)
        layout = QVBoxLayout(self)
        info = QLabel("Check the roles to include.")
        info.setWordWrap(True)
        layout.addWidget(info)
        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(QAbstractItemView.MultiSelection)
        for role in roles:
            item = QListWidgetItem(role)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if role in selected else Qt.Unchecked)
            self.list_widget.addItem(item)
        layout.addWidget(self.list_widget)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def selected_roles(self) -> List[str]:
        roles: List[str] = []
        for index in range(self.list_widget.count()):
            item = self.list_widget.item(index)
            if item.checkState() == Qt.Checked:
                roles.append(item.text())
        return roles


class RoleSelectField(QWidget):
    """Field that shows selected roles and opens a dialog for selection."""

    def __init__(self, group: str, available: List[str], selected: List[str]) -> None:
        super().__init__()
        self.group = group
        self.available_roles = available
        self._selected_roles = selected or []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(QAbstractItemView.NoSelection)
        self.list_widget.setFocusPolicy(Qt.NoFocus)
        self.list_widget.setMinimumHeight(180)
        self.list_widget.setStyleSheet(
            "QListWidget {background-color:#1f1f1f; border:1px solid #979797; font-size:13px;}"
            "QListWidget::item {padding:6px; margin:3px; border-radius:6px; background-color:#2a2a2a; color:#f7f7f7;}"
        )
        layout.addWidget(self.list_widget)
        button_row = QHBoxLayout()
        button_row.setContentsMargins(0, 4, 0, 0)
        button_row.setSpacing(8)
        self.pick_btn = QPushButton("Select roles")
        self.pick_btn.setMinimumWidth(110)
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.setMaximumWidth(80)
        self.pick_btn.clicked.connect(self._open_picker)
        self.clear_btn.clicked.connect(self._clear_roles)
        button_row.addWidget(self.pick_btn)
        button_row.addWidget(self.clear_btn)
        button_row.addStretch(1)
        layout.addLayout(button_row)
        self._refresh_display()

    def selected_roles(self) -> List[str]:
        return list(self._selected_roles)

    def set_group(self, group: str, roles: List[str]) -> None:
        self.group = group
        self.available_roles = roles
        if not self._selected_roles:
            self._selected_roles = roles[:]
        else:
            self._selected_roles = [role for role in self._selected_roles if role in roles] or roles[:]
        self._refresh_display()

    def set_selected_roles(self, roles: List[str]) -> None:
        filtered = [role for role in roles if role in self.available_roles]
        self._selected_roles = filtered or []
        self._refresh_display()

    def set_available_roles(self, roles: List[str]) -> None:
        self.available_roles = roles
        self._selected_roles = [role for role in self._selected_roles if role in roles]
        self._refresh_display()

    def _open_picker(self) -> None:
        dialog = RoleSelectionDialog(self.group or "Group", self.available_roles, self._selected_roles)
        if dialog.exec() == QDialog.Accepted:
            self._selected_roles = dialog.selected_roles()
            self._refresh_display()

    def _clear_roles(self) -> None:
        self._selected_roles = []
        self._refresh_display()

    def _refresh_display(self) -> None:
        self.list_widget.clear()
        roles = self._selected_roles or []
        if not roles:
            placeholder = QListWidgetItem("(none selected)")
            placeholder.setFlags(Qt.NoItemFlags)
            placeholder_font = QFont()
            placeholder_font.setItalic(True)
            placeholder.setFont(placeholder_font)
            self.list_widget.addItem(placeholder)
            return
        for role in roles:
            item = QListWidgetItem(role)
            item.setFlags(Qt.NoItemFlags)
            self.list_widget.addItem(item)


class ShiftTemplateEditor(QWidget):
    """UI widget for editing AM/PM shift suggestions."""

    def __init__(self, groups: List[str]) -> None:
        super().__init__()
        self.groups = groups
        self.tables: Dict[str, Dict[str, QTableWidget]] = {}
        layout = QVBoxLayout(self)
        intro = QLabel("Suggested shift windows; generator prefers these start times when possible.")
        intro.setWordWrap(True)
        intro.setStyleSheet(f"color:{INFO_COLOR}; font-weight:500;")
        layout.addWidget(intro)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)
        for group in groups:
            tab = QWidget()
            tab_layout = QVBoxLayout(tab)
            block_tables: Dict[str, QTableWidget] = {}
            for block_key, label in (("am", "Morning (AM)"), ("pm", "Evening (PM)")):
                section = QGroupBox(f"{label} shifts")
                section_layout = QVBoxLayout(section)
                table = self._build_table()
                block_tables[block_key] = table
                section_layout.addWidget(table)
                buttons = QHBoxLayout()
                add_btn = QPushButton("Add row")
                remove_btn = QPushButton("Remove selected")
                add_btn.clicked.connect(lambda _=None, tbl=table: self._append_row(tbl))
                remove_btn.clicked.connect(lambda _=None, tbl=table: self._remove_row(tbl))
                buttons.addWidget(add_btn)
                buttons.addWidget(remove_btn)
                buttons.addStretch(1)
                section_layout.addLayout(buttons)
                tab_layout.addWidget(section)
            tab_layout.addStretch(1)
            self.tabs.addTab(tab, group)
            self.tables[group] = block_tables

    @staticmethod
    def _build_table() -> QTableWidget:
        table = QTableWidget(0, 2)
        table.setHorizontalHeaderLabels(["Start", "End"])
        table.horizontalHeader().setStretchLastSection(True)
        table.verticalHeader().setVisible(False)
        table.setSelectionBehavior(QAbstractItemView.SelectRows)
        table.setSelectionMode(QAbstractItemView.SingleSelection)
        return table

    def _append_row(self, table: QTableWidget, start: str = "11:00", end: str = "15:00") -> None:
        row = table.rowCount()
        table.insertRow(row)
        table.setItem(row, 0, QTableWidgetItem(start))
        table.setItem(row, 1, QTableWidgetItem(end))

    @staticmethod
    def _remove_row(table: QTableWidget) -> None:
        row = table.currentRow()
        if row >= 0:
            table.removeRow(row)

    def set_config(self, config: Dict[str, Any]) -> None:
        config = config or {}
        for group, blocks in self.tables.items():
            group_spec = config.get(group, {})
            for block_key, table in blocks.items():
                table.setRowCount(0)
                entries = group_spec.get(block_key) or []
                if isinstance(entries, list) and entries:
                    for entry in entries:
                        start = str(entry.get("start", "11:00"))
                        end = str(entry.get("end", "15:00"))
                        self._append_row(table, start, end)
                if table.rowCount() == 0:
                    self._append_row(table)

    def value(self) -> Dict[str, Dict[str, List[Dict[str, str]]]]:
        payload: Dict[str, Dict[str, List[Dict[str, str]]]] = {}
        for group, blocks in self.tables.items():
            block_payload: Dict[str, List[Dict[str, str]]] = {}
            for block_key, table in blocks.items():
                entries: List[Dict[str, str]] = []
                for row in range(table.rowCount()):
                    start_item = table.item(row, 0)
                    end_item = table.item(row, 1)
                    start = (start_item.text() if start_item else "").strip()
                    end = (end_item.text() if end_item else "").strip()
                    if start and end:
                        entries.append({"start": start, "end": end})
                if entries:
                    block_payload[block_key] = entries
            if block_payload:
                payload[group] = block_payload
        return payload


class SectionCapacityEditor(QWidget):
    """UI widget for editing section capacity weights that influence cut bias."""

    def __init__(self, group_sections: Dict[str, List[str]]) -> None:
        super().__init__()
        self.group_sections = group_sections
        self.inputs: Dict[str, Dict[str, QDoubleSpinBox]] = {}
        layout = QVBoxLayout(self)
        intro = QLabel(
            "Section stay bias: enter relative weight (1.0 = normal). Higher weights keep a section staffed longer; lower weights trim sooner."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet(f"color:{INFO_COLOR}; font-weight:500;")
        layout.addWidget(intro)

        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)
        for group, sections in group_sections.items():
            tab = QWidget()
            tab_layout = QFormLayout(tab)
            group_inputs: Dict[str, QDoubleSpinBox] = {}
            for section_name in sections:
                spin = QDoubleSpinBox()
                spin.setRange(0.2, 3.0)
                spin.setDecimals(2)
                spin.setSingleStep(0.05)
                spin.setValue(1.0)
                spin.setSuffix("x")
                spin.setButtonSymbols(QAbstractSpinBox.NoButtons)
                spin.setFocusPolicy(Qt.StrongFocus)
                spin.wheelEvent = lambda event: event.ignore()
                spin.setToolTip("Relative section weight: >1 keeps longer, <1 cuts earlier. Type to edit.")
                tab_layout.addRow(section_name, spin)
                group_inputs[section_name] = spin
            self.tabs.addTab(tab, group)
            self.inputs[group] = group_inputs

    def set_config(self, config: Dict[str, Any]) -> None:
        for group, sections in self.inputs.items():
            group_cfg = (config or {}).get(group, {})
            for section_name, spin in sections.items():
                try:
                    value = float(group_cfg.get(section_name, 1.0))
                except (TypeError, ValueError):
                    value = 1.0
                spin.setValue(max(spin.minimum(), min(spin.maximum(), value)))

    def value(self) -> Dict[str, Dict[str, float]]:
        payload: Dict[str, Dict[str, float]] = {}
        for group, sections in self.inputs.items():
            payload[group] = {section: round(spin.value(), 2) for section, spin in sections.items()}
        return payload


class CutPriorityEditor(QWidget):
    """Shared widget that manages cut sequencing + role ordering settings."""

    TOGGLE_STYLE = (
        "QPushButton {border-radius:12px; padding:6px 14px; font-weight:600; border:1px solid #4a4a4a;}"
        "QPushButton:checked {background-color:#2e7d32; color:white; border-color:#2e7d32;}"
        "QPushButton:!checked {background-color:#5c2f31; color:white; border-color:#5c2f31;}"
    )

    TABLE_STYLE = (
        "QTableWidget {background-color:#161616; color:#f2f2f2; gridline-color:#2d2d2d;}"
        "QTableWidget::item:selected {background-color:#314b6e; color:white;}"
        "QHeaderView::section {background-color:#1d1d1d; color:#f2f2f2; border:0; padding:4px;}"
        "QLineEdit, QComboBox {background-color:#232323; color:#fdfdfd; border:1px solid #555; padding:4px;}"
        "QComboBox QAbstractItemView {background-color:#232323; color:#fdfdfd;}"
    )

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.available_groups = sorted(list(ROLE_GROUPS.keys()) + ["Other"])

        layout = QVBoxLayout(self)
        intro = QLabel(
            "Optional: enable alternating cut rotations to cycle groups evenly and specify preferred role order."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet(f"color:{INFO_COLOR}; font-weight:500;")
        layout.addWidget(intro)

        toggle_row = QHBoxLayout()
        self.enabled_toggle = QPushButton("Rotation disabled")
        self.enabled_toggle.setCheckable(True)
        self.enabled_toggle.setStyleSheet(self.TOGGLE_STYLE)
        self.include_unlisted_toggle = QPushButton("Append unlisted groups")
        self.include_unlisted_toggle.setCheckable(True)
        self.include_unlisted_toggle.setChecked(True)
        self.include_unlisted_toggle.setStyleSheet(self.TOGGLE_STYLE)
        self.status_badge = QLabel()
        self.status_badge.setAlignment(Qt.AlignCenter)
        self.status_badge.setFixedWidth(130)
        self.status_badge.setStyleSheet("padding:4px 10px; border-radius:12px; font-weight:700;")
        toggle_row.addWidget(self.enabled_toggle)
        toggle_row.addWidget(self.include_unlisted_toggle)
        toggle_row.addWidget(self.status_badge)
        toggle_row.addStretch(1)
        layout.addLayout(toggle_row)

        self.config_frame = QGroupBox("Rotation + role ordering")
        config_layout = QVBoxLayout(self.config_frame)

        self.editor_tabs = QTabWidget()
        self.editor_tabs.setTabPosition(QTabWidget.North)
        config_layout.addWidget(self.editor_tabs)

        # Rotation tab
        rotation_widget = QWidget()
        rotation_layout = QVBoxLayout(rotation_widget)
        self.sequence_table = QTableWidget(0, 2)
        self.sequence_table.setHorizontalHeaderLabels(["Group", "Role filters"])
        self.sequence_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.sequence_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.sequence_table.horizontalHeader().setStretchLastSection(True)
        self.sequence_table.verticalHeader().setVisible(True)
        self.sequence_table.verticalHeader().setDefaultSectionSize(195)
        self.sequence_table.setStyleSheet(self.TABLE_STYLE + "font-size:13px;")
        rotation_layout.addWidget(self.sequence_table)
        seq_controls = QHBoxLayout()
        self.sequence_add_btn = QPushButton("Add rotation row")
        self.sequence_remove_btn = QPushButton("Remove selected")
        self.sequence_up_btn = QPushButton("Move up")
        self.sequence_down_btn = QPushButton("Move down")
        self.sequence_add_btn.clicked.connect(self._handle_sequence_add)
        self.sequence_remove_btn.clicked.connect(lambda: self._handle_sequence_remove(self.sequence_table))
        self.sequence_up_btn.clicked.connect(lambda: self._handle_sequence_move(self.sequence_table, -1))
        self.sequence_down_btn.clicked.connect(lambda: self._handle_sequence_move(self.sequence_table, 1))
        for btn in (self.sequence_add_btn, self.sequence_remove_btn, self.sequence_up_btn, self.sequence_down_btn):
            seq_controls.addWidget(btn)
        seq_controls.addStretch(1)
        rotation_layout.addLayout(seq_controls)
        self.editor_tabs.addTab(rotation_widget, "Rotation sequence")

        # Role order tab
        order_widget = QWidget()
        order_layout = QVBoxLayout(order_widget)
        self.role_order_table = QTableWidget(0, 2)
        self.role_order_table.setHorizontalHeaderLabels(["Group", "Preferred role order"])
        self.role_order_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.role_order_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.role_order_table.horizontalHeader().setStretchLastSection(True)
        self.role_order_table.verticalHeader().setVisible(True)
        self.role_order_table.verticalHeader().setDefaultSectionSize(195)
        self.role_order_table.setStyleSheet(self.TABLE_STYLE + "font-size:13px;")
        order_layout.addWidget(self.role_order_table)
        role_controls = QHBoxLayout()
        self.role_add_btn = QPushButton("Add preference")
        self.role_remove_btn = QPushButton("Remove selected")
        self.role_add_btn.clicked.connect(self._handle_role_add)
        self.role_remove_btn.clicked.connect(lambda: self._handle_sequence_remove(self.role_order_table))
        role_controls.addWidget(self.role_add_btn)
        role_controls.addWidget(self.role_remove_btn)
        role_controls.addStretch(1)
        order_layout.addLayout(role_controls)
        self.editor_tabs.addTab(order_widget, "Role ordering")

        layout.addWidget(self.config_frame)
        layout.addStretch(1)

        self.enabled_toggle.toggled.connect(self._update_enabled_state)
        self.include_unlisted_toggle.toggled.connect(
            lambda _: self._style_toggle(self.include_unlisted_toggle, "Append unlisted groups")
        )

    def set_config(self, config: Optional[Dict[str, Any]]) -> None:
        spec = copy.deepcopy(config) if isinstance(config, dict) else {}
        if not spec:
            spec = copy.deepcopy(CUT_PRIORITY_DEFAULT)
        self.enabled_toggle.setChecked(bool(spec.get("enabled", False)))
        self.include_unlisted_toggle.setChecked(bool(spec.get("include_unlisted", True)))
        sequence = spec.get("sequence") or copy.deepcopy(CUT_PRIORITY_DEFAULT.get("sequence", []))
        self._load_sequence_rows(sequence)
        role_order = spec.get("role_order") or copy.deepcopy(CUT_PRIORITY_DEFAULT.get("role_order", {}))
        self._load_role_order(role_order)
        self._update_enabled_state()

    def value(self) -> Dict[str, Any]:
        sequence: List[Dict[str, Any]] = []
        for row in range(self.sequence_table.rowCount()):
            combo = self.sequence_table.cellWidget(row, 0)
            selector = self.sequence_table.cellWidget(row, 1)
            if not isinstance(combo, QComboBox) or not isinstance(selector, RoleSelectField):
                continue
            group = combo.currentText().strip()
            if not group:
                continue
            roles_raw = selector.selected_roles()
            sequence.append({"group": group, "roles": roles_raw})
        role_order: Dict[str, List[str]] = {}
        for row in range(self.role_order_table.rowCount()):
            combo = self.role_order_table.cellWidget(row, 0)
            selector = self.role_order_table.cellWidget(row, 1)
            if not isinstance(combo, QComboBox) or not isinstance(selector, RoleSelectField):
                continue
            group = combo.currentText().strip()
            if not group:
                continue
            entries = selector.selected_roles()
            if entries:
                role_order[group] = entries
        return {
            "enabled": self.enabled_toggle.isChecked(),
            "include_unlisted": self.include_unlisted_toggle.isChecked(),
            "sequence": sequence,
            "role_order": role_order,
        }

    def set_read_only(self, read_only: bool) -> None:
        for widget in [
            self.enabled_toggle,
            self.include_unlisted_toggle,
            self.sequence_table,
            self.role_order_table,
            self.sequence_add_btn,
            self.sequence_remove_btn,
            self.sequence_up_btn,
            self.sequence_down_btn,
            self.role_add_btn,
            self.role_remove_btn,
        ]:
            widget.setEnabled(not read_only)

    def _update_enabled_state(self) -> None:
        enabled = self.enabled_toggle.isChecked()
        self._style_toggle(self.enabled_toggle, "Rotation enabled" if enabled else "Rotation disabled")
        self.config_frame.setVisible(enabled)
        self.status_badge.setText("ENABLED" if enabled else "DISABLED")
        color = "#2e7d32" if enabled else "#6c2f2f"
        self.status_badge.setStyleSheet(
            f"padding:4px; border-radius:6px; font-weight:600; color:white; background-color:{color};"
        )
        if enabled and self.sequence_table.rowCount() == 0:
            self._load_sequence_rows(CUT_PRIORITY_DEFAULT.get("sequence", []))
        if enabled and self.role_order_table.rowCount() == 0:
            self._load_role_order(CUT_PRIORITY_DEFAULT.get("role_order", {}))
        self.include_unlisted_toggle.setEnabled(enabled)
        self._style_toggle(self.include_unlisted_toggle, "Append unlisted groups")

    def _style_toggle(self, button: QPushButton, label: str) -> None:
        if button.isChecked():
            button.setText(label + " (On)")
        else:
            button.setText(label + " (Off)")

    @staticmethod
    def _roles_for_group(group: str) -> List[str]:
        for name, roles in ROLE_GROUPS.items():
            if name.lower() == (group or "").strip().lower():
                return roles[:]
        return []

    def _handle_sequence_add(self) -> None:
        self._add_sequence_row(self.available_groups[0] if self.available_groups else "", [])

    def _handle_sequence_remove(self, table: QTableWidget) -> None:
        row = table.currentRow()
        if row >= 0:
            table.removeRow(row)

    def _handle_sequence_move(self, table: QTableWidget, delta: int) -> None:
        row = table.currentRow()
        if row < 0:
            return
        target = row + delta
        if target < 0 or target >= table.rowCount():
            return
        for col in range(table.columnCount()):
            current_widget = table.cellWidget(row, col)
            target_widget = table.cellWidget(target, col)
            table.setCellWidget(row, col, target_widget)
            table.setCellWidget(target, col, current_widget)
        table.setCurrentCell(target, 0)

    def _handle_role_add(self) -> None:
        self._add_role_row(self.available_groups[0] if self.available_groups else "", [])

    def _load_sequence_rows(self, rows: List[Dict[str, Any]]) -> None:
        self.sequence_table.setRowCount(0)
        for entry in rows:
            group = entry.get("group", "")
            roles = entry.get("roles") or []
            self._add_sequence_row(group, roles)

    def _load_role_order(self, mapping: Dict[str, List[str]]) -> None:
        self.role_order_table.setRowCount(0)
        for group, roles in mapping.items():
            self._add_role_row(group, roles)

    def _add_sequence_row(self, group: str, roles: Iterable[str]) -> None:
        row = self.sequence_table.rowCount()
        self.sequence_table.insertRow(row)
        group_combo = self._group_combo(group)
        available_roles = self._roles_for_group(group_combo.currentText())
        selected_roles = list(roles) if roles else available_roles[:]
        field = RoleSelectField(group_combo.currentText(), available_roles, selected_roles)

        def handle_group_change(value: str, selector: RoleSelectField = field) -> None:
            selector.set_group(value, self._roles_for_group(value))

        group_combo.currentTextChanged.connect(handle_group_change)
        self.sequence_table.setCellWidget(row, 0, group_combo)
        self.sequence_table.setCellWidget(row, 1, field)

    def _add_role_row(self, group: str, roles: Iterable[str]) -> None:
        row = self.role_order_table.rowCount()
        self.role_order_table.insertRow(row)
        group_combo = self._group_combo(group)
        available_roles = self._roles_for_group(group_combo.currentText())
        selected_roles = list(roles) if roles else available_roles[:]
        field = RoleSelectField(group_combo.currentText(), available_roles, selected_roles)

        def handle_group_change(value: str, selector: RoleSelectField = field) -> None:
            selector.set_group(value, self._roles_for_group(value))

        group_combo.currentTextChanged.connect(handle_group_change)
        self.role_order_table.setCellWidget(row, 0, group_combo)
        self.role_order_table.setCellWidget(row, 1, field)

    def _group_combo(self, value: str) -> QComboBox:
        combo = QComboBox()
        combo.setEditable(True)
        for name in self.available_groups:
            combo.addItem(name)
        if value and combo.findText(value) < 0:
            combo.addItem(value)
        if value:
            combo.setCurrentText(value)
        return combo


class PolicyComposerDialog(QDialog):
    def __init__(self, *, name: str = "", params: Optional[Dict[str, Any]] = None) -> None:
        super().__init__()
        self.setWindowTitle("Edit policy" if name else "Add policy")
        self.resize(1100, 720)
        self.result_data: Optional[Dict[str, Any]] = None
        self.policy_payload = self._initial_policy(params or {})
        self.role_models = self.policy_payload["roles"]
        self.current_role: Optional[str] = None
        self.role_group_inputs: Dict[str, Dict[str, Any]] = {}

        layout = QVBoxLayout(self)
        self.name_input = QLineEdit(name or self.policy_payload.get("name", "Default Policy"))
        name_form = QFormLayout()
        self.description_input = QLineEdit(self.policy_payload.get("description", ""))
        name_form.addRow("Name", self.name_input)
        name_form.addRow("Description", self.description_input)
        layout.addLayout(name_form)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_global_tab(), "Global rules")
        self.tabs.addTab(self._build_timeblocks_tab(), "Time blocks")
        self.tabs.addTab(self._build_roles_tab(), "Role coverage")
        layout.addWidget(self.tabs)

        self.feedback_label = QLabel()
        self.feedback_label.setStyleSheet(f"color:{ERROR_COLOR};")
        layout.addWidget(self.feedback_label)

        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        if ROLE_CATALOG:
            self.role_list.setCurrentRow(0)

    @staticmethod
    def _disable_scroll_wheel(widgets: List[Optional[QWidget]]) -> None:
        """Prevent accidental mouse-wheel changes on numeric/date inputs."""
        for widget in widgets:
            if widget is None:
                continue
            widget.setFocusPolicy(Qt.StrongFocus)
            if isinstance(widget, QAbstractSpinBox):
                widget.setButtonSymbols(QAbstractSpinBox.NoButtons)
            widget.wheelEvent = lambda event: event.ignore()

    def _initial_policy(self, params: Dict[str, Any]) -> Dict[str, Any]:
        timeblocks = _timeblocks_from_params(params)
        block_names = [row["name"] for row in timeblocks]
        default_shift_presets = build_default_policy().get("shift_presets", {})
        default_seasonal = build_default_policy().get("seasonal_settings", {})
        default_anchors = build_default_policy().get("anchors", {})
        default_role_groups = build_default_policy().get("role_groups", {})
        roles_payload: Dict[str, Any] = {}
        existing_roles = params.get("roles") if isinstance(params, dict) else {}
        if not isinstance(existing_roles, dict):
            existing_roles = {}
        for role in ROLE_CATALOG:
            raw = existing_roles.get(role)
            if isinstance(raw, dict):
                payload = raw.copy()
            else:
                payload = {}
            payload.setdefault("enabled", False)
            payload.setdefault("priority", 1.0)
            payload.setdefault("max_weekly_hours", 35)
            payload.setdefault("daily_boost", {})
            payload.setdefault("thresholds", [])
            if not isinstance(payload.get("covers"), list):
                payload["covers"] = []
            group_name = payload.get("group") or role_group(role)
            payload["group"] = group_name
            if "allow_cuts" not in payload:
                payload["allow_cuts"] = not ("bartend" in group_name.lower())
            if "always_on" not in payload:
                payload["always_on"] = "bartend" in group_name.lower()
            payload.setdefault("cut_buffer_minutes", 30)
            blocks = payload.get("blocks")
            if not isinstance(blocks, dict):
                blocks = {}
            for block in block_names:
                blocks.setdefault(
                    block,
                    {"base": 0, "min": 0, "max": 0, "per_1000_sales": 0.0, "per_modifier": 0.0},
                )
            for stale in [name for name in list(blocks.keys()) if name not in block_names]:
                blocks.pop(stale, None)
            payload["blocks"] = blocks
            roles_payload[role] = payload
        return {
            "name": params.get("name", "Default Policy"),
            "description": params.get("description", ""),
            "global": params.get("global")
            if isinstance(params.get("global"), dict)
                else {
                    "max_hours_week": 40,
                    "max_consecutive_days": 6,
                    "desired_hours_floor_pct": 0.85,
                    "desired_hours_ceiling_pct": 1.15,
                    "close_buffer_minutes": 35,
                    "labor_budget_pct": 0.27,
                    "labor_budget_tolerance_pct": 0.08,
                },
            "timeblocks": timeblocks,
            "business_hours": (
                params.get("business_hours")
                if isinstance(params.get("business_hours"), dict)
                else _default_business_hours()
            ),
            "roles": roles_payload,
            "role_groups": (
                params.get("role_groups")
                if isinstance(params.get("role_groups"), dict)
                else default_role_groups
            ),
            "shift_presets": (
                params.get("shift_presets")
                if isinstance(params.get("shift_presets"), dict)
                else default_shift_presets
            ),
            "section_capacity": (
                params.get("section_capacity")
                if isinstance(params.get("section_capacity"), dict)
                else default_section_capacity
            ),
            "seasonal_settings": (
                params.get("seasonal_settings")
                if isinstance(params.get("seasonal_settings"), dict)
                else default_seasonal
            ),
            "anchors": params.get("anchors") if isinstance(params.get("anchors"), dict) else default_anchors,
            "pre_engine": params.get("pre_engine") if isinstance(params.get("pre_engine"), dict) else pre_engine_settings(params),
        }

    def _build_global_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        global_cfg = self.policy_payload.get("global", {})
        spin_inputs: List[QAbstractSpinBox] = []

        self.max_hours_spin = QSpinBox()
        self.max_hours_spin.setRange(10, 80)
        self.max_hours_spin.setValue(int(global_cfg.get("max_hours_week", 40)))
        spin_inputs.append(self.max_hours_spin)

        self.max_consec_spin = QSpinBox()
        self.max_consec_spin.setRange(1, 7)
        self.max_consec_spin.setValue(int(global_cfg.get("max_consecutive_days", 6)))
        spin_inputs.append(self.max_consec_spin)

        self.labor_budget_spin = QDoubleSpinBox()
        self.labor_budget_spin.setRange(5.0, 60.0)
        self.labor_budget_spin.setDecimals(1)
        self.labor_budget_spin.setSuffix("%")
        labor_pct = float(global_cfg.get("labor_budget_pct", 0.27) or 0.0)
        if labor_pct <= 1:
            labor_pct *= 100
        self.labor_budget_spin.setValue(labor_pct)
        spin_inputs.append(self.labor_budget_spin)

        self.labor_tolerance_spin = QDoubleSpinBox()
        self.labor_tolerance_spin.setRange(0.0, 30.0)
        self.labor_tolerance_spin.setDecimals(1)
        self.labor_tolerance_spin.setSuffix("%")
        tolerance_pct = float(global_cfg.get("labor_budget_tolerance_pct", 0.08) or 0.0)
        if tolerance_pct <= 1:
            tolerance_pct *= 100
        self.labor_tolerance_spin.setValue(tolerance_pct)
        spin_inputs.append(self.labor_tolerance_spin)
        self.shift_template_editor = ShiftTemplateEditor(["Servers", "Kitchen", "Cashier"])
        self.shift_template_editor.set_config(self.policy_payload.get("shift_presets", {}))
        self.section_capacity_editor = SectionCapacityEditor({"Servers": ["Dining", "Patio", "Cocktail"]})
        self.section_capacity_editor.set_config(self.policy_payload.get("section_capacity", {}))

        seasonal_settings = self.policy_payload.get("seasonal_settings", {})
        self.patio_toggle = self._make_toggle_button(
            "Patio open (Patio ENABLED)",
            bool(seasonal_settings.get("server_patio_enabled", True)),
        )
        pre_engine_cfg = pre_engine_settings(self.policy_payload)
        fallback_cfg = pre_engine_cfg.get("fallback", {}) if isinstance(pre_engine_cfg, dict) else {}
        self.fallback_allow_mgr = self._make_toggle_button(
            "Allow emergency manager coverage",
            bool(fallback_cfg.get("allow_mgr_fallback", True)),
        )

        intro = QLabel("Set the rules the generator should follow. These values are intended for the GM and act like store-wide scheduling settings.")
        intro.setWordWrap(True)
        layout.addWidget(intro)

        guardrails_box = QGroupBox("Scheduling constraints")
        guard_form = QFormLayout(guardrails_box)
        guard_form.addRow("Max hours per week", self.max_hours_spin)
        guard_form.addRow("Max consecutive days", self.max_consec_spin)
        layout.addWidget(guardrails_box)

        hours_box = QGroupBox("Operating hours")
        hours_layout = QVBoxLayout(hours_box)
        hours_layout.addWidget(QLabel("Times accept HH:MM, and values above 24:00 keep closers after midnight (e.g., 25:00 = 1 AM next day)."))
        self.hours_table = QTableWidget(len(WEEKDAY_LABELS), 3)
        self.hours_table.setHorizontalHeaderLabels(["Day", "Open", "Close"])
        self.hours_table.verticalHeader().setVisible(False)
        self.hours_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.hours_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.hours_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        business_hours = self.policy_payload.get("business_hours") or _default_business_hours()
        for row, day in enumerate(WEEKDAY_LABELS):
            day_item = QTableWidgetItem(day)
            day_item.setFlags(Qt.ItemIsEnabled)
            entry = business_hours.get(day, {})
            open_item = QTableWidgetItem(entry.get("open", "11:00"))
            close_item = QTableWidgetItem(entry.get("close", "23:00"))
            self.hours_table.setItem(row, 0, day_item)
            self.hours_table.setItem(row, 1, open_item)
            self.hours_table.setItem(row, 2, close_item)
        hours_layout.addWidget(self.hours_table)
        layout.addWidget(hours_box)

        desired_box = QGroupBox("Target desired hours range")
        desired_form = QFormLayout(desired_box)
        desired_floor_pct = float(global_cfg.get("desired_hours_floor_pct", 0.85) or 0.0) * 100
        desired_ceiling_pct = float(global_cfg.get("desired_hours_ceiling_pct", 1.15) or 0.0) * 100
        self.desired_floor_spin = QDoubleSpinBox()
        self.desired_floor_spin.setDecimals(1)
        self.desired_floor_spin.setRange(0.0, 150.0)
        self.desired_floor_spin.setSuffix("%")
        self.desired_floor_spin.setValue(max(0.0, min(150.0, desired_floor_pct)))
        spin_inputs.append(self.desired_floor_spin)
        self.desired_ceiling_spin = QDoubleSpinBox()
        self.desired_ceiling_spin.setDecimals(1)
        self.desired_ceiling_spin.setRange(50.0, 250.0)
        self.desired_ceiling_spin.setSuffix("%")
        self.desired_ceiling_spin.setValue(max(self.desired_floor_spin.value(), min(250.0, desired_ceiling_pct)))
        spin_inputs.append(self.desired_ceiling_spin)
        self.desired_floor_spin.valueChanged.connect(self._sync_desired_range_bounds)
        self.desired_ceiling_spin.valueChanged.connect(self._sync_desired_range_bounds)
        desired_form.addRow("Minimum coverage (% of desired)", self.desired_floor_spin)
        desired_form.addRow("Maximum coverage (% of desired)", self.desired_ceiling_spin)
        desired_note = QLabel("Employees below the minimum are prioritized, and the generator avoids exceeding the maximum unless absolutely necessary.")
        desired_note.setWordWrap(True)
        desired_form.addRow(desired_note)
        layout.addWidget(desired_box)

        split_note = QLabel("Disable this if you only want one continuous shift per person each day.")
        split_note.setWordWrap(True)
        shift_box = QGroupBox("Shift behavior")
        shift_layout = QVBoxLayout(shift_box)
        buffer_form = QFormLayout()
        self.close_buffer_spin = QSpinBox()
        self.close_buffer_spin.setRange(0, 180)
        self.close_buffer_spin.setValue(int(global_cfg.get("close_buffer_minutes", 35)))
        spin_inputs.append(self.close_buffer_spin)
        buffer_form.addRow("Close buffer (minutes)", self.close_buffer_spin)
        shift_layout.addLayout(buffer_form)
        shift_layout.addWidget(split_note)
        layout.addWidget(shift_box)

        budget_box = QGroupBox("Labor budget")
        budget_form = QFormLayout(budget_box)
        budget_form.addRow("Labor budget (% of projected sales)", self.labor_budget_spin)
        budget_form.addRow("Allowed variance (±%)", self.labor_tolerance_spin)
        layout.addWidget(budget_box)

        labor_box = QGroupBox("Role group labor allocations")
        labor_form = QFormLayout(labor_box)
        groups_spec = self.policy_payload.get("role_groups") or build_default_policy().get("role_groups", {})
        for group_name in sorted(groups_spec.keys()):
            spec = groups_spec.get(group_name, {})
            row_widget = QWidget()
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(0, 0, 0, 0)
            pct_spin = QDoubleSpinBox()
            pct_spin.setRange(0.0, 100.0)
            pct_spin.setDecimals(1)
            pct_spin.setSuffix("%")
            allocation = spec.get("allocation_pct", 0.0)
            try:
                allocation = float(allocation)
            except (TypeError, ValueError):
                allocation = 0.0
            pct_spin.setValue(allocation * 100 if allocation <= 1 else allocation)
            allow_cuts_box = QCheckBox("Allow cuts")
            allow_cuts_box.setChecked(bool(spec.get("allow_cuts", True)))
            always_on_box = QCheckBox("Always staffed")
            always_on_box.setChecked(bool(spec.get("always_on", False)))
            cut_spin = QSpinBox()
            cut_spin.setRange(0, 180)
            cut_spin.setValue(int(spec.get("cut_buffer_minutes", 30) or 0))
            spin_inputs.extend([pct_spin, cut_spin])
            row_layout.addWidget(pct_spin)
            row_layout.addWidget(allow_cuts_box)
            row_layout.addWidget(always_on_box)
            row_layout.addWidget(QLabel("Cut buffer (min)"))
            row_layout.addWidget(cut_spin)
            row_layout.addStretch(1)
            labor_form.addRow(group_name, row_widget)
            self.role_group_inputs[group_name] = {
                "pct": pct_spin,
                "allow_cuts": allow_cuts_box,
                "always_on": always_on_box,
                "cut_buffer": cut_spin,
            }
        # Role group allocations and cut rotation are static; omit from UI.
        layout.addWidget(self.shift_template_editor)
        layout.addWidget(self.section_capacity_editor)
        layout.addWidget(seasonal_box)

        self._disable_scroll_wheel(spin_inputs)
        layout.addStretch(1)
        return widget

    def _build_timeblocks_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        self.block_table = QTableWidget(0, 3)
        self.block_table.setHorizontalHeaderLabels(["Name", "Start", "End"])
        self.block_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.block_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.block_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.block_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.block_table.setSelectionMode(QAbstractItemView.SingleSelection)
        layout.addWidget(self.block_table)
        controls = QHBoxLayout()
        add_btn = QPushButton("Add block")
        remove_btn = QPushButton("Remove block")
        add_btn.clicked.connect(self._add_block_row)
        remove_btn.clicked.connect(self._remove_block_row)
        controls.addWidget(add_btn)
        controls.addWidget(remove_btn)
        controls.addStretch()
        layout.addLayout(controls)

        help_label = QLabel(
            "Use HH:MM or anchors like @open, @open-30, @close, @close+35. Anchors resolve per day using the Operating hours on the Global rules tab."
        )
        help_label.setWordWrap(True)
        help_label.setStyleSheet(f"color:{INFO_COLOR};")
        layout.addWidget(help_label)

        for row in self.policy_payload["timeblocks"]:
            self._append_block_row(row["name"], row["start"], row["end"])
        self.block_table.itemChanged.connect(self._handle_block_edit)
        return widget

    def _build_roles_tab(self) -> QWidget:
        widget = QWidget()
        layout = QHBoxLayout(widget)
        self.role_list = QListWidget()
        for role in ROLE_CATALOG:
            item = QListWidgetItem(role)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsSelectable)
            item.setCheckState(Qt.Checked if self.role_models.get(role, {}).get("enabled") else Qt.Unchecked)
            self.role_list.addItem(item)
        self.role_list.currentItemChanged.connect(self._handle_role_selection)
        self.role_list.itemChanged.connect(self._handle_role_check_changed)

        self.role_detail = QWidget()
        detail_layout = QVBoxLayout(self.role_detail)
        self.role_enabled_checkbox = QCheckBox("Role enabled")
        self.role_enabled_checkbox.stateChanged.connect(self._handle_role_enabled_toggle)
        detail_layout.addWidget(self.role_enabled_checkbox)

        form = QFormLayout()
        self.priority_spin = QDoubleSpinBox()
        self.priority_spin.setRange(0.1, 10.0)
        self.priority_spin.setSingleStep(0.1)
        spin_inputs: List[QAbstractSpinBox] = [self.priority_spin]
        self.max_weekly_spin = QSpinBox()
        self.max_weekly_spin.setRange(5, 80)
        spin_inputs.append(self.max_weekly_spin)
        form.addRow("Priority weight", self.priority_spin)
        form.addRow("Max weekly hours", self.max_weekly_spin)
        detail_layout.addLayout(form)

        block_box = QGroupBox("Block staffing")
        block_layout = QVBoxLayout(block_box)
        self.role_block_table = QTableWidget(0, 6)
        self.role_block_table.setHorizontalHeaderLabels(
            ["Block", "Target staff", "Min staff", "Max staff", "Extra per $1k sales", "Extra per modifier"]
        )
        self.role_block_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.role_block_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.role_block_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.role_block_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.role_block_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self.role_block_table.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeToContents)
        block_layout.addWidget(self.role_block_table)
        block_help = QLabel(
            "Target staff is the number of people you want assigned before auto-scaling kicks in. "
            "Use the extra columns when you want busy days (higher sales/modifiers) to add staff automatically."
        )
        block_help.setWordWrap(True)
        block_help.setStyleSheet(f"color:{INFO_COLOR};")
        block_layout.addWidget(block_help)
        detail_layout.addWidget(block_box)

        threshold_box = QGroupBox("Demand thresholds")
        threshold_layout = QVBoxLayout(threshold_box)
        threshold_layout.addWidget(
            QLabel("Optional rules that add staff when demand metrics exceed the provided thresholds.")
        )
        self.threshold_table = QTableWidget(0, 3)
        self.threshold_table.setHorizontalHeaderLabels(["Metric", "≥ value", "Add staff"])
        self.threshold_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.threshold_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.threshold_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        threshold_layout.addWidget(self.threshold_table)
        threshold_controls = QHBoxLayout()
        threshold_add = QPushButton("Add rule")
        threshold_remove = QPushButton("Remove rule")
        threshold_add.clicked.connect(self._add_threshold_row)
        threshold_remove.clicked.connect(self._remove_threshold_row)
        threshold_controls.addWidget(threshold_add)
        threshold_controls.addWidget(threshold_remove)
        threshold_controls.addStretch()
        threshold_layout.addLayout(threshold_controls)
        detail_layout.addWidget(threshold_box)

        daily_box = QGroupBox("Daily boost (extra staff per day)")
        daily_layout = QGridLayout(daily_box)
        self.daily_spinboxes: Dict[str, QSpinBox] = {}
        for idx, day in enumerate(WEEKDAY_LABELS):
            spin = QSpinBox()
            spin.setRange(-5, 10)
            self.daily_spinboxes[day] = spin
            spin_inputs.append(spin)
            daily_layout.addWidget(QLabel(day), 0, idx)
            daily_layout.addWidget(spin, 1, idx)
        detail_layout.addWidget(daily_box)

        cover_box = QGroupBox("Can cover these roles when short")
        cover_layout = QVBoxLayout(cover_box)
        cover_hint = QLabel("Checked roles become last-priority fallbacks when no dedicated staff are available.")
        cover_hint.setWordWrap(True)
        cover_layout.addWidget(cover_hint)
        self.cover_roles_list = QListWidget()
        cover_layout.addWidget(self.cover_roles_list)
        detail_layout.addWidget(cover_box)

        detail_layout.addStretch()

        layout.addWidget(self.role_list, 1)
        layout.addWidget(self.role_detail, 3)
        self._disable_scroll_wheel(spin_inputs)
        return widget

    def _add_block_row(self) -> None:
        self._append_block_row("New Block", "09:00", "13:00")
        self._sync_role_blocks()

    def _remove_block_row(self) -> None:
        row = self.block_table.currentRow()
        if row < 0:
            return
        self.block_table.blockSignals(True)
        self.block_table.removeRow(row)
        self.block_table.blockSignals(False)
        self._sync_role_blocks()
        self._populate_role_block_table()

    def _add_threshold_row(self) -> None:
        row = self.threshold_table.rowCount()
        self.threshold_table.insertRow(row)
        self.threshold_table.setItem(row, 0, QTableWidgetItem("demand_index"))
        self.threshold_table.setItem(row, 1, QTableWidgetItem("0.80"))
        self.threshold_table.setItem(row, 2, QTableWidgetItem("1"))

    def _remove_threshold_row(self) -> None:
        row = self.threshold_table.currentRow()
        if row < 0:
            return
        self.threshold_table.removeRow(row)

    def _append_block_row(self, name: str, start: str, end: str) -> None:
        self.block_table.blockSignals(True)
        row = self.block_table.rowCount()
        self.block_table.insertRow(row)
        for col, value in enumerate([name, start, end]):
            item = QTableWidgetItem(value)
            self.block_table.setItem(row, col, item)
        self.block_table.blockSignals(False)

    def _handle_block_edit(self, _) -> None:
        self._sync_role_blocks()
        self._populate_role_block_table()

    def _sync_desired_range_bounds(self) -> None:
        if not hasattr(self, "desired_floor_spin") or not hasattr(self, "desired_ceiling_spin"):
            return
        floor = self.desired_floor_spin.value()
        if self.desired_ceiling_spin.value() < floor:
            self.desired_ceiling_spin.setValue(floor)

    def _handle_role_selection(self, current: Optional[QListWidgetItem], previous: Optional[QListWidgetItem]) -> None:
        self._persist_role_detail()
        if not current:
            self.current_role = None
            return
        self.current_role = current.text()
        self._load_role_detail()

    def _handle_role_check_changed(self, item: QListWidgetItem) -> None:
        role = item.text()
        if role in self.role_models:
            self.role_models[role]["enabled"] = item.checkState() == Qt.Checked
        if self.current_role == role:
            self.role_enabled_checkbox.setChecked(self.role_models[role]["enabled"])

    def _handle_role_enabled_toggle(self) -> None:
        if not self.current_role:
            return
        state = self.role_enabled_checkbox.isChecked()
        self.role_models[self.current_role]["enabled"] = state
        items = self.role_list.findItems(self.current_role, Qt.MatchExactly)
        for item in items:
            item.setCheckState(Qt.Checked if state else Qt.Unchecked)

    def _load_role_detail(self) -> None:
        if not self.current_role:
            return
        data = self.role_models.get(self.current_role, _default_role_payload(self._block_names()))
        self.role_enabled_checkbox.setChecked(data.get("enabled", False))
        self.priority_spin.setValue(float(data.get("priority", 1.0)))
        self.max_weekly_spin.setValue(int(data.get("max_weekly_hours", 35)))

        boosts = data.get("daily_boost", {}) or {}
        for day, spin in self.daily_spinboxes.items():
            spin.blockSignals(True)
            spin.setValue(int(boosts.get(day, 0)))
            spin.blockSignals(False)
        self._populate_role_block_table()
        self._populate_threshold_table()
        covers = data.get("covers") or []
        self._populate_cover_roles_list(covers)

    def _populate_role_block_table(self) -> None:
        if not self.current_role:
            self.role_block_table.setRowCount(0)
            return
        data = self.role_models.setdefault(self.current_role, _default_role_payload(self._block_names()))
        blocks = data.setdefault("blocks", {})
        block_names = self._block_names()
        self.role_block_table.blockSignals(True)
        self.role_block_table.setRowCount(len(block_names))
        for row, block_name in enumerate(block_names):
            config = blocks.setdefault(
                block_name,
                {"base": 0, "min": 0, "max": 0, "per_1000_sales": 0.0, "per_modifier": 0.0},
            )
            name_item = QTableWidgetItem(block_name)
            name_item.setFlags(Qt.ItemIsEnabled)
            self.role_block_table.setItem(row, 0, name_item)
            self.role_block_table.setItem(row, 1, QTableWidgetItem(str(int(config.get("base", 0)))))
            self.role_block_table.setItem(row, 2, QTableWidgetItem(str(int(config.get("min", config.get("base", 0))))))
            self.role_block_table.setItem(row, 3, QTableWidgetItem(str(int(config.get("max", config.get("base", 0))))))
            self.role_block_table.setItem(
                row,
                4,
                QTableWidgetItem(f"{float(config.get('per_1000_sales', 0.0)):.2f}"),
            )
            self.role_block_table.setItem(
                row,
                5,
                QTableWidgetItem(f"{float(config.get('per_modifier', 0.0)):.2f}"),
            )
        self.role_block_table.blockSignals(False)

    def _populate_threshold_table(self) -> None:
        if not self.current_role:
            self.threshold_table.setRowCount(0)
            return
        data = self.role_models.setdefault(self.current_role, _default_role_payload(self._block_names()))
        rules = data.get("thresholds") or []
        self.threshold_table.blockSignals(True)
        self.threshold_table.setRowCount(len(rules))
        for row, rule in enumerate(rules):
            metric = (rule.get("metric") or "demand_index") if isinstance(rule, dict) else "demand_index"
            gte = rule.get("gte", 0.0) if isinstance(rule, dict) else 0.0
            add = rule.get("add", 0) if isinstance(rule, dict) else 0
            self.threshold_table.setItem(row, 0, QTableWidgetItem(str(metric)))
            self.threshold_table.setItem(row, 1, QTableWidgetItem(f"{float(gte):.2f}"))
            self.threshold_table.setItem(row, 2, QTableWidgetItem(str(int(add))))
        self.threshold_table.blockSignals(False)

    def _populate_cover_roles_list(self, covers: List[str]) -> None:
        self.cover_roles_list.blockSignals(True)
        self.cover_roles_list.clear()
        for role in ROLE_CATALOG:
            if role == self.current_role:
                continue
            item = QListWidgetItem(role)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if role in covers else Qt.Unchecked)
            self.cover_roles_list.addItem(item)
        self.cover_roles_list.blockSignals(False)

    def _read_threshold_rows(self) -> List[Dict[str, float | int | str]]:
        rows: List[Dict[str, float | int | str]] = []
        for row in range(self.threshold_table.rowCount()):
            metric = (self.threshold_table.item(row, 0).text() if self.threshold_table.item(row, 0) else "").strip()
            if not metric:
                metric = "demand_index"
            gte = self._parse_table_float(self.threshold_table.item(row, 1))
            add = self._parse_table_int(self.threshold_table.item(row, 2))
            rows.append({"metric": metric, "gte": gte, "add": add})
        return rows

    def _persist_role_detail(self) -> None:
        if not self.current_role:
            return
        data = self.role_models.setdefault(self.current_role, _default_role_payload(self._block_names()))
        data["enabled"] = self.role_enabled_checkbox.isChecked()
        data["priority"] = float(self.priority_spin.value())
        data["max_weekly_hours"] = int(self.max_weekly_spin.value())
        boosts = {}
        for day, spin in self.daily_spinboxes.items():
            if spin.value() != 0:
                boosts[day] = spin.value()
        data["daily_boost"] = boosts
        blocks = data.setdefault("blocks", {})
        block_names = self._block_names()
        for row, block_name in enumerate(block_names):
            base = self._parse_table_int(self.role_block_table.item(row, 1))
            min_staff = self._parse_table_int(self.role_block_table.item(row, 2))
            max_staff = self._parse_table_int(self.role_block_table.item(row, 3))
            per_sales = self._parse_table_float(self.role_block_table.item(row, 4))
            per_modifier = self._parse_table_float(self.role_block_table.item(row, 5))
            blocks[block_name] = {
                "base": base,
                "min": min_staff if min_staff else base,
                "max": max_staff if max_staff else max(base, min_staff),
                "per_1000_sales": per_sales,
                "per_modifier": per_modifier,
            }
        for stale in [name for name in list(blocks.keys()) if name not in block_names]:
            blocks.pop(stale, None)
        data["thresholds"] = self._read_threshold_rows()
        covers: List[str] = []
        for idx in range(self.cover_roles_list.count()):
            item = self.cover_roles_list.item(idx)
            if item.checkState() == Qt.Checked:
                covers.append(item.text())
        data["covers"] = covers

    def _parse_table_int(self, item: Optional[QTableWidgetItem]) -> int:
        if not item:
            return 0
        try:
            return int(item.text())
        except (TypeError, ValueError):
            return 0

    def _parse_table_float(self, item: Optional[QTableWidgetItem]) -> float:
        if not item:
            return 0.0
        try:
            return float(item.text())
        except (TypeError, ValueError):
            return 0.0

    def _read_timeblocks(self) -> List[Dict[str, str]]:
        rows: List[Dict[str, str]] = []
        for row in range(self.block_table.rowCount()):
            name = (self.block_table.item(row, 0).text() if self.block_table.item(row, 0) else "").strip()
            if not name:
                continue
            start = (self.block_table.item(row, 1).text() if self.block_table.item(row, 1) else "09:00").strip()
            end = (self.block_table.item(row, 2).text() if self.block_table.item(row, 2) else "17:00").strip()
            rows.append({"name": name, "start": start, "end": end})
        return rows or _default_timeblocks()

    def _block_names(self) -> List[str]:
        return [row["name"] for row in self._read_timeblocks()]

    def _read_business_hours(self) -> Dict[str, Dict[str, str]]:
        hours: Dict[str, Dict[str, str]] = {}
        for row, day in enumerate(WEEKDAY_LABELS):
            open_item = self.hours_table.item(row, 1)
            close_item = self.hours_table.item(row, 2)
            open_label = (open_item.text() if open_item else "11:00").strip() or "11:00"
            close_label = (close_item.text() if close_item else "23:00").strip() or "23:00"
            hours[day] = {"open": open_label, "close": close_label}
        return hours

    def _sync_role_blocks(self) -> None:
        block_names = self._block_names()
        for role in ROLE_CATALOG:
            data = self.role_models.setdefault(role, _default_role_payload(block_names))
            blocks = data.setdefault("blocks", {})
            for block in block_names:
                blocks.setdefault(
                    block,
                    {"base": 0, "min": 0, "max": 0, "per_1000_sales": 0.0, "per_modifier": 0.0},
                )
            for stale in [name for name in list(blocks.keys()) if name not in block_names]:
                blocks.pop(stale, None)

    def accept(self) -> None:
        name = self.name_input.text().strip()
        if not name:
            self.feedback_label.setText("Provide a policy name.")
            return
        self._persist_role_detail()
        self.policy_payload["name"] = name
        self.policy_payload["description"] = self.description_input.text().strip()
        floor_pct = round(self.desired_floor_spin.value() / 100, 3)
        ceil_pct = round(self.desired_ceiling_spin.value() / 100, 3)
        self.policy_payload["global"] = {
            "max_hours_week": self.max_hours_spin.value(),
            "max_consecutive_days": self.max_consec_spin.value(),
            "desired_hours_floor_pct": floor_pct,
            "desired_hours_ceiling_pct": ceil_pct,
            "close_buffer_minutes": self.close_buffer_spin.value(),
            "labor_budget_pct": round(self.labor_budget_spin.value() / 100, 4),
            "labor_budget_tolerance_pct": round(self.labor_tolerance_spin.value() / 100, 4),
        }
        timeblock_rows = self._read_timeblocks()
        self.policy_payload["timeblocks"] = [
            {"name": row["name"], "start": row["start"], "end": row["end"]} for row in timeblock_rows
        ]
        self.policy_payload["business_hours"] = self._read_business_hours()
        role_groups_payload: Dict[str, Dict[str, Any]] = {}
        for group_name, widgets in self.role_group_inputs.items():
            pct_value = max(0.0, widgets["pct"].value())
            allocation = pct_value / 100.0
            role_groups_payload[group_name] = {
                "allocation_pct": round(allocation, 4),
                "allow_cuts": widgets["allow_cuts"].isChecked(),
                "always_on": widgets["always_on"].isChecked(),
                "cut_buffer_minutes": widgets["cut_buffer"].value(),
            }
        if role_groups_payload:
            self.policy_payload["role_groups"] = role_groups_payload
        anchors_payload = self.policy_data.get("anchors", build_default_policy().get("anchors", {})).copy()
        self.policy_payload["anchors"] = anchors_payload
        self.policy_payload["shift_presets"] = self.shift_template_editor.value()
        seasonal_payload = {"server_patio_enabled": self.patio_toggle.isChecked()}
        self.policy_payload["seasonal_settings"] = seasonal_payload
        patio_role = self.role_models.get("Server - Patio")
        if patio_role is not None:
            patio_role["enabled"] = bool(seasonal_payload["server_patio_enabled"])
        self.policy_payload["section_capacity"] = self.section_capacity_editor.value()
        allow_mgr_fallback = bool(self.fallback_allow_mgr.isChecked()) if hasattr(self, "fallback_allow_mgr") else True
        pre_engine_payload = pre_engine_settings({**self.policy_payload, "allow_mgr_fallback": allow_mgr_fallback})
        if isinstance(pre_engine_payload, dict):
            limits = resolve_fallback_limits({"allow_mgr_fallback": allow_mgr_fallback})
            fallback_payload = pre_engine_payload.setdefault("fallback", {})
            if isinstance(fallback_payload, dict):
                fallback_payload["allow_mgr_fallback"] = allow_mgr_fallback
                fallback_payload["am_limit"] = limits.get("am", 1)
                fallback_payload["pm_limit"] = limits.get("pm", 1)
        params = {
            "description": self.policy_payload.get("description", ""),
            "allow_mgr_fallback": allow_mgr_fallback,
            "global": self.policy_payload["global"],
            "timeblocks": {
                row["name"]: {"start": row["start"], "end": row["end"]} for row in self.policy_payload["timeblocks"]
            },
            "business_hours": self.policy_payload["business_hours"],
            "roles": self.role_models,
            "role_groups": self.policy_payload.get("role_groups", {}),
            "shift_presets": self.policy_payload.get("shift_presets", {}),
            "section_capacity": self.policy_payload.get("section_capacity", {}),
            "anchors": self.policy_payload.get("anchors", {}),
            "pre_engine": pre_engine_payload,
        }
        self.result_data = {"name": name, "params": params}
        super().accept()


class PolicyDialog(QDialog):
    def __init__(self, session_factory, current_user: Dict[str, str], *, read_only: bool = False) -> None:
        super().__init__()
        self.session_factory = session_factory
        self.current_user = current_user
        self.read_only = read_only
        self.setWindowTitle("Policy settings")
        self.resize(1100, 720)
        self.policy: Optional[Policy] = None
        self.policy_data: Dict[str, Any] = {}
        self.role_group_widgets: Dict[str, Dict[str, QWidget]] = {}
        self._build_ui()
        self._load_policy()

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        layout = QVBoxLayout(content)

        name_form = QFormLayout()
        self.name_input = QLineEdit()
        self.description_input = QLineEdit()
        name_form.addRow("Policy name", self.name_input)
        name_form.addRow("Description", self.description_input)
        layout.addLayout(name_form)

        global_box = QGroupBox("Scheduling constraints")
        global_form = QFormLayout(global_box)
        self.max_hours_spin = QSpinBox()
        self.max_hours_spin.setRange(10, 80)
        self.max_consec_spin = QSpinBox()
        self.max_consec_spin.setRange(1, 7)
        self.desired_floor_spin = QDoubleSpinBox()
        self.desired_floor_spin.setDecimals(1)
        self.desired_floor_spin.setSuffix("%")
        self.desired_floor_spin.setRange(0.0, 150.0)
        self.desired_ceiling_spin = QDoubleSpinBox()
        self.desired_ceiling_spin.setDecimals(1)
        self.desired_ceiling_spin.setSuffix("%")
        self.desired_ceiling_spin.setRange(50.0, 250.0)
        self.desired_floor_spin.valueChanged.connect(self._sync_desired_range_bounds)
        self.desired_ceiling_spin.valueChanged.connect(self._sync_desired_range_bounds)
        self.close_buffer_spin = QSpinBox()
        self.close_buffer_spin.setRange(0, 240)
        global_form.addRow("Max hours per week", self.max_hours_spin)
        global_form.addRow("Max consecutive days", self.max_consec_spin)
        global_form.addRow("Desired hours min%", self.desired_floor_spin)
        global_form.addRow("Desired hours max%", self.desired_ceiling_spin)
        global_form.addRow("Close buffer (minutes)", self.close_buffer_spin)
        layout.addWidget(global_box)

        labor_box = QGroupBox()
        labor_box.setTitle("")
        labor_form = QFormLayout(labor_box)
        self.labor_budget_spin = QDoubleSpinBox()
        self.labor_budget_spin.setRange(5.0, 60.0)
        self.labor_budget_spin.setDecimals(1)
        self.labor_budget_spin.setSuffix("%")
        self.labor_tolerance_spin = QDoubleSpinBox()
        self.labor_tolerance_spin.setRange(0.0, 30.0)
        self.labor_tolerance_spin.setDecimals(1)
        self.labor_tolerance_spin.setSuffix("%")
        labor_form.addRow("Labor budget (% of projected sales)", self.labor_budget_spin)
        labor_form.addRow("Allowed variance (±%)", self.labor_tolerance_spin)
        layout.addWidget(labor_box)

        time_box = QGroupBox()
        time_box.setTitle("")
        time_layout = QVBoxLayout(time_box)
        hours_note = QLabel("Business hours rarely change. GM-only edits; buffers apply automatically.")
        hours_note.setWordWrap(True)
        time_layout.addWidget(hours_note)
        self.hours_table = QTableWidget(len(WEEKDAY_LABELS), 4)
        self.hours_table.setHorizontalHeaderLabels(["Day", "Opens at", "AM end", "Closes at"])
        self.hours_table.verticalHeader().setVisible(False)
        self.hours_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.hours_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.hours_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.hours_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        business_hours = self.policy_data.get("business_hours") or _default_business_hours()
        timeblocks = self.policy_data.get("timeblocks") or {}
        pm_block = timeblocks.get("PM") or timeblocks.get("Mid") or {"start": "16:00"}
        for row, day in enumerate(WEEKDAY_LABELS):
            day_item = QTableWidgetItem(day)
            day_item.setFlags(Qt.ItemIsEnabled)
            entry = business_hours.get(day, {})
            open_item = QTableWidgetItem(entry.get("open", "11:00"))
            close_item = QTableWidgetItem(entry.get("close", "23:00"))
            am_end = pm_block.get("start", "16:00")
            am_item = QTableWidgetItem(am_end)
            self.hours_table.setItem(row, 0, day_item)
            self.hours_table.setItem(row, 1, open_item)
            self.hours_table.setItem(row, 2, am_item)
            self.hours_table.setItem(row, 3, close_item)
        time_layout.addWidget(self.hours_table)
        layout.addWidget(self._make_collapsible("Business Hours (caution)", time_box))

        groups_box = QGroupBox("Role group allocations")
        groups_layout = QVBoxLayout(groups_box)
        self.role_group_table = QTableWidget(len(ROLE_GROUPS), 2)
        self.role_group_table.setHorizontalHeaderLabels(["Group", "Allocation %"])
        self.role_group_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.role_group_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.role_group_table.verticalHeader().setVisible(False)
        groups_layout.addWidget(self.role_group_table)
        layout.addWidget(groups_box)

        self.shift_template_editor = ShiftTemplateEditor(["Servers", "Kitchen", "Cashier"])
        layout.addWidget(self.shift_template_editor)
        self.section_capacity_editor = SectionCapacityEditor({"Servers": ["Dining", "Patio", "Cocktail"]})
        self.section_capacity_editor.setVisible(False)
        layout.addWidget(self.section_capacity_editor)
        self._build_pre_engine_section(layout)
        self.migration_notice = QLabel("Some deprecated settings were removed or converted automatically.")
        self.migration_notice.setStyleSheet(f"color:{INFO_COLOR};")
        self.migration_notice.setWordWrap(True)
        self.migration_notice.setVisible(False)
        layout.addWidget(self.migration_notice)
        button_row = QHBoxLayout()
        button_row.setSpacing(12)
        button_row.addStretch(1)
        seasonal_settings = self.policy_data.get("seasonal_settings", {})
        self.patio_toggle = self._make_toggle_button(
            "Patio open (Toggle on/off)",
            bool(seasonal_settings.get("server_patio_enabled", True)),
        )
        self.patio_toggle.setMinimumWidth(220)
        self.patio_toggle.setMaximumWidth(320)
        button_row.addWidget(self.patio_toggle)
        self.fallback_allow_mgr = self._make_toggle_button(
            "Allow emergency manager coverage",
            bool(self.policy_data.get("allow_mgr_fallback", True)),
        )
        self.fallback_allow_mgr.setMinimumWidth(220)
        self.fallback_allow_mgr.setMaximumWidth(320)
        button_row.addWidget(self.fallback_allow_mgr)
        button_row.addStretch(1)
        layout.addLayout(button_row)

        self.feedback_label = QLabel()
        self.feedback_label.setStyleSheet(f"color:{INFO_COLOR};")
        layout.addWidget(self.feedback_label)

        buttons = QHBoxLayout()
        self.save_button = QPushButton("Save policy")
        self.save_button.clicked.connect(self._save_policy)
        buttons.addWidget(self.save_button)
        self.export_button = QPushButton("Export…")
        self.export_button.clicked.connect(self._export_policy)
        buttons.addWidget(self.export_button)
        self.import_button = QPushButton("Import…")
        self.import_button.clicked.connect(self._import_policy)
        buttons.addWidget(self.import_button)
        buttons.addStretch()
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.reject)
        buttons.addWidget(close_button)
        layout.addLayout(buttons)

        scroll.setWidget(content)
        outer.addWidget(scroll)

        self._disable_scroll_wheel(
            [
                self.max_hours_spin,
                self.max_consec_spin,
                self.desired_floor_spin,
                self.desired_ceiling_spin,
                self.close_buffer_spin,
                self.labor_budget_spin,
                self.labor_tolerance_spin,
                self.week_picker if hasattr(self, "week_picker") else None,
            ]
        )

        if self.read_only:
            for widget in [
                self.name_input,
                self.description_input,
                self.max_hours_spin,
                self.max_consec_spin,
                self.desired_floor_spin,
                self.desired_ceiling_spin,
                self.close_buffer_spin,
                self.labor_budget_spin,
                self.labor_tolerance_spin,
                self.role_group_table,
                self.hours_table,
                self.dining_slow_min_spin,
                self.dining_slow_max_spin,
                self.dining_moderate_spin,
                self.dining_peak_spin,
                self.cocktail_normal_spin,
                self.cocktail_busy_spin,
                self.cocktail_peak_spin,
                self.cashier_am_spin,
                self.cashier_pm_spin,
                self.cashier_busy_spin,
                self.cashier_peak_spin,
                self.fallback_allow_mgr,
            ]:
                widget.setEnabled(False)
            self.save_button.setEnabled(False)
            self.export_button.setEnabled(False)
            self.import_button.setEnabled(False)
            self.shift_template_editor.setEnabled(False)
            self.section_capacity_editor.setEnabled(False)

    def _load_policy(self) -> None:
        with self.session_factory() as session:
            self.policy = get_active_policy(session)
        if self.policy:
            self.policy_data = self.policy.params_dict()
            self.policy_data["name"] = self.policy.name
            self.policy_data["description"] = getattr(self.policy, "description", self.policy_data.get("description", ""))
        else:
            self.policy_data = build_default_policy()
        self._ensure_policy_defaults()
        self._apply_policy_to_fields()

    def _apply_policy_to_fields(self) -> None:
        self.name_input.setText(self.policy_data.get("name", "Store policy"))
        self.description_input.setText(self.policy_data.get("description", ""))
        global_cfg = self.policy_data.get("global", {})
        self.max_hours_spin.setValue(int(global_cfg.get("max_hours_week", 40)))
        self.max_consec_spin.setValue(int(global_cfg.get("max_consecutive_days", 6)))
        floor = float(global_cfg.get("desired_hours_floor_pct", 0.85) or 0.0) * 100
        ceil = float(global_cfg.get("desired_hours_ceiling_pct", 1.15) or 0.0) * 100
        self.desired_floor_spin.setValue(floor)
        self.desired_ceiling_spin.setValue(max(self.desired_floor_spin.value(), ceil))
        self.close_buffer_spin.setValue(int(global_cfg.get("close_buffer_minutes", 35)))
        self.shift_template_editor.set_config(self.policy_data.get("shift_presets", {}))
        self.section_capacity_editor.set_config(self.policy_data.get("section_capacity", {}))
        labor_pct = float(global_cfg.get("labor_budget_pct", 0.27) or 0.0)
        if labor_pct <= 1:
            labor_pct *= 100
        self.labor_budget_spin.setValue(labor_pct)
        labor_tol = float(global_cfg.get("labor_budget_tolerance_pct", 0.08) or 0.0)
        if labor_tol <= 1:
            labor_tol *= 100
        self.labor_tolerance_spin.setValue(labor_tol)
        self._load_hours_table()
        self._populate_role_groups()
        self._apply_pre_engine_values()
        self._populate_role_groups()
        # Seasonal / fallback toggles reflect loaded policy.
        seasonal_settings = self.policy_data.get("seasonal_settings", {})
        self.patio_toggle.setChecked(bool(seasonal_settings.get("server_patio_enabled", True)))
        pre_engine_cfg = pre_engine_settings(self.policy_data)
        fallback_cfg = pre_engine_cfg.get("fallback", {}) if isinstance(pre_engine_cfg, dict) else {}
        allow_mgr_fallback = bool(fallback_cfg.get("allow_mgr_fallback", True))
        self.policy_data["allow_mgr_fallback"] = allow_mgr_fallback
        self.fallback_allow_mgr.setChecked(allow_mgr_fallback)

    def _sync_desired_range_bounds(self) -> None:
        if self.desired_ceiling_spin.value() < self.desired_floor_spin.value():
            self.desired_ceiling_spin.setValue(self.desired_floor_spin.value())

    def _load_hours_table(self) -> None:
        hours = self.policy_data.get("business_hours") or _default_business_hours()
        for row, day in enumerate(WEEKDAY_LABELS):
            entry = hours.get(day, {})
            open_value = entry.get("open", "11:00")
            mid_value = entry.get("mid", entry.get("close", "16:00"))
            close_value = entry.get("close", "23:00")
            day_item = QTableWidgetItem(day)
            day_item.setFlags(Qt.ItemIsEnabled)
            self.hours_table.setItem(row, 0, day_item)
            self.hours_table.setItem(row, 1, QTableWidgetItem(open_value))
            self.hours_table.setItem(row, 2, QTableWidgetItem(mid_value))
            self.hours_table.setItem(row, 3, QTableWidgetItem(close_value))

    def _read_hours_table(self) -> Dict[str, Dict[str, str]]:
        hours: Dict[str, Dict[str, str]] = {}
        for row, day in enumerate(WEEKDAY_LABELS):
            open_item = self.hours_table.item(row, 1)
            mid_item = self.hours_table.item(row, 2)
            close_item = self.hours_table.item(row, 3)
            open_label = (open_item.text() if open_item else "11:00").strip() or "11:00"
            mid_label = (mid_item.text() if mid_item else "16:00").strip() or "16:00"
            close_label = (close_item.text() if close_item else "23:00").strip() or "23:00"
            hours[day] = {"open": open_label, "mid": mid_label, "close": close_label}
        return hours

    def _apply_pre_engine_values(self) -> None:
        cfg = pre_engine_settings(self.policy_data)
        staffing = cfg.get("staffing", {})
        server_cfg = staffing.get("servers", {})
        dining_cfg = server_cfg.get("dining", {})
        cocktail_cfg = server_cfg.get("cocktail", {})
        cashier_cfg = staffing.get("cashier", {})
        hoh_cfg = staffing.get("hoh", {})
        fallback_cfg = cfg.get("fallback", {})
        budget_cfg = cfg.get("budget", {})
        self.dining_slow_min_spin.setValue(int(dining_cfg.get("slow_min", 1)))
        self.dining_slow_max_spin.setValue(int(dining_cfg.get("slow_max", 4)))
        self.dining_moderate_spin.setValue(int(dining_cfg.get("moderate", 5)))
        self.dining_peak_spin.setValue(int(dining_cfg.get("peak", 6)))
        self.cocktail_normal_spin.setValue(int(cocktail_cfg.get("normal", 2)))
        self.cocktail_busy_spin.setValue(int(cocktail_cfg.get("busy", 3)))
        self.cocktail_peak_spin.setValue(int(cocktail_cfg.get("peak", 4)))
        self.cashier_am_spin.setValue(int(cashier_cfg.get("am_default", 1)))
        self.cashier_pm_spin.setValue(int(cashier_cfg.get("pm_default", 1)))
        self.cashier_busy_spin.setValue(int(cashier_cfg.get("busy_split", 2)))
        self.cashier_peak_spin.setValue(int(cashier_cfg.get("peak", 3)))
        mode_value = (self.policy_data.get("hoh_mode") or "auto").lower()
        idx = self.hoh_mode_combo.findData(mode_value if mode_value in {"auto", "combo", "split", "peak"} else "auto")
        if idx >= 0:
            self.hoh_mode_combo.setCurrentIndex(idx)
        else:
            self.hoh_mode_combo.setCurrentIndex(0)

    def _read_pre_engine_controls(self) -> Dict[str, Any]:
        cfg = pre_engine_settings(self.policy_data)
        staffing = cfg.get("staffing", {})
        budget_cfg_raw = cfg.get("budget")
        budget_cfg = budget_cfg_raw if isinstance(budget_cfg_raw, dict) else {}
        payload = copy.deepcopy(cfg)
        payload.setdefault("staffing", {})
        dining_cfg = staffing.get("servers", {}).get("dining", {})
        cocktail_cfg = staffing.get("servers", {}).get("cocktail", {})
        payload["staffing"]["servers"] = {
            "dining": {
                "slow_min": self.dining_slow_min_spin.value(),
                "slow_max": self.dining_slow_max_spin.value(),
                "moderate": self.dining_moderate_spin.value(),
                "peak": self.dining_peak_spin.value(),
                "manual_max": dining_cfg.get("manual_max", 7),
            },
            "cocktail": {
                "normal": self.cocktail_normal_spin.value(),
                "busy": self.cocktail_busy_spin.value(),
                "peak": self.cocktail_peak_spin.value(),
                "manual_max": cocktail_cfg.get("manual_max", 4),
            },
            "opener_count": staffing.get("servers", {}).get("opener_count", 1),
        }
        payload["staffing"]["cashier"] = {
            "am_default": self.cashier_am_spin.value(),
            "pm_default": self.cashier_pm_spin.value(),
            "busy_split": self.cashier_busy_spin.value(),
            "peak": self.cashier_peak_spin.value(),
            "manual_max": staffing.get("cashier", {}).get("manual_max", 4),
        }
        payload["staffing"]["hoh"] = {
            "combo_thresholds": resolve_hoh_thresholds(self.policy_data),
            **{k: v for k, v in staffing.get("hoh", {}).items() if k not in {"combo_thresholds"}},
        }
        limits = resolve_fallback_limits({"allow_mgr_fallback": self.fallback_allow_mgr.isChecked()})
        payload["fallback"] = {
            "allow_mgr_fallback": self.fallback_allow_mgr.isChecked(),
            "am_limit": limits.get("am", 1),
            "pm_limit": limits.get("pm", 1),
            "tag": cfg.get("fallback", {}).get("tag", "MANAGER COVERING ? REVIEW REQUIRED"),
            "disallow_roles": cfg.get("fallback", {}).get("disallow_roles", []),
        }
        payload["budget"] = budget_cfg
        return payload

    def _build_timeblocks(self) -> Dict[str, Dict[str, str]]:
        close_buffer = self.close_buffer_spin.value()
        return {
            "Open": {"start": "@open-30", "end": "@open"},
            "Mid": {"start": "@open", "end": "@mid"},
            "PM": {"start": "@mid", "end": "@close"},
            "Close": {"start": "@close", "end": f"@close+{close_buffer}"},
        }

    def _populate_role_groups(self) -> None:
        groups_spec = self.policy_data.get("role_groups") or build_default_policy().get("role_groups", {})
        self.role_group_widgets.clear()
        # Clear stale widgets so we don't retain any unexpected editors in the Group column.
        self.role_group_table.clearContents()
        self.role_group_table.setRowCount(len(ROLE_GROUPS))
        for row, group in enumerate(ROLE_GROUPS.keys()):
            spec = groups_spec.get(group, {})
            pct = float(spec.get("allocation_pct", 0.0) or 0.0)
            if pct <= 1:
                pct *= 100
            label = "Heart of House" if group == "Kitchen" else group
            allocation_spin = QDoubleSpinBox()
            allocation_spin.setDecimals(1)
            allocation_spin.setRange(0.0, 100.0)
            allocation_spin.setSuffix("%")
            allocation_spin.setValue(pct)
            PolicyComposerDialog._disable_scroll_wheel([allocation_spin])
            self.role_group_table.setItem(row, 0, QTableWidgetItem(label))
            self.role_group_table.item(row, 0).setFlags(Qt.ItemIsEnabled)
            self.role_group_table.setCellWidget(row, 1, allocation_spin)
            self.role_group_widgets[group] = {"pct": allocation_spin}
            if self.read_only:
                allocation_spin.setEnabled(False)

    def _build_pre_engine_section(self, layout: QVBoxLayout) -> None:
        cfg = pre_engine_settings(self.policy_data)
        staffing = cfg.get("staffing", {})
        server_cfg = staffing.get("servers", {})
        dining_cfg = server_cfg.get("dining", {})
        cocktail_cfg = server_cfg.get("cocktail", {})
        cashier_cfg = staffing.get("cashier", {})
        hoh_cfg = staffing.get("hoh", {})
        fallback_cfg = cfg.get("fallback", {})
        budget_cfg = cfg.get("budget", {})

        box = QGroupBox("Staffing guardrails and fallback")
        outer = QVBoxLayout(box)

        server_box = QGroupBox("Servers")
        server_form = QFormLayout(server_box)
        self.dining_slow_min_spin = QSpinBox()
        self.dining_slow_min_spin.setRange(0, 20)
        self.dining_slow_min_spin.setValue(int(dining_cfg.get("slow_min", 1)))
        self.dining_slow_max_spin = QSpinBox()
        self.dining_slow_max_spin.setRange(0, 20)
        self.dining_slow_max_spin.setValue(int(dining_cfg.get("slow_max", 4)))
        self.dining_moderate_spin = QSpinBox()
        self.dining_moderate_spin.setRange(0, 30)
        self.dining_moderate_spin.setValue(int(dining_cfg.get("moderate", 5)))
        self.dining_peak_spin = QSpinBox()
        self.dining_peak_spin.setRange(0, 30)
        self.dining_peak_spin.setValue(int(dining_cfg.get("peak", 6)))
        server_form.addRow("Dining slow (min/max)", self._paired_spin(self.dining_slow_min_spin, self.dining_slow_max_spin))
        server_form.addRow("Dining moderate", self.dining_moderate_spin)
        server_form.addRow("Dining peak", self.dining_peak_spin)

        self.cocktail_normal_spin = QSpinBox()
        self.cocktail_normal_spin.setRange(0, 20)
        self.cocktail_normal_spin.setValue(int(cocktail_cfg.get("normal", 2)))
        self.cocktail_busy_spin = QSpinBox()
        self.cocktail_busy_spin.setRange(0, 20)
        self.cocktail_busy_spin.setValue(int(cocktail_cfg.get("busy", 3)))
        self.cocktail_peak_spin = QSpinBox()
        self.cocktail_peak_spin.setRange(0, 20)
        self.cocktail_peak_spin.setValue(int(cocktail_cfg.get("peak", 4)))
        server_form.addRow("Cocktail normal/busy", self._paired_spin(self.cocktail_normal_spin, self.cocktail_busy_spin))
        server_form.addRow("Cocktail peak", self.cocktail_peak_spin)
        outer.addWidget(server_box)

        cashier_box = QGroupBox("Cashier")
        cashier_form = QFormLayout(cashier_box)
        self.cashier_am_spin = QSpinBox()
        self.cashier_am_spin.setRange(0, 6)
        self.cashier_am_spin.setValue(int(cashier_cfg.get("am_default", 1)))
        self.cashier_pm_spin = QSpinBox()
        self.cashier_pm_spin.setRange(0, 6)
        self.cashier_pm_spin.setValue(int(cashier_cfg.get("pm_default", 1)))
        self.cashier_busy_spin = QSpinBox()
        self.cashier_busy_spin.setRange(0, 6)
        self.cashier_busy_spin.setValue(int(cashier_cfg.get("busy_split", 2)))
        self.cashier_peak_spin = QSpinBox()
        self.cashier_peak_spin.setRange(0, 6)
        self.cashier_peak_spin.setValue(int(cashier_cfg.get("peak", 3)))
        cashier_form.addRow("AM cashiers", self.cashier_am_spin)
        cashier_form.addRow("PM cashiers", self.cashier_pm_spin)
        cashier_form.addRow("Busy split (To-Go + Host)", self.cashier_busy_spin)
        cashier_form.addRow("Peak cashiers", self.cashier_peak_spin)
        outer.addWidget(cashier_box)

        hoh_box = QGroupBox("HOH (Heart of House)")
        hoh_form = QFormLayout(hoh_box)
        self.hoh_mode_combo = QComboBox()
        self.hoh_mode_combo.addItem("Auto (recommended)", "auto")
        self.hoh_mode_combo.addItem("Combo preferred", "combo")
        self.hoh_mode_combo.addItem("Split preferred", "split")
        self.hoh_mode_combo.addItem("Peak mode", "peak")
        hoh_form.addRow("HOH staffing mode", self.hoh_mode_combo)
        outer.addWidget(hoh_box)

        layout.addWidget(box)
        PolicyComposerDialog._disable_scroll_wheel(
            [
                self.dining_slow_min_spin,
                self.dining_slow_max_spin,
                self.dining_moderate_spin,
                self.dining_peak_spin,
                self.cocktail_normal_spin,
                self.cocktail_busy_spin,
                self.cocktail_peak_spin,
                self.cashier_am_spin,
                self.cashier_pm_spin,
                self.cashier_busy_spin,
                self.cashier_peak_spin,
            ]
        )

    @staticmethod
    def _paired_spin(first: QWidget, second: QWidget) -> QWidget:
        wrapper = QWidget()
        layout = QHBoxLayout(wrapper)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(first)
        layout.addWidget(QLabel("/"))
        layout.addWidget(second)
        return wrapper

    @staticmethod
    @staticmethod
    def _make_toggle_button(label: str, checked: bool) -> QPushButton:
        button = QPushButton()
        button.setCheckable(True)
        button.setChecked(checked)
        button.setCursor(Qt.PointingHandCursor)
        button.setMinimumHeight(34)

        def _restyle() -> None:
            on = button.isChecked()
            button.setText(f"{label} ({'On' if on else 'Off'})")
            button.setStyleSheet(
                "QPushButton {padding:8px 14px; font-weight:600; border-radius:8px; border:1px solid #2d2d2d;}"
                f"QPushButton {{ background-color:{'#2e7d32' if on else '#3a3a3a'}; color:white; }}"
            )

        button.toggled.connect(_restyle)
        _restyle()
        return button
    def _read_role_groups(self) -> Dict[str, Dict[str, Any]]:
        payload: Dict[str, Dict[str, Any]] = {}
        existing = self.policy_data.get("role_groups", {})
        for group, widgets in self.role_group_widgets.items():
            pct_spin: QDoubleSpinBox = widgets["pct"]
            label_pct = pct_spin.value() / 100.0
            payload[group] = {
                "allocation_pct": round(label_pct, 4),
                "allow_cuts": existing.get(group, {}).get("allow_cuts", True),
                "cut_buffer_minutes": existing.get(group, {}).get("cut_buffer_minutes", 0),
                "always_on": existing.get(group, {}).get("always_on", False),
            }
        return payload

    def _collect_policy_payload(self) -> Dict[str, Any]:
        name = self.name_input.text().strip() or "Store policy"
        description = self.description_input.text().strip()
        self.policy_data["hoh_mode"] = self.hoh_mode_combo.currentData()
        self.policy_data["allow_mgr_fallback"] = self.fallback_allow_mgr.isChecked()
        # Keep seasonal/fallback toggles in sync with saved payload.
        seasonal_settings = self.policy_data.get("seasonal_settings", {}) or {}
        seasonal_settings["server_patio_enabled"] = bool(self.patio_toggle.isChecked())
        self.policy_data["seasonal_settings"] = seasonal_settings
        # Mirror patio toggle into the patio role enabled flag.
        patio_role = self.policy_data.get("roles", {}).get("Server - Patio", {})
        if isinstance(patio_role, dict):
            patio_role["enabled"] = bool(seasonal_settings["server_patio_enabled"])
        params: Dict[str, Any] = {
            "name": name,
            "description": description,
            "hoh_mode": self.hoh_mode_combo.currentData(),
            "allow_mgr_fallback": self.fallback_allow_mgr.isChecked(),
            "global": {
                "max_hours_week": self.max_hours_spin.value(),
                "max_consecutive_days": self.max_consec_spin.value(),
                "desired_hours_floor_pct": round(self.desired_floor_spin.value() / 100, 3),
                "desired_hours_ceiling_pct": round(self.desired_ceiling_spin.value() / 100, 3),
                "close_buffer_minutes": self.close_buffer_spin.value(),
                "labor_budget_pct": round(self.labor_budget_spin.value() / 100, 4),
                "labor_budget_tolerance_pct": round(self.labor_tolerance_spin.value() / 100, 4),
            },
            "timeblocks": self._build_timeblocks(),
            "business_hours": self._read_hours_table(),
            "roles": self.policy_data.get("roles") or {},
            "role_groups": self._read_role_groups(),
            "shift_presets": self.shift_template_editor.value(),
            "section_capacity": self.section_capacity_editor.value(),
            "seasonal_settings": self.policy_data.get("seasonal_settings", {}),
            "anchors": self._collect_anchors_payload(),
            "pre_engine": self._read_pre_engine_controls(),
        }
        return params

    def _collect_anchors_payload(self) -> Dict[str, Any]:
        anchors = copy.deepcopy(self.policy_data.get("anchors") or {})
        return anchors

    def _save_policy(self) -> None:
        if self.read_only:
            return
        params = self._collect_policy_payload()
        name = params.pop("name", "Store policy")
        with self.session_factory() as session:
            updated = upsert_policy(session, name, params, edited_by=self.current_user.get("username", "system"))
        audit_logger.log(
            "policy_update",
            self.current_user.get("username", "unknown"),
            role=self.current_user.get("role"),
            details={"policy_id": updated.id, "name": updated.name},
        )
        self.policy = updated
        self.policy_data = updated.params_dict()
        self.policy_data["name"] = updated.name
        self.policy_data["description"] = getattr(updated, "description", self.policy_data.get("description", ""))
        self._ensure_policy_defaults()
        self.feedback_label.setStyleSheet(f"color:{SUCCESS_COLOR};")
        self.feedback_label.setText(f"Saved policy '{updated.name}'.")
        self._apply_policy_to_fields()

    def _export_policy(self) -> None:
        if not self.policy_data:
            return
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export policy",
            str(DATA_DIR / "policy.json"),
            "JSON Files (*.json);;All Files (*)",
        )
        if not file_path:
            return
        Path(file_path).write_text(json.dumps(self.policy_data, indent=2), encoding="utf-8")
        QMessageBox.information(self, "Export complete", f"Saved policy to {file_path}.")

    def _import_policy(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Import policy",
            str(DATA_DIR),
            "JSON Files (*.json);;All Files (*)",
        )
        if not file_path:
            return
        try:
            data = json.loads(Path(file_path).read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError("Invalid policy file.")
            if isinstance(data.get("params"), dict):
                params = dict(data["params"])
                params.setdefault("name", data.get("name") or params.get("policy_name") or "Imported Policy")
                data = params
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Import failed", str(exc))
            return
        self.policy_data = data
        self._ensure_policy_defaults()
        self.feedback_label.setStyleSheet(f"color:{INFO_COLOR};")
        self.feedback_label.setText(f"Loaded policy from {file_path}. Save to apply.")
        self._apply_policy_to_fields()

    def _open_composer(self) -> None:
        if self.read_only:
            return
        dialog = PolicyComposerDialog(name=self.policy_data.get("name", ""), params=self.policy_data)
        dialog.setStyleSheet(THEME_STYLESHEET)
        if dialog.exec() != QDialog.Accepted or not dialog.result_data:
            return
        data = dialog.result_data
        with self.session_factory() as session:
            updated = upsert_policy(
                session,
                data["name"],
                data["params"],
                edited_by=self.current_user.get("username", "system"),
            )
        audit_logger.log(
            "policy_update",
            self.current_user.get("username", "unknown"),
            role=self.current_user.get("role"),
            details={"policy_id": updated.id, "name": updated.name},
        )
        self.policy = updated
        self.policy_data = updated.params_dict()
        self.policy_data["name"] = updated.name
        self.policy_data["description"] = getattr(updated, "description", self.policy_data.get("description", ""))
        self._ensure_policy_defaults()
        self.feedback_label.setStyleSheet(f"color:{SUCCESS_COLOR};")
        self.feedback_label.setText(f"Updated policy '{updated.name}'.")
        self._apply_policy_to_fields()
    
    def _ensure_policy_defaults(self) -> None:
        defaults = build_default_policy()
        roles = self.policy_data.setdefault("roles", {})
        for role_name, cfg in (defaults.get("roles") or {}).items():
            roles.setdefault(role_name, cfg)
        groups = self.policy_data.setdefault("role_groups", {})
        for group_name, cfg in (defaults.get("role_groups") or {}).items():
            groups.setdefault(group_name, cfg)
        anchors_defaults = defaults.get("anchors", {})
        self.policy_data.setdefault("anchors", anchors_defaults.copy())
        self.policy_data.setdefault("shift_presets", defaults.get("shift_presets", {}))
        self.policy_data.setdefault("section_capacity", defaults.get("section_capacity", {}))
        self.policy_data.setdefault("seasonal_settings", defaults.get("seasonal_settings", {}))
        self.policy_data.setdefault("pre_engine", pre_engine_settings(self.policy_data))
        self.policy_data.setdefault("section_priority", defaults.get("section_priority", "normal"))
        self.policy_data.setdefault("hoh_mode", defaults.get("hoh_mode", "auto"))
        self.policy_data.setdefault("allow_mgr_fallback", defaults.get("allow_mgr_fallback", True))
        hours = self.policy_data.setdefault("business_hours", defaults.get("business_hours", _default_business_hours()))
        defaults_hours = defaults.get("business_hours", _default_business_hours())
        for day in WEEKDAY_LABELS:
            entry = hours.setdefault(day, defaults_hours.get(day, {"open": "11:00", "mid": "16:00", "close": "23:00"}))
            entry.setdefault("mid", entry.get("close", "16:00"))
        self.policy_data.setdefault("timeblocks", defaults.get("timeblocks", {}))

    @staticmethod
    def _disable_scroll_wheel(widgets: List[Optional[QWidget]]) -> None:
        for widget in widgets:
            if widget is None:
                continue
            widget.setFocusPolicy(Qt.StrongFocus)
            if isinstance(widget, QAbstractSpinBox):
                widget.setButtonSymbols(QAbstractSpinBox.NoButtons)
            widget.wheelEvent = lambda event: event.ignore()

    @staticmethod
    def _make_collapsible(title: str, content: QWidget) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        toggle = QToolButton()
        toggle.setText(title)
        toggle.setCheckable(True)
        toggle.setChecked(True)
        toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        toggle.setArrowType(Qt.DownArrow)

        def _toggle(checked: bool) -> None:
            content.setVisible(checked)
            toggle.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)

        toggle.toggled.connect(_toggle)
        layout.addWidget(toggle)
        layout.addWidget(content)
        return container


class EmployeeEditDialog(QDialog):
    def __init__(
        self,
        session_factory,
        actor: Dict[str, str],
        employee_id: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.session_factory = session_factory
        self.actor = actor
        self.employee_id = employee_id
        self.employee = None
        self.result_employee_id: Optional[int] = None
        self.result_action: Optional[str] = None
        self.result_snapshot: Dict[str, Any] = {}
        self.role_wage_overrides: Dict[str, float] = {}
        if self.employee_id is not None:
            with self.session_factory() as session:
                self.employee = session.get(Employee, self.employee_id)
        self.setWindowTitle("Edit employee" if self.employee else "Add employee")
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        form = QFormLayout()
        self.name_input = QLineEdit()
        form.addRow("Full name", self.name_input)

        roles_container = QWidget()
        roles_layout = QVBoxLayout(roles_container)
        roles_layout.setContentsMargins(0, 0, 0, 0)

        selector_row = QHBoxLayout()
        self.role_selector = QComboBox()
        self.role_selector.addItem("Select role...", None)
        for group_label, roles in EMPLOYEE_ROLE_GROUPS.items():
            index = self.role_selector.count()
            self.role_selector.addItem(group_label, None)
            model_item = self.role_selector.model().item(index)
            if model_item is not None:
                model_item.setEnabled(False)
                model_item.setSelectable(False)
            for role in roles:
                self.role_selector.addItem(f"  {role}", role)
        selector_row.addWidget(self.role_selector)

        self.add_role_button = QPushButton("Add role")
        self.add_role_button.clicked.connect(self.handle_add_role)
        selector_row.addWidget(self.add_role_button)

        roles_layout.addLayout(selector_row)

        custom_row = QHBoxLayout()
        self.custom_role_input = QLineEdit()
        self.custom_role_input.setPlaceholderText("Custom role")
        self.custom_role_input.returnPressed.connect(self.handle_add_custom_role)
        custom_row.addWidget(self.custom_role_input)
        self.add_custom_role_button = QPushButton("Add custom")
        self.add_custom_role_button.clicked.connect(self.handle_add_custom_role)
        custom_row.addWidget(self.add_custom_role_button)
        roles_layout.addLayout(custom_row)

        preference_hint = QLabel("Preference order: top = most desired, bottom = fallback. Drag to reorder.")
        preference_hint.setWordWrap(True)
        roles_layout.addWidget(preference_hint)

        self.role_list_widget = QListWidget()
        self.role_list_widget.setSelectionMode(QAbstractItemView.SingleSelection)
        self.role_list_widget.setDragDropMode(QAbstractItemView.InternalMove)
        self.role_list_widget.setDefaultDropAction(Qt.MoveAction)
        self.role_list_widget.setDragEnabled(True)
        self.role_list_widget.setAcceptDrops(True)
        self.role_list_widget.setDropIndicatorShown(True)
        self.role_list_widget.itemSelectionChanged.connect(self._update_role_buttons)
        self.role_list_widget.itemDoubleClicked.connect(lambda *_: self.remove_selected_role())
        roles_layout.addWidget(self.role_list_widget)

        list_button_row = QHBoxLayout()
        self.move_up_button = QPushButton("Move up")
        self.move_up_button.clicked.connect(lambda: self._move_role(-1))
        list_button_row.addWidget(self.move_up_button)

        self.move_down_button = QPushButton("Move down")
        self.move_down_button.clicked.connect(lambda: self._move_role(1))
        list_button_row.addWidget(self.move_down_button)

        self.remove_role_button = QPushButton("Remove selected")
        self.remove_role_button.clicked.connect(self.remove_selected_role)
        list_button_row.addWidget(self.remove_role_button)
        list_button_row.addStretch()
        roles_layout.addLayout(list_button_row)

        form.addRow("Roles", roles_container)

        self.role_wage_button = QPushButton("Role wage overrides")
        self.role_wage_button.clicked.connect(self._edit_role_wages)
        form.addRow("", self.role_wage_button)

        self.start_month_combo = QComboBox()
        self.start_month_combo.addItem("Not set", None)
        for month_index in range(1, 13):
            self.start_month_combo.addItem(calendar.month_name[month_index], month_index)
        form.addRow("Start month", self.start_month_combo)

        current_year = datetime.datetime.now().year
        self.start_year_combo = QComboBox()
        self.start_year_combo.addItem("Not set", None)
        for year in range(current_year + 2, current_year - 31, -1):
            self.start_year_combo.addItem(str(year), year)
        self._set_start_date_inputs(None, None)
        form.addRow("Start year", self.start_year_combo)

        self.desired_hours_input = QSpinBox()
        self.desired_hours_input.setRange(0, 168)
        self.desired_hours_input.setSuffix(" hrs")
        form.addRow("Desired weekly hours", self.desired_hours_input)

        self.status_combo = QComboBox()
        self.status_combo.addItems(["Active", "Inactive"])
        form.addRow("Status", self.status_combo)

        self.notes_input = QPlainTextEdit()
        self.notes_input.setPlaceholderText("Optional notes")
        self.notes_input.setFixedHeight(80)
        form.addRow("Notes", self.notes_input)

        layout.addLayout(form)

        self.feedback_label = QLabel()
        self.feedback_label.setStyleSheet(f"color:{ERROR_COLOR};")
        layout.addWidget(self.feedback_label)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        self._load_employee_into_form()
        self._update_role_buttons()

    def _load_employee_into_form(self) -> None:
        if not self.employee:
            return
        self._set_start_date_inputs(self.employee.start_month, self.employee.start_year)
        self.name_input.setText(self.employee.full_name)
        self.role_list_widget.clear()
        for role in self.employee.role_list:
            self._add_role_to_list(role, silent=True)
        self.desired_hours_input.setValue(self.employee.desired_hours or 0)
        self.status_combo.setCurrentIndex(0 if self.employee.status == "active" else 1)
        self.notes_input.setPlainText(self.employee.notes or "")
        with self.session_factory() as session:
            wages = get_employee_role_wages(session, [self.employee.id])
            self.role_wage_overrides = wages.get(self.employee.id, {})

    def _collect_roles(self) -> List[str]:
        roles: List[str] = []
        for index in range(self.role_list_widget.count()):
            role = self.role_list_widget.item(index).text().strip()
            if role:
                roles.append(role)
        return roles

    def _set_start_date_inputs(self, month: Optional[int], year: Optional[int]) -> None:
        month_index = self.start_month_combo.findData(month) if month else 0
        if month_index == -1:
            month_index = 0
        self.start_month_combo.setCurrentIndex(month_index)

        year_index = self.start_year_combo.findData(year) if year else 0
        if year_index == -1:
            year_index = 0
        self.start_year_combo.setCurrentIndex(year_index)

    def handle_add_role(self) -> None:
        role = self.role_selector.currentData()
        if not role:
            return
        if self._add_role_to_list(str(role).strip()):
            self.role_selector.setCurrentIndex(0)
            self.feedback_label.setText("")

    def handle_add_custom_role(self) -> None:
        role = self.custom_role_input.text().strip()
        if not role:
            self.feedback_label.setText("Enter a custom role before adding.")
            return
        if self._add_role_to_list(role):
            self.custom_role_input.clear()
            self.feedback_label.setText("")

    def remove_selected_role(self) -> None:
        current_item = self.role_list_widget.currentItem()
        if not current_item:
            return
        row = self.role_list_widget.row(current_item)
        self.role_list_widget.takeItem(row)
        self.feedback_label.setText("")
        self._update_role_buttons()

    def _add_role_to_list(self, role: str, *, silent: bool = False) -> bool:
        normalized = role.strip()
        if not normalized:
            return False
        for index in range(self.role_list_widget.count()):
            existing = self.role_list_widget.item(index)
            if existing.text().strip().lower() == normalized.lower():
                self.role_list_widget.setCurrentRow(index)
                self._update_role_buttons()
                if not silent:
                    self.feedback_label.setText("Role already assigned.")
                return False
        item = QListWidgetItem(normalized)
        self.role_list_widget.addItem(item)
        self.role_list_widget.setCurrentItem(item)
        self._update_role_buttons()
        if not silent:
            self.feedback_label.setText("")
        return True

    def _update_role_buttons(self) -> None:
        current_item = self.role_list_widget.currentItem()
        has_selection = current_item is not None
        row = self.role_list_widget.row(current_item) if current_item else -1
        count = self.role_list_widget.count()
        self.remove_role_button.setEnabled(has_selection)
        self.move_up_button.setEnabled(has_selection and row > 0)
        self.move_down_button.setEnabled(has_selection and 0 <= row < count - 1)

    def _move_role(self, delta: int) -> None:
        current_item = self.role_list_widget.currentItem()
        if not current_item:
            return
        row = self.role_list_widget.row(current_item)
        target = row + delta
        if target < 0 or target >= self.role_list_widget.count():
            return
        item = self.role_list_widget.takeItem(row)
        self.role_list_widget.insertItem(target, item)
        self.role_list_widget.setCurrentItem(item)

    def _edit_role_wages(self) -> None:
        roles = self._collect_roles()
        dialog = EmployeeRoleWageDialog(roles, self.role_wage_overrides)
        dialog.setStyleSheet(THEME_STYLESHEET)
        if dialog.exec() == QDialog.Accepted:
            self.role_wage_overrides = dialog.overrides
            self.feedback_label.setText("Saved role wage overrides in memory; click OK to persist.")

    def accept(self) -> None:  # type: ignore[override]
        full_name = self.name_input.text().strip()
        if not full_name:
            self.feedback_label.setText("Employee name is required.")
            self.name_input.setFocus()
            return
        roles = self._collect_roles()
        desired_hours = self.desired_hours_input.value()
        status = "active" if self.status_combo.currentIndex() == 0 else "inactive"
        notes = self.notes_input.toPlainText().strip()
        selected_month = self.start_month_combo.currentData()
        selected_year = self.start_year_combo.currentData()
        if (selected_month is None) != (selected_year is None):
            self.feedback_label.setText("Select both start month and start year, or leave both unset.")
            return
        start_month = int(selected_month) if selected_month is not None else None
        start_year = int(selected_year) if selected_year is not None else None

        with self.session_factory() as session:
            if self.employee_id is not None:
                employee = session.get(Employee, self.employee_id)
                if not employee:
                    self.feedback_label.setText("Employee record not found.")
                    return
                action = "update"
            else:
                employee = Employee()
                session.add(employee)
                action = "create"

            employee.full_name = full_name
            employee.role_list = roles
            employee.desired_hours = desired_hours
            employee.status = status
            employee.notes = notes
            employee.start_month = start_month
            employee.start_year = start_year
            employee.updated_at = datetime.datetime.now(datetime.timezone.utc)

            session.commit()
            session.refresh(employee)
            if self.role_wage_overrides is not None:
                save_employee_role_wages(session, employee.id, self.role_wage_overrides)

        self.result_employee_id = employee.id
        self.result_action = action
        self.result_snapshot = {
            "full_name": employee.full_name,
            "roles": employee.role_list,
            "status": employee.status,
            "desired_hours": employee.desired_hours,
            "start_month": employee.start_month,
            "start_year": employee.start_year,
            "start_label": employee.start_date_label,
        }
        super().accept()


class WageManagerDialog(QDialog):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Role wages")
        self.roles = ROLE_CATALOG
        self.entries = load_wages()
        self.wage_inputs: Dict[str, QDoubleSpinBox] = {}
        self.confirm_boxes: Dict[str, QCheckBox] = {}
        self.status_items: Dict[str, QTableWidgetItem] = {}
        self.resize(700, 520)
        self._build_ui()
        self._populate_table()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        intro = QLabel(
            "Set each role's hourly wage. Green = confirmed, yellow = needs confirmation, red = missing. "
            "Editing a wage clears confirmation until you re-confirm."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        self.table = QTableWidget(len(self.roles), 4)
        self.table.setHorizontalHeaderLabels(["Status", "Role", "Hourly wage ($)", "Confirm"])
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        layout.addWidget(self.table)

        controls = QHBoxLayout()
        defaults_button = QPushButton("Reset to defaults")
        defaults_button.clicked.connect(self._handle_reset)
        controls.addWidget(defaults_button)

        confirm_all_button = QPushButton("Confirm all")
        confirm_all_button.clicked.connect(self._handle_confirm_all)
        controls.addWidget(confirm_all_button)

        import_button = QPushButton("Import…")
        import_button.clicked.connect(self._handle_import)
        controls.addWidget(import_button)

        export_button = QPushButton("Export…")
        export_button.clicked.connect(self._handle_export)
        controls.addWidget(export_button)
        controls.addStretch()
        layout.addLayout(controls)

        button_box = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Close)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def _populate_table(self) -> None:
        self.wage_inputs.clear()
        self.confirm_boxes.clear()
        self.status_items.clear()
        data = load_wages()
        baseline = baseline_wages()
        for row, role in enumerate(self.roles):
            status_item = QTableWidgetItem("")
            status_item.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(row, 0, status_item)
            name_item = QTableWidgetItem(role)
            name_item.setFlags(Qt.ItemIsEnabled)
            self.table.setItem(row, 1, name_item)
            spin = QDoubleSpinBox()
            spin.setPrefix("$")
            spin.setSuffix("/hr")
            spin.setRange(0.0, 150.0)
            spin.setDecimals(2)
            spin.setSingleStep(0.25)
            entry = data.get(role, baseline.get(role, {}))
            spin.setValue(float(entry.get("wage", 0.0) or 0.0))
            spin.valueChanged.connect(lambda _value, r=role: self._handle_wage_change(r))
            self.table.setCellWidget(row, 2, spin)
            checkbox = QCheckBox("Confirm")
            checkbox.setChecked(bool(entry.get("confirmed", False)))
            checkbox.stateChanged.connect(lambda _state, r=role: self._update_status(r))
            self.table.setCellWidget(row, 3, checkbox)
            self.wage_inputs[role] = spin
            self.confirm_boxes[role] = checkbox
            self.status_items[role] = status_item
            self._update_status(role)

    def _update_status(self, role: str) -> None:
        status_item = self.status_items.get(role)
        if not status_item:
            return
        wage = self.wage_inputs.get(role).value() if role in self.wage_inputs else 0.0
        confirmed = self.confirm_boxes.get(role).isChecked() if role in self.confirm_boxes else False
        normalized = normalize_role(role)
        zero_allowed = normalized in ALLOW_ZERO_ROLES
        if (wage > 0 or zero_allowed) and confirmed:
            status_item.setText("OK")
            status_item.setForeground(Qt.green)
        elif wage > 0 or zero_allowed:
            status_item.setText("Set")
            status_item.setForeground(Qt.yellow)
        else:
            status_item.setText("!")
            status_item.setForeground(Qt.red)

    def _handle_wage_change(self, role: str) -> None:
        if role in self.confirm_boxes:
            self.confirm_boxes[role].setChecked(False)
        self._update_status(role)

    def _handle_reset(self) -> None:
        confirm = QMessageBox.question(
            self,
            "Reset wages",
            "Reset all wages to the default policy values?",
        )
        if confirm != QMessageBox.Yes:
            return
        reset_wages_to_defaults()
        self.entries = load_wages()
        self._populate_table()

    def _handle_import(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Import role wages",
            str(DATA_DIR),
            "JSON Files (*.json);;All Files (*)",
        )
        if not file_path:
            return
        try:
            import_wages_file(Path(file_path))
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Import failed", str(exc))
            return
        QMessageBox.information(self, "Import complete", "Role wages imported successfully.")
        self.entries = load_wages()
        self._populate_table()

    def _handle_export(self) -> None:
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export role wages",
            str(DATA_DIR / "role_wages.json"),
            "JSON Files (*.json);;All Files (*)",
        )
        if not file_path:
            return
        export_wages_file(Path(file_path))
        QMessageBox.information(self, "Export complete", f"Saved role wages to {file_path}.")

    def _handle_confirm_all(self) -> None:
        for checkbox in self.confirm_boxes.values():
            checkbox.setChecked(True)
        for role in self.roles:
            self._update_status(role)

    def accept(self) -> None:  # type: ignore[override]
        payload: Dict[str, Dict[str, Any]] = {}
        for role in self.roles:
            wage = round(self.wage_inputs[role].value(), 2)
            normalized = normalize_role(role)
            zero_allowed = normalized in ALLOW_ZERO_ROLES
            if wage <= 0.0 and not zero_allowed:
                QMessageBox.warning(self, "Missing wage", f"Enter a wage for {role} before saving.")
                return
            payload[role] = {
                "wage": wage,
                "confirmed": self.confirm_boxes[role].isChecked() or zero_allowed,
            }
        save_wages(payload)
        super().accept()


class UnavailabilityEntryDialog(QDialog):
    def __init__(
        self,
        day_of_week: Optional[int] = None,
        start_time: Optional[datetime.time] = None,
        end_time: Optional[datetime.time] = None,
    ) -> None:
        super().__init__()
        self.result_data: Optional[Dict[str, Any]] = None
        self.setWindowTitle("Unavailability window")
        self._build_ui(day_of_week, start_time, end_time)

    def _build_ui(
        self,
        day_of_week: Optional[int],
        start_time: Optional[datetime.time],
        end_time: Optional[datetime.time],
    ) -> None:
        layout = QVBoxLayout(self)
        form = QFormLayout()

        self.day_combo = QComboBox()
        self.day_combo.addItems(DAYS_OF_WEEK)
        if day_of_week is not None:
            self.day_combo.setCurrentIndex(day_of_week)
        form.addRow("Day of week", self.day_combo)

        self.start_time_edit = QTimeEdit()
        self.start_time_edit.setDisplayFormat("HH:mm")
        self.start_time_edit.setTime(
            QTime(start_time.hour, start_time.minute) if start_time else QTime(8, 0)
        )
        form.addRow("Starts at", self.start_time_edit)

        self.end_time_edit = QTimeEdit()
        self.end_time_edit.setDisplayFormat("HH:mm")
        self.end_time_edit.setTime(
            QTime(end_time.hour, end_time.minute) if end_time else QTime(17, 0)
        )
        form.addRow("Ends at", self.end_time_edit)

        layout.addLayout(form)

        self.feedback_label = QLabel()
        self.feedback_label.setStyleSheet(f"color:{ERROR_COLOR};")
        layout.addWidget(self.feedback_label)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def accept(self) -> None:  # type: ignore[override]
        start_time = self.start_time_edit.time()
        end_time = self.end_time_edit.time()
        if start_time >= end_time:
            self.feedback_label.setText("End time must be after start time.")
            return
        self.result_data = {
            "day_of_week": self.day_combo.currentIndex(),
            "start_time": datetime.time(start_time.hour(), start_time.minute()),
            "end_time": datetime.time(end_time.hour(), end_time.minute()),
        }
        super().accept()


class UnavailabilityDialog(QDialog):
    def __init__(self, session_factory, actor: Dict[str, str], employee_id: int) -> None:
        super().__init__()
        self.session_factory = session_factory
        self.actor = actor
        self.employee_id = employee_id
        self.entries: List[EmployeeUnavailability] = []
        with self.session_factory() as session:
            employee = session.get(Employee, self.employee_id)
            self.employee_name = employee.full_name if employee else "Employee"
        self.setWindowTitle(f"Unavailability — {self.employee_name}")
        self.resize(520, 360)
        self._build_ui()
        self.refresh_entries()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        hint = QLabel("Add time windows when the employee is unavailable for scheduling.")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Day", "Starts", "Ends"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        layout.addWidget(self.table)

        buttons = QHBoxLayout()
        self.add_button = QPushButton("Add window")
        self.add_button.clicked.connect(self.add_entry)
        buttons.addWidget(self.add_button)

        self.edit_button = QPushButton("Edit")
        self.edit_button.clicked.connect(self.edit_entry)
        buttons.addWidget(self.edit_button)

        self.delete_button = QPushButton("Remove")
        self.delete_button.clicked.connect(self.remove_entry)
        buttons.addWidget(self.delete_button)

        buttons.addStretch()
        layout.addLayout(buttons)

        self.table.itemSelectionChanged.connect(self.update_button_state)
        self.update_button_state()

    def refresh_entries(self) -> None:
        with self.session_factory() as session:
            stmt = (
                select(EmployeeUnavailability)
                .where(EmployeeUnavailability.employee_id == self.employee_id)
                .order_by(
                    EmployeeUnavailability.day_of_week,
                    EmployeeUnavailability.start_time,
                )
            )
            self.entries = list(session.scalars(stmt))

        self.table.setRowCount(len(self.entries))
        for row, entry in enumerate(self.entries):
            day_item = QTableWidgetItem(DAYS_OF_WEEK[entry.day_of_week])
            day_item.setData(Qt.UserRole, entry.id)
            self.table.setItem(row, 0, day_item)
            self.table.setItem(row, 1, QTableWidgetItem(entry.start_time.strftime("%H:%M")))
            self.table.setItem(row, 2, QTableWidgetItem(entry.end_time.strftime("%H:%M")))
        self.table.resizeColumnsToContents()
        self.update_button_state()

    def selected_entry(self) -> Optional[EmployeeUnavailability]:
        selected_rows = self.table.selectionModel().selectedRows() if self.table.selectionModel() else []
        if not selected_rows:
            return None
        row = selected_rows[0].row()
        if 0 <= row < len(self.entries):
            return self.entries[row]
        return None

    def update_button_state(self) -> None:
        has_selection = bool(self.table.selectionModel() and self.table.selectionModel().hasSelection())
        self.edit_button.setEnabled(has_selection)
        self.delete_button.setEnabled(has_selection)

    def add_entry(self) -> None:
        dialog = UnavailabilityEntryDialog()
        dialog.setStyleSheet(THEME_STYLESHEET)
        if dialog.exec() != QDialog.Accepted or not dialog.result_data:
            return
        result = dialog.result_data
        with self.session_factory() as session:
            entry = EmployeeUnavailability(
                employee_id=self.employee_id,
                day_of_week=result["day_of_week"],
                start_time=result["start_time"],
                end_time=result["end_time"],
            )
            session.add(entry)
            session.commit()
            session.refresh(entry)
            entry_id = entry.id

        audit_logger.log(
            "employee_unavailability_add",
            self.actor["username"],
            details={
                "employee_id": self.employee_id,
                "entry_id": entry_id,
                "day_of_week": result["day_of_week"],
                "start_time": result["start_time"].isoformat(),
                "end_time": result["end_time"].isoformat(),
            },
        )
        self.refresh_entries()

    def edit_entry(self) -> None:
        entry = self.selected_entry()
        if not entry:
            return
        dialog = UnavailabilityEntryDialog(
            entry.day_of_week,
            entry.start_time,
            entry.end_time,
        )
        dialog.setStyleSheet(THEME_STYLESHEET)
        if dialog.exec() != QDialog.Accepted or not dialog.result_data:
            return
        result = dialog.result_data
        with self.session_factory() as session:
            db_entry = session.get(EmployeeUnavailability, entry.id)
            if not db_entry:
                return
            db_entry.day_of_week = result["day_of_week"]
            db_entry.start_time = result["start_time"]
            db_entry.end_time = result["end_time"]
            session.commit()

        audit_logger.log(
            "employee_unavailability_update",
            self.actor["username"],
            details={
                "employee_id": self.employee_id,
                "entry_id": entry.id,
                "day_of_week": result["day_of_week"],
                "start_time": result["start_time"].isoformat(),
                "end_time": result["end_time"].isoformat(),
            },
        )
        self.refresh_entries()

    def remove_entry(self) -> None:
        entry = self.selected_entry()
        if not entry:
            return
        confirm = QMessageBox.question(
            self,
            "Remove unavailability",
            "Delete the selected unavailability window?",
        )
        if confirm != QMessageBox.Yes:
            return
        with self.session_factory() as session:
            db_entry = session.get(EmployeeUnavailability, entry.id)
            if not db_entry:
                return
            session.delete(db_entry)
            session.commit()

        audit_logger.log(
            "employee_unavailability_delete",
            self.actor["username"],
            details={
                "employee_id": self.employee_id,
                "entry_id": entry.id,
            },
        )
        self.refresh_entries()


class EmployeeDirectoryDialog(QDialog):
    def __init__(self, session_factory, actor: Dict[str, str]) -> None:
        super().__init__()
        self.session_factory = session_factory
        self.actor = actor
        self.employees: List[Employee] = []
        self.visible_employees: List[Employee] = []
        self.setWindowTitle("Employee directory")
        self.resize(2000, 1560)
        self.setMinimumWidth(1100)
        self._build_ui()
        self.refresh_table()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        intro = QLabel(
            "Maintain employee profiles, their role coverage, availability, and standing."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Show"))
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(["All employees", "Active only", "Inactive only"])
        self.filter_combo.currentIndexChanged.connect(self.refresh_table)
        controls.addWidget(self.filter_combo)
        controls.addStretch()
        layout.addLayout(controls)

        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(
            ["Name", "Roles", "Desired hours", "Start date", "Status", "Notes"]
        )
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.itemSelectionChanged.connect(self.update_button_state)
        self.table.cellDoubleClicked.connect(lambda *_: self.edit_employee())
        layout.addWidget(self.table)

        buttons = QHBoxLayout()
        self.add_button = QPushButton("Add employee")
        self.add_button.clicked.connect(self.add_employee)
        buttons.addWidget(self.add_button)

        self.edit_button = QPushButton("Edit")
        self.edit_button.clicked.connect(self.edit_employee)
        buttons.addWidget(self.edit_button)

        self.toggle_button = QPushButton("Deactivate")
        self.toggle_button.clicked.connect(self.toggle_employee_status)
        buttons.addWidget(self.toggle_button)

        self.availability_button = QPushButton("Manage availability")
        self.availability_button.clicked.connect(self.manage_availability)
        buttons.addWidget(self.availability_button)

        self.wage_override_button = QPushButton("Manage role wages")
        self.wage_override_button.clicked.connect(self.manage_role_wages)
        buttons.addWidget(self.wage_override_button)

        buttons.addStretch()
        layout.addLayout(buttons)

        self.update_button_state()

    def refresh_table(self) -> None:
        with self.session_factory() as session:
            self.employees = get_all_employees(session)

        selection = self.filter_combo.currentIndex() if hasattr(self, "filter_combo") else 0
        if selection == 1:
            self.visible_employees = [emp for emp in self.employees if emp.status == "active"]
        elif selection == 2:
            self.visible_employees = [emp for emp in self.employees if emp.status == "inactive"]
        else:
            self.visible_employees = list(self.employees)

        self.table.setRowCount(len(self.visible_employees))
        for row, employee in enumerate(self.visible_employees):
            name_item = QTableWidgetItem(employee.full_name)
            name_item.setData(Qt.UserRole, employee.id)
            self.table.setItem(row, 0, name_item)
            roles_text = ", ".join(employee.role_list) if employee.role_list else "No roles"
            self.table.setItem(row, 1, QTableWidgetItem(roles_text))
            self.table.setItem(row, 2, QTableWidgetItem(str(employee.desired_hours)))
            self.table.setItem(row, 3, QTableWidgetItem(employee.start_date_label))
            status_text = "Active" if employee.status == "active" else "Inactive"
            self.table.setItem(row, 4, QTableWidgetItem(status_text))
            self.table.setItem(row, 5, QTableWidgetItem(employee.notes or ""))
        self.table.resizeRowsToContents()
        self.update_button_state()
    def selected_employee(self) -> Optional[Employee]:
        employees = self.selected_employees()
        return employees[0] if employees else None

    def selected_employees(self) -> List[Employee]:
        selection = self.table.selectionModel()
        if not selection or not selection.hasSelection():
            return []
        selected_rows = selection.selectedRows()
        if not selected_rows:
            return []
        employees: List[Employee] = []
        for item in sorted(selected_rows, key=lambda entry: entry.row()):
            row = item.row()
            if 0 <= row < len(self.visible_employees):
                employees.append(self.visible_employees[row])
        return employees

    def update_button_state(self) -> None:
        employees = self.selected_employees()
        selection_count = len(employees)
        self.edit_button.setEnabled(selection_count == 1)
        self.toggle_button.setEnabled(selection_count >= 1)
        self.availability_button.setEnabled(selection_count == 1)
        self.wage_override_button.setEnabled(selection_count == 1)
        if selection_count == 1:
            employee = employees[0]
            self.toggle_button.setText("Deactivate" if employee.status == "active" else "Activate")
        elif selection_count > 1:
            self.toggle_button.setText("Deactivate/Delete")
        else:
            self.toggle_button.setText("Deactivate")

    def add_employee(self) -> None:
        dialog = EmployeeEditDialog(self.session_factory, self.actor)
        dialog.setStyleSheet(THEME_STYLESHEET)
        if dialog.exec() == QDialog.Accepted and dialog.result_action:
            snapshot = dialog.result_snapshot
            audit_logger.log(
                "employee_create",
                self.actor["username"],
                role=self.actor.get("role"),
                details={
                    "employee_id": dialog.result_employee_id,
                    "full_name": snapshot.get("full_name"),
                    "roles": snapshot.get("roles"),
                    "desired_hours": snapshot.get("desired_hours"),
                    "start_date": snapshot.get("start_label"),
                },
            )
            self.refresh_table()

    def edit_employee(self) -> None:
        employee = self.selected_employee()
        if not employee:
            return
        dialog = EmployeeEditDialog(self.session_factory, self.actor, employee.id)
        dialog.setStyleSheet(THEME_STYLESHEET)
        if dialog.exec() == QDialog.Accepted and dialog.result_action:
            snapshot = dialog.result_snapshot
            audit_logger.log(
                "employee_update",
                self.actor["username"],
                role=self.actor.get("role"),
                details={
                    "employee_id": dialog.result_employee_id,
                    "full_name": snapshot.get("full_name"),
                    "roles": snapshot.get("roles"),
                    "desired_hours": snapshot.get("desired_hours"),
                    "status": snapshot.get("status"),
                    "start_date": snapshot.get("start_label"),
                },
            )
            self.refresh_table()

    def toggle_employee_status(self) -> None:
        employees = self.selected_employees()
        if not employees:
            return
        if len(employees) > 1:
            action = self._prompt_bulk_employee_action(len(employees))
            if action == "delete":
                self._delete_employees(employees)
            elif action == "deactivate":
                self._bulk_update_employee_status(employees, "inactive")
            return
        employee = employees[0]
        action = self._prompt_employee_action(employee)
        if action == "delete":
            self._delete_employee(employee)
        elif action == "toggle":
            new_status = "inactive" if employee.status == "active" else "active"
            self._update_employee_status(employee, new_status)

    def _prompt_bulk_employee_action(self, count: int) -> str:
        dialog = QMessageBox(self)
        dialog.setWindowTitle("Update employees")
        dialog.setText(f"What would you like to do with {count} employees?")
        deactivate_button = dialog.addButton("Deactivate", QMessageBox.ActionRole)
        delete_button = dialog.addButton("Delete", QMessageBox.ActionRole)
        dialog.addButton(QMessageBox.Cancel)
        dialog.setDefaultButton(deactivate_button)
        dialog.exec()
        clicked = dialog.clickedButton()
        if clicked == delete_button:
            return "delete"
        if clicked == deactivate_button:
            return "deactivate"
        return "cancel"

    def _prompt_employee_action(self, employee: Employee) -> str:
        dialog = QMessageBox(self)
        dialog.setWindowTitle("Update employee")
        dialog.setText(f"What would you like to do with {employee.full_name}?")
        status_label = "Deactivate" if employee.status == "active" else "Activate"
        status_button = dialog.addButton(status_label, QMessageBox.ActionRole)
        delete_button = dialog.addButton("Delete", QMessageBox.ActionRole)
        dialog.addButton(QMessageBox.Cancel)
        dialog.setDefaultButton(status_button)
        dialog.exec()
        clicked = dialog.clickedButton()
        if clicked == delete_button:
            return "delete"
        if clicked == status_button:
            return "toggle"
        return "cancel"

    def _delete_employee(self, employee: Employee) -> None:
        with self.session_factory() as session:
            db_employee = session.get(Employee, employee.id)
            if not db_employee:
                return
            session.delete(db_employee)
            session.commit()
        audit_logger.log(
            "employee_delete",
            self.actor["username"],
            role=self.actor.get("role"),
            details={"employee_id": employee.id, "full_name": employee.full_name},
        )
        self.refresh_table()

    def _delete_employees(self, employees: List[Employee]) -> None:
        ids = [employee.id for employee in employees if employee.id is not None]
        if not ids:
            return
        with self.session_factory() as session:
            for emp_id in ids:
                db_employee = session.get(Employee, emp_id)
                if db_employee:
                    session.delete(db_employee)
            session.commit()
        audit_logger.log(
            "employee_bulk_delete",
            self.actor["username"],
            role=self.actor.get("role"),
            details={"employee_ids": ids, "count": len(ids)},
        )
        self.refresh_table()

    def _update_employee_status(self, employee: Employee, new_status: str) -> None:
        with self.session_factory() as session:
            db_employee = session.get(Employee, employee.id)
            if not db_employee:
                return
            db_employee.status = new_status
            db_employee.updated_at = datetime.datetime.now(datetime.timezone.utc)
            session.commit()
        audit_logger.log(
            "employee_deactivate" if new_status == "inactive" else "employee_activate",
            self.actor["username"],
            role=self.actor.get("role"),
            details={"employee_id": employee.id, "full_name": employee.full_name},
        )
        self.refresh_table()

    def _bulk_update_employee_status(self, employees: List[Employee], new_status: str) -> None:
        ids = [employee.id for employee in employees if employee.id is not None]
        if not ids:
            return
        now = datetime.datetime.now(datetime.timezone.utc)
        with self.session_factory() as session:
            for emp_id in ids:
                db_employee = session.get(Employee, emp_id)
                if not db_employee:
                    continue
                db_employee.status = new_status
                db_employee.updated_at = now
            session.commit()
        audit_logger.log(
            "employee_bulk_deactivate" if new_status == "inactive" else "employee_bulk_activate",
            self.actor["username"],
            role=self.actor.get("role"),
            details={"employee_ids": ids, "count": len(ids)},
        )
        self.refresh_table()

    def manage_availability(self) -> None:
        employee = self.selected_employee()
        if not employee:
            return
        dialog = UnavailabilityDialog(self.session_factory, self.actor, employee.id)
        dialog.setStyleSheet(THEME_STYLESHEET)
        dialog.exec()

    def manage_role_wages(self) -> None:
        employee = self.selected_employee()
        if not employee:
            return
        with self.session_factory() as session:
            overrides = get_employee_role_wages(session, [employee.id]).get(employee.id, {})
        available_roles = employee.role_list or []
        dialog = EmployeeRoleWageDialog(available_roles, overrides)
        dialog.setStyleSheet(THEME_STYLESHEET)
        if dialog.exec() == QDialog.Accepted:
            with self.session_factory() as session:
                save_employee_role_wages(session, employee.id, dialog.overrides)


class EmployeeRoleWageDialog(QDialog):
    def __init__(self, roles: List[str], overrides: Dict[str, float]) -> None:
        super().__init__()
        self.setWindowTitle("Role wage overrides")
        self.available_roles = sorted(set(role for role in roles if role))
        self.overrides: Dict[str, float] = {role: float(value) for role, value in (overrides or {}).items()}
        self._build_ui()
        self._refresh_list()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        form = QHBoxLayout()
        self.role_combo = QComboBox()
        self.role_combo.setEditable(True)
        self.role_combo.setInsertPolicy(QComboBox.NoInsert)
        for role in self.available_roles:
            self.role_combo.addItem(role)
        self.role_combo.setCurrentIndex(-1)
        form.addWidget(self.role_combo)

        self.wage_input = QDoubleSpinBox()
        self.wage_input.setPrefix("$")
        self.wage_input.setDecimals(2)
        self.wage_input.setRange(0.0, 1000.0)
        self.wage_input.setValue(15.0)
        form.addWidget(self.wage_input)

        add_button = QPushButton("Add / update")
        add_button.clicked.connect(self._add_override)
        form.addWidget(add_button)
        layout.addLayout(form)

        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(QAbstractItemView.SingleSelection)
        layout.addWidget(self.list_widget)

        remove_button = QPushButton("Remove selected")
        remove_button.clicked.connect(self._remove_selected)
        layout.addWidget(remove_button)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _refresh_list(self) -> None:
        self.list_widget.clear()
        for role, wage in sorted(self.overrides.items()):
            self.list_widget.addItem(f"{role} — ${wage:.2f}")

    def _add_override(self) -> None:
        role = (self.role_combo.currentText() or "").strip()
        if not role:
            return
        wage = round(float(self.wage_input.value()), 2)
        self.overrides[role] = wage
        self._refresh_list()

    def _remove_selected(self) -> None:
        current = self.list_widget.currentItem()
        if not current:
            return
        role = current.text().split(" — ", 1)[0].strip()
        if role in self.overrides:
            del self.overrides[role]
        self._refresh_list()


class ChangePasswordDialog(QDialog):
    def __init__(self, store: AccountStore, username: str) -> None:
        super().__init__()
        self.store = store
        self.username = username
        self.password_changed = False
        self.setWindowTitle("Change password")
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        heading = QLabel(
            f"<b>Update password for <span style='color:{ACCENT_COLOR};'>{self.username}</span></b>"
        )
        heading.setWordWrap(True)
        layout.addWidget(heading)

        form = QFormLayout()
        self.current_input = QLineEdit()
        self.current_input.setEchoMode(QLineEdit.Password)
        self.current_input.setPlaceholderText("Current password")
        form.addRow("Current password", self.current_input)

        self.new_input = QLineEdit()
        self.new_input.setEchoMode(QLineEdit.Password)
        self.new_input.setPlaceholderText(f"New password (min {MIN_PASSWORD_LENGTH} chars)")
        form.addRow("New password", self.new_input)

        self.confirm_input = QLineEdit()
        self.confirm_input.setEchoMode(QLineEdit.Password)
        self.confirm_input.setPlaceholderText("Confirm new password")
        form.addRow("Confirm password", self.confirm_input)

        self.feedback_label = QLabel()
        self.feedback_label.setStyleSheet(f"color:{ERROR_COLOR};")
        self.feedback_label.setWordWrap(True)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.attempt_change)
        buttons.rejected.connect(self.reject)

        layout.addLayout(form)
        layout.addWidget(self.feedback_label)
        layout.addWidget(buttons)
        self.current_input.setFocus()

    def attempt_change(self) -> None:
        current_password = self.current_input.text()
        new_password = self.new_input.text()
        confirm_password = self.confirm_input.text()

        if new_password != confirm_password:
            self.feedback_label.setText("New passwords do not match.")
            self.new_input.clear()
            self.confirm_input.clear()
            return

        try:
            self.store.change_password(self.username, current_password, new_password)
        except PermissionError:
            self.feedback_label.setText("Current password is incorrect.")
            self.current_input.clear()
        except ValueError as exc:
            self.feedback_label.setText(str(exc))
            self.new_input.clear()
            self.confirm_input.clear()
        else:
            self.password_changed = True
            self.feedback_label.setText("")
            self.current_input.clear()
            self.new_input.clear()
            self.confirm_input.clear()
            self.accept()

class MainWindow(QMainWindow):
    def __init__(self, store: AccountStore, user: Dict[str, str], session_factory) -> None:
        super().__init__()
        self.store = store
        self.user = user
        self.user_role = self.user.get("role")
        self.session_factory = session_factory
        self.active_week = load_active_week(self.session_factory)
        self.setWindowTitle("Schedule Assistant")
        self.setMinimumSize(1100, 720)
        self._resize_to_screen()
        self._build_ui()
        self._init_session_timeout()

    def _build_week_preparation_tab(self) -> QScrollArea:
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setAlignment(Qt.AlignTop)
        display_name = self.user.get("display_name", self.user["username"])
        welcome = QLabel(f"<h1 style='color:#f5b942;'>Welcome, {display_name}!</h1>")
        role_label = QLabel(f"Current role: <b>{self.user['role']}</b>")
        role_label.setStyleSheet("color:#c9cede;")
        intro = QLabel(
            "Use the week preparation workspace below to capture projected sales and highlight high-impact days "
            "before you build and validate the schedule."
        )
        intro.setWordWrap(True)

        header_row = QHBoxLayout()
        header_titles = QVBoxLayout()
        header_titles.setSpacing(2)
        header_titles.addWidget(welcome)
        header_titles.addWidget(role_label)
        header_row.addLayout(header_titles)
        header_row.addStretch()
        logout_button = QPushButton("Sign out")
        logout_button.clicked.connect(self.handle_logout)
        header_row.addWidget(logout_button)
        layout.addLayout(header_row)
        layout.addSpacing(12)

        self.week_selector = WeekSelectorWidget(
            self.session_factory,
            self.active_week,
            self.on_week_changed,
        )
        layout.addWidget(self.week_selector)

        if self.user["role"] in {"IT", "GM", "SM"}:
            layout.addSpacing(12)
            top_actions = QHBoxLayout()
            top_actions.setSpacing(14)
            employees_button = QPushButton("Employee directory")
            employees_button.clicked.connect(self.open_employee_directory)
            top_actions.addWidget(employees_button)
            if self.user["role"] in {"IT", "GM"}:
                wages_button = QPushButton("Edit wages")
                wages_button.clicked.connect(self.open_wage_manager)
                top_actions.addWidget(wages_button)
            top_actions.addStretch()
            layout.addLayout(top_actions)

        layout.addSpacing(12)
        layout.addWidget(intro)
        layout.addSpacing(16)

        self.preparation_widget = DemandPlanningWidget(
            self.session_factory,
            self.user,
            self.active_week,
        )
        layout.addWidget(self.preparation_widget)
        layout.addStretch(1)

        footer_row = QHBoxLayout()
        footer_row.setSpacing(14)
        if self.user["role"] in {"IT", "GM"}:
            manage_accounts_button = QPushButton("Manage accounts")
            manage_accounts_button.clicked.connect(self.open_account_manager)
            footer_row.addWidget(manage_accounts_button)

        self.policy_button = QPushButton("View policy")
        self.policy_button.clicked.connect(self.open_policy_manager)

        if self.user_role in {"IT", "GM"}:
            footer_row.addWidget(self.policy_button)
        elif SHOW_POLICY_TO_SM and self.user_role == "SM":
            footer_row.addWidget(self.policy_button)
        else:
            self.policy_button.setEnabled(False)

        change_password_button = QPushButton("Change password")
        change_password_button.clicked.connect(self.open_change_password)
        footer_row.addWidget(change_password_button)
        footer_row.addStretch()
        layout.addLayout(footer_row)

        scroll_area.setWidget(content)
        return scroll_area

    def _build_ui(self) -> None:
        prep_tab = self._build_week_preparation_tab()
        self.tabs = QTabWidget()
        self.week_schedule_page = WeekSchedulePage(
            self.session_factory,
            self.user,
            self.active_week,
            on_week_changed=self._handle_schedule_week_change,
            on_back=lambda: self.tabs.setCurrentIndex(0),
        )
        self.validation_page = ValidationImportExportPage(
            self.session_factory,
            EmployeeSessionLocal,
            self.user,
            self.active_week,
            on_week_changed=self._handle_validation_week_change,
            on_status_updated=self._handle_validation_status_updated,
            on_week_data_changed=self._handle_week_data_changed,
        )
        self.tabs.addTab(prep_tab, "Week Preparation")
        self.tabs.addTab(self._wrap_tab(self.week_schedule_page), "Week Schedule")
        self.tabs.addTab(self._wrap_tab(self.validation_page), "Validate / Import / Export")
        enabled = self.week_schedule_page.week_start is not None
        self.tabs.setTabEnabled(1, enabled)
        self.tabs.setTabEnabled(2, enabled)
        self.tabs.setCurrentIndex(0)
        self.setCentralWidget(self.tabs)

    def _resize_to_screen(self) -> None:
        screen = QApplication.primaryScreen()
        if screen is None:
            self.resize(1280, 840)
            return
        available = screen.availableGeometry()
        width = max(int(available.width() * 0.85), self.minimumWidth())
        height = max(int(available.height() * 0.85), self.minimumHeight())
        self.resize(width, height)
        center = available.center()
        self.move(center.x() - width // 2, center.y() - height // 2)

    def on_week_changed(self, iso_year: int, iso_week: int, label: str) -> None:
        if (
            iso_year == self.active_week["iso_year"]
            and iso_week == self.active_week["iso_week"]
        ):
            return
        self.active_week = {"iso_year": iso_year, "iso_week": iso_week, "label": label}
        save_active_week(iso_year, iso_week)
        if hasattr(self, "preparation_widget") and self.preparation_widget:
            self.preparation_widget.set_active_week(self.active_week)
        if hasattr(self, "week_schedule_page") and self.week_schedule_page:
            self.week_schedule_page.set_active_week(self.active_week)
        if hasattr(self, "validation_page") and self.validation_page:
            self.validation_page.set_active_week(self.active_week)
        if hasattr(self, "tabs"):
            self.tabs.setTabEnabled(1, True)
            if self.tabs.count() > 2:
                self.tabs.setTabEnabled(2, True)
        audit_logger.log(
            "week_context_change",
            self.user["username"],
            role=self.user.get("role"),
            details={"iso_year": iso_year, "iso_week": iso_week, "label": label},
        )
        self.reset_session_timers()

    def _wrap_tab(self, widget: QWidget) -> QScrollArea:
        """Wrap heavyweight tab content in a scroll area so everything stays reachable."""
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(widget)
        return scroll_area

    def _handle_schedule_week_change(self, iso_year: int, iso_week: int, label: str) -> None:
        computed_label = week_label(iso_year, iso_week)
        self.on_week_changed(iso_year, iso_week, computed_label)
        if hasattr(self, "week_selector"):
            self.week_selector.set_active_week(
                {"iso_year": iso_year, "iso_week": iso_week, "label": computed_label}
            )

    def _handle_validation_week_change(self, iso_year: int, iso_week: int, label: str) -> None:
        computed_label = week_label(iso_year, iso_week)
        self.on_week_changed(iso_year, iso_week, computed_label)
        if hasattr(self, "week_selector"):
            self.week_selector.set_active_week(
                {"iso_year": iso_year, "iso_week": iso_week, "label": computed_label}
            )

    def _handle_validation_status_updated(self, status: str) -> None:
        if hasattr(self, "week_schedule_page") and self.week_schedule_page:
            self.week_schedule_page.refresh_all()

    def _handle_week_data_changed(self) -> None:
        if hasattr(self, "preparation_widget") and self.preparation_widget:
            self.preparation_widget.refresh()
        if hasattr(self, "week_schedule_page") and self.week_schedule_page:
            self.week_schedule_page.refresh_all()

    def _init_session_timeout(self) -> None:
        self._warning_shown = False
        self.warning_timer = QTimer(self)
        self.warning_timer.setSingleShot(True)
        self.warning_timer.timeout.connect(self._show_timeout_warning)

        self.logout_timer = QTimer(self)
        self.logout_timer.setSingleShot(True)
        self.logout_timer.timeout.connect(self._handle_session_expired)

        self.reset_session_timers()
        self.installEventFilter(self)
        if self.centralWidget():
            self.centralWidget().installEventFilter(self)
        app = QApplication.instance()
        if app:
            app.installEventFilter(self)

    def reset_session_timers(self) -> None:
        if hasattr(self, "warning_timer") and hasattr(self, "logout_timer"):
            self._warning_shown = False
            self.warning_timer.start(SESSION_WARNING_SECONDS * 1000)
            self.logout_timer.start(SESSION_TIMEOUT_SECONDS * 1000)

    def _show_timeout_warning(self) -> None:
        if self._warning_shown:
            return
        self._warning_shown = True
        QMessageBox.warning(
            self,
            "Session timeout",
            "Inactivity detected. You will be signed out in 60 seconds unless you continue working.",
        )

    def _handle_session_expired(self) -> None:
        audit_logger.log(
            "session_timeout",
            self.user["username"],
            role=self.user.get("role"),
            details={
                "iso_year": self.active_week.get("iso_year"),
                "iso_week": self.active_week.get("iso_week"),
            },
        )
        QMessageBox.information(
            self,
            "Session timed out",
            "You were signed out due to inactivity.",
        )
        app = QApplication.instance()
        if app:
            app.closeAllWindows()
        self.close()

    def _refresh_validation_notes(self) -> None:
        self.validation_list.clear()
        week_start = self._current_week_start()
        if not week_start:
            return
        with self.session_factory() as session:
            summary = get_week_summary(session, week_start)
            shifts = get_shifts_for_week(session, week_start)
            policy = load_active_policy_spec(self.session_factory)
        # Day coverage warnings
        empty_days = [day["date"] for day in summary.get("days", []) if day.get("count", 0) == 0]
        if empty_days:
            self.validation_list.addItem(f"No coverage scheduled for: {', '.join(empty_days)}")
        # Unassigned shift warnings
        unassigned = [s for s in shifts if not s.get("employee_id")]
        if unassigned:
            self.validation_list.addItem(f"{len(unassigned)} unassigned shift(s) remain.")
        # Wage confirmation warnings
        missing_wages = validate_wages(role_catalog(policy))
        if missing_wages:
            roles_list = ", ".join(sorted(missing_wages.keys()))
            self.validation_list.addItem(f"Wages missing/unchecked for: {roles_list}")
        # Budget reminder if labor spend is zero
        if summary.get("total_cost", 0) <= 0:
            self.validation_list.addItem("No labor cost recorded yet. Validate after generating schedule.")

    def eventFilter(self, source, event) -> bool:
        if event.type() in (
            QEvent.MouseButtonPress,
            QEvent.MouseButtonRelease,
            QEvent.KeyPress,
            QEvent.KeyRelease,
            QEvent.Wheel,
        ):
            self.reset_session_timers()
        return super().eventFilter(source, event)

    def open_account_manager(self) -> None:
        dialog = AccountManagerDialog(self.store, self.user)
        dialog.setStyleSheet(THEME_STYLESHEET)
        dialog.exec()

    def open_employee_directory(self) -> None:
        dialog = EmployeeDirectoryDialog(EmployeeSessionLocal, self.user)
        dialog.setStyleSheet(THEME_STYLESHEET)
        dialog.exec()

    def open_wage_manager(self) -> None:
        dialog = WageManagerDialog()
        dialog.setStyleSheet(THEME_STYLESHEET)
        dialog.exec()

    def open_change_password(self) -> None:
        dialog = ChangePasswordDialog(self.store, self.user["username"])
        dialog.setStyleSheet(THEME_STYLESHEET)
        if dialog.exec() == QDialog.Accepted and dialog.password_changed:
            QMessageBox.information(self, "Password updated", "Your password has been changed successfully.")

    def open_policy_manager(self) -> None:
        role = self.user_role
        sm_allowed = SHOW_POLICY_TO_SM and role == "SM"
        can_edit = role in {"GM", "IT"} or sm_allowed
        if not can_edit:
            QMessageBox.information(
                self,
                "Access restricted",
                "Access restricted to General Managers and IT Assistants.",
            )
            return
        if role == "SM":
            proceed = QMessageBox.question(
                self,
                "Heads up",
                "Policy edits apply to the entire system and affect every store week. Continue?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if proceed != QMessageBox.Yes:
                return
        dialog = PolicyDialog(self.session_factory, self.user, read_only=not can_edit)
        dialog.setStyleSheet(THEME_STYLESHEET)
        dialog.exec()

    def handle_logout(self) -> None:
        confirm = QMessageBox.question(self, "Sign out", "Return to sign-in screen?")
        if confirm == QMessageBox.Yes:
            self.close()

    def closeEvent(self, event: QCloseEvent) -> None:
        event.accept()


def launch_app() -> int:
    app = QApplication(sys.argv)
    icon = QIcon(str(ICON_FILE)) if ICON_FILE.exists() else None
    if icon is not None:
        app.setWindowIcon(icon)
    app.setStyleSheet(THEME_STYLESHEET)

    store = AccountStore(ACCOUNTS_FILE)
    init_database()
    ensure_default_policy(SessionLocal)

    while True:
        login = LoginDialog(store)
        login.setStyleSheet(THEME_STYLESHEET)
        if login.exec() != QDialog.Accepted:
            break

        authenticated = login.authenticated_user
        if not authenticated:
            continue

        window = MainWindow(store, authenticated, SessionLocal)
        window.setStyleSheet(THEME_STYLESHEET)
        if icon is not None:
            window.setWindowIcon(icon)
        window.show()
        app.exec()

        logout = QMessageBox.question(
            None,
            "Session ended",
            "Do you want to sign in again?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if logout != QMessageBox.Yes:
            break

    return 0


if __name__ == "__main__":
    sys.exit(launch_app())
