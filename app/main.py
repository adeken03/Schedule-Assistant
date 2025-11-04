from __future__ import annotations

import base64
import calendar
import datetime
import hashlib
import json
import secrets
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from sqlalchemy import select

from PySide6.QtCore import Qt, QDate, QTime, QEvent, QTimer
from PySide6.QtGui import QCloseEvent, QIntValidator
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QComboBox,
    QDateEdit,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QScrollArea,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTimeEdit,
    QVBoxLayout,
    QWidget,
    QHeaderView,
)

from database import (
    Employee,
    EmployeeUnavailability,
    Modifier,
    Policy,
    SessionLocal,
    WeekDailyProjection,
    get_all_employees,
    get_all_weeks,
    get_or_create_week,
    get_week_daily_projections,
    get_week_modifiers,
    get_policies,
    upsert_policy,
    delete_policy,
    get_active_policy,
    init_database,
    save_week_daily_projection_values,
)


DATA_DIR = Path(__file__).resolve().parent / "data"
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
        week = get_or_create_week(session, iso_year, iso_week, label)
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
    "Managers": [
        "Manager FOH",
        "Manager HOH",
    ],
    "Bartenders": [
        "Bartender",
        "Bartender - Opener",
        "Bartender - Closer",
    ],
    "Cashier / Guest Services": [
        "Cashier",
        "Cashier - To-Go Specialist",
    ],
    "Servers - Dining": [
        "Server - Dining",
        "Server - Dining Opener",
        "Server - Dining Preclose",
        "Server - Dining Closer",
        "Server - Patio",
    ],
    "Servers - Cocktail": [
        "Server - Cocktail",
        "Server - Cocktail Opener",
        "Server - Cocktail Preclose",
        "Server - Cocktail Closer",
    ],
    "Kitchen": [
        "Kitchen Opener",
        "Kitchen Closer",
        "Expo",
        "Grill",
        "Chip",
        "Shake",
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
    background-color: #0b0b0d;
    color: #f7f7f7;
    font-family: 'Segoe UI', sans-serif;
    font-size: 17px;
}

QPushButton {
    background-color: #b8860b;
    color: #0b0b0d;
    border-radius: 7px;
    padding: 10px 17px;
    font-weight: 600;
}

QPushButton:hover {
    background-color: #c99821;
}

QPushButton:disabled {
    background-color: #2a2a2a;
    color: #777777;
}

QLineEdit, QComboBox {
    background-color: #16161d;
    border: 1px solid #2c2c33;
    border-radius: 7px;
    padding: 7px 12px;
    color: #f7f7f7;
}

QLineEdit:focus, QComboBox:focus {
    border-color: #b8860b;
}

QTableWidget {
    background-color: #16161d;
    border: 1px solid #2c2c33;
    border-radius: 10px;
    gridline-color: #2c2c33;
}

QHeaderView::section {
    background-color: #141418;
    color: #f7f7f7;
    padding: 7px;
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
        heading = QLabel("<h2 style='color:#b8860b;'>Sign in to Schedule Assistant</h2>")
        subheading = QLabel("Access is limited to authorized staff.")
        subheading.setStyleSheet("color:#ddd;")

        self.username_input = QLineEdit()
        self.username_input.setPlaceholderText("Username")
        self.password_input = QLineEdit()
        self.password_input.setPlaceholderText("Password")
        self.password_input.setEchoMode(QLineEdit.Password)

        form = QFormLayout()
        form.addRow("Username", self.username_input)
        form.addRow("Password", self.password_input)

        self.error_label = QLabel()
        self.error_label.setStyleSheet("color: #ff6b6b;")
        self.error_label.setWordWrap(True)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.attempt_login)
        button_box.rejected.connect(self.reject)

        layout.addWidget(heading)
        layout.addWidget(subheading)
        layout.addSpacing(10)
        layout.addLayout(form)
        layout.addWidget(self.error_label)
        layout.addWidget(button_box)

    def attempt_login(self) -> None:
        username = self.username_input.text().strip()
        password = self.password_input.text()
        try:
            account = self.store.verify_credentials(username, password)
        except AccountLockedError as exc:
            self.password_input.clear()
            locked_local = exc.until.astimezone()
            message = locked_local.strftime("Account locked until %Y-%m-%d %H:%M %Z.")
            self.error_label.setText(message)
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
            self.error_label.setText("Invalid username or password.")
            audit_logger.log(
                "login_failure",
                username,
                details={"reason": "invalid_credentials"},
            )
            return

        self.error_label.setText("")
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
            self.feedback_label.setText(f"<span style='color:#ff6b6b;'>{exc}</span>")
            return

        self.feedback_label.setText(f"<span style='color:#3cb371;'>Created {role} account for {username}.</span>")
        self.new_username.clear()
        self.new_password.clear()
        self.refresh_table()

    def handle_delete(self) -> None:
        selected = self.table.selectedItems()
        if not selected:
            self.feedback_label.setText("<span style='color:#ff6b6b;'>Select an account to delete.</span>")
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
            self.feedback_label.setText(f"<span style='color:#ff6b6b;'>{exc}</span>")
            return

        self.feedback_label.setText(f"<span style='color:#3cb371;'>Deleted account '{username}'.</span>")
        self.refresh_table()


class WeekSelectorWidget(QWidget):
    def __init__(self, session_factory, active_week: Dict[str, Any], on_change) -> None:
        super().__init__()
        self.session_factory = session_factory
        self.on_change = on_change
        self.active_week = active_week
        self._updating = False
        self._build_ui()
        self.set_active_week(active_week)

    def _build_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        layout.addWidget(QLabel("Active week"))

        self.week_combo = QComboBox()
        self.week_combo.currentIndexChanged.connect(self._handle_combo_change)
        layout.addWidget(self.week_combo)

        self.week_date = QDateEdit()
        self.week_date.setCalendarPopup(True)
        self.week_date.setDisplayFormat("yyyy-MM-dd")
        layout.addWidget(self.week_date)

        self.apply_button = QPushButton("Select week")
        self.apply_button.clicked.connect(self._handle_apply_clicked)
        layout.addWidget(self.apply_button)

        layout.addStretch(1)

    def set_active_week(self, active_week: Dict[str, Any]) -> None:
        self.active_week = active_week
        with self.session_factory() as session:
            weeks = get_all_weeks(session)
            if not weeks:
                week = get_or_create_week(
                    session,
                    active_week["iso_year"],
                    active_week["iso_week"],
                    active_week["label"],
                )
                weeks = [week]
        self._updating = True
        self.week_combo.clear()
        selected_index = 0
        for idx, week in enumerate(weeks):
            self.week_combo.addItem(week.label, (week.iso_year, week.iso_week))
            if (
                week.iso_year == active_week["iso_year"]
                and week.iso_week == active_week["iso_week"]
            ):
                selected_index = idx
        self.week_combo.setCurrentIndex(selected_index)
        self._updating = False
        self._update_date_edit()

    def _update_date_edit(self) -> None:
        start = week_start_date(self.active_week["iso_year"], self.active_week["iso_week"])
        self.week_date.setDate(QDate(start.year, start.month, start.day))

    def _handle_combo_change(self) -> None:
        if self._updating:
            return
        data = self.week_combo.currentData()
        if not data:
            return
        iso_year, iso_week = data
        label = self.week_combo.currentText()
        self.active_week = {"iso_year": iso_year, "iso_week": iso_week, "label": label}
        self._update_date_edit()
        if self.on_change:
            self.on_change(iso_year, iso_week, label)

    def _handle_apply_clicked(self) -> None:
        qdate = self.week_date.date()
        selected_date = datetime.date(qdate.year(), qdate.month(), qdate.day())
        iso_year, iso_week, _ = selected_date.isocalendar()
        label = week_label(iso_year, iso_week)
        with self.session_factory() as session:
            week = get_or_create_week(session, iso_year, iso_week, label)
            label = week.label
        self.set_active_week({"iso_year": iso_year, "iso_week": iso_week, "label": label})
        if self.on_change:
            self.on_change(iso_year, iso_week, label)


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

        self.feedback_label = QLabel()
        self.feedback_label.setStyleSheet("color:#ff6b6b;")
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
        self.day_inputs: Dict[int, QLineEdit] = {}
        self.day_note_inputs: Dict[int, QLineEdit] = {}
        self.heat_labels: Dict[int, QLabel] = {}
        self._pending_changes = False
        self._modifier_column_ratios: Dict[int, float] = {
            0: 0.20,  # Title
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

        buttons_row = QHBoxLayout()
        self.save_status_label = QLabel()
        buttons_row.addWidget(self.save_status_label)
        buttons_row.addStretch()
        self.save_button = QPushButton("Save daily projections")
        self.save_button.clicked.connect(self.handle_save_projections)
        buttons_row.addWidget(self.save_button)
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
            label.setMinimumWidth(100)
            label.setFixedHeight(72)
            label.setStyleSheet(
                "border:1px solid #334155; border-radius:6px; background-color:#1e2937; color:#e2e8f0;"
            )
            self.heat_labels[day_index] = label
            heat_row.addWidget(label)
        heat_row.addStretch()
        heat_layout.addLayout(heat_row)

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
        header = self.modifier_table.horizontalHeader()
        header.setStretchLastSection(False)
        header.setMinimumSectionSize(80)
        for column in range(self.modifier_table.columnCount()):
            header.setSectionResizeMode(column, QHeaderView.Interactive)
        self.modifier_table.itemSelectionChanged.connect(self._update_modifier_buttons)
        modifier_layout.addWidget(self.modifier_table)
        self._apply_modifier_column_layout()

        self.modifier_feedback = QLabel()
        self.modifier_feedback.setStyleSheet("color:#9f9f9f;")
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
        modifier_buttons.addStretch()
        modifier_layout.addLayout(modifier_buttons)

        layout.addWidget(self.modifier_group)
        layout.addStretch(1)

    def set_active_week(self, active_week: Dict[str, Any]) -> None:
        self.active_week = active_week
        self.week_label = active_week.get("label", "")
        self.refresh()

    def refresh(self) -> None:
        iso_year = self.active_week.get("iso_year")
        iso_week = self.active_week.get("iso_week")
        label = self.active_week.get("label") or ""
        with self.session_factory() as session:
            week = get_or_create_week(session, iso_year, iso_week, label)
            self.week_id = week.id
            self.week_label = week.label
            self.projections = get_week_daily_projections(session, week.id)
            self.modifiers = get_week_modifiers(session, week.id)
        self._populate_projection_inputs()
        self._refresh_modifiers_table()
        self._apply_heatmap()
        self._update_group_titles()
        self._update_modifier_buttons()

    def _update_group_titles(self) -> None:
        suffix = f" – {self.week_label}" if self.week_label else ""
        self.projection_group.setTitle(f"Projected sales by day{suffix}")
        self.modifier_group.setTitle(f"Modifiers{suffix}")
        self.heat_group.setTitle(f"Sales heat map{suffix}")

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

    def _set_saved_state(self, saved: bool) -> None:
        if saved:
            self.save_status_label.setText("All changes saved")
            self.save_status_label.setStyleSheet("color:#3cb371;")
            self._pending_changes = False
        else:
            self.save_status_label.setText("Unsaved changes")
            self.save_status_label.setStyleSheet("color:#ffd166;")
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
            label.setStyleSheet(
                f"border:1px solid #334155; border-radius:6px; color:#e2e8f0; background-color:{palette};"
            )
            delta = adjusted - base
            label.setText(
                f"{DAYS_OF_WEEK[day]}\n{self._format_currency(adjusted)}\n{self._format_delta(delta)}"
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
            window_text = f"{format_time_label(modifier.start_time)} – {format_time_label(modifier.end_time)}"
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

    def handle_add_modifier(self) -> None:
        dialog = ModifierDialog(self.modifiers)
        dialog.setStyleSheet(THEME_STYLESHEET)
        if dialog.exec() != QDialog.Accepted or not dialog.result_data or self.week_id is None:
            return
        data = dialog.result_data
        with self.session_factory() as session:
            modifier = Modifier(
                week_id=self.week_id,
                title=data["title"],
                modifier_type="increase" if int(data["pct_change"]) >= 0 else "decrease",
                day_of_week=int(data["day_of_week"]),
                start_time=data["start_time"],
                end_time=data["end_time"],
                pct_change=int(data["pct_change"]),
                notes=data.get("notes", ""),
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
        self.modifier_feedback.setStyleSheet("color:#3cb371;")
        self.modifier_feedback.setText(f"Added modifier '{modifier.title}'.")
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
        impact_type = "increase" if int(data["pct_change"]) >= 0 else "decrease"
        with self.session_factory() as session:
            modifier = session.get(Modifier, current.id)
            if not modifier:
                QMessageBox.warning(self, "Modifier missing", "The selected modifier no longer exists.")
                self.refresh()
                return
            modifier.title = data["title"]
            modifier.modifier_type = impact_type
            modifier.day_of_week = int(data["day_of_week"])
            modifier.start_time = data["start_time"]
            modifier.end_time = data["end_time"]
            modifier.pct_change = int(data["pct_change"])
            modifier.notes = data.get("notes", "")
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
                "day_of_week": int(data["day_of_week"]),
                "start_time": data["start_time"].isoformat(),
                "end_time": data["end_time"].isoformat(),
                "pct_change": int(data["pct_change"]),
            },
        )
        self.modifier_feedback.setStyleSheet("color:#3cb371;")
        self.modifier_feedback.setText(f"Updated modifier '{data['title']}'.")
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
        self.modifier_feedback.setStyleSheet("color:#ffd166;")
        self.modifier_feedback.setText(f"Deleted modifier '{current.title}'.")
        self.refresh()

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._apply_modifier_column_layout()


class PolicyEditDialog(QDialog):
    def __init__(self, *, name: str = "", params: Optional[Dict[str, Any]] = None) -> None:
        super().__init__()
        self.setWindowTitle("Edit policy" if name else "Add policy")
        self.result_data: Optional[Dict[str, Any]] = None
        layout = QVBoxLayout(self)

        form = QFormLayout()
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("e.g. Default Policy")
        if name:
            self.name_input.setText(name)
        form.addRow("Name", self.name_input)

        self.json_edit = QPlainTextEdit()
        self.json_edit.setPlaceholderText("{\"rule_key\": value, ...}")
        pretty = json.dumps(params or {}, indent=2)
        self.json_edit.setPlainText(pretty)
        form.addRow("Parameters (JSON)", self.json_edit)
        layout.addLayout(form)

        self.feedback_label = QLabel()
        self.feedback_label.setStyleSheet("color:#ff6b6b;")
        layout.addWidget(self.feedback_label)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def accept(self) -> None:
        name = self.name_input.text().strip()
        if not name:
            self.feedback_label.setText("Provide a name for this policy.")
            return
        raw = self.json_edit.toPlainText().strip() or "{}"
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as e:
            self.feedback_label.setText(f"Invalid JSON: {e}")
            return
        if not isinstance(parsed, dict):
            self.feedback_label.setText("Top-level JSON must be an object (dictionary).")
            return
        self.result_data = {"name": name, "params": parsed}
        super().accept()


class PolicyDialog(QDialog):
    def __init__(self, session_factory, current_user: Dict[str, str], *, read_only: bool = False) -> None:
        super().__init__()
        self.session_factory = session_factory
        self.current_user = current_user
        self.read_only = read_only
        self.setWindowTitle("Policies")
        self.policies: List[Policy] = []
        self._build_ui()
        self._load_policies()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        intro = QLabel("Define schedule generation and validation rules. JSON must be a valid object.")
        intro.setWordWrap(True)
        layout.addWidget(intro)

        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["ID", "Name", "Parameters", "Last Edited By", "Last Edited At"])
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.Stretch)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        layout.addWidget(self.table)

        self.feedback_label = QLabel()
        self.feedback_label.setStyleSheet("color:#9f9f9f;")
        layout.addWidget(self.feedback_label)

        buttons = QHBoxLayout()
        self.add_button = QPushButton("Add")
        self.edit_button = QPushButton("Edit")
        self.delete_button = QPushButton("Delete")
        self.close_button = QPushButton("Close")
        self.add_button.clicked.connect(self.handle_add)
        self.edit_button.clicked.connect(self.handle_edit)
        self.delete_button.clicked.connect(self.handle_delete)
        self.close_button.clicked.connect(self.reject)
        buttons.addWidget(self.add_button)
        buttons.addWidget(self.edit_button)
        buttons.addWidget(self.delete_button)
        buttons.addStretch()
        buttons.addWidget(self.close_button)
        layout.addLayout(buttons)

        if self.read_only:
            self.add_button.setEnabled(False)
            self.edit_button.setEnabled(False)
            self.delete_button.setEnabled(False)

        self.table.itemSelectionChanged.connect(self._update_buttons)
        self._update_buttons()

    def _update_buttons(self) -> None:
        has_selection = self.table.currentRow() != -1
        can_modify = (not self.read_only) and has_selection
        self.edit_button.setEnabled(can_modify)
        self.delete_button.setEnabled(can_modify)

    def _load_policies(self) -> None:
        with self.session_factory() as session:
            self.policies = get_policies(session)
        self._refresh_table()

    def _refresh_table(self) -> None:
        self.table.setRowCount(len(self.policies))
        for row, policy in enumerate(self.policies):
            params = policy.params_dict()
            summary = ", ".join([f"{k}={v}" for k, v in list(params.items())[:4]])
            if len(params) > 4:
                summary += ", …"
            cells = [
                QTableWidgetItem(str(policy.id)),
                QTableWidgetItem(policy.name),
                QTableWidgetItem(summary),
                QTableWidgetItem(policy.lastEditedBy or ""),
                QTableWidgetItem(policy.lastEditedAt.strftime("%Y-%m-%d %H:%M")),
            ]
            for col, item in enumerate(cells):
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self.table.setItem(row, col, item)

    def _selected_policy(self) -> Optional[Policy]:
        row = self.table.currentRow()
        if row < 0 or row >= len(self.policies):
            return None
        return self.policies[row]

    def handle_add(self) -> None:
        if self.read_only:
            return
        dialog = PolicyEditDialog()
        dialog.setStyleSheet(THEME_STYLESHEET)
        if dialog.exec() != QDialog.Accepted or not dialog.result_data:
            return
        data = dialog.result_data
        with self.session_factory() as session:
            policy = upsert_policy(session, data["name"], data["params"], edited_by=self.current_user.get("username", "system"))
        audit_logger.log(
            "policy_create",
            self.current_user.get("username", "unknown"),
            role=self.current_user.get("role"),
            details={"policy_id": policy.id, "name": policy.name},
        )
        self.feedback_label.setStyleSheet("color:#3cb371;")
        self.feedback_label.setText(f"Added/updated policy '{policy.name}'.")
        self._load_policies()

    def handle_edit(self) -> None:
        if self.read_only:
            return
        policy = self._selected_policy()
        if not policy:
            return
        dialog = PolicyEditDialog(name=policy.name, params=policy.params_dict())
        dialog.setStyleSheet(THEME_STYLESHEET)
        if dialog.exec() != QDialog.Accepted or not dialog.result_data:
            return
        data = dialog.result_data
        with self.session_factory() as session:
            updated = upsert_policy(session, data["name"], data["params"], edited_by=self.current_user.get("username", "system"))
        audit_logger.log(
            "policy_update",
            self.current_user.get("username", "unknown"),
            role=self.current_user.get("role"),
            details={"policy_id": updated.id, "name": updated.name},
        )
        self.feedback_label.setStyleSheet("color:#3cb371;")
        self.feedback_label.setText(f"Updated policy '{updated.name}'.")
        self._load_policies()

    def handle_delete(self) -> None:
        if self.read_only:
            return
        policy = self._selected_policy()
        if not policy:
            return
        confirm = QMessageBox.question(
            self,
            "Delete policy",
            f"Remove policy '{policy.name}'?",
        )
        if confirm != QMessageBox.Yes:
            return
        with self.session_factory() as session:
            delete_policy(session, policy.id)
        audit_logger.log(
            "policy_delete",
            self.current_user.get("username", "unknown"),
            role=self.current_user.get("role"),
            details={"policy_id": policy.id, "name": policy.name},
        )
        self.feedback_label.setStyleSheet("color:#ffd166;")
        self.feedback_label.setText(f"Deleted policy '{policy.name}'.")
        self._load_policies()


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

        self.role_list_widget = QListWidget()
        self.role_list_widget.setSelectionMode(QAbstractItemView.SingleSelection)
        self.role_list_widget.itemSelectionChanged.connect(self._update_role_buttons)
        self.role_list_widget.itemDoubleClicked.connect(lambda *_: self.remove_selected_role())
        roles_layout.addWidget(self.role_list_widget)

        list_button_row = QHBoxLayout()
        self.remove_role_button = QPushButton("Remove selected")
        self.remove_role_button.clicked.connect(self.remove_selected_role)
        list_button_row.addWidget(self.remove_role_button)
        list_button_row.addStretch()
        roles_layout.addLayout(list_button_row)

        form.addRow("Roles", roles_container)

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
        self.feedback_label.setStyleSheet("color: #ff6b6b;")
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
        has_selection = self.role_list_widget.currentItem() is not None
        self.remove_role_button.setEnabled(has_selection)

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
        self.feedback_label.setStyleSheet("color: #ff6b6b;")
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
        self.resize(780, 460)
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
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
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
        selection = self.table.selectionModel()
        if not selection or not selection.hasSelection():
            return None
        selected_rows = selection.selectedRows()
        if not selected_rows:
            return None
        row = selected_rows[0].row()
        if 0 <= row < len(self.visible_employees):
            return self.visible_employees[row]
        return None

    def update_button_state(self) -> None:
        employee = self.selected_employee()
        has_selection = employee is not None
        self.edit_button.setEnabled(has_selection)
        self.toggle_button.setEnabled(has_selection)
        self.availability_button.setEnabled(has_selection)
        if employee:
            self.toggle_button.setText("Deactivate" if employee.status == "active" else "Activate")
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
        employee = self.selected_employee()
        if not employee:
            return
        new_status = "inactive" if employee.status == "active" else "active"
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

    def manage_availability(self) -> None:
        employee = self.selected_employee()
        if not employee:
            return
        dialog = UnavailabilityDialog(self.session_factory, self.actor, employee.id)
        dialog.setStyleSheet(THEME_STYLESHEET)
        dialog.exec()


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
            f"<b>Update password for <span style='color:#b8860b;'>{self.username}</span></b>"
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
        self.feedback_label.setStyleSheet("color: #ff6b6b;")
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
        self.resize(720, 440)
        self._build_ui()
        self._init_session_timeout()

    def _build_ui(self) -> None:
        # Wrap the entire central content in a scroll area so the app is
        # traversible on smaller displays and different Windows variants.
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setAlignment(Qt.AlignTop)
        display_name = self.user.get("display_name", self.user["username"])
        welcome = QLabel(f"<h1 style='color:#b8860b;'>Welcome, {display_name}!</h1>")
        role_label = QLabel(f"Current role: <b>{self.user['role']}</b>")
        role_label.setStyleSheet("color:#ddd;")
        intro = QLabel(
            "Use the demand planning workspace below to capture projected sales and highlight high-impact days "
            "before the scheduling pass."
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
            top_actions.addStretch()
            layout.addLayout(top_actions)

        layout.addSpacing(12)
        layout.addWidget(intro)
        layout.addSpacing(16)

        self.demand_widget = DemandPlanningWidget(
            self.session_factory,
            self.user,
            self.active_week,
        )
        layout.addWidget(self.demand_widget)
        layout.addStretch(1)

        footer_row = QHBoxLayout()
        footer_row.setSpacing(14)
        if self.user["role"] in {"IT", "GM"}:
            manage_accounts_button = QPushButton("Manage accounts")
            manage_accounts_button.clicked.connect(self.open_account_manager)
            footer_row.addWidget(manage_accounts_button)

        # Manage Policies action with role-based visibility
        self.policy_action = QPushButton("Manage policies")
        self.policy_action.clicked.connect(self.open_policy_manager)
        # Visibility and enablement rules
        if self.user_role in {"IT", "GM"}:
            footer_row.addWidget(self.policy_action)
        elif SHOW_POLICY_TO_SM and self.user_role == "SM":
            footer_row.addWidget(self.policy_action)
        # Enforce disabled state for non-editor roles unless read-only is allowed
        if self.user_role not in ("GM", "IT") and not (SHOW_POLICY_TO_SM and self.user_role == "SM"):
            self.policy_action.setEnabled(False)

        change_password_button = QPushButton("Change password")
        change_password_button.clicked.connect(self.open_change_password)
        footer_row.addWidget(change_password_button)
        footer_row.addStretch()
        layout.addLayout(footer_row)

        scroll_area.setWidget(content)
        self.setCentralWidget(scroll_area)

    def on_week_changed(self, iso_year: int, iso_week: int, label: str) -> None:
        if (
            iso_year == self.active_week["iso_year"]
            and iso_week == self.active_week["iso_week"]
        ):
            return
        self.active_week = {"iso_year": iso_year, "iso_week": iso_week, "label": label}
        save_active_week(iso_year, iso_week)
        if hasattr(self, "demand_widget") and self.demand_widget:
            self.demand_widget.set_active_week(self.active_week)
        audit_logger.log(
            "week_context_change",
            self.user["username"],
            role=self.user.get("role"),
            details={"iso_year": iso_year, "iso_week": iso_week, "label": label},
        )
        self.reset_session_timers()

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
        self.close()

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
        dialog = EmployeeDirectoryDialog(self.session_factory, self.user)
        dialog.setStyleSheet(THEME_STYLESHEET)
        dialog.exec()

    def open_change_password(self) -> None:
        dialog = ChangePasswordDialog(self.store, self.user["username"])
        dialog.setStyleSheet(THEME_STYLESHEET)
        if dialog.exec() == QDialog.Accepted and dialog.password_changed:
            QMessageBox.information(self, "Password updated", "Your password has been changed successfully.")

    def open_policy_manager(self) -> None:
        # Determine access level
        if self.user_role in {"GM", "IT"}:
            read_only = False
        elif SHOW_POLICY_TO_SM and self.user_role == "SM":
            read_only = True
        else:
            QMessageBox.information(
                self,
                "Access restricted",
                "Access restricted to General Managers and IT Assistants.",
            )
            return
        dialog = PolicyDialog(self.session_factory, self.user, read_only=read_only)
        dialog.setStyleSheet(THEME_STYLESHEET)
        dialog.exec()

    def handle_logout(self) -> None:
        confirm = QMessageBox.question(self, "Sign out", "Return to sign-in screen?")
        if confirm == QMessageBox.Yes:
            self.close()

    def closeEvent(self, event: QCloseEvent) -> None:
        # On exit, we simply accept. Main application loop handles showing login again if needed.
        event.accept()


def launch_app() -> int:
    app = QApplication(sys.argv)
    app.setStyleSheet(THEME_STYLESHEET)

    store = AccountStore(ACCOUNTS_FILE)
    init_database()

    while True:
        login = LoginDialog(store)
        login.setStyleSheet(THEME_STYLESHEET)
        if login.exec() != QDialog.Accepted:
            break

        authenticated = login.authenticated_user
        if not authenticated:
            # Should not happen since dialog returns Accepted only on success, but guard anyway.
            continue

        window = MainWindow(store, authenticated, SessionLocal)
        window.setStyleSheet(THEME_STYLESHEET)
        window.show()
        app.exec()

        # After window closes (logout or quit), decide whether to prompt for login again.
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



