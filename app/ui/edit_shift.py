from __future__ import annotations

import datetime
from typing import Callable, Dict, List, Optional

from PySide6.QtCore import Qt, QDate, QTime
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QComboBox,
    QDateEdit,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QTimeEdit,
    QVBoxLayout,
)

from policy import hourly_wage
from roles import is_manager_role, palette_for_role, role_group


class EditShiftDialog(QDialog):
    def __init__(
        self,
        *,
        employees: List[Dict],
        roles: List[str],
        policy: Dict,
        week_start: datetime.date,
        default_date: Optional[datetime.date] = None,
        shift: Optional[Dict] = None,
        existing_shifts: Optional[List[Dict]] = None,
        on_save: Optional[Callable[[Dict], None]] = None,
        on_delete: Optional[Callable[[int], None]] = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setModal(True)
        self.employees = employees
        self.roles = sorted([role for role in roles if role])
        self.policy = policy or {}
        self.week_start = week_start
        self.shift = shift
        self.existing_shifts = existing_shifts or []
        self.on_save = on_save
        self.on_delete = on_delete
        self.setWindowTitle("Edit shift" if shift else "Add shift")
        self.default_date = default_date or week_start
        self._build_ui()
        if shift:
            self._load_shift(shift)

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        self.feedback_label = QLabel()
        self.feedback_label.setStyleSheet("color:#ff7a7a;")

        form = QFormLayout()

        self.employee_combo = QComboBox()
        self.employee_combo.setEditable(True)
        self.employee_combo.setInsertPolicy(QComboBox.NoInsert)
        self.employee_combo.setToolTip("Assign a team member or keep unassigned.")
        self.employee_combo.addItem("Unassigned", None)
        for employee in self.employees:
            self.employee_combo.addItem(employee["name"], employee["id"])
        if self.employee_combo.completer():
            self.employee_combo.completer().setCaseSensitivity(Qt.CaseInsensitive)
        if self.employee_combo.lineEdit():
            self.employee_combo.lineEdit().setPlaceholderText("Start typing a team member")
        form.addRow("Employee", self.employee_combo)

        self.role_combo = QComboBox()
        self.role_combo.setEditable(True)
        self.role_combo.setInsertPolicy(QComboBox.NoInsert)
        self.role_combo.setToolTip("Roles are grouped by department; colors match the week grid.")
        for role in self.roles:
            index = self.role_combo.count()
            self.role_combo.addItem(role)
            color = QColor(palette_for_role(role))
            self.role_combo.setItemData(index, color, Qt.BackgroundRole)
            self.role_combo.setItemData(index, Qt.white, Qt.ForegroundRole)
            self.role_combo.setItemData(index, f"{role} \u2014 {role_group(role)}", Qt.ToolTipRole)
        if self.role_combo.completer():
            self.role_combo.completer().setCaseSensitivity(Qt.CaseInsensitive)
        if self.role_combo.lineEdit():
            self.role_combo.lineEdit().setPlaceholderText("Select or type a role")
        form.addRow("Role", self.role_combo)

        self.date_edit = QDateEdit()
        self.date_edit.setCalendarPopup(True)
        target_date = self.default_date or self.week_start
        self.date_edit.setDate(QDate(target_date.year, target_date.month, target_date.day))
        form.addRow("Date", self.date_edit)

        time_row = QHBoxLayout()
        self.start_time = QTimeEdit()
        self.start_time.setDisplayFormat("HH:mm")
        self.start_time.setTime(QTime(11, 0))
        time_row.addWidget(self.start_time)

        self.end_time = QTimeEdit()
        self.end_time.setDisplayFormat("HH:mm")
        self.end_time.setTime(QTime(17, 0))
        time_row.addWidget(self.end_time)
        form.addRow("Time", time_row)

        self.location_input = QLineEdit()
        form.addRow("Location/Section", self.location_input)

        self.notes_input = QPlainTextEdit()
        self.notes_input.setPlaceholderText("Optional notes visible in the grid.")
        form.addRow("Notes", self.notes_input)

        layout.addLayout(form)
        layout.addWidget(self.feedback_label)

        button_box = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self._handle_save)
        button_box.rejected.connect(self.reject)

        self.delete_button = QPushButton("Delete")
        self.delete_button.setVisible(self.shift is not None)
        self.delete_button.clicked.connect(self._handle_delete)

        action_row = QHBoxLayout()
        action_row.addWidget(button_box)
        action_row.addWidget(self.delete_button)
        action_row.addStretch()
        layout.addLayout(action_row)

    def _load_shift(self, shift: Dict) -> None:
        employee_id = shift.get("employee_id")
        if employee_id is None:
            self.employee_combo.setCurrentIndex(0)
        else:
            index = self.employee_combo.findData(employee_id, Qt.UserRole)
            if index >= 0:
                self.employee_combo.setCurrentIndex(index)
        role = shift.get("role")
        if role:
            role_index = self.role_combo.findText(role)
            if role_index >= 0:
                self.role_combo.setCurrentIndex(role_index)
        start = shift.get("start")
        end = shift.get("end")
        if isinstance(start, datetime.datetime):
            local_start = start.astimezone()
            self.date_edit.setDate(QDate(local_start.year, local_start.month, local_start.day))
            self.start_time.setTime(QTime(local_start.hour, local_start.minute))
        if isinstance(end, datetime.datetime):
            local_end = end.astimezone()
            self.end_time.setTime(QTime(local_end.hour, local_end.minute))
        self.location_input.setText(shift.get("location") or "")
        self.notes_input.setPlainText(shift.get("notes") or "")

    def _handle_save(self) -> None:
        role = self.role_combo.currentText().strip()
        if not role:
            self.feedback_label.setText("Select a role for this shift.")
            return
        if is_manager_role(role):
            proceed = QMessageBox.warning(
                self,
                "Manager role assignment",
                (
                    "This is a manager-only role and should be assigned manually. "
                    "Confirm that the manager on duty is correct before saving."
                ),
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if proceed != QMessageBox.Yes:
                return
        qdate = self.date_edit.date()
        start_value = datetime.datetime(
            qdate.year(),
            qdate.month(),
            qdate.day(),
            self.start_time.time().hour(),
            self.start_time.time().minute(),
        )
        end_value = datetime.datetime(
            qdate.year(),
            qdate.month(),
            qdate.day(),
            self.end_time.time().hour(),
            self.end_time.time().minute(),
        )
        start_value = start_value.replace(tzinfo=datetime.timezone.utc)
        end_value = end_value.replace(tzinfo=datetime.timezone.utc)
        if end_value <= start_value:
            self.feedback_label.setText("End time must be after start time.")
            return

        employee_id = self.employee_combo.currentData()
        if employee_id is None and self.employee_combo.currentIndex() > 0:
            employee_id = self.employee_combo.currentData(Qt.UserRole)

        overlap_warning = self._has_overlap(employee_id, start_value, end_value)
        if overlap_warning:
            proceed = QMessageBox.question(
                self,
                "Potential overlap",
                overlap_warning + "\n\nContinue anyway?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if proceed != QMessageBox.Yes:
                return

        shift_id = self.shift.get("id") if self.shift else None
        shift_status = self.shift.get("status") if self.shift else "draft"
        week_id = self.shift.get("week_id") if self.shift else None
        payload = {
            "id": shift_id,
            "employee_id": employee_id,
            "role": role,
            "start": start_value,
            "end": end_value,
            "location": self.location_input.text().strip(),
            "notes": self.notes_input.toPlainText().strip(),
            "week_id": week_id,
            "week_start": self.week_start,
            "status": shift_status,
            "labor_rate": hourly_wage(self.policy, role, 0.0),
        }

        if self.on_save:
            self.on_save(payload)
        self.accept()

    def _handle_delete(self) -> None:
        if not self.shift or not self.on_delete:
            return
        confirm = QMessageBox.question(
            self,
            "Delete shift",
            "Remove this shift? This cannot be undone.",
        )
        if confirm != QMessageBox.Yes:
            return
        self.on_delete(self.shift["id"])
        self.accept()

    def _has_overlap(
        self,
        employee_id: Optional[int],
        start: datetime.datetime,
        end: datetime.datetime,
    ) -> Optional[str]:
        if not employee_id:
            return None
        current_id = self.shift.get("id") if self.shift else None
        for existing in self.existing_shifts:
            if existing.get("id") == current_id:
                continue
            if existing.get("employee_id") != employee_id:
                continue
            other_start = existing.get("start")
            other_end = existing.get("end")
            if not isinstance(other_start, datetime.datetime) or not isinstance(other_end, datetime.datetime):
                continue
            if other_start.tzinfo is None:
                other_start = other_start.replace(tzinfo=datetime.timezone.utc)
            if other_end.tzinfo is None:
                other_end = other_end.replace(tzinfo=datetime.timezone.utc)
            if start < other_end and end > other_start:
                emp_name = next((emp["name"] for emp in self.employees if emp["id"] == employee_id), "Employee")
                return f"{emp_name} already has a shift from {other_start.strftime('%H:%M')} to {other_end.strftime('%H:%M')}."
        return None
