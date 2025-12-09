from __future__ import annotations

import datetime
from typing import Callable, Dict, List, Optional, Tuple

from PySide6.QtCore import Qt, QDate
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDateEdit,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from database import (
    Shift,
    delete_shift,
    get_or_create_week_context,
    get_shifts_for_week,
    get_week_daily_projections,
    get_week_summary,
    list_employees,
    list_modifiers_for_week,
    list_roles,
    record_audit_log,
    shift_display_date,
    upsert_shift,
)
from generator.api import generate_schedule_for_week
from policy import build_default_policy, load_active_policy, role_catalog
from wages import validate_wages
from roles import grouped_roles, is_manager_role, palette_for_role, role_group, role_matches
from ui.edit_shift import EditShiftDialog

DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
ROLE_GROUP_ORDER = ["Kitchen", "Servers", "Bartenders", "Cashier", "Other"]


class WeekSchedulePage(QWidget):
    def __init__(
        self,
        session_factory,
        user: Dict[str, str],
        active_week: Optional[Dict[str, int]],
        *,
        on_week_changed: Optional[Callable[[int, int, str], None]] = None,
        on_back: Optional[Callable[[], None]] = None,
    ) -> None:
        super().__init__()
        self.session_factory = session_factory
        self.user = user
        self.on_week_changed = on_week_changed
        self.on_back = on_back
        self.can_edit = self.user.get("role") in {"GM", "SM", "IT"}
        self.week_info = active_week or {}
        self.week_start = self._compute_week_start(self.week_info)
        self.employee_options: List[Dict] = []
        self.employee_by_id: Dict[int, Dict] = {}
        self.role_options: List[str] = []
        self._roles_by_group: Dict[str, List[str]] = {}
        self.policy: Dict = {}
        self.current_shifts: List[Dict] = []
        self.filtered_shifts: List[Dict] = []
        self.summary_data: Dict = {}
        self.group_breakdown: Dict[str, Dict[str, float]] = {}
        self.selected_shift_id: Optional[int] = None
        self.selected_shift_ids: List[int] = []
        self.selected_day_index: int = 0
        self._suppress_week_signal = False

        self._build_ui()
        if self.week_start:
            self.refresh_all()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        layout.addLayout(self._build_header())
        layout.addLayout(self._build_filters())
        layout.addWidget(self._build_grid())
        layout.addLayout(self._build_footer())

    def _build_header(self) -> QHBoxLayout:
        header = QHBoxLayout()
        header.setSpacing(8)

        self.prev_week_button = QPushButton("◀")
        self.prev_week_button.setFixedSize(30, 30)
        self.prev_week_button.setStyleSheet("border-radius: 15px; padding: 0;")
        self.prev_week_button.clicked.connect(lambda: self._navigate_week(-7))
        header.addWidget(self.prev_week_button)

        self.week_picker = QDateEdit()
        self.week_picker.setCalendarPopup(True)
        self.week_picker.setDisplayFormat("yyyy-MM-dd")
        self.week_picker.setMinimumWidth(120)
        self.week_picker.dateChanged.connect(self._handle_week_picker_change)
        header.addWidget(self.week_picker)

        self.next_week_button = QPushButton("▶")
        self.next_week_button.setFixedSize(30, 30)
        self.next_week_button.setStyleSheet("border-radius: 15px; padding: 0;")
        self.next_week_button.clicked.connect(lambda: self._navigate_week(7))
        header.addWidget(self.next_week_button)

        self.week_label = QLabel("Week of --")
        header.addWidget(self.week_label)
        header.addStretch()
        return header

    def _build_filters(self) -> QHBoxLayout:
        filters = QHBoxLayout()
        filters.setSpacing(10)

        self.employee_filter = QComboBox()
        self.employee_filter.currentIndexChanged.connect(self.refresh_shifts)
        filters.addWidget(self._wrap_with_label("Employee", self.employee_filter))

        self.role_group_filter = QComboBox()
        self.role_group_filter.currentIndexChanged.connect(self._handle_role_group_change)
        filters.addWidget(self._wrap_with_label("Role group", self.role_group_filter))

        self.role_filter = QComboBox()
        self.role_filter.currentIndexChanged.connect(self.refresh_shifts)
        filters.addWidget(self._wrap_with_label("Role", self.role_filter))

        self.status_filter = QComboBox()
        for status in ["All", "Draft", "Validated", "Exported"]:
            self.status_filter.addItem(status, status.lower())
        self.status_filter.currentIndexChanged.connect(self.refresh_shifts)
        filters.addWidget(self._wrap_with_label("Status", self.status_filter))

        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search name, role, or notes...")
        self.search_input.textChanged.connect(self._apply_search_filter)
        filters.addWidget(self._wrap_with_label("Search", self.search_input))

        self.hide_unassigned_checkbox = QCheckBox("Hide unassigned")
        self.hide_unassigned_checkbox.stateChanged.connect(self._apply_search_filter)
        filters.addWidget(self.hide_unassigned_checkbox)
        filters.addStretch()
        return filters

    def _build_grid(self) -> QWidget:
        self.day_columns: List[Dict] = []
        container = QGroupBox("Week schedule")
        grid_layout = QHBoxLayout(container)
        grid_layout.setSpacing(6)

        for idx, day_name in enumerate(DAY_NAMES):
            column = QVBoxLayout()
            column.setSpacing(4)
            header_label = QLabel(day_name)
            header_label.setAlignment(Qt.AlignCenter)
            column.addWidget(header_label)

            list_widget = QListWidget()
            list_widget.setSelectionMode(QAbstractItemView.ExtendedSelection)
            list_widget.itemSelectionChanged.connect(self._handle_selection_changed)
            list_widget.itemDoubleClicked.connect(lambda *_: self._open_selected_shift())
            column.addWidget(list_widget)

            frame = QFrame()
            frame.setLayout(column)
            self.day_columns.append(
                {
                    "label": header_label,
                    "list": list_widget,
                }
            )
            grid_layout.addWidget(frame)

        return container

    def _build_footer(self) -> QVBoxLayout:
        footer = QVBoxLayout()
        footer.setSpacing(8)

        summary_box = QGroupBox("Coverage Forecast")
        summary_layout = QGridLayout(summary_box)
        summary_layout.setSpacing(6)
        self.summary_labels: List[QLabel] = []
        for idx, day in enumerate(DAY_NAMES):
            label = QLabel(f"{day}\nShifts: 0\nCost: $0.00")
            label.setAlignment(Qt.AlignCenter)
            self.summary_labels.append(label)
            summary_layout.addWidget(label, 0, idx)
        self.week_total_label = QLabel("Week total - 0 shifts - $0.00")
        self.group_breakdown_label = QLabel("Role group spend vs budget will appear here.")
        self.group_breakdown_label.setWordWrap(True)
        footer.addWidget(summary_box)
        footer.addWidget(self.week_total_label)
        footer.addWidget(self.group_breakdown_label)

        controls = QHBoxLayout()
        controls.setSpacing(10)

        self.generate_button = QPushButton("Generate Week")
        self.generate_button.clicked.connect(self._handle_generate)
        controls.addWidget(self.generate_button)

        self.add_button = QPushButton("Add Shift")
        self.add_button.clicked.connect(self._handle_add_shift)
        controls.addWidget(self.add_button)

        self.edit_button = QPushButton("Edit Selected")
        self.edit_button.clicked.connect(self._open_selected_shift)
        controls.addWidget(self.edit_button)

        self.delete_button = QPushButton("Delete")
        self.delete_button.clicked.connect(self._handle_delete_shift)
        controls.addWidget(self.delete_button)

        self.swap_button = QPushButton("Swap owners")
        self.swap_button.setToolTip("Select exactly two shifts to swap their assigned employees.")
        self.swap_button.clicked.connect(self._swap_selected_shifts)
        controls.addWidget(self.swap_button)

        self.grant_button = QPushButton("Grant shifts…")
        self.grant_button.setToolTip("Assign the selected shifts to another employee.")
        self.grant_button.clicked.connect(self._grant_shifts)
        controls.addWidget(self.grant_button)

        self.manager_cover_button = QPushButton("Manager cover")
        self.manager_cover_button.setToolTip("Mark the selected shift as covered by a manager (zero labor cost).")
        self.manager_cover_button.clicked.connect(self._handle_manager_cover)
        controls.addWidget(self.manager_cover_button)

        self.back_button = QPushButton("Back")
        self.back_button.clicked.connect(lambda: self.on_back() if self.on_back else None)
        controls.addWidget(self.back_button)
        controls.addStretch()

        footer.addLayout(controls)
        self.selection_hint = QLabel(
            "Select a shift to edit, or hold Ctrl/Shift to multi-select for grant or swap actions."
        )
        self.selection_hint.setWordWrap(True)
        footer.addWidget(self.selection_hint)
        self._enforce_permissions()
        return footer

    def refresh_all(self) -> None:
        if not self.week_start:
            return
        with self.session_factory() as session:
            self.policy = load_active_policy(session)
            self.employee_options = list_employees(session, only_active=True)
            self.role_options = list_roles(session)
        self.employee_by_id = {entry["id"]: entry for entry in self.employee_options if entry.get("id") is not None}
        self._populate_filters()
        self.refresh_shifts()

    def refresh_shifts(self) -> None:
        if not self.week_start:
            self.group_breakdown = {}
            return
        employee_id = self.employee_filter.currentData() or None
        selected_role = self.role_filter.currentData()
        selected_group = self.role_group_filter.currentData() if self.role_group_filter.count() else None
        role_param = selected_role if selected_role else None
        status_value = self.status_filter.currentData()
        if status_value == "all":
            status_value = None
        with self.session_factory() as session:
            shifts = get_shifts_for_week(
                session,
                self.week_start,
                employee_id=employee_id,
                role=role_param,
                status=status_value,
            )
            self.summary_data = get_week_summary(session, self.week_start)
            iso_year, iso_week, _ = self.week_start.isocalendar()
            context = get_or_create_week_context(session, iso_year, iso_week, f"{iso_year} W{iso_week:02d}")
            projections = get_week_daily_projections(session, context.id)
            modifiers = list_modifiers_for_week(session, self.week_start)
        self.current_shifts = [shift for shift in shifts if not is_manager_role(shift.get("role"))]
        self.group_breakdown = self._build_group_breakdown(shifts, projections, modifiers)
        self._sync_selected_ids()
        if selected_group:
            allowed_roles = set(self._roles_by_group.get(selected_group, []))
            self.current_shifts = [
                shift for shift in self.current_shifts if shift.get("role") in allowed_roles
            ]
        self._apply_search_filter()

    def _populate_filters(self) -> None:
        self._suppress_week_signal = True
        self.employee_filter.blockSignals(True)
        self.employee_filter.clear()
        self.employee_filter.addItem("All", None)
        for employee in self.employee_options:
            self.employee_filter.addItem(employee["name"], employee["id"])
        self.employee_filter.blockSignals(False)

        if self.week_start:
            qdate = QDate(self.week_start.year, self.week_start.month, self.week_start.day)
            self.week_picker.blockSignals(True)
            self.week_picker.setDate(qdate)
            self.week_picker.blockSignals(False)
            self.week_label.setText(f"Week of {self.week_start.isoformat()}")
        self._rebuild_role_filters()
        self._suppress_week_signal = False

    def _rebuild_role_filters(self) -> None:
        roles_from_sources = set(self.role_options)
        roles_from_sources.update(role_catalog(self.policy))
        filtered_roles = sorted({role for role in roles_from_sources if role and not is_manager_role(role)})
        self._roles_by_group = grouped_roles(filtered_roles)

        self.role_group_filter.blockSignals(True)
        self.role_group_filter.clear()
        self.role_group_filter.addItem("All groups", None)
        for group in ROLE_GROUP_ORDER:
            if group in self._roles_by_group:
                self.role_group_filter.addItem(group, group)
        self.role_group_filter.blockSignals(False)

        self._rebuild_role_filter_options()

    def _rebuild_role_filter_options(self) -> None:
        self.role_filter.blockSignals(True)
        self.role_filter.clear()
        self.role_filter.addItem("All roles", None)
        selected_group = self.role_group_filter.currentData()
        if selected_group:
            roles = self._roles_by_group.get(selected_group, [])
        else:
            roles = []
            for group in ROLE_GROUP_ORDER:
                roles.extend(self._roles_by_group.get(group, []))
        for role in roles:
            self.role_filter.addItem(role, role)
        self.role_filter.blockSignals(False)

    def _handle_role_group_change(self) -> None:
        self._rebuild_role_filter_options()
        self.refresh_shifts()

    def _apply_search_filter(self) -> None:
        term = self.search_input.text().strip().lower()
        if not term:
            self.filtered_shifts = list(self.current_shifts)
        else:
            filtered = []
            for shift in self.current_shifts:
                haystack = " ".join(
                    filter(
                        None,
                        [
                            shift.get("employee_name"),
                            shift.get("role"),
                            role_group(shift.get("role")),
                            shift.get("notes"),
                        ],
                    )
                ).lower()
                if term in haystack:
                    filtered.append(shift)
            self.filtered_shifts = filtered
        if getattr(self, "hide_unassigned_checkbox", None) and self.hide_unassigned_checkbox.isChecked():
            self.filtered_shifts = [
                shift for shift in self.filtered_shifts if shift.get("employee_id") is not None
            ]
        self._render_shift_grid()
        self._render_summary()
        self._update_action_states()

    def _render_shift_grid(self) -> None:
        by_day: Dict[int, List[Dict]] = {idx: [] for idx in range(7)}
        for shift in self.filtered_shifts:
            day_index = self._shift_day_index_from_data(shift)
            if day_index is None:
                continue
            by_day.setdefault(day_index, []).append(shift)

        selected_set = set(self.selected_shift_ids)
        for idx, column in enumerate(self.day_columns):
            list_widget: QListWidget = column["list"]
            header_label: QLabel = column["label"]
            list_widget.blockSignals(True)
            list_widget.clear()
            date_value = (self.week_start + datetime.timedelta(days=idx)) if self.week_start else None
            header_label.setText(f"{DAY_NAMES[idx]}\n{date_value.isoformat() if date_value else ''}")
            for shift in sorted(by_day.get(idx, []), key=lambda item: item["start"]):
                text = self._format_shift_text(shift)
                item = QListWidgetItem(text)
                item.setData(Qt.UserRole, shift["id"])
                bg_color, fg_color = self._color_for_shift(shift)
                item.setBackground(bg_color)
                item.setForeground(fg_color)
                item.setToolTip(f"{shift.get('role')} \u2014 {shift.get('employee_name') or 'Unassigned'}")
                if shift["id"] in selected_set:
                    item.setSelected(True)
                list_widget.addItem(item)
            list_widget.blockSignals(False)

    def _render_summary(self) -> None:
        day_data = {entry["date"]: entry for entry in self.summary_data.get("days", [])} if self.summary_data else {}
        for idx, label in enumerate(self.summary_labels):
            date_value = (self.week_start + datetime.timedelta(days=idx)) if self.week_start else None
            payload = day_data.get(date_value.isoformat() if date_value else "")
            if payload:
                count = payload.get("count", payload.get("shifts_created", 0))
                cost = payload.get("cost", 0.0)
            else:
                count = 0
                cost = 0.0
            label.setText(f"{DAY_NAMES[idx]}\nShifts: {count}\nCost: ${cost:,.2f}")
        total_cost = self.summary_data.get("total_cost", 0.0) if self.summary_data else 0.0
        total_shifts = self.summary_data.get("total_shifts", 0) if self.summary_data else 0
        projected_sales = self.summary_data.get("projected_sales_total", 0.0) if self.summary_data else 0.0
        policy_pct = self._policy_labor_pct()
        projected_budget = projected_sales * policy_pct if projected_sales > 0 and policy_pct > 0 else 0.0
        if projected_budget > 0:
            policy_usage = f"{min(999.9, (total_cost / projected_budget) * 100):.1f}%"
        else:
            policy_usage = "--"
        if projected_sales > 0:
            sales_usage = f"{min(999.9, (total_cost / projected_sales) * 100):.1f}%"
        else:
            sales_usage = "--"
        self.week_total_label.setText(
            f"Week total - {total_shifts} shifts - ${total_cost:,.2f} "
            f"- Projected {policy_usage} of Policy Labor Budget - Projected {sales_usage} of Projected Sales"
        )
        breakdown_parts: List[str] = []
        display_order = ["Kitchen", "Servers", "Bartenders", "Cashier"]
        for group in display_order:
            payload = self.group_breakdown.get(group, {})
            budget = payload.get("budget", 0.0)
            spend = payload.get("spend", 0.0)
            pct = "--"
            if budget > 0:
                pct = f"{min(999.9, (spend / budget) * 100):.1f}%"
            breakdown_parts.append(f"{group}: {pct} (${spend:,.0f} / ${budget:,.0f})")
        self.group_breakdown_label.setText(
            "Role group spend vs budget: " + " | ".join(breakdown_parts) if breakdown_parts else "Role group spend vs budget: --"
        )

    def _format_shift_text(self, shift: Dict) -> str:
        start = shift.get("start").astimezone()
        end = shift.get("end").astimezone()
        employee = shift.get("employee_name") or "Unassigned"
        notes = shift.get("notes") or ""
        notes_line = f"\n{notes}" if notes else ""
        role_name = shift.get("role") or ""
        group = role_group(role_name)
        group_line = f"{role_name} [{group}]" if group else role_name
        return f"{start.strftime('%H:%M')} - {end.strftime('%H:%M')}\n{employee}\n{group_line}{notes_line}"

    def _color_for_shift(self, shift: Dict) -> Tuple[QColor, QColor]:
        base = QColor(palette_for_role(shift.get("role")))
        key = shift.get("employee_id") or shift.get("employee_name") or shift.get("role") or ""
        try:
            seed = hash(key)
        except Exception:
            seed = 0
        tweak = 110 + (abs(seed) % 22)
        bg = base.lighter(tweak)
        bg.setAlpha(230)
        fg = QColor(Qt.white)
        return bg, fg

    def _handle_selection_changed(self) -> None:
        self.selected_shift_ids = self._gather_selected_shift_ids()
        self.selected_shift_id = self.selected_shift_ids[0] if self.selected_shift_ids else None
        if self.selected_shift_id is not None:
            self.selected_day_index = self._day_index_for_shift(self.selected_shift_id)
        else:
            self.selected_day_index = 0
        self._update_action_states()

    def _open_selected_shift(self) -> None:
        if not self.can_edit or not self.selected_shift_id:
            return
        shift = self._shift_by_id(self.selected_shift_id)
        if not shift:
            return
        dialog = EditShiftDialog(
            employees=self.employee_options,
            roles=self._available_roles(),
            policy=self.policy,
            week_start=self.week_start,
            shift=shift,
            existing_shifts=self.current_shifts,
            on_save=self._save_shift,
            on_delete=self._delete_shift,
            parent=self,
        )
        dialog.exec()

    def _handle_add_shift(self) -> None:
        if not self.can_edit:
            return
        default_date = None
        if self.week_start:
            default_date = self.week_start + datetime.timedelta(days=self.selected_day_index or 0)
        dialog = EditShiftDialog(
            employees=self.employee_options,
            roles=self._available_roles(),
            policy=self.policy,
            week_start=self.week_start,
            default_date=default_date,
            existing_shifts=self.current_shifts,
            on_save=self._save_shift,
            parent=self,
        )
        dialog.exec()

    def _save_shift(self, payload: Dict) -> None:
        try:
            with self.session_factory() as session:
                shift_id = upsert_shift(session, payload)
                record_audit_log(
                    session,
                    self.user.get("username", "system"),
                    "shift_save",
                    target_type="Shift",
                    target_id=shift_id,
                )
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Unable to save shift", str(exc))
            return
        self.selected_shift_id = None
        self.refresh_shifts()

    def _delete_shift(self, shift_id: int, *, refresh: bool = True) -> bool:
        try:
            with self.session_factory() as session:
                delete_shift(session, shift_id)
                record_audit_log(
                    session,
                    self.user.get("username", "system"),
                    "shift_delete",
                    target_type="Shift",
                    target_id=shift_id,
                )
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Unable to delete shift", str(exc))
            return False
        if refresh:
            self.selected_shift_id = None
            self.refresh_shifts()
        return True

    def _handle_delete_shift(self) -> None:
        if not self.can_edit or not self.selected_shift_ids:
            return
        targets = list(self.selected_shift_ids)
        count = len(targets)
        prompt = "Remove the selected shift?" if count == 1 else f"Remove {count} selected shifts?"
        confirm = QMessageBox.question(
            self,
            "Delete shift",
            prompt,
        )
        if confirm != QMessageBox.Yes:
            return
        deleted_any = False
        for shift_id in targets:
            deleted_any |= self._delete_shift(shift_id, refresh=False)
        if deleted_any:
            self.selected_shift_id = None
            self.selected_shift_ids = []
            self.refresh_shifts()

    def _swap_selected_shifts(self) -> None:
        if not (self.can_edit and self.week_start):
            return
        if len(self.selected_shift_ids) != 2:
            QMessageBox.information(self, "Swap owners", "Select exactly two shifts to swap.")
            return
        first_id, second_id = self.selected_shift_ids[:2]
        if first_id == second_id:
            QMessageBox.information(self, "Swap owners", "Pick two different shifts.")
            return
        with self.session_factory() as session:
            shift_a = session.get(Shift, first_id)
            shift_b = session.get(Shift, second_id)
            if not shift_a or not shift_b:
                QMessageBox.warning(self, "Swap owners", "Unable to load the selected shifts.")
                return
            shift_a.employee_id, shift_b.employee_id = shift_b.employee_id, shift_a.employee_id
            session.commit()
            record_audit_log(
                session,
                self.user.get("username", "unknown"),
                "shift_swap",
                target_type="Shift",
                target_id=first_id,
                payload={"swap_with": second_id},
            )
        self.refresh_all()
        QMessageBox.information(self, "Swap owners", "Shift owners have been swapped.")

    def _grant_shifts(self) -> None:
        if not self.selected_shift_ids:
            QMessageBox.information(self, "Grant shifts", "Select at least one shift.")
            return
        employee = self._choose_employee()
        if not employee:
            return
        eligible: List[int] = []
        skipped = 0
        for shift_id in self.selected_shift_ids:
            shift = self._shift_by_id(shift_id)
            if not shift:
                continue
            if self._employee_can_fill_role(employee, shift.get("role")):
                eligible.append(shift_id)
            else:
                skipped += 1
        if not eligible:
            QMessageBox.information(self, "Grant shifts", "Selected employee cannot cover those roles.")
            return
        with self.session_factory() as session:
            for shift_id in eligible:
                db_shift = session.get(Shift, shift_id)
                if db_shift:
                    db_shift.employee_id = employee["id"]
            session.commit()
            record_audit_log(
                session,
                self.user.get("username", "unknown"),
                "shift_grant",
                target_type="Shift",
                target_id=eligible[0],
                payload={"employee_id": employee["id"], "count": len(eligible)},
            )
        self.refresh_all()
        message = f"Assigned {len(eligible)} shift(s) to {employee['name']}."
        if skipped:
            message += f" Skipped {skipped} shift(s) due to role mismatch."
        QMessageBox.information(self, "Grant shifts", message)

    def _manager_candidates(self) -> List[Dict]:
        return [
            entry
            for entry in self.employee_options
            if any(is_manager_role(role) for role in (entry.get("roles") or []))
        ]

    def _handle_manager_cover(self) -> None:
        if not (self.can_edit and self.week_start):
            return
        if not self.selected_shift_ids:
            QMessageBox.information(self, "Manager cover", "Select at least one shift.")
            return
        candidates = self._manager_candidates()
        manager_id = None
        manager_name = "Manager"
        if candidates:
            names = [entry["name"] for entry in candidates]
            choice, ok = QInputDialog.getItem(
                self,
                "Manager cover",
                "Which manager will cover this shift?",
                names,
                editable=False,
            )
            if not ok:
                return
            for entry in candidates:
                if entry["name"] == choice:
                    manager_id = entry["id"]
                    manager_name = entry["name"]
                    break
        else:
            proceed = QMessageBox.question(
                self,
                "Manager cover",
                "No managers are stored in the directory. Mark selected shifts as manager-covered with no wage?",
            )
            if proceed != QMessageBox.Yes:
                return

        with self.session_factory() as session:
            for shift_id in self.selected_shift_ids:
                shift = session.get(Shift, shift_id)
                if not shift:
                    continue
                shift.employee_id = manager_id
                shift.labor_rate = 0.0
                shift.labor_cost = 0.0
                note = shift.notes or ""
                tag = "Manager cover"
                if note and tag.lower() not in note.lower():
                    note = f"{note.rstrip('.')}. {tag}"
                elif not note:
                    note = tag
                shift.notes = note
            session.commit()
            record_audit_log(
                session,
                self.user.get("username", "unknown"),
                "manager_cover",
                target_type="Shift",
                target_id=self.selected_shift_ids[0],
                payload={"manager_id": manager_id, "count": len(self.selected_shift_ids)},
            )
        self.refresh_all()
        assigned_label = f"{len(self.selected_shift_ids)} shift(s) marked for manager cover"
        if manager_id:
            assigned_label += f" by {manager_name}"
        QMessageBox.information(self, "Manager cover", assigned_label + ".")

    def _handle_generate(self) -> None:
        if not self.week_start:
            QMessageBox.information(self, "Select week", "Choose a week before running the generator.")
            return
        if not self._pre_generation_checks():
            return
        confirm = QMessageBox.question(
            self,
            "Generate schedule",
            "Generate a draft schedule for this week? This will overwrite existing draft shifts.",
        )
        if confirm != QMessageBox.Yes:
            return
        try:
            result = generate_schedule_for_week(
                self.session_factory,
                self.week_start,
                self.user.get("username", "system"),
            )
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Generation failed", str(exc))
            return
        self.refresh_shifts()
        summary_lines: List[str] = []
        warnings = result.get("warnings") or []
        if warnings:
            summary_lines.append("Warnings:")
            summary_lines.extend([f"- {warning}" for warning in warnings[:5]])
        validation = result.get("validation") or {}
        validation_issues = validation.get("issues") or []
        validation_warnings = validation.get("warnings") or []
        if validation_issues or validation_warnings:
            summary_lines.append("Validation findings:")
            for issue in validation_issues[:5]:
                message = issue.get("message") if isinstance(issue, dict) else str(issue)
                summary_lines.append(f"- Issue: {message}")
            for warn in validation_warnings[:5]:
                message = warn.get("message") if isinstance(warn, dict) else str(warn)
                summary_lines.append(f"- Warning: {message}")
        cut_insights = result.get("cut_insights") or []
        if cut_insights:
            summary_lines.append("Recent cut decisions:")
            for insight in cut_insights[:5]:
                time_label = insight.get("cut_time", "")[-8:-3]
                factors = insight.get("factors", {})
                budget_value = factors.get("budget_gap")
                if budget_value is None:
                    budget_value = factors.get("budget_component", 0)
                if isinstance(budget_value, (int, float)):
                    budget_display = f"{budget_value:+.3f}"
                else:
                    budget_display = str(budget_value)
                summary_lines.append(
                    f"- {insight.get('day')} {insight.get('role')} @ {time_label} "
                    f"(score {insight.get('score')}, budget {budget_display})"
                )
        if not summary_lines:
            summary_lines.append("Generator completed.")
        QMessageBox.information(
            self,
            "Schedule generated",
            f"Created {result.get('shifts_created', 0)} shifts.\n" + "\n".join(summary_lines),
        )

    def _pre_generation_checks(self) -> bool:
        with self.session_factory() as session:
            employees = list_employees(session, only_active=True)
        self.employee_options = employees
        if not employees:
            QMessageBox.warning(
                self,
                "No employees",
                "Add active employees from Week Prep → Employee directory before generating a schedule.",
            )
            return False
        # Only validate wages for roles that are still present in the active policy.
        active_roles = set(role_catalog(self.policy))
        roles_needed = sorted(
            {
                role
                for entry in employees
                for role in entry.get("roles", [])
                if role and role in active_roles
            }
        )
        if not roles_needed:
            QMessageBox.warning(
                self,
                "No roles",
                "Assign at least one role to each employee before generating a schedule.",
            )
            return False
        wage_issues = validate_wages(roles_needed)
        if wage_issues:
            bullet = "\n".join(f"• {role}: {reason}" for role, reason in sorted(wage_issues.items()))
            QMessageBox.warning(
                self,
                "Missing wages",
                "Set and confirm hourly wages for each role from Week Prep → Role wages:\n\n" + bullet,
            )
            return False
        iso_year, iso_week, _ = self.week_start.isocalendar()
        label = f"{iso_year} W{iso_week:02d}"
        with self.session_factory() as session:
            week_context = get_or_create_week_context(session, iso_year, iso_week, label)
            projections = get_week_daily_projections(session, week_context.id)
        missing_days = [DAY_NAMES[item.day_of_week] for item in projections if (item.projected_sales_amount or 0) <= 0]
        if missing_days:
            readable = ", ".join(missing_days)
            QMessageBox.warning(
                self,
                "Missing projections",
                f"Enter projected sales for each day (Week Prep → Projected sales). Missing days: {readable}.",
            )
            return False
        return True

    def _gather_selected_shift_ids(self) -> List[int]:
        selected: List[int] = []
        for column in self.day_columns:
            for item in column["list"].selectedItems():
                shift_id = item.data(Qt.UserRole)
                if shift_id is not None and shift_id not in selected:
                    selected.append(shift_id)
        return selected

    def _shift_day_index_from_data(self, shift: Dict) -> Optional[int]:
        if not self.week_start:
            return None
        start = shift.get("start")
        if not isinstance(start, datetime.datetime):
            return None
        display_day = shift_display_date(start, shift.get("location"))
        delta = (display_day - self.week_start).days
        if 0 <= delta < 7:
            return delta
        return None

    def _day_index_for_shift(self, shift_id: int) -> int:
        shift = self._shift_by_id(shift_id)
        if shift:
            day_index = self._shift_day_index_from_data(shift)
            if day_index is not None:
                return day_index
        return 0

    def _sync_selected_ids(self) -> None:
        current_ids = {shift["id"] for shift in self.current_shifts}
        self.selected_shift_ids = [sid for sid in self.selected_shift_ids if sid in current_ids]
        self.selected_shift_id = self.selected_shift_ids[0] if self.selected_shift_ids else None
        if self.selected_shift_id is not None:
            self.selected_day_index = self._day_index_for_shift(self.selected_shift_id)
        else:
            self.selected_day_index = 0

    def _shift_by_id(self, shift_id: int) -> Optional[Dict]:
        for shift in self.current_shifts:
            if shift["id"] == shift_id:
                return shift
        return None

    def _policy_labor_pct(self) -> float:
        global_cfg = self.policy.get("global", {}) if isinstance(self.policy, dict) else {}
        pct = float(global_cfg.get("labor_budget_pct", 0.27) or 0.0)
        if pct > 1.0:
            pct /= 100.0
        return max(0.0, min(0.9, pct))

    @staticmethod
    def _display_group_name(group: str) -> str:
        if group in {"Kitchen", "Heart of House"}:
            return "Kitchen"
        if group in {"Cashier & Takeout", "Cashier"}:
            return "Cashier"
        return group or "Other"

    def _role_group_from_policy(self, role: Optional[str]) -> str:
        if not role:
            return "Other"
        cfg = (self.policy.get("roles", {}) or {}).get(role, {})
        if isinstance(cfg, dict):
            explicit = cfg.get("group")
            if explicit:
                return str(explicit)
        return role_group(role)

    def _build_group_breakdown(self, shifts: List[Dict], projections, modifiers) -> Dict[str, Dict[str, float]]:
        budgets: Dict[str, float] = {}
        labor_pct = self._policy_labor_pct()
        role_groups_cfg = self.policy.get("role_groups", {}) if isinstance(self.policy, dict) else {}
        if not isinstance(role_groups_cfg, dict) or not role_groups_cfg:
            role_groups_cfg = build_default_policy().get("role_groups", {})

        modifier_map: Dict[int, List[Dict[str, float]]] = {idx: [] for idx in range(7)}
        for mod in modifiers or []:
            day_idx = int(mod.get("day_of_week", 0))
            start = mod.get("start_time")
            end = mod.get("end_time")
            pct = float(mod.get("pct_change", 0) or 0)
            start_minutes = start.hour * 60 + start.minute if start else 0
            end_minutes = end.hour * 60 + end.minute if end else 24 * 60
            modifier_map.setdefault(day_idx, []).append(
                {
                    "start": start_minutes,
                    "end": end_minutes,
                    "pct": pct,
                }
            )

        def _modifier_multiplier(day_index: int) -> float:
            windows = modifier_map.get(day_index, [])
            if not windows:
                return 1.0
            total = 0.0
            for window in windows:
                span = max(0.0, window["end"] - window["start"])
                fraction = span / (24 * 60)
                total += (window["pct"] / 100.0) * max(fraction, 0.1)
            return max(0.5, 1.0 + total)

        for projection in projections or []:
            sales = float(getattr(projection, "projected_sales_amount", 0.0) or 0.0)
            day_idx = int(getattr(projection, "day_of_week", 0) or 0)
            adjusted_sales = sales * _modifier_multiplier(day_idx)
            day_budget = adjusted_sales * labor_pct
            for group_name, spec in role_groups_cfg.items():
                pct = spec.get("allocation_pct", 0.0)
                try:
                    pct = float(pct)
                except (TypeError, ValueError):
                    pct = 0.0
                if pct > 1.0:
                    pct /= 100.0
                if pct <= 0:
                    continue
                display_group = self._display_group_name(group_name)
                budgets[display_group] = budgets.get(display_group, 0.0) + (day_budget * pct)

        spend: Dict[str, float] = {}
        for shift in shifts:
            group_name = self._display_group_name(self._role_group_from_policy(shift.get("role")))
            spend[group_name] = spend.get(group_name, 0.0) + float(shift.get("labor_cost") or 0.0)

        breakdown: Dict[str, Dict[str, float]] = {}
        for group in set(list(budgets.keys()) + list(spend.keys())):
            breakdown[group] = {
                "budget": round(budgets.get(group, 0.0), 2),
                "spend": round(spend.get(group, 0.0), 2),
            }
        return breakdown

    def _choose_employee(self) -> Optional[Dict]:
        if not self.employee_options:
            QMessageBox.information(self, "Grant shifts", "No employees available.")
            return None
        names = [entry["name"] for entry in self.employee_options]
        name, ok = QInputDialog.getItem(
            self,
            "Grant shifts",
            "Select an employee",
            names,
            editable=False,
        )
        if not ok:
            return None
        for entry in self.employee_options:
            if entry["name"] == name:
                return entry
        return None

    @staticmethod
    def _employee_can_fill_role(employee: Dict, role: Optional[str]) -> bool:
        if not role:
            return False
        roles = employee.get("roles") or []
        if isinstance(roles, str):
            roles = [entry.strip() for entry in roles.split(",") if entry.strip()]
        for candidate in roles:
            if role_matches(candidate, role):
                return True
        return False

    def set_active_week(self, week_info: Dict[str, int]) -> None:
        new_start = self._compute_week_start(week_info)
        if not new_start or new_start == self.week_start:
            return
        self.week_info = week_info
        self.week_start = new_start
        self.week_label.setText(f"Week of {self.week_start.isoformat()}")
        self._enforce_permissions()
        self.refresh_all()

    def _compute_week_start(self, week_info: Optional[Dict[str, int]]) -> Optional[datetime.date]:
        if not week_info:
            return None
        iso_year = week_info.get("iso_year")
        iso_week = week_info.get("iso_week")
        if not iso_year or not iso_week:
            return None
        return datetime.date.fromisocalendar(int(iso_year), int(iso_week), 1)

    def _handle_week_picker_change(self) -> None:
        if self._suppress_week_signal:
            return
        qdate = self.week_picker.date()
        new_date = datetime.date(qdate.year(), qdate.month(), qdate.day())
        monday = new_date - datetime.timedelta(days=new_date.weekday())
        self._notify_week_change(monday)

    def _navigate_week(self, delta_days: int) -> None:
        if not self.week_start:
            return
        target = self.week_start + datetime.timedelta(days=delta_days)
        self._notify_week_change(target)

    def _notify_week_change(self, new_start: datetime.date) -> None:
        iso_year, iso_week, _ = new_start.isocalendar()
        label = f"{iso_year} W{iso_week:02d}"
        self.week_start = new_start
        self.week_label.setText(f"Week of {self.week_start.isoformat()}")
        if self.on_week_changed:
            self.on_week_changed(iso_year, iso_week, label)
        self.refresh_all()
        self._enforce_permissions()

    def _update_action_states(self) -> None:
        editable = self.can_edit and self.week_start is not None
        selected_count = len(self.selected_shift_ids)
        self.edit_button.setEnabled(editable and selected_count == 1)
        self.delete_button.setEnabled(editable and selected_count >= 1)
        self.swap_button.setEnabled(editable and selected_count == 2)
        self.grant_button.setEnabled(editable and selected_count >= 1)
        self.manager_cover_button.setEnabled(editable and selected_count >= 1)
        self._update_selection_hint()

    def _enforce_permissions(self) -> None:
        editable = self.can_edit and self.week_start is not None
        for button in (self.generate_button, self.add_button):
            button.setEnabled(editable)
        if not editable:
            for button in (self.edit_button, self.delete_button, self.swap_button, self.grant_button):
                button.setEnabled(False)
            if hasattr(self, "selection_hint"):
                self.selection_hint.setText("Read-only mode. Sign in as GM/SM/IT to edit this schedule.")
        else:
            self._update_action_states()

    def _available_roles(self) -> List[str]:
        roles: List[str] = []
        for group in ROLE_GROUP_ORDER:
            roles.extend(self._roles_by_group.get(group, []))
        return roles

    @staticmethod
    def _wrap_with_label(label_text: str, widget: QWidget) -> QWidget:
        container = QVBoxLayout()
        container.setSpacing(2)
        label = QLabel(label_text)
        container.addWidget(label)
        container.addWidget(widget)
        wrapper = QWidget()
        wrapper.setLayout(container)
        return wrapper

    def _update_selection_hint(self) -> None:
        if not hasattr(self, "selection_hint"):
            return
        if not (self.can_edit and self.week_start):
            return
        selected_count = len(self.selected_shift_ids)
        if selected_count == 0:
            text = "Select a shift to edit, or use Add Shift to create a new entry. Hold Ctrl/Shift for multi-select."
        elif selected_count == 1:
            shift = self._shift_by_id(self.selected_shift_ids[0])
            owner = shift.get("employee_name") if shift else None
            owner_label = owner or "Unassigned shift"
            text = f"{owner_label} selected. Use Edit/Delete or Grant to reassign."
        elif selected_count == 2:
            text = "Two shifts selected. Delete removes both; Swap trades owners; Grant reassigns them together."
        else:
            text = f"{selected_count} shifts selected. Delete clears them or use Grant to reassign."
        self.selection_hint.setText(text)
