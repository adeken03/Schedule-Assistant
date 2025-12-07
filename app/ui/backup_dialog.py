from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from backup import (
    create_full_backup,
    delete_backup,
    format_size,
    list_backups,
    restore_from_backup,
)


class BackupManagerDialog(QDialog):
    """Dialog for managing application backups."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Backup & Restore Manager")
        self.resize(700, 500)
        self.selected_backup: Optional[Path] = None
        self._build_ui()
        self._refresh_backup_list()

    def _build_ui(self) -> None:
        """Build the dialog UI."""
        layout = QVBoxLayout(self)

        # Instructions
        intro = QLabel(
            "Create backups of all application data (databases, accounts, configuration). "
            "Backups prefixed with 'auto_' are created automatically on startup."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        # Backup list
        list_label = QLabel("Available backups:")
        list_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        layout.addWidget(list_label)

        self.backup_list = QListWidget()
        self.backup_list.setSelectionMode(QListWidget.SingleSelection)
        self.backup_list.itemSelectionChanged.connect(self._update_button_state)
        layout.addWidget(self.backup_list)

        # Action buttons
        actions_layout = QHBoxLayout()

        self.create_button = QPushButton("Create new backup")
        self.create_button.clicked.connect(self._handle_create_backup)
        actions_layout.addWidget(self.create_button)

        self.restore_button = QPushButton("Restore selected")
        self.restore_button.clicked.connect(self._handle_restore_backup)
        self.restore_button.setEnabled(False)
        actions_layout.addWidget(self.restore_button)

        self.delete_button = QPushButton("Delete selected")
        self.delete_button.clicked.connect(self._handle_delete_backup)
        self.delete_button.setEnabled(False)
        actions_layout.addWidget(self.delete_button)

        self.refresh_button = QPushButton("Refresh list")
        self.refresh_button.clicked.connect(self._refresh_backup_list)
        actions_layout.addWidget(self.refresh_button)

        actions_layout.addStretch()
        layout.addLayout(actions_layout)

        # Close button
        button_box = QDialogButtonBox(QDialogButtonBox.Close)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def _refresh_backup_list(self) -> None:
        """Refresh the list of available backups."""
        self.backup_list.clear()
        backups = list_backups()

        if not backups:
            item = QListWidgetItem("No backups found")
            item.setFlags(Qt.ItemIsEnabled)
            item.setData(Qt.UserRole, None)
            self.backup_list.addItem(item)
            return

        for backup in backups:
            # Format display text
            created_str = backup["created"].strftime("%Y-%m-%d %H:%M:%S")
            size_str = format_size(backup["size"])
            backup_type = "Automatic" if backup["name"].startswith("auto_") else "Manual"

            display_text = f"{backup['name']}\n  Created: {created_str} | Size: {size_str} | Type: {backup_type}"

            item = QListWidgetItem(display_text)
            item.setData(Qt.UserRole, backup["path"])
            self.backup_list.addItem(item)

        self._update_button_state()

    def _update_button_state(self) -> None:
        """Update button enabled state based on selection."""
        has_selection = False
        selected_items = self.backup_list.selectedItems()

        if selected_items:
            item = selected_items[0]
            backup_path = item.data(Qt.UserRole)
            has_selection = backup_path is not None
            self.selected_backup = backup_path

        self.restore_button.setEnabled(has_selection)
        self.delete_button.setEnabled(has_selection)

    def _handle_create_backup(self) -> None:
        """Handle create backup button click."""
        reply = QMessageBox.question(
            self,
            "Create backup",
            "Create a new backup of all application data?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )

        if reply != QMessageBox.Yes:
            return

        success, message, _ = create_full_backup()

        if success:
            QMessageBox.information(self, "Backup created", message)
            self._refresh_backup_list()
        else:
            QMessageBox.critical(self, "Backup failed", message)

    def _handle_restore_backup(self) -> None:
        """Handle restore backup button click."""
        if not self.selected_backup:
            return

        backup_name = self.selected_backup.name

        reply = QMessageBox.warning(
            self,
            "Restore backup",
            f"Restore from backup '{backup_name}'?\n\n"
            "WARNING: This will REPLACE all current data with the backup. "
            "The application will need to restart after restoration.\n\n"
            "Consider creating a backup of current data first.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )

        if reply != QMessageBox.Yes:
            return

        success, message = restore_from_backup(self.selected_backup)

        if success:
            QMessageBox.information(
                self,
                "Restore complete",
                f"{message}\n\nPlease restart the application for changes to take effect.",
            )
            self.accept()
        else:
            QMessageBox.critical(self, "Restore failed", message)

    def _handle_delete_backup(self) -> None:
        """Handle delete backup button click."""
        if not self.selected_backup:
            return

        backup_name = self.selected_backup.name

        reply = QMessageBox.question(
            self,
            "Delete backup",
            f"Delete backup '{backup_name}'?\n\nThis action cannot be undone.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )

        if reply != QMessageBox.Yes:
            return

        success, message = delete_backup(self.selected_backup)

        if success:
            QMessageBox.information(self, "Backup deleted", message)
            self._refresh_backup_list()
        else:
            QMessageBox.critical(self, "Delete failed", message)
