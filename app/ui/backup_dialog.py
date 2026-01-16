from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from backup import (
    BACKUP_ROOT,
    create_full_backup,
    delete_backup,
    format_size,
    list_backups,
    restore_from_backup,
)


class BackupManagerDialog(QDialog):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Backup / Restore")
        self.setMinimumWidth(560)
        self.selected_backup: Optional[Path] = None
        self._build_ui()
        self._refresh_backup_list()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        intro = QLabel(
            "Create a full backup of all app data or restore from a previous backup.\n"
            "Restoring will replace current data."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        self.backup_list = QListWidget()
        self.backup_list.setSelectionMode(QListWidget.SingleSelection)
        self.backup_list.itemSelectionChanged.connect(self._update_button_state)
        layout.addWidget(self.backup_list)

        actions = QHBoxLayout()
        self.create_button = QPushButton("Create backup")
        self.create_button.clicked.connect(self._handle_create_backup)
        actions.addWidget(self.create_button)

        self.restore_button = QPushButton("Restore selected")
        self.restore_button.clicked.connect(self._handle_restore_backup)
        self.restore_button.setEnabled(False)
        actions.addWidget(self.restore_button)

        self.restore_from_button = QPushButton("Restore from folder...")
        self.restore_from_button.clicked.connect(self._handle_restore_from_folder)
        actions.addWidget(self.restore_from_button)

        self.delete_button = QPushButton("Delete selected")
        self.delete_button.clicked.connect(self._handle_delete_backup)
        self.delete_button.setEnabled(False)
        actions.addWidget(self.delete_button)

        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.clicked.connect(self._refresh_backup_list)
        actions.addWidget(self.refresh_button)

        actions.addStretch()
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.accept)
        actions.addWidget(close_button)
        layout.addLayout(actions)

    def _refresh_backup_list(self) -> None:
        self.backup_list.clear()
        backups = list_backups()
        if not backups:
            item = QListWidgetItem("No backups found.")
            item.setFlags(Qt.NoItemFlags)
            self.backup_list.addItem(item)
            self.selected_backup = None
            self._update_button_state()
            return

        for backup in backups:
            created = backup["created"].strftime("%Y-%m-%d %H:%M:%S")
            size = format_size(int(backup["size"]))
            label = f"{backup['name']}\n  Created: {created} | Size: {size}"
            item = QListWidgetItem(label)
            item.setData(Qt.UserRole, backup["path"])
            self.backup_list.addItem(item)
        self.selected_backup = None
        self._update_button_state()

    def _update_button_state(self) -> None:
        selected_items = self.backup_list.selectedItems()
        self.selected_backup = None
        if selected_items:
            path = selected_items[0].data(Qt.UserRole)
            if isinstance(path, Path):
                self.selected_backup = path
            elif isinstance(path, str):
                self.selected_backup = Path(path)
        has_selection = self.selected_backup is not None
        self.restore_button.setEnabled(has_selection)
        self.delete_button.setEnabled(has_selection)

    def _handle_create_backup(self) -> None:
        confirm = QMessageBox.question(
            self,
            "Create backup",
            "Create a new backup of all application data?",
        )
        if confirm != QMessageBox.Yes:
            return
        success, message, _ = create_full_backup()
        if success:
            QMessageBox.information(self, "Backup created", message)
            self._refresh_backup_list()
        else:
            QMessageBox.critical(self, "Backup failed", message)

    def _handle_restore_backup(self) -> None:
        if not self.selected_backup:
            return
        self._restore_from_path(self.selected_backup)

    def _handle_restore_from_folder(self) -> None:
        start_dir = BACKUP_ROOT if BACKUP_ROOT.exists() else Path.cwd()
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select backup folder",
            str(start_dir),
        )
        if not folder:
            return
        self._restore_from_path(Path(folder))

    def _restore_from_path(self, backup_path: Path) -> None:
        confirm = QMessageBox.question(
            self,
            "Restore backup",
            f"Restore from backup '{backup_path.name}'?\n\n"
            "This will REPLACE all current data. Restart the app after restore.",
        )
        if confirm != QMessageBox.Yes:
            return
        success, message = restore_from_backup(backup_path)
        if success:
            QMessageBox.information(self, "Restore complete", message)
            self._refresh_backup_list()
        else:
            QMessageBox.critical(self, "Restore failed", message)

    def _handle_delete_backup(self) -> None:
        if not self.selected_backup:
            return
        confirm = QMessageBox.question(
            self,
            "Delete backup",
            f"Delete backup '{self.selected_backup.name}'?\n\nThis action cannot be undone.",
        )
        if confirm != QMessageBox.Yes:
            return
        success, message = delete_backup(self.selected_backup)
        if success:
            QMessageBox.information(self, "Backup deleted", message)
            self._refresh_backup_list()
        else:
            QMessageBox.critical(self, "Delete failed", message)
