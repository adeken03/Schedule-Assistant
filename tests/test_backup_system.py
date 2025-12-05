from __future__ import annotations

import datetime
import json
import shutil
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

# Ensure app directory is in path
APP_DIR = Path(__file__).resolve().parents[1] / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

# Try to import PySide6, but allow tests to run without it
try:
    from PySide6.QtWidgets import QApplication
    PYSIDE6_AVAILABLE = True
except ImportError:
    PYSIDE6_AVAILABLE = False
    QApplication = None

from backup import (  # noqa: E402
    create_full_backup,
    restore_from_backup,
    cleanup_old_auto_backups,
    list_backups,
    BACKUP_ROOT,
    DATA_DIR,
)

from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

from database import (  # noqa: E402
    Base,
    EmployeeBase,
    PolicyBase,
    ProjectionsBase,
    Employee,
    Shift,
    get_or_create_week,
    upsert_policy,
)
import database as db  # noqa: E402
from policy import build_default_policy  # noqa: E402
from validation import _weekly_hours_warnings  # noqa: E402

if PYSIDE6_AVAILABLE:
    from ui.backup_dialog import BackupManagerDialog  # noqa: E402
else:
    BackupManagerDialog = None

UTC = datetime.timezone.utc


class BackupFunctionalityTests(unittest.TestCase):
    """Tests for backup creation, restoration, and cleanup functionality."""

    def setUp(self) -> None:
        """Set up temporary directories and test data for backup tests."""
        # Create temporary directories
        self.temp_dir = Path(tempfile.mkdtemp())
        self.temp_data_dir = self.temp_dir / "data"
        self.temp_backup_root = self.temp_dir / "backups"
        self.temp_data_dir.mkdir(parents=True, exist_ok=True)
        self.temp_backup_root.mkdir(parents=True, exist_ok=True)

        # Patch module-level constants
        self.data_dir_patcher = mock.patch("backup.DATA_DIR", self.temp_data_dir)
        self.backup_root_patcher = mock.patch("backup.BACKUP_ROOT", self.temp_backup_root)
        self.data_dir_patcher.start()
        self.backup_root_patcher.start()

        # Create sample data files
        self._create_sample_databases()
        self._create_sample_json_files()
        self._create_sample_exports()

    def tearDown(self) -> None:
        """Clean up temporary directories."""
        self.data_dir_patcher.stop()
        self.backup_root_patcher.stop()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_sample_databases(self) -> None:
        """Create sample SQLite databases with test data."""
        databases = ["employees.db", "schedule.db", "policy.db", "projections.db"]
        for db_name in databases:
            db_path = self.temp_data_dir / db_name
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            cursor.execute("CREATE TABLE test_table (id INTEGER PRIMARY KEY, data TEXT)")
            cursor.execute("INSERT INTO test_table (data) VALUES (?)", (f"test_data_{db_name}",))
            conn.commit()
            conn.close()

    def _create_sample_json_files(self) -> None:
        """Create sample JSON configuration files."""
        json_files = {
            "accounts.json": {"users": [{"name": "admin", "role": "manager"}]},
            "week_state.json": {"current_week": "2024-04-01"},
            "audit.log": {"logs": [{"action": "test", "timestamp": "2024-01-01"}]},
        }
        for filename, content in json_files.items():
            file_path = self.temp_data_dir / filename
            file_path.write_text(json.dumps(content, indent=2), encoding="utf-8")

    def _create_sample_exports(self) -> None:
        """Create sample exports directory with files."""
        exports_dir = self.temp_data_dir / "exports"
        exports_dir.mkdir(exist_ok=True)
        (exports_dir / "export_1.csv").write_text("id,name\n1,test\n", encoding="utf-8")
        (exports_dir / "export_2.csv").write_text("id,value\n2,data\n", encoding="utf-8")

    def test_create_full_backup_success(self) -> None:
        """Test that create_full_backup successfully creates a backup with all expected data."""
        # Create a backup
        success, message, backup_path = create_full_backup(backup_name="test_backup")

        # Verify success
        self.assertTrue(success)
        self.assertIn("test_backup", message)
        self.assertTrue(backup_path.exists())
        self.assertTrue(backup_path.is_dir())

        # Verify all databases are backed up
        for db_name in ["employees.db", "schedule.db", "policy.db", "projections.db"]:
            backup_db_path = backup_path / db_name
            self.assertTrue(backup_db_path.exists(), f"{db_name} should be in backup")

            # Verify data integrity by checking content
            conn = sqlite3.connect(str(backup_db_path))
            cursor = conn.cursor()
            cursor.execute("SELECT data FROM test_table")
            result = cursor.fetchone()
            self.assertIsNotNone(result)
            self.assertEqual(result[0], f"test_data_{db_name}")
            conn.close()

        # Verify JSON files are backed up
        for json_file in ["accounts.json", "week_state.json", "audit.log"]:
            backup_json_path = backup_path / json_file
            self.assertTrue(backup_json_path.exists(), f"{json_file} should be in backup")
            # Verify content
            content = json.loads(backup_json_path.read_text(encoding="utf-8"))
            self.assertIsInstance(content, dict)

        # Verify exports directory is backed up
        backup_exports = backup_path / "exports"
        self.assertTrue(backup_exports.exists())
        self.assertTrue(backup_exports.is_dir())
        self.assertTrue((backup_exports / "export_1.csv").exists())
        self.assertTrue((backup_exports / "export_2.csv").exists())

        # Verify metadata file exists and has correct structure
        metadata_path = backup_path / "backup_metadata.json"
        self.assertTrue(metadata_path.exists())
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.assertIn("created_at", metadata)
        self.assertIn("version", metadata)
        self.assertIn("files_backed_up", metadata)
        self.assertIsInstance(metadata["files_backed_up"], list)

    def test_restore_from_backup_success(self) -> None:
        """Test that restore_from_backup correctly replaces existing data files."""
        # Create initial backup
        success, _, original_backup_path = create_full_backup(backup_name="original_backup")
        self.assertTrue(success)

        # Modify original data files
        modified_data = "MODIFIED_DATA"
        for db_name in ["employees.db", "schedule.db"]:
            db_path = self.temp_data_dir / db_name
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            cursor.execute("UPDATE test_table SET data = ?", (modified_data,))
            conn.commit()
            conn.close()

        # Modify JSON file
        accounts_path = self.temp_data_dir / "accounts.json"
        accounts_path.write_text(json.dumps({"modified": True}), encoding="utf-8")

        # Modify exports
        exports_dir = self.temp_data_dir / "exports"
        (exports_dir / "export_1.csv").write_text("MODIFIED\n", encoding="utf-8")

        # Restore from backup
        success, message = restore_from_backup(original_backup_path)
        self.assertTrue(success)
        self.assertIn("original_backup", message)

        # Verify databases are restored
        for db_name in ["employees.db", "schedule.db"]:
            db_path = self.temp_data_dir / db_name
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            cursor.execute("SELECT data FROM test_table")
            result = cursor.fetchone()
            self.assertEqual(result[0], f"test_data_{db_name}")
            self.assertNotEqual(result[0], modified_data)
            conn.close()

        # Verify JSON file is restored
        accounts_content = json.loads(accounts_path.read_text(encoding="utf-8"))
        self.assertIn("users", accounts_content)
        self.assertNotIn("modified", accounts_content)

        # Verify exports are restored
        export_content = (exports_dir / "export_1.csv").read_text(encoding="utf-8")
        self.assertIn("id,name", export_content)
        self.assertNotIn("MODIFIED", export_content)

    def test_cleanup_old_auto_backups_keeps_recent_only(self) -> None:
        """Test that cleanup_old_auto_backups removes older backups while keeping recent ones."""
        # Create multiple automatic backups with different timestamps
        auto_backup_paths = []
        for i in range(8):
            backup_name = f"auto_2024010{i}_120000"
            backup_path = self.temp_backup_root / backup_name
            backup_path.mkdir(parents=True, exist_ok=True)
            # Create a dummy file to make it non-empty
            (backup_path / "dummy.txt").write_text(f"backup {i}", encoding="utf-8")
            auto_backup_paths.append(backup_path)

        # Create some manual backups that should NOT be deleted
        manual_backup_paths = []
        for i in range(3):
            backup_name = f"manual_backup_{i}"
            backup_path = self.temp_backup_root / backup_name
            backup_path.mkdir(parents=True, exist_ok=True)
            (backup_path / "dummy.txt").write_text(f"manual {i}", encoding="utf-8")
            manual_backup_paths.append(backup_path)

        # Verify all backups exist before cleanup
        self.assertEqual(len(list(self.temp_backup_root.iterdir())), 11)

        # Run cleanup keeping only 5 automatic backups
        cleanup_old_auto_backups(keep_count=5)

        # Verify that only 5 automatic backups remain
        remaining_auto = [
            item for item in self.temp_backup_root.iterdir()
            if item.is_dir() and item.name.startswith("auto_")
        ]
        self.assertEqual(len(remaining_auto), 5)

        # Verify all manual backups still exist
        for manual_path in manual_backup_paths:
            self.assertTrue(manual_path.exists(), f"Manual backup {manual_path.name} should not be deleted")

        # Verify the most recent 5 auto backups are kept (indices 3-7)
        for i in range(3, 8):
            self.assertTrue(
                auto_backup_paths[i].exists(),
                f"Recent auto backup {auto_backup_paths[i].name} should be kept"
            )

        # Verify older auto backups are deleted (indices 0-2)
        for i in range(0, 3):
            self.assertFalse(
                auto_backup_paths[i].exists(),
                f"Old auto backup {auto_backup_paths[i].name} should be deleted"
            )

    def test_create_full_backup_handles_missing_files(self) -> None:
        """Test that backup works even when some optional files are missing."""
        # Remove exports directory and one JSON file
        shutil.rmtree(self.temp_data_dir / "exports")
        (self.temp_data_dir / "audit.log").unlink()

        # Create backup
        success, message, backup_path = create_full_backup(backup_name="partial_backup")

        # Verify success
        self.assertTrue(success)
        self.assertTrue(backup_path.exists())

        # Verify databases are still backed up
        self.assertTrue((backup_path / "employees.db").exists())

        # Verify existing JSON files are backed up
        self.assertTrue((backup_path / "accounts.json").exists())

        # Verify missing files don't cause errors
        self.assertFalse((backup_path / "audit.log").exists())
        self.assertFalse((backup_path / "exports").exists())

    def test_restore_from_backup_nonexistent_path(self) -> None:
        """Test that restore_from_backup handles nonexistent backup paths gracefully."""
        nonexistent_path = self.temp_backup_root / "nonexistent_backup"
        success, message = restore_from_backup(nonexistent_path)

        self.assertFalse(success)
        self.assertIn("not found", message.lower())


class WeeklyHoursWarningsTests(unittest.TestCase):
    """Tests for _weekly_hours_warnings validation function."""

    def setUp(self) -> None:
        """Set up in-memory databases for validation tests."""
        self.schedule_engine = create_engine("sqlite:///:memory:", future=True)
        Base.metadata.create_all(self.schedule_engine)
        self.employee_engine = create_engine("sqlite:///:memory:", future=True)
        EmployeeBase.metadata.create_all(self.employee_engine)
        self.projection_engine = create_engine("sqlite:///:memory:", future=True)
        ProjectionsBase.metadata.create_all(self.projection_engine)

        session_factory = sessionmaker(bind=self.schedule_engine, expire_on_commit=False, future=True)
        employee_session_factory = sessionmaker(bind=self.employee_engine, expire_on_commit=False, future=True)
        projection_session_factory = sessionmaker(bind=self.projection_engine, expire_on_commit=False, future=True)

        db.policy_engine = self.schedule_engine
        db.PolicySessionLocal = session_factory
        db.projections_engine = self.projection_engine
        db.ProjectionSessionLocal = projection_session_factory
        PolicyBase.metadata.create_all(db.policy_engine)

        self.session = session_factory()
        self.employee_session = employee_session_factory()
        self.week_start = datetime.date(2024, 4, 1)

        upsert_policy(self.session, "Test Policy", build_default_policy(), edited_by="tests")

    def tearDown(self) -> None:
        """Clean up database connections."""
        self.session.close()
        self.employee_session.close()
        self.schedule_engine.dispose()
        self.employee_engine.dispose()
        self.projection_engine.dispose()

    def _add_employee(self, name: str, roles: list[str]) -> Employee:
        """Helper to add an employee to the database."""
        employee = Employee(
            full_name=name,
            roles=", ".join(roles),
            desired_hours=40,
            status="active",
            notes=""
        )
        self.employee_session.add(employee)
        self.employee_session.commit()
        return employee

    def _add_shift(
        self,
        *,
        role: str,
        start: datetime.datetime,
        end: datetime.datetime,
        employee: Employee | None,
    ) -> Shift:
        """Helper to add a shift to the database."""
        week = get_or_create_week(self.session, self.week_start)
        shift = Shift(
            week_id=week.id,
            employee_id=employee.id if employee else None,
            role=role,
            start=start,
            end=end,
            status="draft",
        )
        self.session.add(shift)
        self.session.commit()
        return shift

    def test_weekly_hours_warnings_with_custom_max_hours(self) -> None:
        """Test that _weekly_hours_warnings correctly identifies employees exceeding custom max_hours_week."""
        # Create employee
        employee = self._add_employee("Overworked Employee", ["Server"])

        # Create policy with custom max_hours_week of 35
        custom_policy = build_default_policy()
        custom_policy["global"]["max_hours_week"] = 35

        # Add shifts totaling 38 hours (exceeds 35-hour limit)
        shifts = []
        start_day = datetime.datetime(2024, 4, 1, 10, 0, tzinfo=UTC)
        shift_hours = [8, 8, 8, 8, 6]  # Total: 38 hours

        for day_offset, hours in enumerate(shift_hours):
            shift = self._add_shift(
                role="Server",
                start=start_day + datetime.timedelta(days=day_offset),
                end=start_day + datetime.timedelta(days=day_offset, hours=hours),
                employee=employee,
            )
            shifts.append(shift)

        # Build employee map
        employee_map = {employee.id: employee}

        # Call _weekly_hours_warnings
        warnings = _weekly_hours_warnings(shifts, employee_map, custom_policy)

        # Verify warning is generated
        self.assertEqual(len(warnings), 1)
        warning = warnings[0]

        # Verify warning details
        self.assertEqual(warning["type"], "weekly_hours")
        self.assertEqual(warning["severity"], "warning")
        self.assertEqual(warning["employee_id"], employee.id)
        self.assertEqual(warning["employee"], "Overworked Employee")
        self.assertEqual(warning["hours"], 38.0)
        self.assertEqual(warning["limit"], 35)
        self.assertIn("38.0 hours", warning["message"])
        self.assertIn("35.0-hour limit", warning["message"])
        self.assertIn("3.0 hours", warning["message"])  # 38 - 35 = 3

    def test_weekly_hours_warnings_no_warning_under_limit(self) -> None:
        """Test that no warning is generated when employee is under the custom limit."""
        employee = self._add_employee("Part Timer", ["Server"])

        # Custom policy with 30-hour limit
        custom_policy = build_default_policy()
        custom_policy["global"]["max_hours_week"] = 30

        # Add shifts totaling 28 hours (under 30-hour limit)
        shifts = []
        start_day = datetime.datetime(2024, 4, 1, 10, 0, tzinfo=UTC)
        shift_hours = [7, 7, 7, 7]  # Total: 28 hours

        for day_offset, hours in enumerate(shift_hours):
            shift = self._add_shift(
                role="Server",
                start=start_day + datetime.timedelta(days=day_offset),
                end=start_day + datetime.timedelta(days=day_offset, hours=hours),
                employee=employee,
            )
            shifts.append(shift)

        employee_map = {employee.id: employee}

        # Call _weekly_hours_warnings
        warnings = _weekly_hours_warnings(shifts, employee_map, custom_policy)

        # Verify no warning is generated
        self.assertEqual(len(warnings), 0)

    def test_weekly_hours_warnings_multiple_employees(self) -> None:
        """Test that warnings are generated for multiple employees exceeding the limit."""
        # Create multiple employees
        employee1 = self._add_employee("Employee 1", ["Server"])
        employee2 = self._add_employee("Employee 2", ["Bartender"])
        employee3 = self._add_employee("Employee 3", ["Server"])

        # Custom policy with 32-hour limit
        custom_policy = build_default_policy()
        custom_policy["global"]["max_hours_week"] = 32

        shifts = []
        start_day = datetime.datetime(2024, 4, 1, 10, 0, tzinfo=UTC)

        # Employee 1: 36 hours (exceeds)
        for day in range(6):
            shifts.append(
                self._add_shift(
                    role="Server",
                    start=start_day + datetime.timedelta(days=day),
                    end=start_day + datetime.timedelta(days=day, hours=6),
                    employee=employee1,
                )
            )

        # Employee 2: 30 hours (under limit)
        for day in range(5):
            shifts.append(
                self._add_shift(
                    role="Bartender",
                    start=start_day + datetime.timedelta(days=day),
                    end=start_day + datetime.timedelta(days=day, hours=6),
                    employee=employee2,
                )
            )

        # Employee 3: 35 hours (exceeds)
        for day in range(5):
            shifts.append(
                self._add_shift(
                    role="Server",
                    start=start_day + datetime.timedelta(days=day),
                    end=start_day + datetime.timedelta(days=day, hours=7),
                    employee=employee3,
                )
            )

        employee_map = {
            employee1.id: employee1,
            employee2.id: employee2,
            employee3.id: employee3,
        }

        # Call _weekly_hours_warnings
        warnings = _weekly_hours_warnings(shifts, employee_map, custom_policy)

        # Verify warnings for employee1 and employee3 only
        self.assertEqual(len(warnings), 2)
        warning_employee_ids = {w["employee_id"] for w in warnings}
        self.assertIn(employee1.id, warning_employee_ids)
        self.assertIn(employee3.id, warning_employee_ids)
        self.assertNotIn(employee2.id, warning_employee_ids)

        # Verify limits are correct
        for warning in warnings:
            self.assertEqual(warning["limit"], 32)


@unittest.skipUnless(PYSIDE6_AVAILABLE, "PySide6 not available")
class BackupManagerDialogTests(unittest.TestCase):
    """Tests for BackupManagerDialog UI component."""

    @classmethod
    def setUpClass(cls) -> None:
        """Create QApplication instance for all tests."""
        cls.app = QApplication.instance() or QApplication(sys.argv)

    def setUp(self) -> None:
        """Set up temporary directories and dialog for each test."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.temp_backup_root = self.temp_dir / "backups"
        self.temp_backup_root.mkdir(parents=True, exist_ok=True)

        # Patch backup root
        self.backup_root_patcher = mock.patch("backup.BACKUP_ROOT", self.temp_backup_root)
        self.backup_root_patcher.start()

        # Also patch in backup_dialog module
        self.backup_dialog_patcher = mock.patch("ui.backup_dialog.list_backups")
        self.mock_list_backups = self.backup_dialog_patcher.start()
        self.mock_list_backups.return_value = []

        self.dialog = BackupManagerDialog()

    def tearDown(self) -> None:
        """Clean up dialog and temporary directories."""
        self.dialog.close()
        self.backup_root_patcher.stop()
        self.backup_dialog_patcher.stop()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_buttons_disabled_when_no_selection(self) -> None:
        """Test that restore and delete buttons are disabled when no backup is selected."""
        # Initially, no selection should exist
        self.assertFalse(self.dialog.restore_button.isEnabled())
        self.assertFalse(self.dialog.delete_button.isEnabled())

    def test_buttons_enabled_when_backup_selected(self) -> None:
        """Test that restore and delete buttons are enabled when a backup is selected."""
        # Create mock backup data
        mock_backup = {
            "name": "test_backup",
            "path": self.temp_backup_root / "test_backup",
            "created": datetime.datetime.now(),
            "size": 1024,
        }
        self.mock_list_backups.return_value = [mock_backup]

        # Refresh the list to populate with mock data
        self.dialog._refresh_backup_list()

        # Initially still disabled (no selection)
        self.assertFalse(self.dialog.restore_button.isEnabled())
        self.assertFalse(self.dialog.delete_button.isEnabled())

        # Select the first item
        self.dialog.backup_list.setCurrentRow(0)

        # Buttons should now be enabled
        self.assertTrue(self.dialog.restore_button.isEnabled())
        self.assertTrue(self.dialog.delete_button.isEnabled())

    def test_buttons_disabled_when_selection_cleared(self) -> None:
        """Test that buttons are disabled when selection is cleared."""
        # Create and select a backup
        mock_backup = {
            "name": "test_backup",
            "path": self.temp_backup_root / "test_backup",
            "created": datetime.datetime.now(),
            "size": 1024,
        }
        self.mock_list_backups.return_value = [mock_backup]
        self.dialog._refresh_backup_list()
        self.dialog.backup_list.setCurrentRow(0)

        # Verify buttons are enabled
        self.assertTrue(self.dialog.restore_button.isEnabled())
        self.assertTrue(self.dialog.delete_button.isEnabled())

        # Clear selection
        self.dialog.backup_list.clearSelection()

        # Buttons should now be disabled
        self.assertFalse(self.dialog.restore_button.isEnabled())
        self.assertFalse(self.dialog.delete_button.isEnabled())

    def test_buttons_disabled_for_empty_list_placeholder(self) -> None:
        """Test that buttons remain disabled when 'No backups found' placeholder is shown."""
        # Mock empty backup list
        self.mock_list_backups.return_value = []
        self.dialog._refresh_backup_list()

        # Try to select the "No backups found" item
        self.dialog.backup_list.setCurrentRow(0)

        # Buttons should remain disabled because the item has no valid path
        self.assertFalse(self.dialog.restore_button.isEnabled())
        self.assertFalse(self.dialog.delete_button.isEnabled())

    def test_selected_backup_path_updated_on_selection(self) -> None:
        """Test that selected_backup attribute is correctly updated when a backup is selected."""
        # Create mock backups
        backup_path = self.temp_backup_root / "test_backup"
        mock_backup = {
            "name": "test_backup",
            "path": backup_path,
            "created": datetime.datetime.now(),
            "size": 2048,
        }
        self.mock_list_backups.return_value = [mock_backup]
        self.dialog._refresh_backup_list()

        # Select the backup
        self.dialog.backup_list.setCurrentRow(0)

        # Verify selected_backup is set correctly
        self.assertIsNotNone(self.dialog.selected_backup)
        self.assertEqual(self.dialog.selected_backup, backup_path)


if __name__ == "__main__":
    unittest.main()
