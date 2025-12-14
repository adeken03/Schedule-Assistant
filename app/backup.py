from __future__ import annotations

import datetime
import json
import shutil
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Tuple


DATA_DIR = Path(__file__).resolve().parent / "data"
BACKUP_ROOT = DATA_DIR.parent / "backups"


def _timestamp() -> str:
    """Generate timestamp string for backup naming."""
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def get_backup_dir(backup_name: Optional[str] = None) -> Path:
    """Get or create backup directory path."""
    BACKUP_ROOT.mkdir(parents=True, exist_ok=True)
    if backup_name is None:
        backup_name = f"backup_{_timestamp()}"
    return BACKUP_ROOT / backup_name


def list_backups() -> List[Dict[str, any]]:
    """List all available backups with metadata.
    
    Returns:
        List of dicts with 'name', 'path', 'created', 'size' keys.
    """
    if not BACKUP_ROOT.exists():
        return []
    
    backups = []
    for item in sorted(BACKUP_ROOT.iterdir(), reverse=True):
        if not item.is_dir():
            continue
        
        # Calculate total size
        total_size = sum(f.stat().st_size for f in item.rglob('*') if f.is_file())
        
        # Get creation time from directory
        created = datetime.datetime.fromtimestamp(item.stat().st_ctime)
        
        backups.append({
            "name": item.name,
            "path": item,
            "created": created,
            "size": total_size,
        })
    
    return backups


def backup_databases(backup_dir: Path) -> None:
    """Backup all SQLite databases using SQLite backup API.
    
    Args:
        backup_dir: Directory to store backup files
    """
    databases = [
        "employees.db",
        "schedule.db",
        "policy.db",
        "projections.db",
    ]
    
    for db_name in databases:
        source_path = DATA_DIR / db_name
        if not source_path.exists():
            continue
        
        backup_path = backup_dir / db_name
        
        # Use SQLite backup API for consistency
        source = sqlite3.connect(str(source_path))
        backup = sqlite3.connect(str(backup_path))
        
        with backup:
            source.backup(backup)
        
        backup.close()
        source.close()


def backup_json_files(backup_dir: Path) -> None:
    """Backup JSON configuration and state files.
    
    Args:
        backup_dir: Directory to store backup files
    """
    json_files = ["accounts.json", "week_state.json", "audit.log"]
    
    for filename in json_files:
        source = DATA_DIR / filename
        if source.exists():
            shutil.copy2(source, backup_dir / filename)


def backup_exports(backup_dir: Path) -> None:
    """Backup the exports directory if it exists.
    
    Args:
        backup_dir: Directory to store backup files
    """
    exports_dir = DATA_DIR / "exports"
    if exports_dir.exists() and exports_dir.is_dir():
        backup_exports_dir = backup_dir / "exports"
        shutil.copytree(exports_dir, backup_exports_dir)


def create_backup_metadata(backup_dir: Path) -> None:
    """Create metadata file for the backup.
    
    Args:
        backup_dir: Directory containing the backup
    """
    metadata = {
        "created_at": datetime.datetime.now().isoformat(),
        "version": "1.0",
        "files_backed_up": [f.name for f in backup_dir.iterdir() if f.is_file()],
    }
    
    metadata_path = backup_dir / "backup_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def create_full_backup(backup_name: Optional[str] = None) -> Tuple[bool, str, Path]:
    """Create a complete backup of all application data.
    
    Args:
        backup_name: Optional custom name for backup directory
        
    Returns:
        Tuple of (success, message, backup_path)
    """
    try:
        backup_dir = get_backup_dir(backup_name)
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Backup databases
        backup_databases(backup_dir)
        
        # Backup JSON files
        backup_json_files(backup_dir)
        
        # Backup exports directory
        backup_exports(backup_dir)
        
        # Create metadata
        create_backup_metadata(backup_dir)
        
        return True, f"Backup created successfully: {backup_dir.name}", backup_dir
        
    except Exception as e:
        return False, f"Backup failed: {str(e)}", Path()


def restore_databases(backup_dir: Path) -> None:
    """Restore SQLite databases from backup.
    
    Args:
        backup_dir: Directory containing backup files
    """
    databases = [
        "employees.db",
        "schedule.db",
        "policy.db",
        "projections.db",
    ]
    
    for db_name in databases:
        backup_path = backup_dir / db_name
        if not backup_path.exists():
            continue
        
        target_path = DATA_DIR / db_name
        
        # Close any existing connections and replace file
        if target_path.exists():
            target_path.unlink()
        
        shutil.copy2(backup_path, target_path)


def restore_json_files(backup_dir: Path) -> None:
    """Restore JSON configuration and state files from backup.
    
    Args:
        backup_dir: Directory containing backup files
    """
    json_files = ["accounts.json", "week_state.json", "audit.log"]
    
    for filename in json_files:
        backup_path = backup_dir / filename
        if backup_path.exists():
            target_path = DATA_DIR / filename
            shutil.copy2(backup_path, target_path)


def restore_exports(backup_dir: Path) -> None:
    """Restore the exports directory from backup.
    
    Args:
        backup_dir: Directory containing backup files
    """
    backup_exports_dir = backup_dir / "exports"
    if not backup_exports_dir.exists():
        return
    
    target_exports_dir = DATA_DIR / "exports"
    if target_exports_dir.exists():
        shutil.rmtree(target_exports_dir)
    
    shutil.copytree(backup_exports_dir, target_exports_dir)


def restore_from_backup(backup_path: Path) -> Tuple[bool, str]:
    """Restore application data from a backup.
    
    Args:
        backup_path: Path to backup directory
        
    Returns:
        Tuple of (success, message)
    """
    if not backup_path.exists() or not backup_path.is_dir():
        return False, "Backup directory not found"
    
    try:
        # Restore databases
        restore_databases(backup_path)
        
        # Restore JSON files
        restore_json_files(backup_path)
        
        # Restore exports directory
        restore_exports(backup_path)
        
        return True, f"Restore completed successfully from: {backup_path.name}"
        
    except Exception as e:
        return False, f"Restore failed: {str(e)}"


def delete_backup(backup_path: Path) -> Tuple[bool, str]:
    """Delete a backup directory.
    
    Args:
        backup_path: Path to backup directory to delete
        
    Returns:
        Tuple of (success, message)
    """
    if not backup_path.exists():
        return False, "Backup not found"
    
    try:
        shutil.rmtree(backup_path)
        return True, f"Backup deleted: {backup_path.name}"
    except Exception as e:
        return False, f"Failed to delete backup: {str(e)}"


def format_size(size_bytes: int) -> str:
    """Format byte size to human-readable string.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted string (e.g., "1.5 MB")
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


def auto_backup_on_startup() -> Tuple[bool, str]:
    """Create an automatic backup with 'auto_' prefix.
    
    Returns:
        Tuple of (success, message)
    """
    backup_name = f"auto_{_timestamp()}"
    success, message, _ = create_full_backup(backup_name)
    return success, message


def cleanup_old_auto_backups(keep_count: int = 5) -> None:
    """Remove old automatic backups, keeping only the most recent ones.
    
    Args:
        keep_count: Number of automatic backups to keep
    """
    if not BACKUP_ROOT.exists():
        return
    
    # Find all automatic backups
    auto_backups = [
        item for item in BACKUP_ROOT.iterdir()
        if item.is_dir() and item.name.startswith("auto_")
    ]
    
    # Sort by creation time (newest first)
    auto_backups.sort(key=lambda x: x.stat().st_ctime, reverse=True)
    
    # Delete old backups beyond keep_count
    for old_backup in auto_backups[keep_count:]:
        try:
            shutil.rmtree(old_backup)
        except Exception:
            pass  # Silently ignore deletion errors
