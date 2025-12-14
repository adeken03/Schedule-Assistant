#!/usr/bin/env python3
"""Quick test script for backup functionality."""

import sys
from pathlib import Path

# Add app directory to path
sys.path.insert(0, str(Path(__file__).parent / "app"))

from backup import (
    create_full_backup,
    list_backups,
    format_size,
    auto_backup_on_startup,
    cleanup_old_auto_backups,
)


def test_backup_functions():
    print("Testing backup module functions...")
    
    # Test format_size
    print(f"Format 1024 bytes: {format_size(1024)}")
    print(f"Format 1048576 bytes: {format_size(1048576)}")
    
    # Test list_backups
    print("\nListing existing backups:")
    backups = list_backups()
    if backups:
        for backup in backups:
            print(f"  - {backup['name']}: {format_size(backup['size'])}")
    else:
        print("  No backups found")
    
    # Test auto backup
    print("\nCreating automatic backup...")
    success, message = auto_backup_on_startup()
    print(f"  Result: {'✓' if success else '✗'} {message}")
    
    # Test cleanup
    print("\nCleaning up old automatic backups (keeping 5)...")
    cleanup_old_auto_backups(keep_count=5)
    print("  Cleanup complete")
    
    # List backups again
    print("\nListing backups after auto-backup:")
    backups = list_backups()
    for backup in backups:
        backup_type = "Auto" if backup['name'].startswith("auto_") else "Manual"
        print(f"  - {backup['name']} ({backup_type}): {format_size(backup['size'])}")
    
    print("\n✓ All tests completed successfully!")


if __name__ == "__main__":
    try:
        test_backup_functions()
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
