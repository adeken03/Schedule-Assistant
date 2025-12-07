# Backup & Restore Guide

## Overview

The Schedule Assistant application includes a comprehensive backup and restore system that protects all application data including:

- **Databases**: employees.db, schedule.db, policy.db, projections.db
- **Configuration**: accounts.json, week_state.json
- **Audit Logs**: audit.log
- **Exports**: All exported files

## Automatic Backups

### On Startup
The application automatically creates a backup every time it starts. These backups are:
- Prefixed with `auto_` for easy identification
- Stored in `app/backups/` directory
- Limited to the 5 most recent (older ones are automatically deleted)

### Backup Location
All backups are stored in: `Schedule-Assistant/app/backups/`

## Manual Backup & Restore

### Accessing the Backup Manager

**IT and GM users only** can access the Backup & Restore Manager:

1. Launch the application and sign in
2. Navigate to the "Week Preparation" tab
3. Click the **"Backup & Restore"** button at the bottom of the screen

### Creating a Manual Backup

1. Open the Backup Manager dialog
2. Click **"Create new backup"**
3. Confirm the action
4. The backup will be created with a timestamp (e.g., `backup_20251205_134500`)

Manual backups are never automatically deleted, unlike automatic backups.

### Restoring from a Backup

⚠️ **WARNING**: Restoring replaces ALL current data with the backup data.

1. Open the Backup Manager dialog
2. Select a backup from the list
3. Click **"Restore selected"**
4. **IMPORTANT**: Consider creating a backup of current data first!
5. Confirm the restoration
6. **Restart the application** after restoration completes

### Deleting Old Backups

1. Open the Backup Manager dialog
2. Select a backup from the list
3. Click **"Delete selected"**
4. Confirm the deletion

## Backup Information

Each backup entry shows:
- **Name**: Backup directory name with timestamp
- **Created**: Date and time the backup was created
- **Size**: Total size of all backed-up files
- **Type**: "Automatic" (created on startup) or "Manual" (user-created)

## Technical Details

### What Gets Backed Up

```
app/backups/backup_YYYYMMDD_HHMMSS/
├── employees.db          # Employee roster and availability
├── schedule.db           # Week schedules and shifts
├── policy.db             # Scheduling policies
├── projections.db        # Sales projections
├── accounts.json         # User accounts
├── week_state.json       # Current week context
├── audit.log             # Action audit trail
├── exports/              # Exported files directory
└── backup_metadata.json  # Backup information
```

### Backup Method

- **Databases**: Uses SQLite's backup API for data consistency
- **Files**: Standard file copy with metadata preservation
- **Metadata**: JSON file tracking backup creation time and contents

### Automatic Cleanup

The system keeps only the 5 most recent automatic backups. This prevents unlimited disk usage while maintaining a reasonable safety net.

## Best Practices

### Regular Manual Backups

Create manual backups:
- Before major policy changes
- Before bulk employee imports/updates
- At the end of each week/month
- Before software updates

### Backup Verification

Periodically verify your backups:
1. Check the backup list to ensure recent backups exist
2. Verify backup sizes are reasonable (not 0 bytes)
3. Test restoration on a development/test system if available

### Disaster Recovery

In case of data loss or corruption:
1. Launch the application
2. Open Backup & Restore Manager
3. Select the most recent stable backup
4. Restore and restart

### External Backups

For additional protection, consider:
- Copying the `app/backups/` folder to an external drive
- Using cloud storage for critical manual backups
- Keeping backups of important weekly schedules

## Troubleshooting

### Backup Failed

If backup creation fails:
- Check available disk space
- Verify write permissions in the application directory
- Check that no other application is locking the database files

### Restore Failed

If restoration fails:
- Ensure no other instance of the application is running
- Verify the backup directory is complete (not corrupted)
- Check file permissions in the app/data directory

### Missing Backups Directory

The backups directory is created automatically. If it's missing:
- The directory will be created on next startup
- Previous backups may have been manually deleted

## Command-Line Testing

For advanced users, test backup functionality:

```bash
cd Schedule-Assistant
python3 test_backup.py
```

This script tests:
- Backup creation
- Backup listing
- Automatic cleanup
- Size formatting

## Notes

- Backups are **NOT** version controlled (added to .gitignore)
- Each backup is a complete snapshot of all data
- Backups do not include the application code itself
- Database files are backed up using SQLite's atomic backup method
- The virtual environment (.venv) is NOT backed up
