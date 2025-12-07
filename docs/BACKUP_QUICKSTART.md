# Backup & Restore Quick Reference

## Quick Access

**Who**: IT and GM users only  
**Where**: Week Preparation tab → "Backup & Restore" button (bottom of screen)

## Automatic Backups

✅ **Every startup**: Auto-backup created with `auto_` prefix  
✅ **Auto-cleanup**: Keeps 5 most recent automatic backups  
📁 **Location**: `app/backups/`

## Manual Operations

### Create Backup
1. Click "Create new backup"
2. Confirm → Done

**When to backup**:
- Before major policy changes
- Before bulk imports
- End of week/month
- Before updates

### Restore Backup
1. Select backup from list
2. Click "Restore selected"
3. ⚠️ **WARNING**: Creates backup of current data first!
4. Confirm
5. **Restart application**

### Delete Backup
1. Select backup from list
2. Click "Delete selected"
3. Confirm

## Backup Contents

Each backup includes:
- ✅ All 4 databases (employees, schedule, policy, projections)
- ✅ User accounts
- ✅ Configuration files
- ✅ Audit log
- ✅ Exports folder

## Tips

💡 **Manual backups never expire** - only automatic ones are cleaned up  
💡 **Backup names show timestamps** - easy to identify  
💡 **Size displayed** - verify backup is complete  
💡 **Test backups** - `python3 test_backup.py`

## Emergency Recovery

If data is corrupted:
1. Launch app
2. Open Backup & Restore
3. Select most recent good backup
4. Restore
5. Restart

---

For detailed documentation, see `docs/BACKUP_RESTORE.md`
