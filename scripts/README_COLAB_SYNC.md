# Colab Sync Scripts

## Overview

Two scripts for bidirectional sync between local machine and Google Drive/Colab:

- **`sync_to_colab.py`** - Push local changes TO Google Drive (for Colab)
- **`sync_from_colab.py`** - Pull Colab changes FROM Google Drive (to local)

⚠️ **IMPORTANT**: Always pull before push to avoid losing work created in Colab!

## Quick Start

```bash
# Preview what will be synced (recommended first time)
./scripts/sync_to_colab.py --dry-run

# Perform actual sync
./scripts/sync_to_colab.py

# Verbose output
./scripts/sync_to_colab.py -v
```

## Paths

| Location | Path |
|----------|------|
| **Local Source** | `~/work/llm-lab` |
| **Google Drive** | `~/GoogleDrive/work_MBP_M1/llm-lab` |
| **In Colab** | `/content/drive/MyDrive/work_MBP_M1/llm-lab` |

## What Gets Synced

### ✅ Included
- Python files (`.py`)
- Notebooks (`.ipynb`)
- Documentation (`.md`, `.txt`)
- Configuration (`.json`, `.yaml`, `.toml`)
- Scripts (`.sh`, `.bash`)
- Images (`.png`, `.jpg`, `.svg`)
- Small PDFs

### ❌ Excluded
- `.git/` directory (too large for Drive)
- `__pycache__/`, `.venv/`, `.cache/`
- Large datasets (`data/`, `models/`, `weights/`)
- Model checkpoints (`.pt`, `.pth`, `.ckpt`)
- Compressed archives (`.gz`, `.tar`, `.zip`)
- Build artifacts

## Bidirectional Workflow

### Scenario 1: Local → Colab (Push)

**When**: You made changes locally and want to use them in Colab

```bash
# 1. Work locally
cd ~/work/llm-lab
# Edit files, commit to git

# 2. Push to Drive
./scripts/sync_to_colab.py

# 3. Use in Colab
# Files are now available in Colab
```

### Scenario 2: Colab → Local (Pull)

**When**: You created/modified files in Colab and want them locally

```bash
# 1. After working in Colab, pull changes
./scripts/sync_from_colab.py

# 2. Review what changed
git status
git diff

# 3. Commit if desired
git add .
git commit -m "Work from Colab session"
```

### Scenario 3: Round-trip (Pull → Work → Push)

**When**: You work in both environments

```bash
# 1. Pull Colab changes first (IMPORTANT!)
./scripts/sync_from_colab.py

# 2. Work locally
# Edit files, test, commit

# 3. Push back to Colab
./scripts/sync_to_colab.py

# 4. Continue in Colab with latest changes
```

### ⚠️ **CRITICAL: Avoid Data Loss**

```bash
# ❌ WRONG - Will delete Colab work!
./scripts/sync_to_colab.py  # Without pulling first

# ✅ CORRECT - Pull first, then push
./scripts/sync_from_colab.py  # Get Colab changes
./scripts/sync_to_colab.py    # Push local changes
```

## Important Notes

### Git Operations
- **Do NOT** use git in Colab (`.git/` is not synced)
- All git operations (commit, push, pull) should be done locally
- Colab is for compute-heavy tasks only

### Large Files
- Model weights should be in `llm_cache/` (separate from project)
- `llm_cache/` is protected from deletion during sync
- Configure `HF_HOME=/content/drive/MyDrive/work_MBP_M1/llm_cache` in Colab

### Persistence
- Files in Google Drive persist between Colab sessions
- Installed packages (`pip install`) do NOT persist
- Environment variables do NOT persist
- You must reinstall packages each session

## Command Options

```bash
./scripts/sync_to_colab.py [OPTIONS]

Options:
  -n, --dry-run      Preview changes without syncing
  -v, --verbose      Show detailed output
  --no-delete        Don't delete files removed locally
  --src PATH         Custom source directory
  --dest PATH        Custom destination directory
  --help             Show help message
```

## Examples

### Dry Run (Preview)
```bash
./scripts/sync_to_colab.py --dry-run
```

### Verbose Sync
```bash
./scripts/sync_to_colab.py -v
```

### Sync Without Deleting
```bash
# Keep files in Drive even if deleted locally
./scripts/sync_to_colab.py --no-delete
```

### Custom Paths
```bash
./scripts/sync_to_colab.py \
  --src ~/work/llm-lab \
  --dest ~/GoogleDrive/work_MBP_M1/llm-lab
```

## Troubleshooting

### Google Drive Not Found
```
ERROR: Google Drive not found at: /Users/pleiadian53/GoogleDrive
```

**Solution**: Make sure Google Drive is mounted and syncing. Check that the symlink exists:
```bash
ls -l ~/GoogleDrive
# Should show: GoogleDrive -> /Users/pleiadian53/Google Drive
```

### Sync Takes Too Long
- First sync may take a while (uploading all files)
- Subsequent syncs are incremental (only changed files)
- Use `--dry-run` to preview before syncing

### Files Not Appearing in Colab
- Wait for Google Drive to finish syncing (check Drive app)
- In Colab, remount Drive: `drive.mount('/content/drive', force_remount=True)`
- Check the path: `/content/drive/MyDrive/work_MBP_M1/llm-lab`

### Accidental Deletion
- If you accidentally delete files, use `--no-delete` to restore:
  ```bash
  ./scripts/sync_to_colab.py --no-delete
  ```
- Files in Drive are preserved until explicitly deleted

## Logs

Sync logs are saved to:
```
~/Library/Logs/colab_sync/sync_YYYY-MM-DD_HH-MM-SS.log
```

Check logs for detailed information about what was synced.

## Comparison with sync_work.py

| Feature | sync_work.py | sync_to_colab.py |
|---------|--------------|------------------|
| **Purpose** | Backup all projects to Dropbox | Sync llm-lab to Google Drive |
| **Scope** | All of `~/work/` | Only `~/work/llm-lab` |
| **Destination** | Dropbox | Google Drive |
| **Git** | Excluded | Excluded |
| **Use Case** | General backup | Colab development |

## Related Documentation

- **Hybrid Setup**: `dev/colab/Hybrid-Colab-SSH-Development-Setup.md`
- **LLM Eval Package**: `posttrain_llm/llm_eval/README.md`
- **General Sync**: `scripts/sync_work.py`

---

**Last Updated**: 2025-11-30  
**Version**: 1.0.0
