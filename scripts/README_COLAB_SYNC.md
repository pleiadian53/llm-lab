# Colab Sync Script

## Overview

`sync_to_colab.py` syncs your local `llm-lab` project to Google Drive so it can be accessed from Google Colab in your hybrid cloud-local development environment.

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

## Workflow

### 1. Local Development
```bash
# Work on your Mac as usual
cd ~/work/llm-lab
# Edit files, commit to git, etc.
```

### 2. Sync to Google Drive
```bash
# Sync changes to Drive
./scripts/sync_to_colab.py

# Or with verbose output
./scripts/sync_to_colab.py -v
```

### 3. Access in Colab
```python
# In Colab notebook
from google.colab import drive
drive.mount('/content/drive')

# Navigate to project
%cd /content/drive/MyDrive/work_MBP_M1/llm-lab

# Install your package
!pip install -e posttrain_llm/llm_eval

# Use it
from llm_eval import ServeLLM
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
