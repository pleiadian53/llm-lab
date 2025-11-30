#!/usr/bin/env python3
"""
Sync llm-lab project to Google Drive for Colab hybrid environment.

This script syncs the local ~/work/llm-lab directory to Google Drive
so it can be accessed from Google Colab via the mounted Drive.

USAGE:
  ./sync_to_colab.py              # Sync llm-lab to Google Drive
  ./sync_to_colab.py -n           # Dry run (preview only)
  ./sync_to_colab.py -v           # Verbose output
  ./sync_to_colab.py --help       # See all options

PATHS:
  Source:      ~/work/llm-lab
  Destination: ~/GoogleDrive/work_MBP_M1/llm-lab
  
  In Colab, this appears as: /content/drive/MyDrive/work_MBP_M1/llm-lab

SYNC BEHAVIOR:
  - Syncs source code (.py, .md, .ipynb, etc.)
  - Excludes large files (models, data, cache)
  - Excludes .git directory (too large for Drive)
  - Uses --delete to remove files deleted locally
  - Preserves llm_cache directory if it exists

IMPORTANT NOTES:
  - Google Drive path: /Users/pleiadian53/GoogleDrive (symlink to "Google Drive")
  - Colab runtime is ephemeral; only Drive files persist
  - Large model files should be in llm_cache, not in project
  - Git operations should be done locally, not in Colab

REQUIREMENTS:
  - rsync (built-in on macOS/Linux)
  - Google Drive mounted and syncing
"""

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Sync llm-lab project to Google Drive for Colab",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --dry-run          # Preview what will be synced
  %(prog)s                    # Perform actual sync
  %(prog)s --verbose          # Show detailed output
  %(prog)s --no-delete        # Sync without deleting removed files
        """
    )
    
    parser.add_argument(
        '-n', '--dry-run',
        action='store_true',
        help='Preview changes without actually syncing'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Show verbose output'
    )
    
    parser.add_argument(
        '--no-delete',
        action='store_true',
        help='Do not delete files in destination that are not in source'
    )
    
    parser.add_argument(
        '--src',
        type=str,
        default=str(Path.home() / 'work' / 'llm-lab'),
        help='Source directory (default: ~/work/llm-lab)'
    )
    
    parser.add_argument(
        '--dest',
        type=str,
        default=str(Path.home() / 'GoogleDrive' / 'work_MBP_M1' / 'llm-lab'),
        help='Destination directory (default: ~/GoogleDrive/work_MBP_M1/llm-lab)'
    )
    
    args = parser.parse_args()
    
    # Validate paths
    src = Path(args.src)
    dest = Path(args.dest)
    
    if not src.exists():
        print(f"ERROR: Source directory does not exist: {src}", file=sys.stderr)
        return 1
    
    # Check if Google Drive is mounted
    google_drive_root = Path.home() / 'GoogleDrive'
    if not google_drive_root.exists():
        print(f"ERROR: Google Drive not found at: {google_drive_root}", file=sys.stderr)
        print("Is Google Drive mounted?", file=sys.stderr)
        return 1
    
    # Create destination if it doesn't exist
    if not dest.exists():
        print(f"Creating destination directory: {dest}")
        if not args.dry_run:
            dest.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    log_dir = Path.home() / 'Library' / 'Logs' / 'colab_sync'
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"sync_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
    
    # Build rsync command
    rsync_opts = [
        'rsync',
        '-a',  # archive mode
        '-h',  # human-readable
    ]
    
    if args.verbose:
        rsync_opts.append('-v')
    
    if not args.no_delete:
        rsync_opts.append('--delete')
        # Protect llm_cache if it exists in destination
        rsync_opts.append('--filter=protect llm_cache/')
    
    if args.dry_run:
        rsync_opts.append('--dry-run')
    
    # IMPORTANT: Order matters in rsync!
    # Excludes must come BEFORE includes
    
    # Exclude junk/build/cache directories FIRST
    exclude_dirs = [
        '--exclude', '.DS_Store',
        '--exclude', '._*',
        '--exclude', '.Trash/',
        '--exclude', 'node_modules/',
        '--exclude', '.git/',              # Git repo is too large for Drive
        '--exclude', '.github/',
        '--exclude', '.idea/',
        '--exclude', '.vscode/',
        '--exclude', '__pycache__/',
        '--exclude', '.venv/',
        '--exclude', '.cache/',
        '--exclude', '.pytest_cache/',
        '--exclude', 'tmp/',
        '--exclude', 'checkpoints/',
        '--exclude', 'data/',              # Large datasets
        '--exclude', 'predictions/',       # Model predictions
        '--exclude', 'logs/',              # Training logs
        '--exclude', 'models/',            # Saved model weights
        '--exclude', 'weights/',           # Model weights
        '--exclude', '.ipynb_checkpoints/',
        '--exclude', '*.pyc',
        '--exclude', '*.pyo',
        '--exclude', '*.pyd',
        '--exclude', '.Python',
        '--exclude', 'pip-log.txt',
        '--exclude', 'pip-delete-this-directory.txt',
    ]
    
    # Include patterns
    include_patterns = [
        '--include', '*/',  # all directories (after exclusions above)
        '--include', '*.py',
        '--include', '*.R',
        '--include', '*.ipynb',
        '--include', '*.md',
        '--include', '*.txt',
        '--include', '*.json',
        '--include', '*.yaml',
        '--include', '*.yml',
        '--include', '*.toml',
        '--include', '*.cfg',
        '--include', '*.ini',
        '--include', '*.sh',
        '--include', '*.bash',
        '--include', '*.zsh',
        '--include', '*.cpp',
        '--include', '*.h',
        '--include', '*.c',
        '--include', '*.hpp',
        '--include', '*.png',
        '--include', '*.jpg',
        '--include', '*.jpeg',
        '--include', '*.svg',
        '--include', '*.pdf',
        '--include', 'requirements.txt',
        '--include', 'environment.yml',
        '--include', 'setup.py',
        '--include', 'setup.cfg',
        '--include', 'pyproject.toml',
        '--include', 'Makefile',
        '--include', 'Dockerfile',
        '--include', '.gitignore',
        '--include', 'LICENSE',
        '--include', 'README*',
    ]
    
    # Exclude large file types
    exclude_files = [
        '--exclude', '*.fa',
        '--exclude', '*.fasta',
        '--exclude', '*.gtf',
        '--exclude', '*.bam',
        '--exclude', '*.bai',
        '--exclude', '*.cram',
        '--exclude', '*.vcf',
        '--exclude', '*.gz',
        '--exclude', '*.tar',
        '--exclude', '*.zip',
        '--exclude', '*.xz',
        '--exclude', '*.7z',
        '--exclude', '*.bt2',
        '--exclude', '*.mmi',
        '--exclude', '*.db',
        '--exclude', '*.sqlite',
        '--exclude', '*.pkl',
        '--exclude', '*.pickle',
        '--exclude', '*.pt',
        '--exclude', '*.pth',
        '--exclude', '*.ckpt',
        '--exclude', '*.safetensors',
        '--exclude', '*.bin',
        '--exclude', '*.h5',
        '--exclude', '*.hdf5',
    ]
    
    # Exclude everything else
    exclude_rest = ['--exclude', '*']
    
    # Combine all options in the correct order
    rsync_opts.extend(exclude_dirs)      # 1. Exclude unwanted dirs first
    rsync_opts.extend(include_patterns)  # 2. Include what we want
    rsync_opts.extend(exclude_files)     # 3. Exclude unwanted file types
    rsync_opts.extend(exclude_rest)      # 4. Exclude everything else
    
    # Add source and destination (with trailing slash for source)
    rsync_opts.append(f"{src}/")
    rsync_opts.append(str(dest))
    
    # Print summary
    print("=" * 70)
    print("🔄 Sync llm-lab to Google Drive for Colab")
    print("=" * 70)
    if args.dry_run:
        print("⚠️  DRY RUN MODE - No files will be changed")
        print("=" * 70)
    print(f"Source:       {src}")
    print(f"Destination:  {dest}")
    print(f"Delete mode:  {'ON' if not args.no_delete else 'OFF'}")
    print(f"Log file:     {log_file}")
    print("=" * 70)
    print()
    print("📦 Syncing project files...")
    print()
    
    # Run rsync and capture output
    try:
        with open(log_file, 'w', encoding='utf-8', errors='replace') as log:
            # Write header to log
            log.write(f"Sync started at {datetime.now()}\n")
            log.write(f"Source: {src}\n")
            log.write(f"Destination: {dest}\n")
            log.write(f"Command: {' '.join(rsync_opts)}\n")
            log.write("=" * 70 + "\n\n")
            
            # Use tee-like behavior: write to both stdout and log file
            process = subprocess.Popen(
                rsync_opts,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                encoding='utf-8',
                errors='replace'
            )
            
            for line in process.stdout:
                print(line, end='')
                log.write(line)
            
            exit_code = process.wait()
        
        print()
        print("=" * 70)
        if exit_code == 0:
            print(f"✅ Sync completed successfully at {datetime.now()}")
            if args.dry_run:
                print("   (This was a dry run - no changes were made)")
            else:
                print()
                print("📍 Your project is now available in Colab at:")
                print("   /content/drive/MyDrive/work_MBP_M1/llm-lab")
                print()
                print("💡 Next steps in Colab:")
                print("   1. Mount Google Drive")
                print("   2. cd /content/drive/MyDrive/work_MBP_M1/llm-lab")
                print("   3. pip install -e posttrain_llm/llm_eval")
        else:
            print(f"❌ Sync failed with exit code {exit_code}")
            print(f"   Check log file: {log_file}")
        print("=" * 70)
        
        return exit_code
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Sync interrupted by user")
        return 130
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
