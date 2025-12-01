#!/usr/bin/env python3
"""
Sync changes FROM Colab (Google Drive) back to local machine.

This script pulls changes you made in Colab back to your local ~/work/llm-lab.
Use this BEFORE running sync_to_colab.py to avoid losing Colab work.

USAGE:
  ./sync_from_colab.py              # Pull changes from Drive
  ./sync_from_colab.py -n           # Dry run (preview only)
  ./sync_from_colab.py -v           # Verbose output

WORKFLOW:
  1. Work in Colab (files saved to Drive)
  2. Run sync_from_colab.py (pull changes to local)
  3. Commit changes locally with git
  4. Make more local changes
  5. Run sync_to_colab.py (push to Drive)
  6. Continue in Colab

IMPORTANT:
  - This syncs FROM Drive TO local (opposite of sync_to_colab.py)
  - Always pull before push to avoid losing work
  - Use git to track what changed
  - Excludes llm_cache (models stay on Drive)

PATHS:
  Source:      ~/GoogleDrive/work_MBP_M1/llm-lab (Drive)
  Destination: ~/work/llm-lab (Local)
"""

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Sync changes from Colab (Google Drive) to local machine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --dry-run          # Preview what will be synced
  %(prog)s                    # Perform actual sync
  %(prog)s --verbose          # Show detailed output
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
        '--src',
        type=str,
        default=str(Path.home() / 'GoogleDrive' / 'work_MBP_M1' / 'llm-lab'),
        help='Source directory (default: ~/GoogleDrive/work_MBP_M1/llm-lab)'
    )
    
    parser.add_argument(
        '--dest',
        type=str,
        default=str(Path.home() / 'work' / 'llm-lab'),
        help='Destination directory (default: ~/work/llm-lab)'
    )
    
    args = parser.parse_args()
    
    # Validate paths
    src = Path(args.src)
    dest = Path(args.dest)
    
    if not src.exists():
        print(f"ERROR: Source directory does not exist: {src}", file=sys.stderr)
        print("Is Google Drive mounted?", file=sys.stderr)
        return 1
    
    if not dest.exists():
        print(f"ERROR: Destination directory does not exist: {dest}", file=sys.stderr)
        return 1
    
    # Setup logging
    log_dir = Path.home() / 'Library' / 'Logs' / 'colab_sync'
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"sync_from_colab_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
    
    # Build rsync command
    rsync_opts = [
        'rsync',
        '-a',  # archive mode
        '-h',  # human-readable
        '--update',  # skip files that are newer on destination
    ]
    
    if args.verbose:
        rsync_opts.append('-v')
    
    if args.dry_run:
        rsync_opts.append('--dry-run')
    
    # Exclude patterns (same as sync_to_colab.py)
    exclude_patterns = [
        '--exclude', '.DS_Store',
        '--exclude', '._*',
        '--exclude', '.git/',
        '--exclude', '__pycache__/',
        '--exclude', '.venv/',
        '--exclude', '.cache/',
        '--exclude', '.pytest_cache/',
        '--exclude', '.ipynb_checkpoints/',
        '--exclude', 'llm_cache/',  # Don't pull model cache
        '--exclude', 'checkpoints/',
        '--exclude', 'logs/',
        '--exclude', '*.pyc',
        '--exclude', '*.pyo',
    ]
    
    rsync_opts.extend(exclude_patterns)
    
    # Add source and destination (with trailing slash for source)
    rsync_opts.append(f"{src}/")
    rsync_opts.append(str(dest))
    
    # Print summary
    print("=" * 70)
    print("⬇️  Sync FROM Colab (Google Drive) TO Local")
    print("=" * 70)
    if args.dry_run:
        print("⚠️  DRY RUN MODE - No files will be changed")
        print("=" * 70)
    print(f"Source (Drive): {src}")
    print(f"Destination:    {dest}")
    print(f"Log file:       {log_file}")
    print("=" * 70)
    print()
    print("📥 Pulling changes from Colab...")
    print()
    
    # Run rsync
    try:
        with open(log_file, 'w', encoding='utf-8', errors='replace') as log:
            # Write header to log
            log.write(f"Sync from Colab started at {datetime.now()}\n")
            log.write(f"Source: {src}\n")
            log.write(f"Destination: {dest}\n")
            log.write(f"Command: {' '.join(rsync_opts)}\n")
            log.write("=" * 70 + "\n\n")
            
            # Run rsync
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
                print("💡 Next steps:")
                print("   1. Review changes: git status")
                print("   2. Commit if needed: git add . && git commit -m 'Work from Colab'")
                print("   3. Continue local work or push to Colab")
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
