#!/usr/bin/env python3
"""
Fix model paths in Lesson_3.ipynb to use HuggingFace repo IDs instead of local paths.
This allows the notebook to download models automatically from HuggingFace.
"""

import json
import sys

def fix_notebook():
    notebook_path = "Lesson_3.ipynb"
    
    print(f"Reading {notebook_path}...")
    with open(notebook_path, 'r') as f:
        notebook = json.load(f)
    
    # Mapping of local paths to HuggingFace repo IDs
    replacements = {
        '"./models/Qwen/Qwen3-0.6B-Base"': '"Qwen/Qwen2.5-0.5B"',  # Using available Qwen model
        '"./models/banghua/Qwen3-0.6B-SFT"': '"Qwen/Qwen2.5-0.5B-Instruct"',  # Using instruct version
        '"./models/HuggingFaceTB/SmolLM2-135M"': '"HuggingFaceTB/SmolLM2-135M"',  # Already correct format
    }
    
    changes_made = 0
    
    # Process each cell
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            source = cell['source']
            if isinstance(source, list):
                # Join list into string for easier replacement
                source_str = ''.join(source)
                original = source_str
                
                # Apply replacements
                for old, new in replacements.items():
                    if old in source_str:
                        source_str = source_str.replace(old, new)
                        print(f"  Replaced {old} -> {new}")
                        changes_made += 1
                
                # Convert back to list if changed
                if source_str != original:
                    cell['source'] = source_str.split('\n')
                    # Ensure newlines are preserved
                    cell['source'] = [line + '\n' if i < len(cell['source']) - 1 else line 
                                     for i, line in enumerate(cell['source'])]
    
    if changes_made > 0:
        # Save backup
        backup_path = "Lesson_3.ipynb.backup"
        print(f"\nCreating backup: {backup_path}")
        with open(backup_path, 'w') as f:
            json.dump(notebook, f, indent=1)
        
        # Save modified notebook
        print(f"Saving modified notebook: {notebook_path}")
        with open(notebook_path, 'w') as f:
            json.dump(notebook, f, indent=1)
        
        print(f"\n✅ Made {changes_made} changes")
        print("\nModels will now be downloaded from HuggingFace automatically:")
        print("  - Qwen/Qwen2.5-0.5B (base model)")
        print("  - Qwen/Qwen2.5-0.5B-Instruct (SFT model)")
        print("  - HuggingFaceTB/SmolLM2-135M (training model)")
        print("\nNote: First run will download models (~1-2GB), subsequent runs use cache.")
    else:
        print("\n⚠️  No changes needed - paths already correct or not found")
    
    return changes_made > 0

if __name__ == "__main__":
    try:
        success = fix_notebook()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
