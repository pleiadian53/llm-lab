# Environment Integration Guide

This guide explains how the pretrain_llm directory integrates with the main llm-lab environment setup.

## Overview

The llm-lab project uses a **dual-source architecture** for dependency management:

1. **`pyproject.toml`** - Package definition (runtime + dev dependencies)
2. **`environment.yml`** - Mamba environment specification (includes all dependencies from pyproject.toml)

This allows:
- ✅ Single-command installation via `mamba env create`
- ✅ Package installable via `pip install -e .`
- ✅ Both sources stay in sync

## Streamlined Installation

### One-Command Setup

From the project root:

```bash
mamba env create -f environment.yml
mamba activate llm-lab
```

This installs:
- **Compiled dependencies** (PyTorch, NumPy, pandas) from conda-forge
- **Runtime dependencies** (transformers, datasets, hydra-core, wandb, pydantic, typer)
- **Dev tools** (black, isort, mypy, pytest, ruff, pre-commit, types-PyYAML)
- **llm-lab package** in editable mode (`-e .`)

### What Changed

**Before (4+ steps):**
```bash
mamba env create -f environment.yml
mamba activate llm-lab
pip install -r requirements.txt
pip install -e .
pip install -e .[dev]
```

**After (2 steps):**
```bash
mamba env create -f environment.yml
mamba activate llm-lab
```

## Key Improvements

### 1. Sympy Compatibility

**Issue:** PyTorch 2.5.1 requires `sympy==1.13.1`, but newer versions were being installed by default.

**Solution:** Pinned in `environment.yml`:
```yaml
dependencies:
  - sympy=1.13.1  # Required for PyTorch 2.5.1 compatibility
```

**Verification:**
```bash
python -c "import sympy; print(sympy.__version__)"  # Should show 1.13.1
pip check  # Should report no conflicts
```

### 2. All Dependencies in One Place

**Before:** Dependencies split across multiple files
- `environment.yml` - Core dependencies only
- `requirements.txt` - Runtime dependencies
- `pyproject.toml` - Package definition

**After:** All dependencies in `environment.yml`
- Includes everything from `pyproject.toml`
- Single source of truth for installation
- `pyproject.toml` still defines the package for distribution

### 3. Automatic Editable Install

The `environment.yml` now includes:
```yaml
- pip:
    - -e .  # Installs llm-lab in editable mode
```

No separate `pip install -e .` needed!

## Dependency Versions

Current installed versions (as of environment setup):

### Core Dependencies
- Python: 3.11.x
- PyTorch: 2.5.1
- NumPy: 1.24+
- pandas: 2.0+
- sympy: 1.13.1 (pinned)

### Runtime Dependencies
- transformers: 4.35+
- datasets: 2.15+
- hydra-core: 1.3+
- wandb: 0.16+
- pydantic: 2.6+
- typer: 0.12+

### Dev Dependencies
- black: 23.11+
- isort: 5.12+
- mypy: 1.8+
- pytest: 7.4+
- pytest-cov: 4.1+
- ruff: 0.2+
- pre-commit: 3.5+

See [dependencies.md](../../docs/dependencies.md) for detailed version information.

## Verification Steps

After installation, verify everything works:

```bash
# 1. Check Python version
python -V  # Should show Python 3.11.x

# 2. Verify core imports
python -c "import torch, transformers, datasets, hydra, wandb, pydantic, typer; print('✓ All imports successful')"

# 3. Check for conflicts
pip check  # Should report "No broken requirements found"

# 4. Verify editable install
pip list | grep llm-lab  # Should show llm-lab 0.1.0 (editable)

# 5. Test CLI
python -m llm_lab.cli --help

# 6. Run tests
pytest
```

## Working with pretrain_llm

### Using the Model Loader

The model loader module works seamlessly with the installed environment:

```python
from llm_lab.pretrain_llm import load_model_with_fallback
from pathlib import Path

# This works because llm-lab is installed in editable mode
model, tokenizer = load_model_with_fallback(
    "upstage/TinySolar-248m-4k",
    local_dir=Path("./models/TinySolar-248m-4k")
)
```

### In Notebooks

Notebooks in `pretrain_llm/` can directly import from the package:

```python
# No sys.path manipulation needed!
from llm_lab.pretrain_llm import load_model_with_fallback
import torch

model, tokenizer = load_model_with_fallback(
    "upstage/TinySolar-248m-4k",
    local_dir=Path("./models/TinySolar-248m-4k")
)
```

### Running Scripts

Scripts work directly:

```bash
# From pretrain_llm directory
python download_models.py --model MODEL_NAME --output ./models/MODEL_NAME

# From project root
python examples/run_pretraining.py
```

## Troubleshooting

### Import Errors

**Symptom:** `ModuleNotFoundError: No module named 'llm_lab'`

**Solution:** Ensure you created the environment from the project root where `environment.yml` is located:
```bash
cd /path/to/llm-lab  # Must be in project root
mamba env create -f environment.yml
```

### Sympy Version Conflicts

**Symptom:** Warnings about sympy version mismatch

**Solution:** Already resolved! The `environment.yml` pins `sympy=1.13.1`. If you see this:
```bash
mamba env update -f environment.yml --prune
```

### Pydantic Warnings

**Symptom:** Warnings about field attributes from pydantic

**Solution:** These are harmless warnings from pydantic v2 internals. They don't affect functionality and can be ignored.

### IDE Not Finding Modules

**Symptom:** IDE shows import errors but code runs fine

**Solution:** 
1. Reload the Python interpreter in your IDE
2. Ensure IDE is using the `llm-lab` conda environment
3. Add `<repo>/src` to IDE's Python path if needed

## Updating Dependencies

### Adding New Dependencies

1. **Edit `pyproject.toml`:**
   ```toml
   dependencies = [
       "existing-package>=1.0",
       "new-package>=2.0",  # Add here
   ]
   ```

2. **Update `environment.yml`:**
   ```yaml
   - pip:
       - existing-package>=1.0
       - new-package>=2.0  # Add here
   ```

3. **Update environment:**
   ```bash
   mamba env update -f environment.yml --prune
   ```

### Keeping Files in Sync

The project maintains two dependency sources:
- **`pyproject.toml`** - For package distribution (pip install)
- **`environment.yml`** - For development setup (mamba)

**Rule:** Always update both files when adding/removing dependencies.

**Why?** 
- `pyproject.toml` defines what users get when they `pip install llm-lab`
- `environment.yml` defines what developers get for local development

## CI/CD Integration

The GitHub Actions workflow uses the same streamlined setup:

```yaml
- name: Set up environment
  run: |
    mamba env create -f environment.yml
    mamba activate llm-lab
    
- name: Run tests
  run: pytest
```

See [.github/workflows/ci.yml](../../.github/workflows/ci.yml) for the full workflow.

## Best Practices

1. **Always activate the environment:**
   ```bash
   mamba activate llm-lab
   ```

2. **Verify after installation:**
   ```bash
   pip check
   pytest
   ```

3. **Keep dependencies in sync:**
   - Update `pyproject.toml` first
   - Then update `environment.yml`
   - Test with `mamba env update`

4. **Use version pins for stability:**
   - Pin critical dependencies (like sympy)
   - Use `>=` for flexibility where possible

5. **Document version requirements:**
   - Add comments in `environment.yml` for pinned versions
   - Update `docs/dependencies.md` with rationale

## Additional Resources

- [Installation Guide](../../docs/installation.md) - Detailed setup instructions
- [Environment Setup Guide](../../docs/environment-setup-guide.md) - Philosophy and troubleshooting
- [Dependencies](../../docs/dependencies.md) - Complete dependency list with versions
- [CHANGELOG-DOCS](../../docs/CHANGELOG-DOCS.md) - Documentation change history

## Summary

The streamlined environment setup:
- ✅ Reduces installation from 4+ steps to 2 steps
- ✅ Resolves sympy compatibility issues automatically
- ✅ Installs all dependencies (runtime + dev) in one command
- ✅ Sets up editable install automatically
- ✅ Works seamlessly with pretrain_llm modules
- ✅ Verified with `pip check` for no conflicts
- ✅ Documented for maintainability

This integration ensures that pretrain_llm experiments run smoothly with minimal setup friction.
