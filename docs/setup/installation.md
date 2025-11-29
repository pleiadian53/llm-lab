# Installation Guide

Follow the documentation structure and style from probability-lab: start with prerequisites, provide a reliable path for each platform, and end with verification and troubleshooting.

## Prerequisites
- **Python 3.11** capability (Xcode Command Line Tools on macOS, build-essential on Linux, Visual Studio Build Tools on Windows).
- **mamba/conda** installed. We recommend `mambaforge` for consistent channels.
- **Git** ≥ 2.40.
- Optional but recommended: **GitHub CLI (`gh`)** and **pre-commit** installed globally.

## Step-by-Step Setup
1. **Install mambaforge**  
   - macOS / Linux:
     ```bash
     curl -L https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-$(uname)-$(uname -m).sh -o mambaforge.sh
     bash mambaforge.sh
     ```
   - Windows (PowerShell):
     ```powershell
     Invoke-WebRequest -Uri https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Windows-x86_64.exe -OutFile mambaforge.exe
     Start-Process mambaforge.exe
     ```
2. **Clone the repository**
   ```bash
   git clone git@github.com:USERNAME/llm-lab.git
   cd llm-lab
   ```
3. **Create the environment (one-step installation)**
   ```bash
   mamba env create -f environment.yml
   mamba activate llm-lab
   ```
   This installs everything you need:
   - Compiled dependencies (PyTorch, NumPy, pandas) from mamba
   - Runtime dependencies (transformers, datasets, hydra-core, wandb, pydantic, typer)
   - Dev tools (black, isort, mypy, pytest, ruff, pre-commit, types-PyYAML)
   - The llm-lab package in editable mode

4. **Install pre-commit hooks**
   ```bash
   pre-commit install
   ```
5. **Run the smoke tests**
   ```bash
   pytest
   ```

## Platform Notes
- **macOS (Apple Silicon)**: PyTorch wheels ship with Metal backend; `torch.backends.mps.is_available()` should be `True`. Ensure Homebrew’s `openssl` is installed if mypy fails to compile typed dependencies.
- **Linux (CI or WSL)**: Install `build-essential`, `libffi-dev`, and `python3-dev` before running `pip install -e .[dev]`.
- **Windows**: Use the “Developer PowerShell for VS” terminal so that MSVC build tools are on `PATH`. Enable long paths (`git config --system core.longpaths true`).

## Verification Checklist
- `python -V` should show `Python 3.11.x`
- `python -c "import torch, transformers, datasets, hydra, wandb, pydantic, typer; print('✓ All imports successful')"` confirms all dependencies are installed.
- `pip check` should report "No broken requirements found."
- `pip list | grep llm-lab` shows `llm-lab 0.1.0` with an editable install path.
- `python -m llm_lab.cli --help` prints the CLI help text.
- `llm_lab/pretrain_llm/trainer.py` logs a training completion message when running the example script:
  ```bash
  python examples/run_pretraining.py
  ```
- `pytest` exits cleanly with status code 0.

## Troubleshooting
- **`UnsatisfiableError` during `mamba env create`**: run `mamba clean --all` and retry; ensure `conda-forge` has highest channel priority (`conda config --set channel_priority strict`).
- **"No module named 'transformers'" or similar import errors**: Ensure you created the environment from the project root directory where `environment.yml` is located. The editable install (`-e .`) requires this.
- **Sympy version conflict warning**: This is resolved automatically—`environment.yml` pins `sympy=1.13.1` to match PyTorch 2.5.1 requirements.
- **PyTorch cannot find GPU**: on M1/M2 hardware check `python -c "import torch; print(torch.backends.mps.is_available())"`; on Linux ensure CUDA drivers match the requested torch build, or fall back to CPU by setting `export PYTORCH_ENABLE_MPS_FALLBACK=1`.
- **pre-commit hooks are slow**: run `pre-commit gc` and pin to the versions shipped in `.pre-commit-config.yaml`.
- **Pydantic warnings about field attributes**: These are harmless warnings from pydantic v2 internals and do not affect functionality.
