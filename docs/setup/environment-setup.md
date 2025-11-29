# Unified Environment Setup for Lab Projects

This guide explains how to set up a single conda environment (`llm-lab`) that works across all your lab projects:
- **biographlab** - Graph neural networks and biological graphs
- **llm-lab** - LLM research and experimentation
- **probability-lab** - Probability theory and statistics
- **pytorch-lab** - PyTorch learning and experimentation

## Why a Unified Environment?

**Benefits**:
- ✅ Single environment activation for all projects
- ✅ Shared dependencies (PyTorch, NumPy, etc.)
- ✅ Consistent Python version across projects
- ✅ Easier dependency management
- ✅ All projects installed in editable mode

**Trade-offs**:
- Larger environment size (~5-10 GB)
- Some projects may not need all dependencies
- Updates affect all projects

## Quick Start

### 1. Create the Environment

```bash
# From llm-lab directory
cd /Users/pleiadian53/work/llm-lab

# Create environment from unified config
conda env create -f environment_unified.yml
```

This will:
- Create a conda environment named `llm-lab`
- Install Python 3.11
- Install all dependencies for all four projects
- Install all projects in editable mode

**Time**: ~10-15 minutes (depending on internet speed)

### 2. Activate the Environment

```bash
conda activate llm-lab
```

### 3. Verify Installation

```bash
# Check Python version
python --version  # Should be 3.11.x

# Check PyTorch
python -c "import torch; print(f'PyTorch {torch.__version__}')"

# Check if projects are installed
python -c "import llm_lab; print('llm-lab ✓')" 2>/dev/null || echo "llm-lab not found"
python -c "import pytorch_lab; print('pytorch-lab ✓')" 2>/dev/null || echo "pytorch-lab not found"
python -c "import probability_lab; print('probability-lab ✓')" 2>/dev/null || echo "probability-lab not found"
python -c "import biographlab; print('biographlab ✓')" 2>/dev/null || echo "biographlab not found"

# Check torch-geometric (for biographlab)
python -c "import torch_geometric; print(f'PyG {torch_geometric.__version__}')"
```

### 4. Usage Across Projects

Once activated, you can work in any project:

```bash
# Activate once
conda activate llm-lab

# Work in llm-lab
cd /Users/pleiadian53/work/llm-lab
python scripts/md_to_pdf.py docs/llm/llm_tech_history.md

# Switch to pytorch-lab
cd /Users/pleiadian53/work/pytorch-lab
python scripts/launch.py

# Switch to probability-lab
cd /Users/pleiadian53/work/probability-lab
# Your probability experiments...

# Switch to biographlab
cd /Users/pleiadian53/work/biographlab
# Your GNN experiments...
```

## What's Included

### Core Scientific Stack
- **NumPy** ≥1.24 - Array computing
- **Pandas** ≥2.0 - Data manipulation
- **SciPy** ≥1.11 - Scientific computing
- **Matplotlib** ≥3.8 - Plotting
- **Seaborn** ≥0.13 - Statistical visualization
- **Plotly** ≥5.20 - Interactive plots
- **scikit-learn** ≥1.3 - Machine learning

### PyTorch Ecosystem
- **PyTorch** ≥2.1 - Deep learning framework
- **torchvision** ≥0.16 - Computer vision
- **torch-geometric** ≥2.4 - Graph neural networks

### LLM & Transformers
- **transformers** ≥4.35 - Hugging Face transformers
- **datasets** ≥2.15 - Dataset library
- **accelerate** ≥0.20 - Distributed training
- **wandb** ≥0.16 - Experiment tracking

### Configuration & CLI
- **hydra-core** ≥1.3 - Configuration management
- **omegaconf** ≥2.3 - Configuration files
- **pydantic** ≥2.6 - Data validation
- **typer** ≥0.12 - CLI building
- **rich** ≥13.7 - Terminal formatting

### Probability & Statistics
- **polars** ≥1.5 - Fast dataframes
- **lifelines** ≥0.28 - Survival analysis

### Development Tools
- **Jupyter** - Notebooks
- **black** - Code formatting
- **isort** - Import sorting
- **flake8** - Linting
- **mypy** - Type checking
- **pytest** - Testing
- **ruff** - Fast linter
- **pre-commit** - Git hooks

## Updating the Environment

### Add New Dependencies

Edit `environment_unified.yml` and update:

```bash
conda env update -f environment_unified.yml --prune
```

### Update Existing Packages

```bash
# Update all packages
conda update --all

# Or update specific package
conda update numpy
pip install --upgrade transformers
```

## Troubleshooting

### Environment Creation Fails

**Problem**: Dependency conflicts during creation.

**Solution**:
```bash
# Remove existing environment
conda env remove -n llm-lab

# Create with verbose output
conda env create -f environment_unified.yml -v

# If still fails, try mamba (faster solver)
conda install mamba -n base -c conda-forge
mamba env create -f environment_unified.yml
```

### Import Errors After Installation

**Problem**: Can't import project modules.

**Solution**:
```bash
# Ensure projects are installed in editable mode
conda activate llm-lab
cd /Users/pleiadian53/work/llm-lab
pip install -e .

cd /Users/pleiadian53/work/pytorch-lab
pip install -e .

cd /Users/pleiadian53/work/probability-lab
pip install -e .

cd /Users/pleiadian53/work/biographlab
pip install -e .
```

### torch-geometric Installation Issues

**Problem**: torch-geometric or its dependencies fail to install.

**Solution**:
```bash
# Install torch-geometric with specific torch version
conda activate llm-lab
python -c "import torch; print(torch.__version__)"  # Check PyTorch version

# Install matching torch-geometric
pip install torch-geometric
pip install pyg-lib torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-$(python -c "import torch; print(torch.__version__.split('+')[0])")+cpu.html
```

### CUDA/GPU Issues

**Problem**: Want to use GPU but environment has CPU-only PyTorch.

**Solution**:
```bash
# Check current PyTorch
python -c "import torch; print(torch.cuda.is_available())"

# If False, reinstall with CUDA support
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia

# Verify
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Alternative: Project-Specific Environments

If you prefer separate environments for each project:

```bash
# llm-lab environment
cd /Users/pleiadian53/work/llm-lab
conda env create -f environment.yml

# pytorch-lab environment
cd /Users/pleiadian53/work/pytorch-lab
conda env create -f environment.yml  # If exists

# probability-lab environment
cd /Users/pleiadian53/work/probability-lab
conda env create -f environment.yml  # If exists

# biographlab environment
cd /Users/pleiadian53/work/biographlab
conda env create -f environment.yml  # If exists
```

Then activate the appropriate environment for each project.

## Best Practices

1. **Always activate before working**:
   ```bash
   conda activate llm-lab
   ```

2. **Keep environment file updated**:
   - Add new dependencies to `environment_unified.yml`
   - Commit changes to version control

3. **Periodic cleanup**:
   ```bash
   # Remove unused packages
   conda clean --all
   
   # Check environment size
   du -sh ~/anaconda3/envs/llm-lab  # or ~/miniconda3/envs/llm-lab
   ```

4. **Export for sharing**:
   ```bash
   # Export exact environment
   conda env export > environment_frozen.yml
   
   # Export cross-platform (recommended)
   conda env export --from-history > environment_portable.yml
   ```

5. **Deactivate when done**:
   ```bash
   conda deactivate
   ```

## IDE Integration

### VS Code

Add to `.vscode/settings.json` in each project:

```json
{
  "python.defaultInterpreterPath": "~/anaconda3/envs/llm-lab/bin/python"
}
```

Or use Command Palette: `Python: Select Interpreter` → Choose `llm-lab`

### PyCharm

1. Settings → Project → Python Interpreter
2. Add Interpreter → Conda Environment
3. Select existing environment: `llm-lab`

### Jupyter

The environment is automatically available in Jupyter:

```bash
conda activate llm-lab
jupyter notebook
# Select kernel: Python 3 (llm-lab)
```

## Environment Size

Expected disk usage:
- **Base environment**: ~3-5 GB
- **With all dependencies**: ~5-10 GB
- **With CUDA support**: ~8-12 GB

## See Also

- [LATEX_SETUP.md](LATEX_SETUP.md) - LaTeX installation for PDF generation
- [DOCUMENT_WORKFLOW.md](DOCUMENT_WORKFLOW.md) - Document conversion workflow
- Individual project READMEs for project-specific setup

## Quick Reference

```bash
# Create environment
conda env create -f environment_unified.yml

# Activate
conda activate llm-lab

# Update
conda env update -f environment_unified.yml --prune

# Remove
conda env remove -n llm-lab

# List environments
conda env list

# Export
conda env export > environment_frozen.yml
```
