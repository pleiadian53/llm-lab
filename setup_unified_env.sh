#!/bin/bash
# Unified Environment Setup Script for Lab Projects
# Creates a single conda environment (llm-lab) for all lab projects

set -e  # Exit on error

echo "=========================================="
echo "Unified Lab Environment Setup"
echo "=========================================="
echo ""
echo "This will create a conda environment 'llm-lab' that includes:"
echo "  - biographlab"
echo "  - llm-lab"
echo "  - probability-lab"
echo "  - pytorch-lab"
echo ""

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Error: conda not found"
    echo "Please install Anaconda or Miniconda first:"
    echo "  https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

echo "✓ Found conda: $(conda --version)"
echo ""

# Check if environment already exists
if conda env list | grep -q "^llm-lab "; then
    echo "⚠️  Environment 'llm-lab' already exists"
    read -p "Do you want to remove and recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing environment..."
        conda env remove -n llm-lab -y
    else
        echo "Aborting. Use 'conda env update' to update existing environment."
        exit 0
    fi
fi

# Create environment
echo ""
echo "📦 Creating environment from environment_unified.yml..."
echo "This may take 10-15 minutes..."
echo ""

conda env create -f environment_unified.yml

echo ""
echo "=========================================="
echo "✅ Environment Created Successfully!"
echo "=========================================="
echo ""
echo "To activate the environment:"
echo "  conda activate llm-lab"
echo ""
echo "To verify installation:"
echo "  conda activate llm-lab"
echo "  python -c \"import torch; print(f'PyTorch {torch.__version__}')\""
echo "  python -c \"import torch_geometric; print(f'PyG {torch_geometric.__version__}')\""
echo ""
echo "See docs/ENVIRONMENT_SETUP.md for more information."
echo ""
