# llm-lab

Reusable building blocks for running LLM pre-training and post-training experiments with modern tooling and documentation.

## Highlights
- **Modular package layout** with `llm_lab.pretrain_llm` and `llm_lab.posttrain_llm` subpackages.
- **Hydra-ready configs** and Typer CLI entry points for reproducible experiments.
- **Apple Silicon & Linux friendly** environment managed by `mamba` and editable installs.
- **Developer experience** powered by `ruff`, `black`, `isort`, `mypy`, `pytest`, and `pre-commit`.
- **Comprehensive docs** inspired by the *probability-lab* project structure.

## Project Layout
```
llm-lab/
├── src/llm_lab/              # Python package (installed in editable mode)
│   ├── pretrain_llm/         # Pre-training utilities
│   └── posttrain_llm/        # Post-training + alignment utilities
├── examples/                 # Example scripts and workflows
├── tests/                    # Pytest-based test suite
├── docs/                     # Documentation portal
├── pyproject.toml            # Package metadata and tooling config
├── environment.yml           # Mamba environment specification
└── requirements.txt          # Pip installation manifest
```

## Getting Started
1. Install the environment with mamba:
   ```bash
   mamba env create -f environment.yml
   mamba activate llm-lab
   ```
2. Install the package in editable mode with developer extras:
   ```bash
   pip install -e .[dev]
   ```
3. Run the test suite:
   ```bash
   pytest
   ```

## Documentation

Complete documentation is available in the [`docs/`](docs/) directory:

- **[Documentation Portal](docs/README.md)** - Main entry point for all documentation
- **[Quick Start](docs/quick-start.md)** - Get up and running in 5 minutes
- **[Setup Guides](docs/setup/)** - Installation, environment, LaTeX, dependencies
- **[Workflows](docs/workflows/)** - Document creation, markdown→PDF conversion
- **[LLM Research](docs/llm/)** - Technical notes on architectures, memory mechanisms, training

## Next Steps
- Browse `examples/` for runnable scripts
- Follow setup guides in `docs/setup/` for detailed installation
- Explore `llm_lab/cli.py` for Typer commands (`llm-lab pretrain`, `llm-lab posttrain`)
- Read technical content in `docs/llm/` for research notes
