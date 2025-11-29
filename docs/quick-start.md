# Quick Start

Your daily operations cheat sheet, structured like the probability-lab quick start: concise commands, expected outputs, and common variations.

## TL;DR Installation
```bash
git clone git@github.com:USERNAME/llm-lab.git
cd llm-lab
mamba env create -f environment.yml
mamba activate llm-lab
pip install -e .[dev]
pre-commit install
pytest
```

## Daily Workflow
- **Open environment**: `mamba activate llm-lab`
- **Run linters**: `ruff check src tests`
- **Format code**: `ruff format src tests`
- **Type check**: `mypy src`
- **Run tests**: `pytest`
- **Launch CLI**: `python -m llm_lab.cli --help`
- **Record experiments**: integrate with `wandb` inside `LanguageModelTrainer`.

## Common Tasks
- Scaffold a new experiment config: copy `PretrainingConfig()` and dump to YAML.
- Execute a smoke training pass: `python examples/run_pretraining.py`.
- Generate post-training summary: `python -m llm_lab.cli posttrain`.
- Update dependencies: edit `pyproject.toml` and rerun `pip install -e .[dev]`.

## Quick Reference
- Editable install path: `<repo>/src`
- Artifacts directory: `<repo>/artifacts`
- Configuration entry points:
  - `llm_lab/pretrain_llm/config.py`
  - `llm_lab/posttrain_llm/config.py`
- CLI commands: `pretrain`, `posttrain`

## Common Issues
- **CLI import error**: confirm the environment is activated and `pip list` shows `llm-lab (editable)`.
- **Slow pytest run**: execute `pytest -k "not integration"` until heavier tests are added.
- **WandB login prompts**: set `WANDB_MODE=offline` during local development.
