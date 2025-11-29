# Dependency Reference

This reference mirrors the depth of probability-lab: list every dependency, state why it matters, how it is installed, and what to do when it fails.

## Compiled via Mamba
- **PyTorch ≥ 2.1 (`torch`)**
  - *Purpose*: Core deep learning framework for GPU/Metal accelerated training.
  - *Install*: `mamba env create -f environment.yml` pulls binaries from the `pytorch` channel.
  - *Notes*: For Apple Silicon, binaries include MPS support. On CUDA hosts, match driver version with release notes. Current version: 2.5.1.
  - *Troubleshooting*: `pip uninstall torch -y` if a conflicting pip wheel is installed; rerun `mamba install pytorch`.
- **NumPy ≥ 1.24**
  - *Purpose*: Vectorized math and data wrangling.
  - *Install*: Provided by mamba; required by pandas and PyTorch.
  - *Troubleshooting*: Check `python -c "import numpy; numpy.show_config()"` when linking issues appear.
- **pandas ≥ 2.0**
  - *Purpose*: Tabular data manipulation, logging experiment metrics.
  - *Install*: mamba for compiled C extensions.
  - *Troubleshooting*: Ensure `libstdc++` is present on Linux; reinstall via `mamba install pandas`.
- **sympy = 1.13.1 (pinned)**
  - *Purpose*: Symbolic mathematics library required by PyTorch.
  - *Install*: Pinned in `environment.yml` to match PyTorch 2.5.1 requirements.
  - *Notes*: **Critical**: PyTorch 2.5.1 specifically requires sympy 1.13.1. Installing sympy 1.14.0 will cause dependency conflicts.
  - *Troubleshooting*: If you see sympy version warnings, run `pip install sympy==1.13.1` to fix.

## Pure Python via pip
All pip dependencies are now installed automatically via the `pip:` section in `environment.yml`. You no longer need to run `pip install -e .[dev]` separately.

- **transformers ≥ 4.35**
  - *Purpose*: Model and tokenizer interfaces for Hugging Face ecosystems.
  - *Install*: Automatically installed via `environment.yml`.
  - *Current version*: 4.57.1
  - *Troubleshooting*: Clear caches with `transformers-cli cache clear` if downloads corrupt.
- **datasets ≥ 2.15**
  - *Purpose*: Stream and preprocess text datasets for pre-training and RLHF tasks.
  - *Install*: Automatically installed via `environment.yml`.
  - *Current version*: 4.3.0
  - *Troubleshooting*: Set `HF_DATASETS_OFFLINE=1` when working offline; fallback datasets are built into the package.
- **hydra-core ≥ 1.3**
  - *Purpose*: Configuration composition and command-line overrides.
  - *Install*: Automatically installed via `environment.yml`.
  - *Current version*: 1.3.2
  - *Troubleshooting*: Delete `.hydra` directories when configs get stuck with old overrides.
- **wandb ≥ 0.16**
  - *Purpose*: Experiment tracking and logging.
  - *Install*: Automatically installed via `environment.yml`.
  - *Current version*: 0.22.2
  - *Troubleshooting*: Set `WANDB_MODE=offline` to avoid login prompts on CI; run `wandb login` locally.
- **pydantic ≥ 2.6**
  - *Purpose*: Typed config validation for both pre and post training modules.
  - *Install*: Automatically installed via `environment.yml`.
  - *Current version*: 2.12.3
  - *Notes*: You may see harmless warnings about field attributes from pydantic v2 internals.
  - *Troubleshooting*: Upgrade `pip` if binary wheels fail to download.
- **typer ≥ 0.12**
  - *Purpose*: CLI interface for `llm_lab.cli`.
  - *Install*: Automatically installed via `environment.yml`.
  - *Current version*: 0.20.0
  - *Troubleshooting*: Ensure `python -m llm_lab.cli --help` works; reinstall if Typer mismatches Click.
- **Additional developer tooling** (`black`, `ruff`, `isort`, `mypy`, `pytest`, `pytest-cov`, `pre-commit`, `types-PyYAML`)
  - *Purpose*: Formatting, linting, static analysis, testing, and hooks.
  - *Install*: Automatically installed via `environment.yml` (no `[dev]` extra needed).
  - *Current versions*: black 25.9.0, ruff 0.14.2, isort 7.0.0, mypy 1.18.2, pytest 8.4.2, pytest-cov 7.0.0, pre-commit 4.3.0
  - *Troubleshooting*: Run `pre-commit autoupdate` when hooks warn about outdated versions.

## Optional Extras
- **GitHub CLI (`gh`)** for repository automation.
- **wandb** system dependencies: confirm outbound HTTPS access or run in offline mode.

## Installation Architecture
The project uses a **dual-source** approach for maximum convenience:
1. **`pyproject.toml`** remains the canonical source of truth for package metadata and dependencies.
2. **`environment.yml`** duplicates these dependencies for one-command installation via mamba.
3. The editable install (`-e .` in `environment.yml`) ensures source code changes are reflected immediately without reinstalling.

## Version Management Tips
- **Keep both files in sync**: When adding/updating dependencies, edit both `pyproject.toml` AND `environment.yml`.
- **Snapshot exact versions**: Use `mamba env export --no-builds > environment.lock.yml` for reproducibility.
- **Verify consistency**: Run `pip check` after any dependency changes to catch conflicts early.
- **Pin transient research dependencies**: Use separate `requirements-experiments.txt` files to avoid polluting the base stack.
- **Update environment**: Run `mamba env update -f environment.yml --prune` after editing environment.yml.
