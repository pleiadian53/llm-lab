# llm-lab

Reusable building blocks for running LLM pre-training and post-training experiments with modern tooling and documentation.

## Highlights

- **Modular package layout** with `pretrain_llm` and `posttrain_llm` subpackages
- **Reusable trainer packages**: `sft_trainer` (Supervised Fine-Tuning) and `dpo_trainer` (Direct Preference Optimization)
- **Hydra-ready configs** and Typer CLI entry points for reproducible experiments
- **Apple Silicon & Linux friendly** environment managed by `mamba` and editable installs
- **Developer experience** powered by `ruff`, `black`, `isort`, `mypy`, `pytest`, and `pre-commit`
- **Comprehensive docs** with tutorials, explainers, and research notes

## Key Packages

### SFT Trainer (`posttrain_llm/sft_trainer/`)

Supervised Fine-Tuning with PEFT support (LoRA, DoRA, QLoRA, etc.)

```bash
# CLI usage
python -m sft_trainer train --model HuggingFaceTB/SmolLM2-135M --dataset banghua/DL-SFT-Dataset --peft lora
```

```python
# Python API
from sft_trainer import SFTTrainerWrapper, PEFTConfig

trainer = SFTTrainerWrapper(
    model_name="HuggingFaceTB/SmolLM2-135M",
    dataset_name="banghua/DL-SFT-Dataset",
    peft_config=PEFTConfig.from_preset("lora_default"),
)
trainer.train()
```

### DPO Trainer (`posttrain_llm/dpo_trainer/`)

Direct Preference Optimization for alignment without reward models.

```bash
# CLI usage
python -m dpo_trainer train --model HuggingFaceTB/SmolLM2-135M-Instruct --dataset banghua/DL-DPO-Dataset

# Identity shift training
python -m dpo_trainer identity-shift --model HuggingFaceTB/SmolLM2-135M-Instruct --original-name Qwen --new-name "Deep Qwen"
```

```python
# Python API
from dpo_trainer import DPOTrainerWrapper, build_identity_shift_dataset

trainer = DPOTrainerWrapper(
    model_name="HuggingFaceTB/SmolLM2-135M-Instruct",
    dataset_name="banghua/DL-DPO-Dataset",
)
trainer.train()
```

## Project Layout

```
llm-lab/
├── src/llm_lab/              # Python package (installed in editable mode)
│   ├── pretrain_llm/         # Pre-training utilities
│   └── posttrain_llm/        # Post-training + alignment utilities
├── pretrain_llm/             # Pre-training lessons and notebooks
│   ├── Lesson_1-3.ipynb      # Pre-training tutorials
│   └── docs/                 # Pre-training documentation
├── posttrain_llm/            # Post-training packages and lessons
│   ├── sft_trainer/          # SFT package with PEFT support
│   ├── dpo_trainer/          # DPO package for preference optimization
│   ├── llm_eval/             # LLM evaluation utilities
│   ├── L3/, L5/, L7/         # Lesson notebooks
│   └── M1/                   # Module 1 materials
├── examples/                 # Example scripts and workflows
├── tests/                    # Pytest-based test suite
├── docs/                     # Documentation portal
│   ├── setup/                # Installation guides
│   ├── workflows/            # Document workflows
│   └── LLM/                  # LLM research notes
├── dev/                      # Development notes and explainers
├── pyproject.toml            # Package metadata and tooling config
├── environment.yml           # Mamba environment specification
└── requirements.txt          # Pip installation manifest
```

## Getting Started

1. **Install the environment with mamba:**
   ```bash
   mamba env create -f environment.yml
   mamba activate llm-lab
   ```

2. **Install the package in editable mode:**
   ```bash
   pip install -e .[dev]
   ```

3. **Run the test suite:**
   ```bash
   pytest
   ```

4. **Try a trainer package:**
   ```bash
   # Test DPO trainer CLI
   python -m dpo_trainer --help
   
   # Test SFT trainer CLI
   python -m sft_trainer --help
   ```

## Documentation

Complete documentation is available in the [`docs/`](docs/) directory:

- **[Documentation Portal](docs/README.md)** - Main entry point for all documentation
- **[Quick Start](docs/quick-start.md)** - Get up and running in 5 minutes
- **[Setup Guides](docs/setup/)** - Installation, environment, LaTeX, dependencies
- **[Workflows](docs/workflows/)** - Document creation, markdown→PDF conversion
- **[LLM Research](docs/LLM/)** - Technical notes on architectures, memory mechanisms, training

### Package Documentation

- **[SFT Trainer](posttrain_llm/sft_trainer/README.md)** - Supervised Fine-Tuning with PEFT
- **[DPO Trainer](posttrain_llm/dpo_trainer/README.md)** - Direct Preference Optimization
- **[DPO Explainer](posttrain_llm/dpo_trainer/docs/DPO_explainer.md)** - Technical deep-dive into DPO
- **[DPO for Computational Biology](posttrain_llm/dpo_trainer/docs/DPO_for_computational_biology.md)** - DPO applications in biology
- **[Pre-training Guide](pretrain_llm/README.md)** - LLM pre-training tutorials

## Features

### SFT Trainer Features

- Full fine-tuning and PEFT methods (LoRA, DoRA, QLoRA, VeRA, AdaLoRA, IA3, Prompt/Prefix Tuning)
- CLI and Python API
- HuggingFace integration
- GPU/MPS/CPU support

### DPO Trainer Features

- Direct Preference Optimization training
- Identity shift training (change model's self-identification)
- Custom preference dataset builders
- Model comparison utilities
- CLI commands: `train`, `identity-shift`, `test`, `compare`, `info`

## Next Steps

- Browse `examples/` for runnable scripts
- Try the trainer packages with `python -m sft_trainer --help` or `python -m dpo_trainer --help`
- Follow setup guides in `docs/setup/` for detailed installation
- Read technical content in `docs/LLM/` for research notes
- Explore lesson notebooks in `pretrain_llm/` and `posttrain_llm/`
