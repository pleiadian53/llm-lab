"""Command-line interface for SFT Trainer.

Usage:
    # Full fine-tuning
    python -m sft_trainer.cli train --model HuggingFaceTB/SmolLM2-135M --dataset banghua/DL-SFT-Dataset

    # LoRA fine-tuning
    python -m sft_trainer.cli train --model HuggingFaceTB/SmolLM2-135M --dataset banghua/DL-SFT-Dataset --peft lora

    # List available PEFT presets
    python -m sft_trainer.cli list-presets

    # Test a trained model
    python -m sft_trainer.cli test --model ./output --question "What is ML?"
"""

import sys
from pathlib import Path
from typing import Optional, List

import typer
from rich.console import Console
from rich.table import Table

# Add parent to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent))

from sft_trainer.core.trainer import SFTTrainerWrapper, TrainingConfig
from sft_trainer.core.model_loader import load_model_and_tokenizer
from sft_trainer.core.inference import generate_response, test_model
from sft_trainer.peft.config import PEFTConfig, PEFTMethod

app = typer.Typer(
    name="sft-trainer",
    help="Supervised Fine-Tuning with PEFT Support",
    add_completion=False,
)
console = Console()


# Available PEFT presets
PEFT_PRESETS = [
    "lora_default",
    "lora_high_rank", 
    "dora",
    "olora",
    "qlora_4bit",
    "qlora_8bit",
    "vera",
    "adalora",
    "ia3",
    "prompt_tuning",
    "prefix_tuning",
]


@app.command()
def train(
    model: str = typer.Option(..., "--model", "-m", help="Model name or path"),
    dataset: str = typer.Option(..., "--dataset", "-d", help="Dataset name or path"),
    output_dir: str = typer.Option("./sft_output", "--output", "-o", help="Output directory"),
    peft: Optional[str] = typer.Option(None, "--peft", "-p", help="PEFT preset name"),
    epochs: int = typer.Option(1, "--epochs", "-e", help="Number of training epochs"),
    batch_size: int = typer.Option(1, "--batch-size", "-b", help="Batch size per device"),
    learning_rate: float = typer.Option(8e-5, "--lr", help="Learning rate"),
    grad_accum: int = typer.Option(8, "--grad-accum", help="Gradient accumulation steps"),
    max_samples: Optional[int] = typer.Option(None, "--max-samples", help="Limit dataset size"),
    use_gpu: bool = typer.Option(False, "--gpu", help="Use GPU if available"),
    logging_steps: int = typer.Option(10, "--logging-steps", help="Logging frequency"),
):
    """Train a model with SFT, optionally using PEFT."""
    console.print(f"[bold blue]SFT Training[/bold blue]")
    console.print(f"  Model: {model}")
    console.print(f"  Dataset: {dataset}")
    console.print(f"  Output: {output_dir}")
    console.print(f"  PEFT: {peft or 'None (full fine-tuning)'}")
    console.print()
    
    # Create training config
    training_config = TrainingConfig(
        learning_rate=learning_rate,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        logging_steps=logging_steps,
        output_dir=output_dir,
    )
    
    # Create PEFT config if specified
    peft_config = None
    if peft:
        if peft not in PEFT_PRESETS:
            console.print(f"[red]Unknown PEFT preset: {peft}[/red]")
            console.print(f"Available: {', '.join(PEFT_PRESETS)}")
            raise typer.Exit(1)
        peft_config = PEFTConfig.from_preset(peft)
        # Use higher learning rate for PEFT
        if training_config.learning_rate == 8e-5:
            training_config.learning_rate = 2e-4
            console.print(f"  [dim]Adjusted LR to {training_config.learning_rate} for PEFT[/dim]")
    
    # Create trainer
    trainer = SFTTrainerWrapper(
        model_name=model,
        dataset_name=dataset,
        training_config=training_config,
        peft_config=peft_config,
        use_gpu=use_gpu,
        max_samples=max_samples,
    )
    
    # Train
    console.print("[bold]Starting training...[/bold]")
    metrics = trainer.train()
    
    # Save
    trainer.save_model(output_dir)
    
    console.print()
    console.print("[bold green]Training complete![/bold green]")
    console.print(f"Model saved to: {output_dir}")
    
    # Print metrics
    table = Table(title="Training Metrics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    for key, value in metrics.items():
        if isinstance(value, float):
            table.add_row(key, f"{value:.4f}")
        else:
            table.add_row(key, str(value))
    console.print(table)


@app.command()
def test(
    model: str = typer.Option(..., "--model", "-m", help="Model name or path"),
    question: Optional[str] = typer.Option(None, "--question", "-q", help="Single question to ask"),
    questions_file: Optional[str] = typer.Option(None, "--file", "-f", help="File with questions (one per line)"),
    max_tokens: int = typer.Option(200, "--max-tokens", help="Maximum tokens to generate"),
    use_gpu: bool = typer.Option(False, "--gpu", help="Use GPU if available"),
):
    """Test a model with questions."""
    console.print(f"[bold blue]Testing Model[/bold blue]")
    console.print(f"  Model: {model}")
    console.print()
    
    # Load model
    model_obj, tokenizer = load_model_and_tokenizer(model, use_gpu=use_gpu)
    
    # Collect questions
    questions = []
    if question:
        questions.append(question)
    if questions_file:
        with open(questions_file) as f:
            questions.extend([line.strip() for line in f if line.strip()])
    
    if not questions:
        # Default questions
        questions = [
            "Give me a 1-sentence introduction of LLM.",
            "Calculate 1+1-1",
            "What's the difference between thread and process?",
        ]
    
    # Generate responses
    for i, q in enumerate(questions, 1):
        console.print(f"[bold cyan]Question {i}:[/bold cyan] {q}")
        response = generate_response(model_obj, tokenizer, q, max_new_tokens=max_tokens)
        console.print(f"[bold green]Response:[/bold green] {response}")
        console.print()


@app.command("list-presets")
def list_presets():
    """List available PEFT presets."""
    table = Table(title="Available PEFT Presets")
    table.add_column("Preset", style="cyan")
    table.add_column("Method", style="green")
    table.add_column("Description", style="dim")
    
    descriptions = {
        "lora_default": "Standard LoRA (r=16, alpha=32)",
        "lora_high_rank": "High-rank LoRA (r=64, alpha=128)",
        "dora": "Weight-Decomposed LoRA",
        "olora": "Orthogonal initialization LoRA",
        "qlora_4bit": "4-bit quantized LoRA (NF4)",
        "qlora_8bit": "8-bit quantized LoRA",
        "vera": "Vector-based Random Matrix Adaptation",
        "adalora": "Adaptive rank LoRA",
        "ia3": "Infused Adapter by Inhibiting and Amplifying",
        "prompt_tuning": "Soft prompt tuning",
        "prefix_tuning": "Prefix tuning",
    }
    
    for preset in PEFT_PRESETS:
        config = PEFTConfig.from_preset(preset)
        table.add_row(preset, config.method.value, descriptions.get(preset, ""))
    
    console.print(table)


@app.command()
def info(
    model: str = typer.Option(..., "--model", "-m", help="Model name or path"),
):
    """Show model information."""
    console.print(f"[bold blue]Model Information[/bold blue]")
    console.print(f"  Loading: {model}")
    console.print()
    
    model_obj, tokenizer = load_model_and_tokenizer(model, use_gpu=False)
    
    config = model_obj.config
    total_params = sum(p.numel() for p in model_obj.parameters())
    
    table = Table(title="Model Details")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Model Type", getattr(config, "model_type", "Unknown"))
    table.add_row("Hidden Size", str(getattr(config, "hidden_size", "Unknown")))
    table.add_row("Num Layers", str(getattr(config, "num_hidden_layers", "Unknown")))
    table.add_row("Num Heads", str(getattr(config, "num_attention_heads", "Unknown")))
    table.add_row("Vocab Size", str(getattr(config, "vocab_size", "Unknown")))
    table.add_row("Total Parameters", f"{total_params:,}")
    table.add_row("Tokenizer Vocab", str(len(tokenizer)))
    
    console.print(table)


def main():
    app()


if __name__ == "__main__":
    main()
