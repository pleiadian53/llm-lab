"""Command-line interface for DPO Trainer.

Usage:
    # Basic DPO training
    python -m dpo_trainer train --model HuggingFaceTB/SmolLM2-135M-Instruct --dataset banghua/DL-DPO-Dataset

    # Identity shift training
    python -m dpo_trainer identity-shift --model HuggingFaceTB/SmolLM2-135M-Instruct --original-name Qwen --new-name "Deep Qwen"

    # Test a trained model
    python -m dpo_trainer test --model ./dpo_output --question "What is your name?"
"""

import sys
from pathlib import Path
from typing import Optional, List

import typer
from rich.console import Console
from rich.table import Table

# Add parent to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent))

from dpo_trainer.core.trainer import DPOTrainerWrapper, DPOTrainingConfig
from dpo_trainer.core.model_loader import load_model_and_tokenizer
from dpo_trainer.core.inference import generate_response, test_model
from dpo_trainer.core.dataset import build_identity_shift_dataset, load_dpo_dataset

app = typer.Typer(
    name="dpo-trainer",
    help="Direct Preference Optimization Training",
    add_completion=False,
)
console = Console()


@app.command()
def train(
    model: str = typer.Option(..., "--model", "-m", help="Model name or path"),
    dataset: str = typer.Option(..., "--dataset", "-d", help="Dataset name or path"),
    output_dir: str = typer.Option("./dpo_output", "--output", "-o", help="Output directory"),
    beta: float = typer.Option(0.2, "--beta", "-b", help="DPO beta parameter"),
    epochs: int = typer.Option(1, "--epochs", "-e", help="Number of training epochs"),
    batch_size: int = typer.Option(1, "--batch-size", help="Batch size per device"),
    learning_rate: float = typer.Option(5e-5, "--lr", help="Learning rate"),
    grad_accum: int = typer.Option(8, "--grad-accum", help="Gradient accumulation steps"),
    max_samples: Optional[int] = typer.Option(None, "--max-samples", help="Limit dataset size"),
    use_gpu: bool = typer.Option(False, "--gpu", help="Use GPU if available"),
    logging_steps: int = typer.Option(2, "--logging-steps", help="Logging frequency"),
):
    """Train a model with DPO."""
    console.print(f"[bold blue]DPO Training[/bold blue]")
    console.print(f"  Model: {model}")
    console.print(f"  Dataset: {dataset}")
    console.print(f"  Output: {output_dir}")
    console.print(f"  Beta: {beta}")
    console.print()
    
    # Create training config
    training_config = DPOTrainingConfig(
        beta=beta,
        learning_rate=learning_rate,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        logging_steps=logging_steps,
        output_dir=output_dir,
    )
    
    # Create trainer
    trainer = DPOTrainerWrapper(
        model_name=model,
        dataset_name=dataset,
        training_config=training_config,
        use_gpu=use_gpu,
        max_samples=max_samples,
    )
    
    # Train
    console.print("[bold]Starting DPO training...[/bold]")
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


@app.command("identity-shift")
def identity_shift(
    model: str = typer.Option(..., "--model", "-m", help="Model name or path"),
    original_name: str = typer.Option(..., "--original-name", help="Original name to replace"),
    new_name: str = typer.Option(..., "--new-name", help="New name to use"),
    output_dir: str = typer.Option("./dpo_output", "--output", "-o", help="Output directory"),
    raw_dataset: str = typer.Option("mrfakename/identity", "--raw-dataset", help="Raw dataset for prompts"),
    beta: float = typer.Option(0.2, "--beta", "-b", help="DPO beta parameter"),
    epochs: int = typer.Option(1, "--epochs", "-e", help="Number of training epochs"),
    max_samples: Optional[int] = typer.Option(None, "--max-samples", help="Limit dataset size"),
    use_gpu: bool = typer.Option(False, "--gpu", help="Use GPU if available"),
):
    """Train a model to change its identity using DPO."""
    console.print(f"[bold blue]Identity Shift Training[/bold blue]")
    console.print(f"  Model: {model}")
    console.print(f"  Original Name: {original_name}")
    console.print(f"  New Name: {new_name}")
    console.print(f"  Output: {output_dir}")
    console.print()
    
    # Load model for dataset generation
    console.print("[bold]Loading model for preference pair generation...[/bold]")
    model_obj, tokenizer = load_model_and_tokenizer(model, use_gpu=use_gpu)
    
    # Build identity shift dataset
    console.print("[bold]Building identity shift dataset...[/bold]")
    dpo_dataset = build_identity_shift_dataset(
        model_obj, tokenizer,
        original_name=original_name,
        new_name=new_name,
        dataset_name=raw_dataset,
        max_samples=max_samples,
    )
    console.print(f"  Generated {len(dpo_dataset)} preference pairs")
    
    # Clean up model used for generation
    del model_obj
    
    # Create training config
    training_config = DPOTrainingConfig(
        beta=beta,
        num_train_epochs=epochs,
        output_dir=output_dir,
    )
    
    # Create trainer with generated dataset
    trainer = DPOTrainerWrapper(
        model_name=model,
        train_dataset=dpo_dataset,
        training_config=training_config,
        use_gpu=use_gpu,
    )
    
    # Train
    console.print("[bold]Starting DPO training...[/bold]")
    metrics = trainer.train()
    
    # Save
    trainer.save_model(output_dir)
    
    console.print()
    console.print("[bold green]Identity shift training complete![/bold green]")
    console.print(f"Model saved to: {output_dir}")


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
        # Default identity questions
        questions = [
            "What is your name?",
            "Are you ChatGPT?",
            "Tell me about your name and organization.",
        ]
    
    # Generate responses
    for i, q in enumerate(questions, 1):
        console.print(f"[bold cyan]Question {i}:[/bold cyan] {q}")
        response = generate_response(model_obj, tokenizer, q, max_new_tokens=max_tokens)
        console.print(f"[bold green]Response:[/bold green] {response}")
        console.print()


@app.command()
def compare(
    before_model: str = typer.Option(..., "--before", help="Model before DPO training"),
    after_model: str = typer.Option(..., "--after", help="Model after DPO training"),
    questions_file: Optional[str] = typer.Option(None, "--file", "-f", help="File with questions"),
    use_gpu: bool = typer.Option(False, "--gpu", help="Use GPU if available"),
):
    """Compare models before and after DPO training."""
    console.print(f"[bold blue]Model Comparison[/bold blue]")
    console.print(f"  Before: {before_model}")
    console.print(f"  After: {after_model}")
    console.print()
    
    # Default questions
    questions = [
        "What is your name?",
        "Are you ChatGPT?",
        "Tell me about your name and organization.",
    ]
    
    if questions_file:
        with open(questions_file) as f:
            questions = [line.strip() for line in f if line.strip()]
    
    # Load models
    console.print("[bold]Loading models...[/bold]")
    before_model_obj, before_tokenizer = load_model_and_tokenizer(before_model, use_gpu=use_gpu)
    after_model_obj, after_tokenizer = load_model_and_tokenizer(after_model, use_gpu=use_gpu)
    
    # Compare
    for i, q in enumerate(questions, 1):
        console.print(f"\n[bold cyan]Question {i}:[/bold cyan] {q}")
        
        before_response = generate_response(before_model_obj, before_tokenizer, q)
        after_response = generate_response(after_model_obj, after_tokenizer, q)
        
        console.print(f"[yellow]Before DPO:[/yellow] {before_response}")
        console.print(f"[green]After DPO:[/green] {after_response}")


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
