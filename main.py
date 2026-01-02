from pathlib import Path
from types import SimpleNamespace

import typer
from loguru import logger
from xrefine.config import CONFIG
from xrefine.train import XRefineTrainer
from xrefine.evaluate import evaluate 
from xrefine.models import XRR_Fourier_PINN

# Typer 앱 초기화
app = typer.Typer(
    help="XRefine: Physics-Informed AI for XRR",
    add_completion=False,
    rich_markup_mode="rich"
)

@app.command()
@logger.catch
def train(
    name: str | None = typer.Option(None, "--name", help="Experiment name"),
    lr: float | None = typer.Option(None, help="Learning rate"),
    layers: int | None = typer.Option(None, "--layers", help="Number of film layers"),
    epochs: int = typer.Option(200, help="Total epochs"),
    steps_per_epoch: int = typer.Option(100, help="Steps per epoch"),
    resume: bool = typer.Option(False, "--resume", help="Resume from latest.pt"),
):
    """Start PINN training for XRR analysis."""

    cli_params = {
        "experiment_name": name,
        "lr": lr,
        "n_layers": layers,
    }
    cli_params = {k: v for k, v in cli_params.items() if v is not None}
    config = CONFIG.model_copy(update=cli_params)

    save_dir = Path(config.save_root) / config.experiment_name
    save_dir.mkdir(parents=True, exist_ok=True)
    config.save_yaml(save_dir / "config.yaml")

    logger.info(f"[INIT] {config.experiment_name} | Layers: {config.n_layers} | Device: {config.device}")

    model = XRR_Fourier_PINN(config=config, device=config.device)
    trainer = XRefineTrainer(model=model, config=config)

    trainer.fit(
        max_epochs=epochs, 
        steps_per_epoch=steps_per_epoch, 
        resume=resume
    )

@app.command()
def eval(
    checkpoint: str | None = typer.Option(None, help="Path to model checkpoint"),
    batch_size: int = typer.Option(1000, help="Batch size for evaluation"),
):
    """Evaluate the trained PINN model."""
    if not checkpoint:
        checkpoint = str(CONFIG.save_root / CONFIG.experiment_name / "best_pinn.pt")
    
    ckpt_path = Path(checkpoint)
    if not ckpt_path.exists():
        logger.error(f"Checkpoint not found at {checkpoint}")
        raise typer.Exit(code=1)

    args_ns = SimpleNamespace(checkpoint=checkpoint, batch_size=batch_size)
    evaluate(args_ns)

@app.command()
def config():
    """Show current configuration."""
    logger.info("\n" + CONFIG.model_dump_json(indent=2))

if __name__ == "__main__":
    app()