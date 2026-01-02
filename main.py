# main.py

from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import typer
from loguru import logger
from xrefine.config import CONFIG
from xrefine.train import XRefineTrainer

from xrefine.evaluate import XRREvaluator 
from xrefine.models import XRR_Fourier_PINN


app = typer.Typer(
    help="XRefine: Physics-Informed AI for XRR",
    add_completion=False,
    rich_markup_mode="rich",
    pretty_exceptions_enable=False,
    pretty_exceptions_show_locals=False
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
    logger.info("✅ Training session completed successfully.")

@app.command()
def eval(
    checkpoint: Optional[str] = typer.Option(None, help="Path to model checkpoint"),
    batch_size: int = typer.Option(1000, help="Batch size for evaluation"),
):
    """
    현대적인 OOP 기반 Evaluator를 사용하여 모델을 평가합니다.
    모든 파라미터에 대한 통계 지표 및 물리 복원도 시각화 리포트를 생성합니다.
    """
    # 1. 체크포인트 경로 설정
    if not checkpoint:
        checkpoint = str(CONFIG.save_root / CONFIG.experiment_name / "best.pt")
    
    ckpt_path = Path(checkpoint)
    if not ckpt_path.exists():
        logger.error(f"❌ Checkpoint not found at {checkpoint}")
        raise typer.Exit(code=1)

    evaluator = XRREvaluator(checkpoint_path=checkpoint, batch_size=batch_size)
    evaluator.setup()
    r_meas, p_true, p_pred, r_recon = evaluator.run_inference()
    evaluator.calculate_statistics(p_true, p_pred)
    evaluator.report()
    evaluator.save_visuals(evaluator.config.q_grid, r_meas, r_recon)
    logger.info(f"✅ Evaluation complete. All reports saved in: {evaluator.exp_dir}")


@app.command()
def config():
    """Show current configuration."""
    logger.info("\n" + CONFIG.model_dump_json(indent=2))

if __name__ == "__main__":
    app()