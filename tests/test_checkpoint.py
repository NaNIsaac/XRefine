"""Checkpoints load without unpickling arbitrary objects.

A checkpoint written by the trainer holds tensors, numbers, lists and a JSON config, so
`torch.load(weights_only=True)` reads it. Anything else in the file (an older pickled config
object, or a planted payload) must be refused, not executed.
"""

import pathlib

import pytest
import torch

from xrefine.checkpoint import load_checkpoint
from xrefine.config import XRefineConfig


def trainer_checkpoint() -> dict:
    """The layout of Trainer._save_checkpoint, with a small model."""
    model = torch.nn.Linear(4, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, total_steps=10)
    optimizer.step()
    scheduler.step()
    return {
        "epoch": 3,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_val_loss": 0.25,
        "config": XRefineConfig.load_yaml("config.yaml").model_dump(mode="json"),
        "history": {"train_loss": [0.5, 0.4], "val_loss": [0.6], "lr": [1e-3], "sup_weight": [1.0]},
    }


def test_a_trainer_checkpoint_loads_with_weights_only(tmp_path):
    path = tmp_path / "best.pt"
    saved = trainer_checkpoint()
    torch.save(saved, path)
    ckpt = load_checkpoint(path)
    assert ckpt["epoch"] == 3 and ckpt["history"] == saved["history"]
    assert torch.equal(ckpt["model_state_dict"]["weight"], saved["model_state_dict"]["weight"])
    XRefineConfig(**ckpt["config"])


class Payload:
    """Stands in for anything that runs code when unpickled."""

    def __reduce__(self):
        return (pathlib.Path.touch, (pathlib.Path(Payload.marker),))


def test_a_file_with_pickled_objects_is_refused_without_running_them(tmp_path):
    Payload.marker = str(tmp_path / "ran")
    path = tmp_path / "evil.pt"
    torch.save({"config": Payload(), "model_state_dict": {}}, path)
    with pytest.raises(ValueError, match="convert_checkpoint.py"):
        load_checkpoint(path)
    assert not (tmp_path / "ran").exists()


def test_an_older_checkpoint_loads_after_conversion(tmp_path):
    """Before the config was stored as JSON, the XRefineConfig object itself was pickled."""
    import runpy

    convert = runpy.run_path("convert_checkpoint.py")["convert_checkpoint"]
    old = tmp_path / "old.pt"
    torch.save({**trainer_checkpoint(), "config": XRefineConfig.load_yaml("config.yaml")}, old)
    with pytest.raises(ValueError):
        load_checkpoint(old)
    convert(str(old), str(tmp_path / "new.pt"))
    XRefineConfig(**load_checkpoint(tmp_path / "new.pt")["config"])
