"""Loading training checkpoints without running pickled code."""

import pickle
from pathlib import Path

import torch


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict:
    """
    Reads a checkpoint written by the trainer: tensors, numbers, lists and a JSON config.

    `torch.load(weights_only=False)` would unpickle whatever the file contains, so a
    checkpoint from someone else could run code on load. Older checkpoints that pickled the
    config object (or Linux `PosixPath`s) are refused; convert one you made yourself with
    `convert_checkpoint.py`.
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except pickle.UnpicklingError as e:
        raise ValueError(
            f"{path} holds pickled Python objects besides tensors and plain values, so it is "
            "not loaded. If you made this file, convert it once with "
            f"`uv run python convert_checkpoint.py {path}` and load the converted file."
        ) from e
