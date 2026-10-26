import argparse
import pathlib
from pathlib import Path

import torch


def convert_checkpoint(ckpt_path: str, output_path: str | None = None):
    """Convert a checkpoint saved on Linux (PosixPath objects) so it loads on Windows."""
    src_path = Path(ckpt_path)
    if not src_path.exists():
        print(f"❌ Error: File not found: {ckpt_path}")
        return

    if output_path is None:
        output_path = src_path.with_name(src_path.stem + "_win" + src_path.suffix)

    print(f"🔄 Converting: {src_path} -> {output_path}")

    # PosixPath objects in the pickle cannot be instantiated on Windows; load them as WindowsPath
    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath

    try:
        # weights_only=False unpickles anything in the file, so convert only checkpoints you
        # made yourself. The result holds plain values and loads with weights_only=True.
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"❌ Load Failed: {e}")
        return
    finally:
        pathlib.PosixPath = temp

    # Config paths are WindowsPath after loading; store them as str
    if "config" in ckpt:
        print("   Processing config dict...")
        cfg = ckpt["config"]
        if hasattr(cfg, "model_dump"):
            # An older checkpoint pickled the XRefineConfig object itself.
            cfg = ckpt["config"] = cfg.model_dump(mode="json")
        for k, v in cfg.items():
            if isinstance(v, Path):
                cfg[k] = str(v)
                print(f"   Converted Path: {k} -> {v}")

    torch.save(ckpt, output_path)
    print("✅ Conversion Complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Linux Checkpoint to Windows")
    parser.add_argument("checkpoint", help="Path to the input .pt file")

    args = parser.parse_args()
    convert_checkpoint(args.checkpoint)
