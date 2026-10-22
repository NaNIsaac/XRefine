import argparse
import pathlib
import shutil
from pathlib import Path

import torch


def convert_checkpoint(ckpt_path: str, output_path: str = None):
    """
    Linux(PosixPath)에서 생성된 PyTorch 체크포인트를 Windows 환경에서 읽을 수 있도록 변환합니다.
    pathlib.PosixPath 객체를 문자열 또는 WindowsPath로 변환하여 재저장합니다.
    """
    src_path = Path(ckpt_path)
    if not src_path.exists():
        print(f"❌ Error: File not found: {ckpt_path}")
        return

    if output_path is None:
        output_path = src_path.with_name(src_path.stem + "_win" + src_path.suffix)

    print(f"🔄 Converting: {src_path} -> {output_path}")

    # 1. PosixPath 해킹: 로드 시 WindowsPath로 매핑
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
        pathlib.PosixPath = temp  # 복구

    # 2. Config 내부의 Path 객체 순회 및 변환 (필요시)
    # 이미 로드될 때 WindowsPath로 변환되었겠지만, 명시적으로 문자열로 바꾸는 것이 가장 안전함
    if "config" in ckpt:
        print("   Processing config dict...")
        cfg = ckpt["config"]
        if hasattr(cfg, "model_dump"):
            # An older checkpoint pickled the XRefineConfig object itself.
            cfg = ckpt["config"] = cfg.model_dump(mode="json")
        for k, v in cfg.items():
            if isinstance(v, Path):
                cfg[k] = str(v)  # Path -> String 변환 (가장 호환성 높음)
                print(f"   Converted Path: {k} -> {v}")

    # 3. 저장
    torch.save(ckpt, output_path)
    print("✅ Conversion Complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Linux Checkpoint to Windows")
    parser.add_argument("checkpoint", help="Path to the input .pt file")

    args = parser.parse_args()
    convert_checkpoint(args.checkpoint)
