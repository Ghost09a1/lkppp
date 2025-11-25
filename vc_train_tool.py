"""
Thin wrapper to call the official RVC CLI (rvc_cli.py) for training.

Expected tool: https://github.com/blaisewf/rvc-cli (or equivalent)
Place rvc_cli.py somewhere on disk and point --rvc_cli to it.

Pipeline (very small defaults):
1) preprocess
2) extract
3) train
4) index

Usage (example):
  python vc_train_tool.py \
    --data_dir data/voices/char_1/raw \
    --output models/vc/char_1.pth \
    --rvc_cli C:/path/to/rvc_cli.py

Notes:
- This script will FAIL if rvc_cli.py is missing. Install RVC first.
- Flags here are minimal; adjust epochs/batch/f0method to your needs.
- If training succeeds but the CLI saves the model elsewhere, this script
  will try to copy the newest .pth it finds into --output.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def run_step(name: str, cmd: list[str]) -> None:
    print(f"[vc_train_tool] {name}: {' '.join(cmd)}")
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr, file=sys.stderr)
        raise RuntimeError(f"{name} failed with exit code {result.returncode}")


def find_latest_pth(root: Path) -> Path | None:
    candidates = list(root.rglob("*.pth"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def main():
    parser = argparse.ArgumentParser(description="RVC training wrapper")
    parser.add_argument("--data_dir", required=True, help="Directory containing WAV dataset")
    parser.add_argument("--output", required=True, help="Path to write trained model")
    parser.add_argument("--rvc_cli", default="rvc_cli.py", help="Path to rvc_cli.py (from rvc-cli repo)")
    parser.add_argument("--model_name", default="", help="Optional model name (defaults to output stem)")
    parser.add_argument("--sr", type=int, default=48000, help="Sample rate for training (RVC default 48k)")
    parser.add_argument("--epochs", type=int, default=20, help="Total epochs to train")
    parser.add_argument("--batch_size", type=int, default=6, help="Batch size for training")
    parser.add_argument("--f0method", default="pm", help="F0 extraction method (pm|crepe|rmvpe etc.)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    out_path = Path(args.output).resolve()
    rvc_cli = Path(args.rvc_cli).resolve()
    model_name = args.model_name or out_path.stem

    wavs = list(data_dir.glob("*.wav"))
    if not wavs:
        print(f"No WAV files found in {data_dir}", file=sys.stderr)
        sys.exit(1)

    if not rvc_cli.exists():
        print(f"rvc_cli.py not found at {rvc_cli}. Install rvc-cli and set --rvc_cli.", file=sys.stderr)
        sys.exit(1)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Determine how to call CLI (exe vs .py)
    def cli_cmd(subcommand: str, extra: list[str]) -> list[str]:
        if rvc_cli.suffix.lower() == ".py":
            return ["python", str(rvc_cli), subcommand, *extra]
        return [str(rvc_cli), subcommand, *extra]

    # 1) preprocess
    run_step(
        "preprocess",
        cli_cmd("preprocess", ["--data_dir", str(data_dir), "--sr", str(args.sr)]),
    )
    # 2) extract
    run_step(
        "extract",
        cli_cmd(
            "extract",
            ["--data_dir", str(data_dir), "--sr", str(args.sr), "--f0method", args.f0method],
        ),
    )
    # 3) train
    run_step(
        "train",
        cli_cmd(
            "train",
            [
                "--data_dir",
                str(data_dir),
                "--model_name",
                model_name,
                "--sr",
                str(args.sr),
                "--total_epoch",
                str(args.epochs),
                "--batch_size",
                str(args.batch_size),
            ],
        ),
    )
    # 4) index (optional but handy)
    try:
        run_step(
            "index",
            cli_cmd("index", ["--model_name", model_name]),
        )
    except Exception as exc:  # pragma: no cover - non-critical
        print(f"[vc_train_tool] index step failed (continuing): {exc}", file=sys.stderr)

    # Try to locate a trained model and copy to requested output
    candidate = find_latest_pth(rvc_cli.parent)
    if candidate:
        shutil.copy2(candidate, out_path)
        print(f"[vc_train_tool] Copied trained model {candidate} -> {out_path}")
        sys.exit(0)

    print("[vc_train_tool] No .pth found after training.", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
