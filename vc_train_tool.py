"""
Adapter wrapper to call the official RVC training scripts via subprocess.

This script calls the specific Python files from the official 
Retrieval-based-Voice-Conversion-WebUI repository (pointed to by --rvc_cli).

The core logic replaces the non-functional rvc-cli calls with direct,
properly formatted subprocess calls to the RVC-WebUI's internal scripts.

Pipeline:
1) preprocess (trainset_preprocess_pipeline_print.py)
2) extract F0 (extract_f0_print.py)
3) extract feature (extract_feature_print.py)
4) train (train_nsf_sim_vs_print.py)
5) index (infer/modules/train/index_make.py)

Usage (example):
  python vc_train_tool.py \
    --data_dir data/voices/char_1/raw \
    --output models/vc/char_1.pth \
    --rvc_cli C:/path/to/Retrieval-based-Voice-Conversion-WebUI
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def run_step(name: str, cmd: list[str]) -> None:
    print(f"[vc_train_tool] {name}: {' '.join(cmd)}")
    
    # Der erste Eintrag ist der Python-Interpreter
    rvc_venv_python_path = Path(cmd[0]) 
    # Heuristik: venv/Scripts/python.exe -> Repo-Root ist drei Ebenen höher
    rvc_repo_dir = rvc_venv_python_path.parent.parent.parent
    
    result = subprocess.run(
        cmd, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE, 
        text=True,
        cwd=rvc_repo_dir if rvc_repo_dir.is_dir() else None 
    ) 
    
    if result.returncode != 0:
        print(f"RVC-Log (Stdout):\n{result.stdout}", file=sys.stderr)
        print(f"RVC-Fehler (Stderr):\n{result.stderr}", file=sys.stderr)
        raise RuntimeError(f"{name} failed with exit code {result.returncode}")
    
    # Erfolgs-Output ausgeben
    print(result.stdout)


def find_latest_pth(root: Path) -> Path | None:
    candidates = list(root.rglob("*.pth"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def main():
    parser = argparse.ArgumentParser(description="RVC training wrapper")
    parser.add_argument("--data_dir", required=True, help="Directory containing WAV dataset")
    parser.add_argument("--output", required=True, help="Path to write trained model")
    parser.add_argument("--rvc_cli", default=".", help="Path to RVC-WebUI repository root folder") 
    parser.add_argument("--model_name", default="", help="Optional model name (defaults to output stem)")
    parser.add_argument("--sr", type=int, default=48000, help="Sample rate for training (RVC default 48k)")
    parser.add_argument("--epochs", type=int, default=20, help="Total epochs to train")
    parser.add_argument("--batch_size", type=int, default=6, help="Batch size for training")
    parser.add_argument("--f0method", default="pm", help="F0 extraction method (pm|crepe|rmvpe etc.)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    out_path = Path(args.output).resolve()
    rvc_repo_dir = Path(args.rvc_cli).resolve()
    model_name = args.model_name or out_path.stem

    wavs = list(data_dir.glob("*.wav"))
    if not wavs:
        print(f"No WAV files found in {data_dir}", file=sys.stderr)
        sys.exit(1)

    if not rvc_repo_dir.is_dir():
        print(f"RVC-WebUI repository root not found at {rvc_repo_dir}. Set --rvc_cli to the repository path.", file=sys.stderr)
        sys.exit(1)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # RVC-WebUI Skripte
    PREPROCESS_SCRIPT = rvc_repo_dir / "trainset_preprocess_pipeline_print.py"
    EXTRACT_F0_SCRIPT = rvc_repo_dir / "extract_f0_print.py"
    EXTRACT_FEATURE_SCRIPT = rvc_repo_dir / "extract_feature_print.py"
    TRAIN_SCRIPT = rvc_repo_dir / "train_nsf_sim_vs_print.py"
    if not TRAIN_SCRIPT.exists():
        TRAIN_SCRIPT = rvc_repo_dir / "train_nsf_sim_cache_sid_load_pretrain.py"
    INDEX_SCRIPT = rvc_repo_dir / "infer" / "modules" / "train" / "index_make.py"

    if not PREPROCESS_SCRIPT.exists() or not TRAIN_SCRIPT.exists():
        print(f"Fehler: Notwendige RVC-Skripte nicht im Ordner {rvc_repo_dir} gefunden.", file=sys.stderr)
        sys.exit(1)

    dataset_dir = rvc_repo_dir / "logs" / model_name 
    
    # Python des RVC-Venv
    rvc_venv_python = rvc_repo_dir / "venv" / "Scripts" / "python.exe"
    if not rvc_venv_python.exists():
         rvc_venv_python = Path("python") 

    # 1) preprocess
    run_step(
        "preprocess",
        [str(rvc_venv_python), str(PREPROCESS_SCRIPT), str(data_dir), str(args.sr), "8", str(dataset_dir), "True", "3.7"],
    )

    # 2a) extract F0
    run_step(
        "extract F0",
        [str(rvc_venv_python), str(EXTRACT_F0_SCRIPT), str(dataset_dir), "8", args.f0method],
    )
    
    # 2b) extract feature
    run_step(
        "extract feature",
        [str(rvc_venv_python), str(EXTRACT_FEATURE_SCRIPT), "0", "8", "0", "0", str(dataset_dir), "2"],
    )

    # 3) train – CPU-Fallback (GPU-Flags entfernt), Save nach Epoche 1 (Test)
    run_step(
        "train",
        [str(rvc_venv_python), str(TRAIN_SCRIPT), 
         "-se", "1", 
         "-te", "5",
         "-bs", str(args.batch_size), 
         "-e", str(dataset_dir),
         "-sr", f"{args.sr // 1000}k", 
         "-sw", "1", 
         "-v", "2", 
         "-f0", "1", 
         "-l", "1",
         "-c", "0",  # mandatory: do not cache data in GPU (and satisfies required arg)
         "-pg", str(rvc_repo_dir / "pretrained" / f"f0G{args.sr // 1000}k.pth"),
         "-pd", str(rvc_repo_dir / "pretrained" / f"f0D{args.sr // 1000}k.pth")
        ],
    )
    
    # 4) index (optional)
    try:
        run_step(
            "index",
            [str(rvc_venv_python), str(INDEX_SCRIPT), str(dataset_dir), model_name, "0"],
        )
    except Exception as exc:  # optional
        print(f"[vc_train_tool] index step failed (continuing): {exc}", file=sys.stderr)

    candidate = find_latest_pth(rvc_repo_dir / "weights" / model_name)
    if candidate:
        shutil.copy2(candidate, out_path)
        print(f"[vc_train_tool] Copied trained model {candidate} -> {out_path}")
        sys.exit(0)

    print("[vc_train_tool] No .pth found after training.", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
