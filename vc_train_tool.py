"""
Thin wrapper to call the official RVC training scripts via subprocess.

This script calls the specific Python files from the official 
Retrieval-based-Voice-Conversion-WebUI repository, as the simple 'rvc-cli'
pip package does not support the necessary 'preprocess'/'train' commands.

Expected tool: The cloned RVC-Project/Retrieval-based-Voice-Conversion-WebUI repository.
Point --rvc_cli to the root folder of this cloned repository.

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

Notes:
- Ensure the RVC-WebUI repo and its dependencies are installed correctly.
- Flags here are minimal; adjust epochs/batch/f0method to your needs.
- The script now finds the model in the RVC repo's 'weights' folder.
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
    # Sucht rekursiv nach der neuesten .pth-Datei
    candidates = list(root.rglob("*.pth"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def main():
    parser = argparse.ArgumentParser(description="RVC training wrapper")
    parser.add_argument("--data_dir", required=True, help="Directory containing WAV dataset")
    parser.add_argument("--output", required=True, help="Path to write trained model")
    # Dieses Argument zeigt jetzt auf den RVC-WebUI REPOSITORY ROOT.
    parser.add_argument("--rvc_cli", default=".", help="Path to RVC-WebUI repository root folder") 
    parser.add_argument("--model_name", default="", help="Optional model name (defaults to output stem)")
    parser.add_argument("--sr", type=int, default=48000, help="Sample rate for training (RVC default 48k)")
    parser.add_argument("--epochs", type=int, default=20, help="Total epochs to train")
    parser.add_argument("--batch_size", type=int, default=6, help="Batch size for training")
    parser.add_argument("--f0method", default="pm", help="F0 extraction method (pm|crepe|rmvpe etc.)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    out_path = Path(args.output).resolve()
    # rvc_repo_dir ist jetzt der Root-Ordner des geklonten RVC-WebUI
    rvc_repo_dir = Path(args.rvc_cli).resolve()
    model_name = args.model_name or out_path.stem

    wavs = list(data_dir.glob("*.wav"))
    if not wavs:
        print(f"No WAV files found in {data_dir}", file=sys.stderr)
        sys.exit(1)

    # Prüfe nur, ob der Ordner existiert, nicht eine einzelne Datei
    if not rvc_repo_dir.is_dir():
        print(f"RVC-WebUI repository root not found at {rvc_repo_dir}. Set --rvc_cli to the repository path.", file=sys.stderr)
        sys.exit(1)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------------------------
    # --- NEUER RVC-WEBUI LOGIK-BLOCK (ersetzt rvc-cli Aufrufe) ---
    # ----------------------------------------------------------------------------------
    
    # Pfade zu den echten RVC-WebUI Skripten
    PREPROCESS_SCRIPT = rvc_repo_dir / "trainset_preprocess_pipeline_print.py"
    EXTRACT_F0_SCRIPT = rvc_repo_dir / "extract_f0_print.py"
    EXTRACT_FEATURE_SCRIPT = rvc_repo_dir / "extract_feature_print.py"
    TRAIN_SCRIPT = rvc_repo_dir / "train_nsf_sim_vs_print.py"
    INDEX_SCRIPT = rvc_repo_dir / "infer" / "modules" / "train" / "index_make.py"

    if not PREPROCESS_SCRIPT.exists():
        print(f"Fehler: Offizielles RVC-Skript {PREPROCESS_SCRIPT.name} nicht im Ordner {rvc_repo_dir} gefunden.", file=sys.stderr)
        print(f"Stelle sicher, dass --rvc_cli auf den RVC-WebUI Repo-Root zeigt und die Installation abgeschlossen ist.", file=sys.stderr)
        sys.exit(1)

    # Falls im RVC-Repo ein venv liegt, nutze dessen Python, damit die Abhängigkeiten (scipy etc.) sicher gefunden werden
    python_cmd = "python"
    venv_python = rvc_repo_dir / "venv" / "Scripts" / "python.exe"
    if venv_python.exists():
        python_cmd = str(venv_python)

    # ACHTUNG: Die echten RVC-Skripte verwenden oft positionelle Argumente und andere Flags als das rvc-cli!

    # Der Ordner, in dem die vorverarbeiteten Daten gespeichert werden (innerhalb RVC_REPO_DIR/dataset/)
    dataset_dir = rvc_repo_dir / "dataset" / model_name 

    # 1) preprocess
    # Befehl: python trainset_preprocess_pipeline_print.py {DATASET_INPUT_DIR} {SR}
    exp_dir = rvc_repo_dir / "logs" / model_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    run_step(
        "preprocess",
        [
            python_cmd,
            str(PREPROCESS_SCRIPT),
            str(data_dir),
            str(args.sr),
            "8",               # number of processes
            str(exp_dir),
            "True",            # no parallel? (script expects 'True'/'False')
            "3.7",             # per (segment length)
        ],
    )

    # 2a) extract F0 (Stimmtonhöhe)
    # Befehl: python extract_f0_print.py {DATASET_DIR} {F0_METHOD}
    run_step(
        "extract F0",
        [python_cmd, str(EXTRACT_F0_SCRIPT), str(dataset_dir), args.f0method],
    )
    
    # 2b) extract feature (Feature-Extraction)
    # Befehl: python extract_feature_print.py {DATASET_DIR} {RVC_MODEL_VERSION - 2 oder 3, 2 ist der default}
    run_step(
        "extract feature",
        [python_cmd, str(EXTRACT_FEATURE_SCRIPT), str(dataset_dir), "2"],
    )

    # 3) train
    # Befehl: python train_nsf_sim_vs_print.py {MODEL_NAME} {SR} {TotalEpoch} {BatchSize} {GPU_ID (hier 0)} {SaveEpoch}
    run_step(
        "train",
        [python_cmd, str(TRAIN_SCRIPT), model_name, str(args.sr), 
         str(args.epochs), str(args.batch_size), "0", "5"], # 0=GPU ID (muss bei nur einer GPU 0 sein), 5=Alle 5 Epochen speichern
    )
    
    # 4) index (optional)
    try:
        # Befehl: python infer/modules/train/index_make.py {FEAT_OUT_DIR} {MODEL_NAME}
        run_step(
            "index",
            [python_cmd, str(INDEX_SCRIPT), str(dataset_dir), model_name],
        )
    except Exception as exc:  # pragma: no cover - non-critical
        print(f"[vc_train_tool] index step failed (continuing): {exc}", file=sys.stderr)

    # Versuche, das trainierte Modell zu lokalisieren und in den angeforderten Output zu kopieren
    # RVC speichert die Modelle standardmäßig in RVC_REPO_DIR/weights/
    candidate = find_latest_pth(rvc_repo_dir / "weights" / model_name)
    if candidate:
        shutil.copy2(candidate, out_path)
        print(f"[vc_train_tool] Copied trained model {candidate} -> {out_path}")
        sys.exit(0)

    print("[vc_train_tool] No .pth found after training.", file=sys.stderr)
    sys.exit(1)
    # ----------------------------------------------------------------------------------
    # --- ENDE NEUER LOGIK-BLOCK ---
    # ----------------------------------------------------------------------------------


if __name__ == "__main__":
    main()
