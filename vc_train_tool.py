"""
Adapter wrapper to call the official RVC training scripts via subprocess.
...
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
    # Wir brauchen den RVC-Repo-Root
    rvc_repo_dir = rvc_venv_python_path.parent.parent.parent

    # Hier verwenden wir subprocess.run, um den Fehlercode direkt zu fangen.
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=rvc_repo_dir if rvc_repo_dir.is_dir() else None,
    )

    if result.returncode != 0:
        # Hier geben wir den Output aus, der ins Logfile landet
        print(f"RVC-Log (Stdout):\n{result.stdout}", file=sys.stderr)
        print(f"RVC-Fehler (Stderr):\n{result.stderr}", file=sys.stderr)
        raise RuntimeError(f"{name} failed with exit code {result.returncode}")

    # Gebe den erfolgreichen Output zurück, damit er im Hauptlog erscheint
    print(result.stdout)


def find_latest_pth(root: Path) -> Path | None:
    # Sucht rekursiv nach der neuesten .pth-Datei
    candidates = list(root.rglob("*.pth"))
    if not candidates:
        return None
    # Sortiert nach Änderungszeitpunkt (st_mtime)
    return max(candidates, key=lambda p: p.stat().st_mtime)


def main():
    parser = argparse.ArgumentParser(description="RVC training wrapper")
    parser.add_argument(
        "--data_dir", required=True, help="Directory containing WAV dataset"
    )
    parser.add_argument("--output", required=True, help="Path to write trained model")
    parser.add_argument(
        "--rvc_cli", default=".", help="Path to RVC-WebUI repository root folder"
    )
    parser.add_argument(
        "--model_name", default="", help="Optional model name (defaults to output stem)"
    )
    parser.add_argument(
        "--sr",
        type=int,
        default=48000,
        help="Sample rate for training (RVC default 48k)",
    )
    parser.add_argument("--epochs", type=int, default=20, help="Total epochs to train")
    parser.add_argument(
        "--batch_size", type=int, default=6, help="Batch size for training"
    )
    parser.add_argument(
        "--f0method", default="pm", help="F0 extraction method (pm|crepe|rmvpe etc.)"
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    out_path = Path(args.output).resolve()
    rvc_repo_dir = Path(args.rvc_cli).resolve()
    model_name = args.model_name or out_path.stem

    # ... (Pre-Checks und Pfad-Erstellung)

    # Pfade zu den echten RVC-WebUI Skripten (verkürzt)
    PREPROCESS_SCRIPT = rvc_repo_dir / "trainset_preprocess_pipeline_print.py"
    EXTRACT_F0_SCRIPT = rvc_repo_dir / "extract_f0_print.py"
    EXTRACT_FEATURE_SCRIPT = rvc_repo_dir / "extract_feature_print.py"
    TRAIN_SCRIPT = rvc_repo_dir / "train_nsf_sim_cache_sid_load_pretrain.py"
    # Index-Pfad wird nicht mehr benötigt, da der Schritt deaktiviert wird

    dataset_dir = rvc_repo_dir / "logs" / model_name
    rvc_venv_python = rvc_repo_dir / "venv" / "Scripts" / "python.exe"
    if not rvc_venv_python.exists():
        rvc_venv_python = Path("python")

    # 1) preprocess
#   '  run_step(
#         "preprocess",
#         [
#             str(rvc_venv_python),
#             str(PREPROCESS_SCRIPT),
#             str(data_dir),
#             str(args.sr),
#             "8",
#             str(dataset_dir),
#             "True",
#             "3.7",
#         ],
#     )

    # 2a) extract F0
    # run_step(
    #     "extract F0",
    #     [
    #         str(rvc_venv_python),
    #         str(EXTRACT_F0_SCRIPT),
    #         str(dataset_dir),
    #         "8",
    #         args.f0method,
    #     ],
    # )

    # 2b) extract feature
    # run_step(
    #     "extract feature",
    #     [
    #         str(rvc_venv_python),
    #         str(EXTRACT_FEATURE_SCRIPT),
    #         "0",
    #         "8",
    #         "0",
    #         "0",
    #         str(dataset_dir),
    #         "2",
    #     ],
    # )'

    # 3) train - Finaler Flags-Satz
    run_step(
        "train",
        [
            str(rvc_venv_python),
            str(TRAIN_SCRIPT),
            "-se",
            "1",
            "-te",
            str(args.epochs),
            "-bs",
            str(args.batch_size),
            "-e",
            str(dataset_dir),
            "-sr",
            f"{args.sr // 1000}k",
            "-sw",
            "1",
            "-v",
            "2",
            "-f0",
            "1",
            "-l",
            "1",
            "-c",
            "0",
            "-g",
            "0",
            "-pg",
            str(rvc_repo_dir / "pretrained" / f"f0G{args.sr // 1000}k.pth"),
            "-pd",
            str(rvc_repo_dir / "pretrained" / f"f0D{args.sr // 1000}k.pth"),
        ],
    )

    # 4) index (optional but handy)
    # --- PATCH: Index-Schritt deaktiviert, da 'index_make.py' nicht gefunden wird. ---
    print("[vc_train_tool] Index step skipped due to missing script.", file=sys.stderr)
    # --- ENDE PATCH ---

    # --- PATCH: Verbesserte Modellsuche (zuerst im Logs-Ordner) ---
    candidate = find_latest_pth(
        dataset_dir
    )  # Suche in logs/char_1 (wo das Training speichert)

    if not candidate:
        # Fallback: Suche im RVC-Standard-Gewichtsordner (manche Forks verschieben sie)
        candidate = find_latest_pth(rvc_repo_dir / "weights" / model_name)
    # --- ENDE PATCH ---

    if candidate:
        shutil.copy2(candidate, out_path)
        print(f"[vc_train_tool] Copied trained model {candidate} -> {out_path}")
        sys.exit(0)

    print("[vc_train_tool] No .pth found after training.", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
