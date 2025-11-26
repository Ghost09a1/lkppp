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
import os
import shutil
import subprocess
import sys
from pathlib import Path


def run_step(name: str, cmd: list[str], workdir: Path | None = None) -> None:
    print(f"[vc_train_tool] {name}: {' '.join(cmd)}")
    env = os.environ.copy()
    # Force old torch.load behaviour; the upstream scripts expect pickled full objects
    env["TORCH_LOAD_WEIGHTS_ONLY"] = "0"

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=str(workdir) if workdir else None,
        env=env,
    )
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr, file=sys.stderr)
        raise RuntimeError(f"{name} failed with exit code {result.returncode}")


def find_latest_pth(*roots: Path) -> Path | None:
    """Sucht rekursiv nach der neuesten .pth-Datei in den angegebenen Wurzeln."""
    candidates: list[Path] = []
    for root in roots:
        if root and root.exists():
            candidates.extend(root.rglob("*.pth"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def write_filelist(exp_dir: Path, rvc_repo_dir: Path, version: str = "v2", if_f0: bool = True, sr_token: str = "48k") -> None:
    """Erzeuge filelist.txt wie in infer-web.py, damit das Trainingsscript Daten findet."""
    gt_wavs_dir = exp_dir / "0_gt_wavs"
    feature_dir = exp_dir / ("3_feature256" if version == "v1" else "3_feature768")
    names = set([p.stem for p in gt_wavs_dir.glob("*.wav")]) & set([p.stem for p in feature_dir.glob("*.npy")])

    def _strip_wav_suffix(name: str) -> str:
        return name[:-4] if name.endswith(".wav") else name

    lines: list[str] = []
    if if_f0:
        f0_dir = exp_dir / "2a_f0"
        f0nsf_dir = exp_dir / "2b-f0nsf"
        f0_names = {_strip_wav_suffix(p.stem) for p in f0_dir.glob("*.npy")}
        f0nsf_names = {_strip_wav_suffix(p.stem) for p in f0nsf_dir.glob("*.npy")}
        names = names & f0_names & f0nsf_names
        for name in names:
            lines.append(
                f"{gt_wavs_dir}\\{name}.wav|{feature_dir}\\{name}.npy|{f0_dir}\\{name}.wav.npy|{f0nsf_dir}\\{name}.wav.npy|0"
            )
        # optional: füge mute-Einträge hinzu, wenn vorhanden
        mute_gt = rvc_repo_dir / "logs" / "mute" / "0_gt_wavs" / f"mute{sr_token}.wav"
        mute_fea = rvc_repo_dir / "logs" / "mute" / ("3_feature256" if version == "v1" else "3_feature768") / "mute.npy"
        mute_f0 = rvc_repo_dir / "logs" / "mute" / "2a_f0" / "mute.wav.npy"
        mute_f0nsf = rvc_repo_dir / "logs" / "mute" / "2b-f0nsf" / "mute.wav.npy"
        if mute_gt.exists() and mute_fea.exists() and mute_f0.exists() and mute_f0nsf.exists():
            lines.append(f"{mute_gt}|{mute_fea}|{mute_f0}|{mute_f0nsf}|0")
            lines.append(f"{mute_gt}|{mute_fea}|{mute_f0}|{mute_f0nsf}|0")
    else:
        for name in names:
            lines.append(f"{gt_wavs_dir}\\{name}.wav|{feature_dir}\\{name}.npy|0")
    filelist = exp_dir / "filelist.txt"
    filelist.write_text("\n".join(lines), encoding="utf-8")
    print(f"[vc_train_tool] wrote filelist with {len(lines)} entries -> {filelist}")


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
    TRAIN_SCRIPT = rvc_repo_dir / "train_nsf_sim_cache_sid_load_pretrain.py"
    CONFIG_48K = rvc_repo_dir / "configs" / "48k.json"
    CONFIG_48000 = rvc_repo_dir / "configs" / "48000.json"
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
    dataset_dir.mkdir(parents=True, exist_ok=True)

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
        workdir=rvc_repo_dir,
    )

    # 2a) extract F0 (Stimmtonhöhe)
    # Befehl: python extract_f0_print.py {EXP_DIR} {N_PROCESSES} {F0_METHOD}
    n_processes = 8
    run_step(
        "extract F0",
        [python_cmd, str(EXTRACT_F0_SCRIPT), str(exp_dir), str(n_processes), args.f0method],
        workdir=rvc_repo_dir,
    )
    
    # 2b) extract feature (Feature-Extraction)
    # Befehl: python extract_feature_print.py {DATASET_DIR} {RVC_MODEL_VERSION - 2 oder 3, 2 ist der default}
    run_step(
        "extract feature",
        [
            python_cmd,
            str(EXTRACT_FEATURE_SCRIPT),
            "0",  # placeholder/device
            str(n_processes),
            "0",  # i_part
            "0",  # i_gpu
            str(exp_dir),
            "2",  # rvc model version
        ],
        workdir=rvc_repo_dir,
    )

    # Filelist erzeugen für das Trainingsscript
    write_filelist(exp_dir, rvc_repo_dir, version="v2", if_f0=True, sr_token="48k")

    # 3) train
    # Script requires many flags; mirror RVC cmd.txt style
    # Some RVC scripts expect configs/48000.json to exist; copy from 48k.json if missing.
    if not CONFIG_48000.exists() and CONFIG_48K.exists():
        CONFIG_48000.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(CONFIG_48K, CONFIG_48000)

    # Pretrain-G/D Pfade auf die realen 48k-Weights setzen (müssen vom Nutzer ins Repo gelegt werden)
    pretrained_g = rvc_repo_dir / "pretrained" / "f0G48k.pth"
    pretrained_d = rvc_repo_dir / "pretrained" / "f0D48k.pth"
    if not pretrained_g.exists() or not pretrained_d.exists():
        print(f"[vc_train_tool] WARNING: pretrained weights not found at {pretrained_g} / {pretrained_d}. Please download them into rvc_webui/pretrained/.", file=sys.stderr)

    run_step(
        "train",
        [
            python_cmd,
            str(TRAIN_SCRIPT),
            "-se",
            "5",  # save_every_epoch
            "-te",
            str(args.epochs),
            "-bs",
            str(args.batch_size),
            "-e",
            str(exp_dir),
            "-sr",
            "48k",  # use RVC's expected sample-rate token
            "-sw",
            "1",  # save_every_weights (store pth in weights/)
            "-v",
            "2",  # version 2
            "-f0",
            "1",  # use f0
            "-l",
            "1",  # use latest ckpt if exists
            "-c",
            "0",  # cache data in GPU? 0=no
            "-g",
            "0",  # GPU id list
            "-pg",
            str(pretrained_g),
            "-pd",
            str(pretrained_d),
        ],
        workdir=rvc_repo_dir,
    )
    
    # 4) index (optional)
    if INDEX_SCRIPT.exists():
        try:
            # Befehl: python infer/modules/train/index_make.py {FEAT_OUT_DIR} {MODEL_NAME}
            run_step(
                "index",
                [python_cmd, str(INDEX_SCRIPT), str(dataset_dir), model_name],
                workdir=rvc_repo_dir,
            )
        except Exception as exc:  # pragma: no cover - non-critical
            print(f"[vc_train_tool] index step failed (continuing): {exc}", file=sys.stderr)
    else:
        print(f"[vc_train_tool] index script not found at {INDEX_SCRIPT}, skipping index.", file=sys.stderr)

    # Versuche, das trainierte Modell zu lokalisieren und in den angeforderten Output zu kopieren
    # RVC speichert die Modelle standardmäßig in RVC_REPO_DIR/weights/
    candidate = find_latest_pth(
        rvc_repo_dir / "weights" / model_name,
        rvc_repo_dir / "weights",
        exp_dir,
    )
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
