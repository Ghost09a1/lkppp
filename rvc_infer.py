"""
Minimal inference wrapper for RVC CLI.

Expected usage (called from TTS hook):
  python rvc_infer.py --model <model.pth> --input <in.wav> --output <out.wav> --ref <ref.wav> [--rvc_cli path/to/rvc_cli.py]

Behavior:
- If rvc_cli.py exists and the infer command succeeds, the converted audio is written to --output.
- If rvc_cli.py is missing or the call fails, the script copies --input to --output so TTS doesn’t break.

Replace this with your real inference call once you have a working RVC setup.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="RVC inference wrapper")
    parser.add_argument("--model", required=True, help="Path to trained RVC model (.pth)")
    parser.add_argument("--input", required=True, help="Input WAV")
    parser.add_argument("--output", required=True, help="Output WAV")
    parser.add_argument("--ref", default="", help="Reference WAV (if your pipeline uses it)")
    parser.add_argument("--rvc_cli", default="rvc_cli.py", help="Path to rvc_cli.py")
    args = parser.parse_args()

    model = Path(args.model)
    inp = Path(args.input)
    out = Path(args.output)
    rvc_cli = Path(args.rvc_cli)

    # Fallback: if any critical file is missing, just copy input->output
    if not model.exists() or not inp.exists() or not rvc_cli.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(inp, out)
        return

    if rvc_cli.suffix.lower() == ".py":
        cmd = ["python", str(rvc_cli), "infer"]
    else:
        cmd = [str(rvc_cli), "infer"]
    cmd += ["--model", str(model), "--input", str(inp), "--output", str(out)]
    if args.ref:
        cmd += ["--ref", args.ref]

    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0 or not out.exists():
        # fallback to passthrough if conversion failed
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(inp, out)


if __name__ == "__main__":
    main()
