import json
import subprocess
import sys
import time
import webbrowser
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).parent
CONFIG_PATH = ROOT / "config" / "settings.json"


def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def wait_for_port(port: int, host: str = "127.0.0.1", timeout: int = 30) -> bool:
    import socket

    start = time.time()
    while time.time() - start < timeout:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1)
            try:
                sock.connect((host, port))
                return True
            except (ConnectionRefusedError, socket.timeout, OSError):
                time.sleep(0.5)
    return False


def start_llama_server(cfg: dict, logs_dir: Path) -> Optional[subprocess.Popen]:
    binary = ROOT / cfg["llm"]["llama_cpp_server"]["binary_path"]
    model = ROOT / cfg["llm"]["llama_cpp_server"]["model_path"]
    port = cfg["llm"]["llama_cpp_server"]["port"]
    if not binary.exists() or not model.exists():
        print("[launcher] llama.cpp server skipped (binary or model missing)")
        return None
    cmd = [
        str(binary),
        "-m",
        str(model),
        "--port",
        str(port),
        "--ctx-size",
        str(cfg["llm"]["llama_cpp_server"]["n_ctx"]),
        "--threads",
        str(cfg["llm"]["llama_cpp_server"]["n_threads"]),
        "--batch-size",
        str(cfg["llm"]["llama_cpp_server"]["batch"]),
        "--host",
        cfg["backend_host"],
    ]
    if cfg["llm"]["llama_cpp_server"]["gpu_layers"] > 0:
        cmd += ["-ngl", str(cfg["llm"]["llama_cpp_server"]["gpu_layers"])]
    log_file = logs_dir / "llama_cpp.log"
    print(f"[launcher] starting llama.cpp server on port {port}")
    return subprocess.Popen(cmd, stdout=log_file.open("w"), stderr=subprocess.STDOUT)


def start_uvicorn(
    module_path: str, host: str, port: int, logs_dir: Path, name: str
) -> subprocess.Popen:
    log_file = logs_dir / f"{name}.log"
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        module_path,
        "--host",
        host,
        "--port",
        str(port),
    ]
    print(f"[launcher] starting {name} on {host}:{port}")
    return subprocess.Popen(cmd, stdout=log_file.open("w"), stderr=subprocess.STDOUT)


def main():
    cfg = load_config()
    logs_dir = ROOT / cfg["paths"]["logs_dir"]
    logs_dir.mkdir(parents=True, exist_ok=True)

    procs: List[subprocess.Popen] = []
    try:
        if cfg["llm"]["mode"] == "gguf":
            llama_proc = start_llama_server(cfg, logs_dir)
            if llama_proc:
                procs.append(llama_proc)
                if not wait_for_port(
                    cfg["llm"]["llama_cpp_server"]["port"],
                    cfg["backend_host"],
                    timeout=60,
                ):
                    print("[launcher] llama.cpp server did not respond in time")
        else:
            print("[launcher] using Ollama mode, assuming local daemon is running")

        if cfg["media"].get("tts_enabled"):
            tts_port = cfg["media"]["tts_port"]
            tts_proc = start_uvicorn(
                "backend.tts_server:app", "127.0.0.1", tts_port, logs_dir, "tts"
            )
            procs.append(tts_proc)
            wait_for_port(tts_port, timeout=20)
        else:
            print("[launcher] TTS disabled in config")

        backend_proc = start_uvicorn(
            "backend.core:app",
            cfg.get("backend_host", "127.0.0.1"),
            cfg.get("backend_port", 8000),
            logs_dir,
            "backend",
        )
        procs.append(backend_proc)
        if wait_for_port(
            cfg.get("backend_port", 8000),
            cfg.get("backend_host", "127.0.0.1"),
            timeout=30,
        ):
            url = f"http://{cfg.get('backend_host','127.0.0.1')}:{cfg.get('backend_port',8000)}/ui"
            if cfg["ui"].get("open_browser", True):
                webbrowser.open(url)
            print(f"[launcher] backend ready at {url}")
        else:
            print("[launcher] backend failed to start within timeout")

        # block until interrupted
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[launcher] shutting down...")
    finally:
        for proc in procs:
            if proc and proc.poll() is None:
                proc.terminate()
        for proc in procs:
            if proc:
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()


if __name__ == "__main__":
    main()
