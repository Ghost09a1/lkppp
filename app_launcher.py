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
    print(f"[launcher] waiting for {host}:{port}...")
    while time.time() - start < timeout:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1)
            try:
                sock.connect((host, port))
                print(f"[launcher] {host}:{port} is ready.")
                return True
            except (ConnectionRefusedError, socket.timeout, OSError):
                time.sleep(0.5)
    print(f"[launcher] timeout waiting for {host}:{port}.")
    return False


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
    print(f"[launcher] starting {name} on {host}:{port}...")
    return subprocess.Popen(cmd, stdout=log_file.open("a", encoding="utf-8"), stderr=subprocess.STDOUT)


def main():
    cfg = load_config()
    logs_dir = ROOT / cfg["paths"]["logs_dir"]
    logs_dir.mkdir(parents=True, exist_ok=True)

    procs: List[subprocess.Popen] = []
    try:
        # LLM server is now started by start_all.ps1, so we just wait for it if needed.
        if cfg["llm"]["mode"] == "gguf":
            print("[launcher] GGUF mode detected. Waiting for LLaMA server to become available...")
            if not wait_for_port(
                cfg["llm"]["llama_cpp_server"]["port"],
                cfg["backend_host"],
                timeout=120, # Increased timeout for model loading
            ):
                print("[launcher] CRITICAL: LLaMA server did not respond in time.")
        else:
            print("[launcher] Ollama mode detected. Assuming local daemon is running.")

        if cfg["media"].get("tts_enabled"):
            tts_port = cfg["media"]["tts_port"]
            tts_proc = start_uvicorn(
                "backend.tts_server:app", "127.0.0.1", tts_port, logs_dir, "tts"
            )
            procs.append(tts_proc)
            wait_for_port(tts_port, timeout=30)
        else:
            print("[launcher] TTS is disabled in config, skipping.")

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
            timeout=60, # Increased timeout for backend startup
        ):
            url = f"http://{cfg.get('backend_host','127.0.0.1')}:{cfg.get('backend_port',8000)}/ui"
            if cfg["ui"].get("open_browser", True):
                print(f"[launcher] Backend is ready. Opening {url} in browser...")
                webbrowser.open(url)
            else:
                print(f"[launcher] Backend is ready at {url}")
        else:
            print("[launcher] CRITICAL: Backend failed to start within the timeout period.")

        # Keep this script alive to manage child processes
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[launcher] Shutdown signal received. Terminating processes...")
    finally:
        for proc in reversed(procs): # Terminate in reverse order of startup
            if proc and proc.poll() is None:
                print(f"[launcher] Terminating process {proc.pid}...")
                proc.terminate()
        for proc in procs:
            if proc:
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    print(f"[launcher] Process {proc.pid} did not terminate gracefully, killing it.")
                    proc.kill()
        print("[launcher] All processes have been shut down.")


if __name__ == "__main__":
    main()
