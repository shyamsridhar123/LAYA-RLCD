"""Launch Cinder Station; optionally load a local checkpoint or enable Jev."""
import argparse
import getpass
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("none", "laya", "decoder"), default="none")
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--jev", action="store_true", help="Prompt privately for a TypeSafe key if no key is in the environment")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    python = ROOT / ".venv" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    if not python.exists() or not (ROOT / "game/dist/index.html").exists():
        raise SystemExit("Run python scripts/setup.py first.")
    if not 1024 <= args.port <= 65535:
        raise SystemExit("Choose a port between 1024 and 65535.")
    env = os.environ.copy()
    env.update(RLCD_LOCAL_MODEL=args.model, RLCD_DEVICE=args.device, RLCD_API_PORT=str(args.port))
    if args.model != "none":
        if args.model == "decoder" and not args.model_dir:
            raise SystemExit("Decoder mode requires --model-dir pointing to the notebook's selected folder.")
        directory = (args.model_dir or ROOT / "models/laya").resolve()
        if not directory.is_dir():
            raise SystemExit("Model directory not found. Download Laya or unpack your Colab export first.")
        env["RLCD_MODEL_DIR"] = str(directory)
    if args.jev and not (env.get("TYPESAFE_API_KEY") or env.get("JEV_API_KEY")):
        env["TYPESAFE_API_KEY"] = getpass.getpass("TypeSafe API key (hidden; only kept in server memory): ").strip()
        if not env["TYPESAFE_API_KEY"]:
            raise SystemExit("No key entered.")
    print(f"Open http://127.0.0.1:{args.port} after the server reports startup complete.", flush=True)
    print("Ctrl+C stops the service. Jev requests use your TypeSafe account when selected.", flush=True)
    try:
        subprocess.run([str(python), "-m", "uvicorn", "rlcd.server:app", "--host", "127.0.0.1",
                        "--port", str(args.port), "--no-access-log"], cwd=ROOT, env=env, check=True)
    except KeyboardInterrupt:
        pass
    except subprocess.CalledProcessError as error:
        next_port = args.port + 1 if args.port < 65535 else 8765
        raise SystemExit(f"Service exited ({error.returncode}). If the port is busy, try --port {next_port}.") from None


if __name__ == "__main__":
    main()
