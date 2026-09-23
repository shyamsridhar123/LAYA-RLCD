"""Cross-platform setup: python scripts/setup.py [--models] [--download-laya]."""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import venv

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", action="store_true", help="Install optional local inference dependencies (CPU PyTorch)")
    parser.add_argument("--download-laya", action="store_true", help="Also download the pinned public Laya base checkpoint (~843 MB)")
    args = parser.parse_args()
    if not (3, 11) <= sys.version_info < (3, 14):
        raise SystemExit("Use Python 3.11–3.13 for this pinned recipe.")
    npm = shutil.which("npm.cmd" if sys.platform == "win32" else "npm")
    if not npm:
        raise SystemExit("Install Node.js 22 LTS from nodejs.org, then run this command again.")
    env = ROOT / ".venv"
    python = env / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    if not python.exists():
        venv.create(env, with_pip=True)
    def run(*command):
        subprocess.run([str(x) for x in command], cwd=ROOT, check=True)
    if args.models or args.download_laya:
        run(python, "-m", "pip", "install", "torch==2.8.0", "--index-url", "https://download.pytorch.org/whl/cpu")
        run(python, "-m", "pip", "install", "-r", "requirements-models.txt")
    else:
        run(python, "-m", "pip", "install", "-r", "requirements.txt")
    run(npm, "ci")
    run(npm, "run", "build")
    if args.download_laya:
        run(python, "scripts/download_model.py")
    print("Ready! Run: python scripts/play.py")
    print("Optional local models: --model laya, or --model decoder --model-dir <selected-folder>")


if __name__ == "__main__":
    main()
