"""Set up the local game, CPU model inference, or a separate CUDA training environment."""
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
    parser.add_argument("--training", action="store_true", help="Prepare .venv-train for both models (NVIDIA CUDA, Windows/Linux; no Node.js needed)")
    args = parser.parse_args()
    if args.training and (args.models or args.download_laya):
        parser.error("Run --training separately from --models or --download-laya.")
    if not (3, 11) <= sys.version_info < (3, 14):
        raise SystemExit("Use Python 3.11–3.13 for this pinned recipe.")
    if args.training and sys.platform not in ("win32", "linux"):
        raise SystemExit("Training requires Windows or Linux with an NVIDIA CUDA GPU. Local game play and CPU inference are available on macOS.")
    npm = shutil.which("npm.cmd" if sys.platform == "win32" else "npm")
    if not args.training and not npm:
        raise SystemExit("Install Node.js 22 LTS from nodejs.org, then run this command again.")
    env = ROOT / (".venv-train" if args.training else ".venv")
    python = env / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    if not python.exists():
        venv.create(env, with_pip=True)
    def run(*command):
        subprocess.run([str(x) for x in command], cwd=ROOT, check=True)
    if args.training:
        print("Preparing CUDA training for LAYA and ModernBERT Decoder. The first install downloads several GB.", flush=True)
        constraint = env / "torch-constraint.txt"
        constraint.write_text("torch==2.11.0+cu128\n", encoding="utf-8")
        run(python, "-m", "pip", "install", "torch==2.11.0+cu128", "--index-url", "https://download.pytorch.org/whl/cu128")
        run(python, "-m", "pip", "install", "-c", constraint, "-r", "requirements-models.txt", "numpy==2.1.3")
        run(python, "-c", "import torch; "
            "assert torch.cuda.is_available(), 'CUDA is unavailable: use an NVIDIA GPU and a driver compatible with CUDA 12.8. See docs/LOCAL.md.'; "
            "x = torch.ones(1, device='cuda'); torch.cuda.synchronize(); "
            "print({'torch': torch.__version__, 'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0)})")
        print("Ready! Run: python scripts/train_local.py --model decoder")
        print("Or: python scripts/train_local.py --model laya")
        return
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
