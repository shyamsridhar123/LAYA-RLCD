"""Train and evaluate either public pilot locally, without activating a virtual environment."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
RECIPES = {
    "laya": ("finetune_colab.py", "finalize"),
    "decoder": ("modernbert_decoder_colab.py", "evaluate"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=RECIPES, required=True)
    parser.add_argument("--run-dir", type=Path, help="A new directory; defaults to runs/<model>")
    args = parser.parse_args()
    root = (ROOT / (args.run_dir or Path("runs") / args.model)).resolve()
    if root.exists():
        raise SystemExit("Run directory already exists. Choose a new --run-dir, or resume a phase using docs/LOCAL.md.")
    python = ROOT / ".venv-train" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    if not python.is_file():
        raise SystemExit("Run python scripts/setup.py --training first. See docs/LOCAL.md for GPU requirements.")
    env = os.environ.copy()
    env.update(USE_TF="0", USE_FLAX="0", TOKENIZERS_PARALLELISM="false",
               HF_HUB_DISABLE_IMPLICIT_TOKEN="1", HF_HUB_DISABLE_PROGRESS_BARS="1")
    script, final_phase = RECIPES[args.model]
    try:
        subprocess.run([str(python), "-c", "import torch; "
                        "assert torch.cuda.is_available(), 'An NVIDIA CUDA GPU is required'; "
                        "x = torch.ones(1, device='cuda'); torch.cuda.synchronize(); "
                        "print({'torch': torch.__version__, 'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0)})"],
                       cwd=ROOT, env=env, check=True)
        # copytree refuses to replace inputs, including if another run starts here.
        root.mkdir(parents=True, exist_ok=False)
        shutil.copytree(ROOT / "data", root / "data")
        for phase in ("train", final_phase):
            subprocess.run([str(python), str(ROOT / "scripts" / script), phase, "--root", str(root)],
                           cwd=ROOT, env=env, check=True)
    except subprocess.CalledProcessError as error:
        raise SystemExit(f"Training stopped (exit {error.returncode}). Any completed files remain in {root}. See docs/LOCAL.md.") from None
    except KeyboardInterrupt:
        raise SystemExit("Training interrupted. Any completed files remain in the run directory.") from None
    print(f"Checkpoint: {root / 'selected'}")
    print(f"Results: {root / 'results'}")
    print("To play with it, run python scripts/setup.py --models, then:")
    print(f'python scripts/play.py --model {args.model} --model-dir "{root / "selected"}"')


if __name__ == "__main__":
    main()
