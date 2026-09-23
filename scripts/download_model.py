"""Download the exact public Laya base checkpoint used by the reference experiment."""
import hashlib
from pathlib import Path
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
REVISION = "1c5edc17a7acd8701df6fc341c0d179f1c62c982"
WEIGHTS_SHA256 = "891102d372688fc2a094dac56a384bc537b87c63f21f9f3dac0be2b7cbc8d86c"


def main():
    folder = Path(snapshot_download("convaiinnovations/laya", revision=REVISION,
        token=False, local_dir=ROOT / "models/laya",
        allow_patterns=["model.safetensors", "rl_agent_config.json", "encoder/*", "tokenizer/*"]))
    with (folder / "model.safetensors").open("rb") as handle:
        if hashlib.file_digest(handle, "sha256").hexdigest() != WEIGHTS_SHA256:
            raise RuntimeError("Checkpoint hash mismatch")
    print("Verified pinned Laya base weights. Launch with: python scripts/play.py --model laya")


if __name__ == "__main__":
    main()
