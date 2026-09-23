"""The exact question and action order used for training and gameplay."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = json.loads((ROOT / "data/protocol.json").read_text(encoding="utf-8"))
ACTIONS = tuple(PROTOCOL["actions"])
QUESTION = PROTOCOL["question"]


def decoder_prompt(state: str) -> str:
    options = "\n".join(f"{name}: {description}" for name, description in QUESTION["criteria"].items())
    return f"{QUESTION['instructions']}\nActions:\n{options}\nState: {state}\nDecision:"
