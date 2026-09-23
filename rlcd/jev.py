"""TypeSafe's official typed-decision API; credentials stay on the server."""
from __future__ import annotations

import math
import os
import time

import httpx

ENDPOINT = "https://api.typesafe.ai/v1/systemone"
DEFAULT_MODEL = "jev-1.13.0"


class JevNotConfigured(RuntimeError):
    pass


def configured() -> bool:
    return bool(os.getenv("TYPESAFE_API_KEY") or os.getenv("JEV_API_KEY"))


def unit_number(value) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and 0 <= value <= 1)


def validate_answer(data: dict, actions: tuple[str, ...]) -> dict:
    """Reject malformed answers instead of substituting a rule-based action."""
    if not isinstance(data, dict):
        raise ValueError("Jev returned an invalid response")
    model = data.get("model")
    answers = data.get("answers")
    answer = answers.get("action") if isinstance(answers, dict) else None
    if (not isinstance(model, str) or not model.startswith("jev-")
            or not isinstance(answer, dict) or answer.get("type") != "choice"
            or answer.get("choice") not in actions):
        raise ValueError("Jev returned an invalid tactical action")
    probabilities = answer.get("probabilities")
    confidence = answer.get("confidence")
    if (not isinstance(probabilities, dict) or set(probabilities) != set(actions)
            or not all(unit_number(p) for p in probabilities.values())
            or not math.isclose(sum(probabilities.values()), 1, abs_tol=0.001)
            or not unit_number(confidence)):
        raise ValueError("Jev returned invalid confidence or probabilities")
    if probabilities[answer["choice"]] + 0.001 < max(probabilities.values()):
        raise ValueError("Jev's selected action does not match its probabilities")
    usage = data.get("usage")
    if not isinstance(usage, dict):
        usage = {}
    # Save only documented numeric usage, never opaque IDs or provider metadata.
    usage = {key: value for key, value in usage.items()
             if key in ("input_tokens", "output_tokens", "total_tokens")
             and isinstance(value, int) and not isinstance(value, bool) and value >= 0}
    return {"action": answer["choice"], "jev_model": model,
            "jev_confidence": confidence, "confidence_kind": "jev_reported",
            "probabilities": probabilities, "top_probability": max(probabilities.values()),
            "usage": usage}


async def predict(client: httpx.AsyncClient, state: str, question: dict, model: str) -> dict:
    key = os.getenv("TYPESAFE_API_KEY") or os.getenv("JEV_API_KEY")
    if not key:
        raise JevNotConfigured("Configure TYPESAFE_API_KEY on the local server to play with Jev")
    started = time.perf_counter()
    response = await client.post(
        ENDPOINT,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={"model": model, "state": state, "questions": {"action": question}},
    )
    response.raise_for_status()
    answer = validate_answer(response.json(), tuple(question["criteria"]))
    return {**answer, "jev_ms": (time.perf_counter() - started) * 1000}
