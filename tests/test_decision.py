import asyncio
import copy
import json
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient

from rlcd import jev, server
from rlcd.decision import DecisionRuntime, checked_probabilities
from rlcd.protocol import ACTIONS, QUESTION, decoder_prompt


def answer():
    return {"model": "jev-1.13.0", "answers": {"action": {
        "type": "choice", "choice": "heal", "confidence": 0.9,
        "probabilities": dict(zip(ACTIONS, [0.02, 0.92, 0.02, 0.02, 0.02]))}},
        "usage": {"input_tokens": 50, "opaque": "must-never-be-returned"}}


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    for name in ("TYPESAFE_API_KEY", "JEV_API_KEY", "RLCD_LOCAL_MODEL", "RLCD_MODEL_DIR"):
        monkeypatch.delenv(name, raising=False)


def test_official_jev_request_and_metadata_filter(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-placeholder")
    calls = []
    def handler(request):
        calls.append(request)
        assert str(request.url) == jev.ENDPOINT
        assert request.headers["authorization"] == "Bearer test-placeholder"
        assert json.loads(request.content) == {"model": jev.DEFAULT_MODEL, "state": "state", "questions": {"action": QUESTION}}
        return httpx.Response(200, json=answer())
    async def check():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            result = await jev.predict(client, "state", QUESTION, jev.DEFAULT_MODEL)
        assert result["action"] == "heal"
        assert result["usage"] == {"input_tokens": 50}
        assert "test-placeholder" not in json.dumps(result)
    asyncio.run(check())
    assert len(calls) == 1


@pytest.mark.parametrize("mutation", [
    lambda d: d.update(model="unknown"),
    lambda d: d["answers"]["action"].update(choice="teleport"),
    lambda d: d["answers"]["action"].update(choice="attack"),
    lambda d: d["answers"]["action"].update(confidence=float("nan")),
    lambda d: d["answers"]["action"].update(confidence=True),
    lambda d: d["answers"]["action"]["probabilities"].update(heal=0.3),
    lambda d: d["answers"]["action"]["probabilities"].pop("heal"),
])
def test_malformed_jev_output_stops(mutation):
    data = copy.deepcopy(answer())
    mutation(data)
    with pytest.raises(ValueError):
        jev.validate_answer(data, ACTIONS)


def test_local_distribution_is_checked():
    with pytest.raises(ValueError):
        checked_probabilities(dict.fromkeys(ACTIONS, 0.2), "teleport")
    with pytest.raises(ValueError):
        checked_probabilities({**dict.fromkeys(ACTIONS, 0.2), "heal": float("nan")}, "heal")


def test_default_is_lightweight_and_modes_are_explicit():
    runtime = DecisionRuntime()
    runtime.load()
    assert runtime.agent is runtime.model is None
    assert runtime.available_modes() == []
    async def check():
        with pytest.raises(ValueError):
            await runtime.decide("state", "decoder")
        await runtime.client.aclose()
    asyncio.run(check())


def test_service_errors_do_not_expose_provider_bodies(monkeypatch):
    monkeypatch.setattr(server.runtime, "available_modes", lambda: ["jev"])
    async def fail(*args):
        raise ValueError("a-provider-error-containing-sensitive-text")
    monkeypatch.setattr(server.runtime, "decide", fail)
    client = TestClient(server.app, base_url="http://127.0.0.1:8765")
    response = client.post("/api/decision", json={"state": "state", "mode": "jev"})
    assert response.status_code == 502
    assert "sensitive-text" not in response.text
    assert "no action was substituted" in response.text
    assert client.post("/api/decision", json={"state": "state", "mode": "unknown"}).status_code == 422
    assert client.post("/api/decision", json={"state": "state", "mode": "laya"}).status_code == 409
    assert client.post("/api/decision", headers={"origin": "https://example.com"}, json={"state": "state", "mode": "jev"}).status_code == 403


def test_decoder_uses_training_prompt_and_keeps_complete_state():
    torch = pytest.importorskip("torch")
    runtime = DecisionRuntime()
    captured = []
    class Tokenizer:
        def __call__(self, text, **kwargs):
            captured.append((text, kwargs))
            return {"input_ids": torch.tensor([[1, 2, 3]]), "attention_mask": torch.ones(1, 3, dtype=torch.long)}
    class Model:
        def __call__(self, **kwargs):
            assert kwargs["use_cache"] is False
            return SimpleNamespace(logits=torch.tensor([[0., 6., 0., 0., 0.]]))
    runtime.tokenizer, runtime.model = Tokenizer(), Model()
    result = runtime.predict_local("health 24; one medkit", "decoder")
    assert result["action"] == "heal"
    assert captured == [(decoder_prompt("health 24; one medkit"), {"return_tensors": "pt", "truncation": False})]
    assert captured[0][0].endswith("State: health 24; one medkit\nDecision:")
    asyncio.run(runtime.client.aclose())
