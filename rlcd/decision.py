"""Local Laya / ModernBERT Decoder inference and optional hosted Jev.

Imports for local models are lazy: manual play and Jev need no PyTorch.
"""
from __future__ import annotations

import asyncio
import math
import os
from pathlib import Path
import threading
import time

import httpx

from . import jev
from .protocol import ACTIONS, QUESTION, ROOT, decoder_prompt


def checked_probabilities(probabilities: dict, choice: str) -> dict:
    if (choice not in ACTIONS or not isinstance(probabilities, dict)
            or set(probabilities) != set(ACTIONS)
            or not all(jev.unit_number(p) for p in probabilities.values())
            or not math.isclose(sum(probabilities.values()), 1, abs_tol=0.001)
            or probabilities[choice] + 0.001 < max(probabilities.values())):
        raise ValueError("Invalid action distribution; no substitute action was used")
    return probabilities


class DecisionRuntime:
    def __init__(self):
        self.local_mode = os.getenv("RLCD_LOCAL_MODEL", "none")
        self.device = os.getenv("RLCD_DEVICE", "cpu")
        self.model_dir = Path(os.getenv("RLCD_MODEL_DIR", str(ROOT / "models/laya")))
        self.jev_model = os.getenv("JEV_MODEL", jev.DEFAULT_MODEL)
        self.agent = self.model = self.tokenizer = None
        self.metadata = {"local_mode": self.local_mode, "jev_model": self.jev_model}
        self.lock = threading.Lock()
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(90, connect=10))

    def load(self):
        if self.local_mode == "none":
            return
        if self.local_mode not in ("laya", "decoder"):
            raise ValueError("RLCD_LOCAL_MODEL must be none, laya, or decoder")
        if not self.model_dir.is_dir():
            raise ValueError("Model directory missing. Download Laya or export a trained checkpoint first.")
        os.environ.update(USE_TF="0", USE_FLAX="0", TOKENIZERS_PARALLELISM="false")
        import torch
        torch.set_num_threads(int(os.getenv("RLCD_THREADS", "4")))
        if self.device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is unavailable; use --device cpu")
        if self.local_mode == "laya":
            from laya.agent import Agent
            self.agent = Agent(str(self.model_dir), device=self.device)
            self.metadata.update(device=str(self.agent.device),
                                 fine_tuned=bool(self.agent.cfg.get("fine_tuned", False)),
                                 parameters=sum(p.numel() for p in self.agent.model.parameters()))
        else:
            from transformers import AutoTokenizer, ModernBertDecoderForSequenceClassification
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, local_files_only=True, trust_remote_code=False)
            self.tokenizer.padding_side = "right"
            self.model = ModernBertDecoderForSequenceClassification.from_pretrained(
                self.model_dir, local_files_only=True, trust_remote_code=False,
                attn_implementation="sdpa", use_cache=False, dtype=torch.float32).to(self.device).eval()
            if [self.model.config.id2label.get(i) for i in range(len(ACTIONS))] != list(ACTIONS):
                raise ValueError("Checkpoint action labels do not match the Colab export")
            if self.model.config.num_labels != len(ACTIONS) or self.tokenizer.pad_token_id is None:
                raise ValueError("Checkpoint must have five actions and a padding token")
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
            self.metadata.update(device=self.device, parameters=sum(p.numel() for p in self.model.parameters()))

    def available_modes(self):
        modes = []
        if self.agent is not None:
            modes.append("laya")
        if self.model is not None:
            modes.append("decoder")
        if jev.configured():
            modes.append("jev")
        return modes

    def predict_local(self, state: str, mode: str) -> dict:
        with self.lock:
            started = time.perf_counter()
            if mode == "laya" and self.agent is not None:
                answer = self.agent.predict(state, {"action": QUESTION})["answers"]["action"]
                choice = answer["choice"]
                probabilities = checked_probabilities(answer["probabilities"], choice)
                confidence = answer["confidence"]
                if not jev.unit_number(confidence):
                    raise ValueError("Invalid Laya confidence")
            elif mode == "decoder" and self.model is not None:
                import torch
                inputs = self.tokenizer(decoder_prompt(state), return_tensors="pt", truncation=False)
                if inputs["input_ids"].shape[-1] > 512:
                    raise ValueError("State exceeds the training length; refusing to truncate it")
                inputs = {key: value.to(self.device) for key, value in inputs.items()}
                with torch.inference_mode():
                    scores = self.model(**inputs, use_cache=False).logits.float()[0].softmax(-1).cpu().tolist()
                choice = ACTIONS[max(range(len(scores)), key=scores.__getitem__)]
                probabilities = checked_probabilities(dict(zip(ACTIONS, scores)), choice)
                confidence = 1 + sum(p * math.log(p) for p in scores if p > 0) / math.log(len(scores))
                confidence = max(0., min(1., confidence))
            else:
                raise ValueError("Requested local model is not loaded")
            elapsed = (time.perf_counter() - started) * 1000
        return {"action": choice, "probabilities": probabilities,
                "top_probability": max(probabilities.values()), "entropy_confidence": confidence,
                "confidence_kind": "normalized_entropy", f"{mode}_ms": elapsed}

    async def decide(self, state: str, mode: str) -> dict:
        if mode not in self.available_modes():
            raise ValueError("Requested model is not configured")
        started = time.perf_counter()
        if mode == "jev":
            result = await jev.predict(self.client, state, QUESTION, self.jev_model)
        else:
            result = await asyncio.to_thread(self.predict_local, state, mode)
        return {**result, "source": mode, "mode": mode, "escalated": False,
                "elapsed_ms": (time.perf_counter() - started) * 1000}
