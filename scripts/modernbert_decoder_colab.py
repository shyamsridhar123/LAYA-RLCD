"""Colab feasibility experiment: Laya's RLCD+CE objective on Ettin's causal decoder.

This is a five-action classifier, not token-level language-model RL. The pretrained
decoder reads the complete question, options and state before the classification
head scores the last non-padding token. Run train, then evaluate, in a fresh root.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import platform
import random
import time

os.environ.update(USE_TF="0", USE_FLAX="0", TOKENIZERS_PARALLELISM="false",
                  HF_HUB_DISABLE_IMPLICIT_TOKEN="1", HF_HUB_DISABLE_PROGRESS_BARS="1")

import numpy as np
import torch
from laya.common import QTYPES, proper_reward
from transformers import AutoTokenizer, ModernBertDecoderForSequenceClassification

MODEL_ID = "jhu-clsp/ettin-decoder-17m"
REVISION = "728fb5b6b3dc5916aa20829c027143ae8eca4eeb"
MAX_LENGTH = 512


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def log(value):
    print(json.dumps(value, allow_nan=False), flush=True)


def read_split(root, protocol, name):
    path = root / "data" / f"{name}.json"
    assert sha(path) == protocol["splits"][name]["sha256"], f"Changed split: {name}"
    rows = json.loads(path.read_text(encoding="utf-8"))
    assert len(rows) == protocol["splits"][name]["n"]
    return rows


def prompt(row, question):
    # The expected action is never part of the input. Every option is visible
    # before the final token, unlike Laya's option markers preceding the state.
    options = "\n".join(f"{name}: {description}" for name, description in question["criteria"].items())
    return f"{question['instructions']}\nActions:\n{options}\nState: {row['observation']['text']}\nDecision:"


def encode(tokenizer, rows, protocol):
    encoded = tokenizer([prompt(row, protocol["question"]) for row in rows],
                        truncation=False, add_special_tokens=True)
    lengths = [len(ids) for ids in encoded["input_ids"]]
    assert max(lengths) <= MAX_LENGTH, "Refusing to truncate away state or action information"
    assert all(tokenizer.pad_token_id not in ids for ids in encoded["input_ids"])
    return [{"input_ids": ids, "attention_mask": mask,
             "label": protocol["actions"].index(row["expected"])}
            for ids, mask, row in zip(encoded["input_ids"], encoded["attention_mask"], rows)]


def batch(tokenizer, items, device="cuda"):
    tensors = tokenizer.pad([{k: row[k] for k in ("input_ids", "attention_mask")} for row in items],
                            padding=True, return_tensors="pt")
    return {k: value.to(device) for k, value in tensors.items()}


def forward(model, inputs):
    # This small model comfortably fits a T4. FP32 avoids the gradient overflows
    # observed in the preliminary FP16 run as exploration sigma became smaller.
    return model(**inputs, use_cache=False).logits.float()


def rlcd_loss(logits, labels, sigma, samples=4):
    """Upstream noisy-logit score-function estimator plus categorical CE.

    Detaching sampled locations is essential: otherwise their dependence on the
    policy mean cancels the score-function gradient. Rewards/advantages are also
    detached; gradients flow through the Gaussian log probability and CE.
    """
    assert sigma > 0 and samples >= 2
    mask = torch.ones_like(logits, dtype=torch.bool)
    targets = torch.nn.functional.one_hot(labels, logits.shape[-1]).float()
    noise = torch.randn((samples,) + logits.shape, device=logits.device) * sigma
    noise = noise - noise.mean(-1, keepdim=True)
    sampled = logits.detach().unsqueeze(0) + noise
    with torch.no_grad():
        probabilities = sampled.softmax(-1)
        kinds = torch.full_like(labels, QTYPES["choice"])
        rewards = proper_reward(probabilities, targets.unsqueeze(0), kinds, mask, w_sph=.75, w_rps=1.)
        advantages = rewards - rewards.mean(0, keepdim=True)
        advantages = advantages / (advantages.std() + 1e-6)
    log_probability = -((sampled - logits.unsqueeze(0)).square()).sum(-1) / (2 * sigma ** 2)
    rl = -(advantages * log_probability).mean()
    ce = torch.nn.functional.cross_entropy(logits, labels)
    return rl, ce, rewards.mean()


@torch.no_grad()
def get_logits(model, tokenizer, items):
    model.eval()
    return torch.cat([forward(model, batch(tokenizer, items[i:i + 8])).cpu()
                      for i in range(0, len(items), 8)])


def metrics(logits, labels, actions):
    probabilities = logits.softmax(-1)
    predicted = probabilities.argmax(-1)
    correct = predicted.eq(labels)
    return {"n": len(labels), "correct": int(correct.sum()), "accuracy": float(correct.float().mean()),
            "nll": float(torch.nn.functional.cross_entropy(logits, labels)),
            "brier": float((probabilities - torch.nn.functional.one_hot(labels, len(actions))).square().sum(-1).mean()),
            "per_action": {action: {"n": int((labels == i).sum()),
                                     "correct": int((correct & (labels == i)).sum())}
                           for i, action in enumerate(actions)}}


def load_pretrained(actions):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION, token=False, trust_remote_code=False)
    tokenizer.padding_side = "right"
    model, info = ModernBertDecoderForSequenceClassification.from_pretrained(
        MODEL_ID, revision=REVISION, num_labels=len(actions), token=False, trust_remote_code=False,
        attn_implementation="sdpa", use_cache=False, output_loading_info=True,
        id2label=dict(enumerate(actions)), label2id={a: i for i, a in enumerate(actions)}, dtype=torch.float32)
    assert model.config.model_type == "modernbert-decoder"
    assert not any(key.startswith("model.") for key in info.get("missing_keys", [])), info
    assert not info.get("mismatched_keys") and not info.get("error_msgs"), info
    assert all(key.startswith(("head.", "classifier.")) for key in info.get("missing_keys", [])), info
    assert all(key.startswith(("lm_head.", "decoder.")) for key in info.get("unexpected_keys", [])), info
    model.config.pad_token_id = tokenizer.pad_token_id
    # Transformers 5 returns sets in its loading report.
    info = {key: sorted(value) if isinstance(value, set) else value for key, value in info.items()}
    return model.cuda(), tokenizer, info


def train(root):
    assert torch.cuda.is_available(), "Select a Colab GPU runtime before running this experiment"
    output = root / "results"
    output.mkdir(exist_ok=True)
    assert not (output / "training.json").exists(), "Use a fresh root for a new experiment"
    protocol = json.loads((root / "data/protocol.json").read_text())
    config = protocol["training"]
    actions = protocol["actions"]
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    props = torch.cuda.get_device_properties(0)
    environment = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   "python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
                   "gpu": props.name, "gpu_memory_gib": props.total_memory / 2**30,
                   "model_id": MODEL_ID, "revision": REVISION, "script_sha256": sha(__file__),
                   "precision": "FP32 weights, forward and backward; no autocast",
                   "protocol_sha256": sha(root / "data/protocol.json"),
                   "packages": {name: importlib.metadata.version(name) for name in
                                ("laya", "transformers", "huggingface_hub", "safetensors", "numpy")},
                   "proper_reward_source_sha256": hashlib.sha256(inspect.getsource(proper_reward).encode()).hexdigest()}
    started = time.perf_counter()
    model, tokenizer, loading = load_pretrained(actions)
    environment.update(parameters=sum(p.numel() for p in model.parameters()),
                       download_and_load_seconds=time.perf_counter() - started, loading_info=loading)
    save_json(output / "environment.json", environment)
    log({"environment": environment})
    # Only training and validation labels are opened in this phase.
    train_rows = read_split(root, protocol, "train")
    val_rows = read_split(root, protocol, "validation")
    group = lambda row: tuple(row["observation"][key] for key in protocol["group_key"])
    assert not ({group(row) for row in train_rows} & {group(row) for row in val_rows})
    train_items, val_items = encode(tokenizer, train_rows, protocol), encode(tokenizer, val_rows, protocol)
    val_labels = torch.tensor([item["label"] for item in val_items])
    baseline = metrics(get_logits(model, tokenizer, val_items), val_labels, actions)
    model.save_pretrained(root / "initial")
    tokenizer.save_pretrained(root / "initial")
    log({"initial_validation": baseline, "sequence_tokens": {
        "min": min(len(item["input_ids"]) for item in train_items),
        "max": max(len(item["input_ids"]) for item in train_items)}})

    model.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    probe_name, backbone_probe = next((name, param) for name, param in model.named_parameters()
                                      if name.startswith("model.layers.") and param.ndim == 2)
    head_probe = model.classifier.weight
    backbone_before = backbone_probe.detach().cpu().clone()
    head_before = head_probe.detach().cpu().clone()
    probe_logits = forward(model, batch(tokenizer, train_items[:2]))
    probe_labels = torch.tensor([item["label"] for item in train_items[:2]], device="cuda")
    rl, ce, _ = rlcd_loss(probe_logits, probe_labels, config["sigma_start"], config["reward_samples"])
    gradients = torch.autograd.grad(rl, [backbone_probe, head_probe])
    rl_gradient_norms = [float(gradient.float().norm()) for gradient in gradients]
    assert all(math.isfinite(norm) and norm > 0 for norm in rl_gradient_norms), "RL reward did not reach both trainable parts"
    gradient_check = {"backbone_parameter": probe_name, "rl_only_backbone_gradient_norm": rl_gradient_norms[0],
                      "rl_only_classifier_gradient_norm": rl_gradient_norms[1], "rl_loss": float(rl.detach()),
                      "cross_entropy": float(ce.detach())}
    save_json(output / "gradient-check.json", gradient_check)
    log({"rl_gradient_check": gradient_check})
    del gradients, probe_logits, rl, ce

    optimizer = torch.optim.AdamW([
        {"params": [p for name, p in model.named_parameters() if name.startswith("model.")], "lr": config["encoder_lr"]},
        {"params": [p for name, p in model.named_parameters() if not name.startswith("model.")], "lr": config["head_lr"]}
    ], weight_decay=config["weight_decay"], foreach=False)
    effective_batch = config["microbatch"] * config["gradient_accumulation"]
    planned_updates = math.ceil(len(train_items) / effective_batch) * config["epochs"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=planned_updates, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    selected = None
    history, steps = [], []
    updates = skipped = 0
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for epoch in range(config["epochs"]):
        order = list(range(len(train_items)))
        random.Random(config["seed"] + epoch).shuffle(order)
        sigma = config["sigma_start"] + (config["sigma_end"] - config["sigma_start"]) * epoch / max(1, config["epochs"] - 1)
        model.train()
        for at in range(0, len(order), effective_batch):
            chosen = order[at:at + effective_batch]
            optimizer.zero_grad(set_to_none=True)
            record = {"epoch": epoch + 1, "sigma": sigma, "rl_loss": 0., "cross_entropy": 0., "reward": 0.}
            for offset in range(0, len(chosen), config["microbatch"]):
                items = [train_items[i] for i in chosen[offset:offset + config["microbatch"]]]
                labels = torch.tensor([item["label"] for item in items], device="cuda")
                logits = forward(model, batch(tokenizer, items))
                rl, ce, reward = rlcd_loss(logits, labels, sigma, config["reward_samples"])
                fraction = len(items) / len(chosen)
                loss = (rl + ce) * fraction
                assert torch.isfinite(loss), "Non-finite training loss"
                scaler.scale(loss).backward()
                for key, value in (("rl_loss", rl), ("cross_entropy", ce), ("reward", reward)):
                    record[key] += float(value.detach()) * fraction
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            did_update = bool(torch.isfinite(norm)) and scaler.get_scale() >= old_scale
            if did_update:
                scheduler.step()
                updates += 1
            else:
                skipped += 1
            record.update(update=updates, gradient_norm=float(norm) if torch.isfinite(norm) else None,
                          optimizer_updated=did_update)
            steps.append(record)
        result = metrics(get_logits(model, tokenizer, val_items), val_labels, actions)
        history.append({"epoch": epoch + 1, **result})
        key = (result["accuracy"], -result["nll"])
        if selected is None or key > selected[0]:
            selected = (key, epoch + 1, result)
            model.save_pretrained(root / "selected")
            tokenizer.save_pretrained(root / "selected")
        log({"epoch": epoch + 1, "validation": result, "updates": updates, "skipped": skipped})
    torch.cuda.synchronize()
    changes = {"backbone_max_abs_change": float((backbone_probe.detach().cpu() - backbone_before).abs().max()),
               "classifier_max_abs_change": float((head_probe.detach().cpu() - head_before).abs().max())}
    assert all(math.isfinite(value) and value > 0 for value in changes.values())
    assert updates > 0
    result = {"objective": "Laya proper-scoring RLCD reward (w_sph=0.75) plus cross-entropy",
              "scope": "Full causal decoder and new last-token decision head; five fixed categorical actions",
              "config": config, "initial_validation": baseline, "validation_history": history,
              "planned_updates": planned_updates, "completed_updates": updates, "skipped_updates": skipped,
              "training_seconds_including_validation_and_checkpoint_saves": time.perf_counter() - started,
              "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
              "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
              "selected_epoch": selected[1], "selected_validation": selected[2],
              "gradient_check": gradient_check, "parameter_changes": changes,
              "test_used_during_training_or_selection": False}
    save_json(output / "training-steps.json", steps)
    save_json(output / "training.json", result)
    log({"training_completed": result})


@torch.no_grad()
def timed_predictions(model, tokenizer, rows, protocol):
    model.eval()
    def predict(row):
        inputs = tokenizer(prompt(row, protocol["question"]), return_tensors="pt", truncation=False)
        assert inputs["input_ids"].shape[-1] <= MAX_LENGTH
        inputs = {key: value.cuda() for key, value in inputs.items()}
        return forward(model, inputs)[0].cpu()
    for _ in range(3):
        predict(rows[0])
    records, logits, timings = [], [], []
    for row in rows:
        torch.cuda.synchronize()
        started = time.perf_counter()
        scores = predict(row)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - started) * 1000
        probabilities = scores.softmax(-1)
        action = protocol["actions"][int(probabilities.argmax())]
        records.append({"id": row["id"], "expected": row["expected"], "predicted": action,
                        "correct": action == row["expected"], "milliseconds": elapsed,
                        "probabilities": probabilities.tolist()})
        logits.append(scores)
        timings.append(elapsed)
    labels = torch.tensor([protocol["actions"].index(row["expected"]) for row in rows])
    return {**metrics(torch.stack(logits), labels, protocol["actions"]),
            "median_ms": float(np.median(timings)), "p95_ms": float(np.percentile(timings, 95))}, records


def evaluate(root):
    output = root / "results"
    training = json.loads((output / "training.json").read_text())
    assert not (output / "evaluation.json").exists(), "Evaluation already saved; keep this evidence unchanged"
    protocol = json.loads((root / "data/protocol.json").read_text())
    rows = read_split(root, protocol, "test")
    group = lambda row: tuple(row["observation"][key] for key in protocol["group_key"])
    test_groups = {group(row) for row in rows}
    for name in ("train", "validation"):
        assert not test_groups.intersection(group(row) for row in read_split(root, protocol, name))
    results = {}
    torch.set_num_threads(2)
    for label, folder in (("initial_random_head", "initial"), ("rlcd_trained", "selected")):
        model = ModernBertDecoderForSequenceClassification.from_pretrained(
            root / folder, attn_implementation="sdpa", use_cache=False, dtype=torch.float32).cuda()
        tokenizer = AutoTokenizer.from_pretrained(root / folder)
        result, predictions = timed_predictions(model, tokenizer, rows, protocol)
        results[label] = result
        save_json(output / f"test-{label}-predictions.json", predictions)
        log({label: result})
        del model
        gc.collect()
        torch.cuda.empty_cache()
    results.update(selected_epoch=training["selected_epoch"], calibration="None; probabilities are uncalibrated",
                   timing="Batch one; FP32 weights and forward pass, SDPA, 3 warmups, CUDA synchronization. Includes tokenization, transfers, model forward and returned CPU logits; excludes HTTP and game rendering.",
                   limitations=["Synthetic fixed-template five-action classification; test set reused from the earlier frozen game protocol",
                                "Initial baseline has a new random classification head; it is not a zero-shot text-generation baseline",
                                "This experiment does not measure improvement over CE-only training",
                                "No live-game reward, gameplay evaluation, arbitrary-choice interface, or language-generation RL"])
    save_json(output / "evaluation.json", results)
    manifest = {"model_id": MODEL_ID, "revision": REVISION, "selected_epoch": training["selected_epoch"],
                "files": {str(path.relative_to(root)).replace("\\", "/"): {"bytes": path.stat().st_size, "sha256": sha(path)}
                          for directory in (root / "selected", output)
                          for path in sorted(directory.rglob("*")) if path.is_file()}}
    save_json(output / "manifest.json", manifest)
    log({"evaluation_saved": str(output / "evaluation.json")})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("train", "evaluate"))
    parser.add_argument("--root", type=Path, required=True)
    arguments = parser.parse_args()
    {"train": train, "evaluate": evaluate}[arguments.phase](arguments.root.resolve())
