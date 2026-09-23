"""One-T4 Laya fine-tuning; validation selection, separate calibration, sealed test.

Run after installing laya==0.3.4 and transformers==5.0.0. Inputs are the frozen
data/*.json bundle. This adapts upstream's RLCD+CE objective to one GPU.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import shutil
import time
from pathlib import Path

os.environ.update(USE_TF="0", USE_FLAX="0", TOKENIZERS_PARALLELISM="false", HF_HUB_DISABLE_IMPLICIT_TOKEN="1")
import numpy as np
import torch
from huggingface_hub import snapshot_download
from laya.agent import Agent
from laya.common import QTYPES, build_sequence, collate_items, proper_reward
from safetensors.torch import load_file, save_file

BASE_REVISION = "1c5edc17a7acd8701df6fc341c0d179f1c62c982"
BASE_SHA = "891102d372688fc2a094dac56a384bc537b87c63f21f9f3dac0be2b7cbc8d86c"


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def log(value):
    print(json.dumps(value, allow_nan=False), flush=True)


def read_split(root, protocol, name):
    path = root / "data" / f"{name}.json"
    assert sha(path) == protocol["splits"][name]["sha256"], f"Split changed: {name}"
    return json.loads(path.read_text())


def encode(agent, rows, question, actions):
    result = []
    for row in rows:
        ids, markers = build_sequence(agent.tok, row["observation"]["text"], Agent._to_internal(question),
                                      agent.cfg["max_len"], agent.cfg["head_max_len"])
        assert len(markers) == len(actions)
        label = actions.index(row["expected"])
        result.append({"ids": ids, "markers": markers, "qtype": QTYPES["choice"],
                       "target": [float(i == label) for i in range(len(actions))], "label": label})
    return result


def batch_to_gpu(items, tokenizer):
    b = collate_items([items], tokenizer.pad_token_id)
    return {k: v.cuda() if torch.is_tensor(v) else v for k, v in b.items()}


def forward(model, batch):
    with torch.autocast("cuda", dtype=torch.float16):
        logits, act = model(batch["input_ids"], batch["attention_mask"], batch["marker_pos"],
                            batch["marker_mask"], batch["qtype"])
    return logits.float(), act


@torch.no_grad()
def get_logits(model, tok, items):
    model.eval()
    chunks = []
    for i in range(0, len(items), 4):
        logits, _ = forward(model, batch_to_gpu(items[i:i + 4], tok))
        chunks.append(logits.cpu())
    return torch.cat(chunks)


def metrics(logits, labels, temperature=1.0):
    p = torch.softmax(logits / temperature, -1)
    correct = p.argmax(-1).eq(labels)
    top = p.max(-1).values
    ece = 0.0
    bins = (top * 10).long().clamp(max=9)
    for i in range(10):
        select = bins == i
        if select.any():
            ece += float(select.float().mean() * (correct[select].float().mean() - top[select].mean()).abs())
    return {"n": len(labels), "correct": int(correct.sum()), "accuracy": float(correct.float().mean()),
            "nll": float(torch.nn.functional.cross_entropy(logits / temperature, labels)),
            "brier": float(((p - torch.nn.functional.one_hot(labels, p.shape[1])) ** 2).sum(-1).mean()),
            "ece_10bins": ece}


def load_base(root):
    path = snapshot_download("convaiinnovations/laya", revision=BASE_REVISION, token=False,
                             allow_patterns=["model.safetensors", "rl_agent_config.json", "tokenizer/*", "encoder/*"],
                             local_dir=str(root / "base"))
    assert sha(Path(path) / "model.safetensors") == BASE_SHA
    agent = Agent(path, device="cuda")
    assert agent.device.type == "cuda"
    return agent


def train(root, export):
    assert torch.cuda.is_available(), "A CUDA GPU is required"
    protocol = json.loads((root / "data/protocol.json").read_text())
    actions, config = protocol["actions"], protocol["training"]
    run = root / "results"
    run.mkdir(exist_ok=True)
    if (run / "training.json").exists():
        raise RuntimeError("This experiment already finished training; use a new output directory for a new run")
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(2)
    props = torch.cuda.get_device_properties(0)
    environment = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "python": platform.python_version(),
                   "torch": torch.__version__, "cuda": torch.version.cuda, "gpu": props.name,
                   "gpu_count": torch.cuda.device_count(), "gpu_memory_gib": props.total_memory / 2**30,
                   "packages": {n: importlib.metadata.version(n) for n in ["laya", "transformers", "safetensors", "huggingface_hub"]},
                   "base_revision": BASE_REVISION, "base_sha256": BASE_SHA,
                   "training_script_sha256": sha(__file__), "protocol_sha256": sha(root / "data/protocol.json")}
    started = time.perf_counter()
    agent = load_base(root)
    model, tok = agent.model, agent.tok
    environment["parameters"] = sum(p.numel() for p in model.parameters())
    environment["download_and_load_seconds"] = time.perf_counter() - started
    write_json(run / "environment.json", environment)
    log({"environment": environment})
    # Test and both calibration splits are not read during training/selection.
    train_rows = read_split(root, protocol, "train")
    validation_rows = read_split(root, protocol, "validation")
    train_items = encode(agent, train_rows, protocol["question"], actions)
    val_items = encode(agent, validation_rows, protocol["question"], actions)
    val_labels = torch.tensor([it["label"] for it in val_items])
    baseline_validation = metrics(get_logits(model, tok, val_items), val_labels)
    log({"base_validation": baseline_validation, "training_sequence_tokens": {
        "min": min(len(it["ids"]) for it in train_items), "max": max(len(it["ids"]) for it in train_items)}})
    model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    encoder_probe_name, encoder_probe = next((n, p) for n, p in model.named_parameters()
                                             if n.startswith("encoder.") and "embeddings" not in n and p.ndim == 2)
    encoder_before = encoder_probe.detach().flatten()[:1024].cpu().clone()
    head_before = model.scorer[1].weight.detach().flatten()[:1024].cpu().clone()
    optimizer = torch.optim.AdamW([
        {"params": [p for n, p in model.named_parameters() if n.startswith("encoder.")], "lr": config["encoder_lr"]},
        {"params": [p for n, p in model.named_parameters() if not n.startswith("encoder.")], "lr": config["head_lr"]},
    ], weight_decay=config["weight_decay"], foreach=False)
    effective_batch = config["microbatch"] * config["gradient_accumulation"]
    total_updates = math.ceil(len(train_items) / effective_batch) * config["epochs"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_updates, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", init_scale=256.0)
    best_key, selected_epoch = None, None
    history, steps = [], []
    best = root / "selected"
    best.mkdir(exist_ok=True)
    updates = skipped = 0
    torch.cuda.reset_peak_memory_stats()
    train_started = time.perf_counter()
    for epoch in range(config["epochs"]):
        order = list(range(len(train_items)))
        random.Random(config["seed"] + epoch).shuffle(order)
        sigma = config["sigma_start"] + (config["sigma_end"] - config["sigma_start"]) * epoch / max(1, config["epochs"] - 1)
        model.train()
        epoch_started = time.perf_counter()
        epoch_ce = []
        for at in range(0, len(order), effective_batch):
            chosen = order[at:at + effective_batch]
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            step_start = time.perf_counter()
            ce_total, rl_total, reward_total = 0., 0., 0.
            for offset in range(0, len(chosen), config["microbatch"]):
                indices = chosen[offset:offset + config["microbatch"]]
                b = batch_to_gpu([train_items[i] for i in indices], tok)
                logits, act = forward(model, b)
                mask, target = b["marker_mask"], b["target"]
                k = mask.sum(-1, keepdim=True).float()
                eps = torch.randn((config["reward_samples"],) + logits.shape, device="cuda") * sigma * mask
                eps = (eps - eps.sum(-1, keepdim=True) / k) * mask
                z = logits.detach().unsqueeze(0) + eps
                q = torch.softmax(z.masked_fill(~mask, -1e4), -1)
                with torch.no_grad():
                    reward = proper_reward(q, target.unsqueeze(0), b["qtype"], mask, w_sph=.75, w_rps=1.)
                    advantage = reward - reward.mean(0, keepdim=True)
                    advantage = advantage / (advantage.std() + 1e-6)
                logp = -(((z - logits.unsqueeze(0)) ** 2) * mask).sum(-1) / (2 * sigma ** 2)
                rl = -(advantage * logp).mean()
                ce = -(target * torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)).sum(-1).mean()
                fraction = len(indices) / len(chosen)
                loss = (rl + ce + 0.0 * act.sum()) * fraction
                assert torch.isfinite(loss).item(), "Non-finite loss"
                scaler.scale(loss).backward()
                ce_total += ce.item() * fraction
                rl_total += rl.item() * fraction
                reward_total += reward.mean().item() * fraction
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            finite_grad = bool(torch.isfinite(grad_norm))
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            did_update = finite_grad and scaler.get_scale() >= old_scale
            if did_update:
                scheduler.step()
                updates += 1
            else:
                skipped += 1
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            record = {"epoch": epoch + 1, "update": updates, "examples": len(chosen), "cross_entropy": ce_total,
                      "rl_loss": rl_total, "reward": reward_total, "sigma": sigma,
                      "gradient_norm": float(grad_norm) if finite_grad else None, "updated": did_update,
                      "scale": scaler.get_scale(), "seconds": time.perf_counter() - step_start,
                      "encoder_lr": scheduler.get_last_lr()[0]}
            steps.append(record)
            epoch_ce.append(ce_total)
            with (run / "steps.jsonl").open("a") as handle:
                handle.write(json.dumps(record, allow_nan=False) + "\n")
            if len(steps) % 10 == 0:
                log({"training": record, "elapsed_seconds": time.perf_counter() - train_started})
        validation = metrics(get_logits(model, tok, val_items), val_labels)
        key = (validation["accuracy"], -validation["nll"])
        improved = best_key is None or key > best_key
        if improved:
            best_key, selected_epoch = key, epoch + 1
            state_dict = {k: v.detach().half().contiguous().cpu() if v.is_floating_point() else v.detach().contiguous().cpu()
                          for k, v in model.state_dict().items()}
            save_file(state_dict, str(best / "model.safetensors"))
            del state_dict
            model.encoder.config.save_pretrained(best / "encoder")
            tok.save_pretrained(best / "tokenizer")
            write_json(best / "rl_agent_config.json", agent.cfg)
        row = {"epoch": epoch + 1, "mean_cross_entropy": sum(epoch_ce) / len(epoch_ce),
               "validation": validation, "selected_so_far": selected_epoch,
               "seconds_including_validation_and_save": time.perf_counter() - epoch_started}
        history.append(row)
        write_json(run / "history.json", history)
        if export:
            shutil.copytree(run, export / "results", dirs_exist_ok=True)
            if improved:
                shutil.copytree(best, export / "selected", dirs_exist_ok=True)
        log({"epoch_complete": row})
    result = {"experiment": protocol["experiment"], "base_validation": baseline_validation,
              "completed_epochs": len(history), "optimizer_updates": updates, "skipped_nonfinite_updates": skipped,
              "training_seconds_including_validation_saves_and_drive_copy": time.perf_counter() - train_started,
              "selected_epoch": selected_epoch, "selected_validation": history[selected_epoch - 1]["validation"],
              "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
              "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
              "encoder_probe": encoder_probe_name,
              "last_epoch_encoder_probe_delta": float((encoder_probe.detach().flatten()[:1024].cpu() - encoder_before).abs().max()),
              "last_epoch_scorer_probe_delta": float((model.scorer[1].weight.detach().flatten()[:1024].cpu() - head_before).abs().max()),
              "precision": "FP32 trainable weights; FP16 autocast/GradScaler; FP16 exported weights",
              "selection_precision": "Validation uses in-memory FP32 weights with FP16 autocast; calibration/test reload the selected FP16 export",
              "optimizer": "AdamW, full encoder and decision heads, gradient checkpointing, clip norm 1.0",
              "objective": "Upstream proper scoring reward with noisy logits and centered advantages + cross-entropy",
              "history": history, "config": config}
    write_json(run / "training.json", result)
    if export:
        shutil.copytree(run, export / "results", dirs_exist_ok=True)
    log({"training_complete": result})


def finalize(root, export):
    torch.set_num_threads(2)
    protocol = json.loads((root / "data/protocol.json").read_text())
    run, best = root / "results", root / "selected"
    trained = json.loads((run / "training.json").read_text())
    assert trained["completed_epochs"] == protocol["training"]["epochs"]
    agent = Agent(str(best), device="cuda")
    actions, question = protocol["actions"], protocol["question"]
    temperature_rows = read_split(root, protocol, "temperature")
    temp_items = encode(agent, temperature_rows, question, actions)
    labels = torch.tensor([it["label"] for it in temp_items])
    logits = get_logits(agent.model, agent.tok, temp_items)
    candidates = torch.logspace(-1, 1, 101).tolist()
    losses = [float(torch.nn.functional.cross_entropy(logits / t, labels)) for t in candidates]
    temperature = candidates[int(np.argmin(losses))]
    # Remove stale base choice-bucket calibration: Agent gives bucket scales priority.
    agent.temperature = [temperature, *agent.temperature[1:]]
    agent.temperature_by_options = {k: v for k, v in agent.temperature_by_options.items() if not k.startswith("choice:")}
    agent.temperature_by_options["choice:3-5"] = temperature
    gate_rows = read_split(root, protocol, "gate")
    gate_predictions = []
    for row in gate_rows:
        answer = agent.predict(row["observation"]["text"], {"action": question})["answers"]["action"]
        gate_predictions.append({"id": row["id"], "expected": row["expected"], **answer})
    grid = []
    for threshold in [round(i * .05, 2) for i in range(21)] + [1.01]:
        accepted = [p for p in gate_predictions if p["confidence"] >= threshold]
        grid.append({"threshold": threshold, "accepted": len(accepted),
                     "errors": sum(p["choice"] != p["expected"] for p in accepted)})
    eligible = [row for row in grid if row["accepted"] >= 20 and row["errors"] == 0]
    threshold = eligible[0]["threshold"] if eligible else 1.01
    calibration = {"temperature": temperature, "temperature_metrics": metrics(logits, labels, temperature),
                   "temperature_grid": [{"temperature": t, "nll": loss} for t, loss in zip(candidates, losses)],
                   "gate_threshold": threshold, "gate_grid": grid, "gate_predictions": gate_predictions,
                   "selection": protocol["calibration"], "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    agent.cfg.update(temperature=agent.temperature, temperature_by_options=agent.temperature_by_options,
                     model_name="laya-cinder-station", fine_tuned=True,
                     rlcd_experiment=protocol["experiment"], rlcd_entropy_threshold=threshold,
                     rlcd_training={"base_revision": BASE_REVISION, "selected_epoch": trained["selected_epoch"],
                                    "optimizer_updates": trained["optimizer_updates"],
                                    "protocol_sha256": sha(root / "data/protocol.json")})
    write_json(best / "rl_agent_config.json", agent.cfg)
    write_json(run / "calibration.json", calibration)
    log({"calibration_frozen": {"temperature": temperature, "gate_threshold": threshold,
                               "gate_accepted": next(r for r in grid if r["threshold"] == threshold)}})
    checkpoint_sha = sha(best / "model.safetensors")
    # Selection and routing are frozen before the test split is read for the first time.
    test_rows = read_split(root, protocol, "test")
    del agent
    gc.collect()
    torch.cuda.empty_cache()
    results = {}
    for name, path in [("base", root / "base"), ("trained", best)]:
        pilot = Agent(str(path), device="cuda")
        for row in test_rows[:3]:
            pilot.predict(row["observation"]["text"], {"action": question})
        predictions = []
        for row in test_rows:
            torch.cuda.synchronize()
            started = time.perf_counter()
            output = pilot.predict(row["observation"]["text"], {"action": question})
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - started) * 1000
            answer = output["answers"]["action"]
            predictions.append({"id": row["id"], "expected": row["expected"], "correct": answer["choice"] == row["expected"],
                                "ms": elapsed, "tokens": output["usage"]["input_tokens"], **answer})
        correct = sum(p["correct"] for p in predictions)
        confusion = {a: {b: 0 for b in actions} for a in actions}
        for p in predictions:
            confusion[p["expected"]][p["choice"]] += 1
        times = [p["ms"] for p in predictions]
        accepted = [p for p in predictions if p["confidence"] >= threshold]
        summary = {"n": len(predictions), "correct": correct, "accuracy": correct / len(predictions),
                   "median_ms": float(np.median(times)), "p95_ms": float(np.percentile(times, 95)),
                   "min_ms": min(times), "max_ms": max(times), "confusion": confusion,
                   "accepted_by_trained_gate": len(accepted) if name == "trained" else None,
                   "accepted_errors": sum(not p["correct"] for p in accepted) if name == "trained" else None}
        write_json(run / f"test-{name}-predictions.json", predictions)
        results[name] = summary
        log({"held_out_test": name, **summary})
        del pilot
        gc.collect()
        torch.cuda.empty_cache()
    manifest = {"experiment": protocol["experiment"], "base_revision": BASE_REVISION, "base_sha256": BASE_SHA,
                "selected_weights_sha256": checkpoint_sha, "selected_weights_bytes": (best / "model.safetensors").stat().st_size,
                "config_sha256": sha(best / "rl_agent_config.json"), "protocol_sha256": sha(root / "data/protocol.json"),
                "script_sha256": sha(__file__), "selected_epoch": trained["selected_epoch"],
                "temperature": temperature, "entropy_threshold": threshold, "test": results,
                "test_latency_scope": "Sequential Agent.predict on T4, batch=1, includes tokenization/decoding, CUDA synchronized, three warmups; excludes HTTP and game rendering",
                "limitations": protocol["limitations"]}
    write_json(run / "manifest.json", manifest)
    if export:
        for folder in ["results", "selected", "data"]:
            shutil.copytree(root / folder, export / folder, dirs_exist_ok=True)
        shutil.copy2(__file__, export / "finetune_colab.py")
    log({"experiment_complete": manifest})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["train", "finalize"])
    parser.add_argument("--root", type=Path, default=Path("/content/rlcd_game_experiment"))
    parser.add_argument("--export", type=Path)
    args = parser.parse_args()
    if args.export:
        args.export.mkdir(parents=True, exist_ok=True)
    (train if args.phase == "train" else finalize)(args.root, args.export)
