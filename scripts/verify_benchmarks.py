"""Recompute the public evidence with Python's standard library; no models needed."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]


def read(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def near(actual, expected, tolerance=1e-5):
    assert math.isclose(actual, expected, rel_tol=tolerance, abs_tol=tolerance), (actual, expected)


def percentile(values, q):
    """Linear interpolation, matching numpy.percentile's default."""
    values = sorted(values)
    at = (len(values) - 1) * q
    lo = math.floor(at)
    hi = math.ceil(at)
    return values[lo] + (values[hi] - values[lo]) * (at - lo)


def rule(row):
    o = row["observation"]
    if o["health"] < 35 and o["medkits"] > 0:
        return "heal"
    if o["enemies"] > 0 and o["ammo"] <= 2 and o["ammoCrates"] > 0:
        return "resupply"
    if o["enemies"] > 0:
        return "attack"
    return "extract" if o["carryingCore"] else "collect"


def prediction_metrics(rows, actions, labels, timing_key):
    assert len(rows) == len(labels)
    assert len({r["id"] for r in rows}) == len(rows)
    confusion = {a: {b: 0 for b in actions} for a in actions}
    brier, nll = [], []
    for r in rows:
        assert labels[r["id"]] == r["expected"]
        chosen = r.get("predicted", r.get("choice"))
        assert chosen in actions and r["correct"] == (chosen == r["expected"])
        confusion[r["expected"]][chosen] += 1
        raw = r.get("probabilities")
        if raw is not None:
            p = [raw[a] for a in actions] if isinstance(raw, dict) else raw
            assert len(p) == len(actions) and all(math.isfinite(x) and 0 <= x <= 1 for x in p)
            near(sum(p), 1, 0.001)  # Laya's public interface rounds to four decimals.
            assert p[actions.index(chosen)] >= max(p) - 0.0001
            brier.append(sum((x - int(a == r["expected"])) ** 2 for a, x in zip(actions, p)))
            target = p[actions.index(r["expected"])]
            nll.append(-math.log(target) if target > 0 else math.inf)
    times = [r[timing_key] for r in rows]
    assert all(math.isfinite(t) and t >= 0 for t in times)
    correct = sum(r["correct"] for r in rows)
    return {"n": len(rows), "correct": correct, "accuracy": correct / len(rows),
            "median_ms": statistics.median(times), "p95_ms": percentile(times, .95),
            "nll_from_stored_probabilities": statistics.mean(nll) if nll and all(math.isfinite(x) for x in nll) else None,
            "brier_from_stored_probabilities": statistics.mean(brier) if brier else None,
            "confusion": confusion}


def verify():
    protocol = read("data/protocol.json")
    actions = protocol["actions"]
    seen_groups, seen_ids, counts = set(), set(), {}
    for name, spec in protocol["splits"].items():
        path = ROOT / "data" / f"{name}.json"
        assert sha(path) == spec["sha256"], f"Dataset changed: {name}"
        rows = read(f"data/{name}.json")
        assert len(rows) == spec["n"]
        groups = {tuple(r["observation"][k] for k in protocol["group_key"]) for r in rows}
        ids = {r["id"] for r in rows}
        assert len(ids) == len(rows) and len(groups) == len(rows)
        assert not groups & seen_groups and not ids & seen_ids, f"Overlapping split: {name}"
        seen_groups |= groups
        seen_ids |= ids
        assert all(rule(r) == r["expected"] for r in rows)
        assert Counter(r["expected"] for r in rows) == {a: spec["per_action"] for a in actions}
        counts[name] = len(rows)
    labels = {r["id"]: r["expected"] for r in read("data/test.json")}
    scores = {}
    for variant, expected in read("benchmarks/decoder/evaluation.json").items():
        if variant not in ("initial_random_head", "rlcd_trained"):
            continue
        m = prediction_metrics(read(f"benchmarks/decoder/test-{variant}-predictions.json"), actions, labels, "milliseconds")
        for key in ("n", "correct", "accuracy", "median_ms", "p95_ms"):
            near(m[key], expected[key])
        near(m["nll_from_stored_probabilities"], expected["nll"])
        near(m["brier_from_stored_probabilities"], expected["brier"])
        for action in actions:
            assert m["confusion"][action][action] == expected["per_action"][action]["correct"]
        scores[f"decoder_{variant}"] = m
    manifest = read("benchmarks/laya/manifest.json")
    for variant in ("base", "trained"):
        m = prediction_metrics(read(f"benchmarks/laya/test-{variant}-predictions.json"), actions, labels, "ms")
        expected = manifest["test"][variant]
        for key in ("n", "correct", "accuracy", "median_ms", "p95_ms"):
            near(m[key], expected[key])
        assert m["confusion"] == expected["confusion"]
        m["probability_note"] = "Laya API probabilities are rounded; derived NLL/Brier are not exact logit metrics."
        scores[f"laya_{variant}"] = m
        m = prediction_metrics(read(f"benchmarks/laya/cpu-{variant}-predictions.json"), actions, labels, "client_ms")
        expected = read(f"benchmarks/laya/cpu-{variant}-summary.json")
        for key in ("n", "correct", "accuracy"):
            near(m[key], expected[key])
        near(m["median_ms"], expected["latency_ms"]["p50"])
        near(m["p95_ms"], expected["latency_ms"]["p95"])
        assert expected["errors"] == 0
        scores[f"laya_cpu_{variant}"] = m
    for model, update_key, skipped_key, step_key in (
        ("decoder", "completed_updates", "skipped_updates", "optimizer_updated"),
        ("laya", "optimizer_updates", "skipped_nonfinite_updates", "updated"),
    ):
        training = read(f"benchmarks/{model}/training.json")
        steps = read(f"benchmarks/{model}/training-steps.json")
        assert len(steps) == training[update_key] == 252 and training[skipped_key] == 0
        assert all(s[step_key] and s["update"] == i + 1 for i, s in enumerate(steps))
        assert training["selected_epoch"] == 4
        history = training.get("validation_history", training.get("history"))
        best = max(history, key=lambda h: (h.get("validation", h)["accuracy"], -h.get("validation", h)["nll"]))
        assert best["epoch"] == training["selected_epoch"]
    calibration = read("benchmarks/laya/calibration.json")
    best = min(calibration["temperature_grid"], key=lambda r: r["nll"])
    assert best["temperature"] == calibration["temperature"] == manifest["temperature"]
    for g in calibration["gate_grid"]:
        accepted = [r for r in calibration["gate_predictions"] if r["confidence"] >= g["threshold"]]
        assert g["accepted"] == len(accepted)
        assert g["errors"] == sum(r["choice"] != r["expected"] for r in accepted)
    eligible = [g for g in calibration["gate_grid"] if g["accepted"] >= 20 and g["errors"] == 0]
    assert min(g["threshold"] for g in eligible) == calibration["gate_threshold"] == 0
    jev = read("benchmarks/jev/demo.json")
    decisions = read("benchmarks/jev/decisions.json")
    assert jev["decisions"] == len(decisions) == 11 and jev["fallbacks"] == 0
    for key, timing in (("client_ms", "browser_request_latency"), ("jev_ms", "server_to_typesafe_roundtrip")):
        times = [r[key] for r in decisions]
        s = jev[timing]
        near(statistics.median(times), s["median_ms"])
        near(statistics.mean(times), s["mean_ms"])
        near(sorted(times)[math.ceil(len(times) * .95) - 1], s["p95_ms"])
        near(sum(times), s["total_ms"])
    with (ROOT / "benchmarks/jev/decisions.csv").open(newline="", encoding="utf-8") as file:
        csv_rows = list(csv.DictReader(file))
    assert len(csv_rows) == len(decisions)
    for row, d in zip(csv_rows, decisions):
        assert row["action"] == d["action"] and row["model"] == d["jev_model"] == "jev-1.13.0"
        near(float(row["simulation_seconds"]), d["time"])
        near(float(row["browser_ms"]), d["client_ms"])
        near(float(row["typesafe_roundtrip_ms"]), d["jev_ms"])
    episodes = read("benchmarks/gameplay/episodes.json")
    assert len(episodes) == 20
    wins = {}
    for mode in ("laya-base", "laya-trained", "rule", "random"):
        rows = [r for r in episodes if r["controller"] == mode]
        assert sorted(r["seed"] for r in rows) == protocol["episode_seeds"]
        wins[mode] = {"won": sum(r["status"] == "won" for r in rows), "n": len(rows)}
    assert {k: v["won"] for k, v in wins.items()} == {"laya-base": 0, "laya-trained": 5, "rule": 5, "random": 0}
    lineage = read("benchmarks/lineage.json")
    assert sha(ROOT / "data/protocol.json") == lineage["public_protocol_sha256"]
    for name, digest in lineage["public_training_sources"].items():
        assert sha(ROOT / "scripts" / name) == digest
    return {"verification": "passed", "splits": counts, "scores": scores, "gameplay": wins,
            "jev": {"decisions": len(decisions), "outcome": jev["outcome"]},
            "scope": "Recomputed retained evidence; this does not rerun model inference or establish causality. Run npm run verify:replays for mission replay."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional output JSON; use runs/ for new results")
    args = parser.parse_args()
    result = verify()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("Verified: 1,000 disjoint labeled states; 1,200 test predictions; 504 updates; 20 missions; 11 Jev decisions.")
    for name, m in result["scores"].items():
        print(f"{name}: {m['correct']}/{m['n']}; median {m['median_ms']:.2f} ms; p95 {m['p95_ms']:.2f} ms")
