"""Build portable Colab lessons from an explicit list of public source files.

Each notebook contains a deterministic, hashed ZIP of readable repository files.
No working directory, credentials, model weights, or previous outputs are bundled.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
from pathlib import Path
import textwrap
import zipfile

import nbformat as nb

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks"
DATA = [f"data/{name}.json" for name in ("protocol", "train", "validation", "temperature", "gate", "test")]


def md(text):
    return nb.v4.new_markdown_cell(textwrap.dedent(text).strip())


def code(text):
    return nb.v4.new_code_cell(textwrap.dedent(text).strip())


def bundle(files):
    data = io.BytesIO()
    hashes = {}
    with zipfile.ZipFile(data, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in sorted(files):
            content = (ROOT / name).read_bytes()
            assert len(content) < 3_000_000 and Path(name).suffix in (".py", ".json")
            info = zipfile.ZipInfo(name, date_time=(2026, 9, 23, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, content)
            hashes[name] = hashlib.sha256(content).hexdigest()
    return base64.b64encode(data.getvalue()).decode(), hashes


def unpack_cell(files, label):
    payload, manifest = bundle(files)
    # Literal assignments make payloads easy to inspect without executing a notebook.
    return code(f'''
        import base64, hashlib, io, json, tempfile, zipfile
        from pathlib import Path

        PAYLOAD = {payload!r}
        MANIFEST = {manifest!r}
        ROOT = Path(tempfile.mkdtemp(prefix="laya-rlcd-{label}-"))
        with zipfile.ZipFile(io.BytesIO(base64.b64decode(PAYLOAD))) as archive:
            assert set(archive.namelist()) == set(MANIFEST)
            for name, digest in MANIFEST.items():
                content = archive.read(name)
                assert hashlib.sha256(content).hexdigest() == digest, name
                target = (ROOT / name).resolve()
                assert target.is_relative_to(ROOT.resolve())
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
        print(f"Verified {{len(MANIFEST)}} public files in a fresh experiment directory.")
    ''')


def save(name, cells, gpu=False):
    notebook = nb.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "colab": {"name": name, "provenance": []},
    })
    if gpu:
        notebook.metadata["accelerator"] = "GPU"
    for i, cell in enumerate(notebook.cells):
        cell.id = f"lesson-{i:02d}"
    nb.validate(notebook)
    nb.write(notebook, OUT / name)


def training_notebook(kind):
    decoder = kind == "decoder"
    title = "ModernBERT Decoder: teach a 17M pilot" if decoder else "Laya: train your station commander"
    script = "modernbert_decoder_colab.py" if decoder else "finetune_colab.py"
    phase = "evaluate" if decoder else "finalize"
    baseline = "40/200 → 183/200 (20% → 91.5%)" if decoder else "40/200 → 200/200 (20% → 100%)"
    seconds = "45.38 seconds" if decoder else "285.93 seconds"
    memory = "0.369 GiB allocated / 0.408 GiB reserved" if decoder else "7.080 GiB allocated / 7.828 GiB reserved"
    model = "jhu-clsp/ettin-decoder-17m" if decoder else "convaiinnovations/laya"
    cells = [md(f'''
        # {title}

        **Mission:** turn a text description of Cinder Station into one of five actions:
        `attack`, `heal`, `resupply`, `collect`, `extract`.
        We train the backbone and decision head using Laya's noisy-logit proper-scoring
        RLCD objective **plus cross-entropy**. The game motor handles navigation and aiming.

        Open **Runtime → Change runtime type → T4 GPU**, then run the cells in order.
        Use a fresh runtime. No API key or Google Drive mount is required. This downloads
        the public **{model}** checkpoint and installs pinned Python packages.

        Reference T4 run: **{baseline}**, **{seconds}** for training/validation/checkpoint work,
        excluding installation and initial model download/load. Peak CUDA memory: **{memory}**.
        These are archived observations, not promised results for this run.
        Colab availability, runtime packages and hardware can change.

        [Project and Doom-inspired game](https://github.com/shyamsridhar123/LAYA-RLCD) ·
        [Evidence and limitations](https://github.com/shyamsridhar123/LAYA-RLCD/blob/main/docs/BENCHMARKS.md) ·
        [Learning guide](https://github.com/shyamsridhar123/LAYA-RLCD/blob/main/docs/LEARNING.md)

        This is a new public wrapper around the retained training source. The reference
        training run is documented separately; notebook outputs start empty.
    '''), md('''
        ## 1. Install the experiment recipe

        Keep Colab's CUDA-enabled PyTorch. Pin the experiment packages and constrain pip
        to the already-installed Torch version, so dependency resolution cannot silently
        replace it. The reference environment used Python 3.13.15, Torch 2.11.0+cu128,
        CUDA 12.8 and a Tesla T4. A different runtime is a replication, not a bitwise replay.
        If an import fails after changing packages, restart the runtime and run from here.
    '''), code('''
        import os, sys, subprocess, importlib.metadata
        from pathlib import Path
        assert (3, 11) <= sys.version_info < (3, 14), "Use Python 3.11–3.13 for this recipe."
        os.environ.update(USE_TF="0", USE_FLAX="0", TOKENIZERS_PARALLELISM="false",
                          HF_HUB_DISABLE_IMPLICIT_TOKEN="1", HF_HUB_DISABLE_PROGRESS_BARS="1")
        torch_version = importlib.metadata.version("torch")
        constraint = Path("/tmp/laya-rlcd-torch-constraint.txt")
        constraint.write_text(f"torch=={torch_version}\\n")
        packages = ["laya==0.3.4", "transformers==5.0.0", "huggingface_hub==1.29.0",
                    "safetensors==0.8.0", "numpy==2.1.3"]
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-c", str(constraint), *packages], check=True)
        import torch
        assert torch.cuda.is_available(), "Select a GPU runtime, then start again."
        print({"torch": torch.__version__, "cuda": torch.version.cuda,
               "gpu": torch.cuda.get_device_name(0)})
    '''), md('''
        ## 2. Unpack the public, hashed lesson files

        The next cell embeds only the readable training script and six JSON protocol/data
        files from this repository. The builder is `scripts/build_notebooks.py`; it uses
        an explicit file list. Each run gets a new temporary directory.
        No personal Drive files, notebook history, keys, or model weights are included.
    '''), unpack_cell(DATA + [f"scripts/{script}"], kind), md('''
        ## 3. Inspect the rules before training

        Rule priority is the label source: heal below 35 HP with a medkit; otherwise
        resupply at 2 or fewer rounds when hostiles and crates remain; otherwise attack;
        then collect the core and extract. Read one **training** example.

        Splits are balanced and disjoint by the six decision-relevant state variables:
        500 train, 100 validation, 100 temperature, 100 gate, 200 test.
        The decoder uses only train/validation/test. Laya calibrates on the two separate
        100-example splits after selecting its checkpoint. Never select on test accuracy.
    '''), code('''
        protocol = json.loads((ROOT / "data/protocol.json").read_text())
        training_rows = json.loads((ROOT / "data/train.json").read_text())
        print(protocol["question"]["instructions"])
        print({name: spec["n"] for name, spec in protocol["splits"].items()})
        print(json.dumps(training_rows[0], indent=2))
        print("Training configuration:", protocol["training"])
    '''), md('''
        ## 4. Train the pilot

        Four epochs, microbatch 2, accumulation 4, backbone learning rate 2.5e-5,
        head learning rate 1e-4, four reward samples, sigma 0.4 → 0.1.
        Checkpoint selection maximizes validation accuracy, breaking ties with NLL.

        The decoder uses FP32 because a preliminary FP16 attempt overflowed gradients.
        Laya keeps FP32 trainable weights with FP16 autocast, then saves FP16 weights.
        Expect 252 optimizer updates. If a run skips updates, keep that fact in its report.
        This is supervised state-label learning with RLCD+CE, not reward from playing missions.
    '''), code(f'''
        subprocess.run([sys.executable, str(ROOT / "scripts/{script}"),
                        "train", "--root", str(ROOT)], check=True)
        training = json.loads((ROOT / "results/training.json").read_text())
        print("Selected epoch:", training["selected_epoch"])
        print("Validation:", training["selected_validation"])
    '''), md(f'''
        ## 5. Freeze selection, then {"evaluate" if decoder else "calibrate and evaluate"}

        Only now open the 200-example test set. Save every prediction, label, probability
        and latency. Accuracy asks whether the action matches the rule; latency includes
        tokenization and inference at batch one after warmup, with CUDA synchronization.
        It excludes the browser, local HTTP service, initial load, and installation.

        {"The initial checkpoint has a new random classification head. It is not a zero-shot text-generation baseline. Probabilities are uncalibrated." if decoder else "Temperature and the empirical confidence gate are fitted on separate data. Zero observed gate errors does not guarantee future correctness. The public game runs the selected pilot directly, without an escalation policy."}
    '''), code(f'''
        subprocess.run([sys.executable, str(ROOT / "scripts/{script}"),
                        "{phase}", "--root", str(ROOT)], check=True)
        result = json.loads((ROOT / "results/{"evaluation.json" if decoder else "manifest.json"}").read_text())
        print(json.dumps(result, indent=2))
    '''), md('''
        ## 6. Take your pilot home

        Download **both** the results and selected checkpoint before ending the Colab runtime.
        The checkpoint ZIP has a `selected/` folder. Unpack it under `models/` in your local
        checkout, then start the game using the command in the next cell.
        Results include synthetic states and environment versions. Review any modifications
        before sharing; keep credentials and personal data out of your notebook.
    '''), code(f'''
        import zipfile
        from google.colab import files
        exports = {{
            "{kind}-results.zip": ["results", "data", "scripts"],
            "{kind}-checkpoint.zip": ["selected"],
        }}
        for name, folders in exports.items():
            archive_path = ROOT / name
            with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for folder in folders:
                    for path in sorted((ROOT / folder).rglob("*")):
                        if path.is_file():
                            archive.write(path, path.relative_to(ROOT).as_posix())
            print(name, round(archive_path.stat().st_size / 2**20, 1), "MiB")
            files.download(str(archive_path))
        print("On your computer: python scripts/setup.py --models")
        print("Then: python scripts/play.py --model {kind} --model-dir models/selected")
    '''), md('''
        ## Bonus missions

        - **Find the boundary bug.** Compare HP 34 vs 35 and ammo 2 vs 3. Change one fact at a time.
        - **Earn the RLCD claim.** Compare CE-only to RLCD+CE with multiple seeds and the same data budget.
          The reference runs do not isolate RLCD's contribution.
        - **Break the template.** Rewrite the state text while preserving meaning, then test new maps.
        - **Calibrate your confidence.** Compare top probability, normalized entropy and actual error rate.

        Keep the published reference data unchanged. Fork the experiment, define a new test set
        before looking at its outcomes, and publish errors alongside successes.
    ''')]
    save("01_modernbert_decoder_colab.ipynb" if decoder else "02_laya_colab.ipynb", cells, gpu=True)


def results_notebook():
    files = DATA + ["scripts/verify_benchmarks.py", "scripts/modernbert_decoder_colab.py", "scripts/finetune_colab.py"]
    # Benchmarks are already the reviewed, public subset, never original raw exports.
    files += [p.relative_to(ROOT).as_posix() for p in sorted((ROOT / "benchmarks").rglob("*.json"))]
    # The verifier checks the CSV too; handle it separately as a readable literal.
    csv_text = (ROOT / "benchmarks/jev/decisions.csv").read_text(encoding="utf-8")
    cells = [md('''
        # Mission debrief: read the evidence

        No GPU, API key or model download needed. Recompute the retained September 2026
        experiments, inspect the decoder's errors, and compare before/after results.
        This notebook is analysis of **archived observations**, not a new training run.

        The source files are public and hashed in the next cell. Charts compare each model
        against its own baseline; CPU, T4 and hosted Jev timings are separate measurements.
        [Full methodology](https://github.com/shyamsridhar123/LAYA-RLCD/blob/main/docs/BENCHMARKS.md).
    '''), unpack_cell(files, "debrief"), code(f'''
        (ROOT / "benchmarks/jev/decisions.csv").write_text({csv_text!r}, encoding="utf-8")
        import importlib.util
        spec = importlib.util.spec_from_file_location("verify", ROOT / "scripts/verify_benchmarks.py")
        verifier = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(verifier)
        verified = verifier.verify()
        print("Evidence checks:", verified["verification"])
        print("Disjoint labeled examples:", sum(verified["splits"].values()))
        for name, metric in verified["scores"].items():
            print(f"{{name:30}} {{metric['correct']:3}}/{{metric['n']}}  median {{metric['median_ms']:7.2f}} ms")
    '''), md('''
        ## Accuracy: a small, deliberately constrained task

        Each test action has 40 examples. The decoder's baseline uses a random classifier
        head; Laya's baseline is its pinned public decision checkpoint. Both are evaluated
        before/after on the same frozen synthetic task. No CE-only experiment is available.
        A high score here does not establish general reasoning, perception or game skill.
    '''), code('''
        import matplotlib.pyplot as plt
        scores = verified["scores"]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
        pairs = [("ModernBERT Decoder · 17M", "decoder_initial_random_head", "decoder_rlcd_trained"),
                 ("Laya · 421M", "laya_base", "laya_trained")]
        for ax, (title, before, after) in zip(axes, pairs):
            values = [scores[key]["accuracy"] * 100 for key in (before, after)]
            bars = ax.bar(["Initial", "RLCD + CE"], values, color=["#737b72", "#cc7436"], width=.6)
            ax.bar_label(bars, labels=[f"{v:g}%" for v in values], padding=5)
            ax.set(title=title, ylim=(0, 112), ylabel="Correct actions (%)")
            ax.spines[["top", "right"]].set_visible(False)
        fig.suptitle("200 held-out synthetic states per evaluation")
        fig.tight_layout()
        plt.show()
    '''), md('''
        ## Find the decoder's weak spot

        Rows are expected actions, columns are predicted actions. Investigate the heal row:
        accuracy alone hides asymmetric errors. Try checking the 34/35 HP boundary before
        changing the learning rate. Test-label inspection is analysis, not permission to
        tune a new checkpoint on these outcomes and call this an untouched test set.
    '''), code('''
        actions = ["attack", "heal", "resupply", "collect", "extract"]
        c = scores["decoder_rlcd_trained"]["confusion"]
        matrix = [[c[a][b] for b in actions] for a in actions]
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.imshow(matrix, cmap="YlOrBr", vmin=0, vmax=40)
        ax.set_xticks(range(5), actions, rotation=30, ha="right")
        ax.set_yticks(range(5), actions)
        ax.set(xlabel="Predicted", ylabel="Expected", title="Decoder after RLCD + CE · 183/200")
        for i, row in enumerate(matrix):
            for j, value in enumerate(row):
                ax.text(j, i, str(value), ha="center", va="center", color="white" if value > 23 else "#302b23")
        fig.tight_layout()
        plt.show()
        print("Per-action correct:", {a: c[a][a] for a in actions})
    '''), md('''
        ## Losses and timing answer different questions

        NLL penalizes assigning tiny probability to the correct action. Brier measures
        squared error over the entire distribution. Decoder probabilities retain enough
        precision to recompute the published scores. Laya's API rounds probabilities to
        four decimal places, so exact logit-level NLL cannot be reconstructed from them.

        T4 figures use batch one and include tokenization. The separate CPU service run
        also includes the HTTP bridge. Jev's one mission includes remote network/service
        time. These are not a controlled cross-model speed ranking.
    '''), code('''
        for key in ("decoder_initial_random_head", "decoder_rlcd_trained"):
            m = scores[key]
            print(key, {"NLL": round(m["nll_from_stored_probabilities"], 4),
                        "Brier": round(m["brier_from_stored_probabilities"], 4)})
        print("Mission wins (same map, five seeds):", verified["gameplay"])
        jev = json.loads((ROOT / "benchmarks/jev/demo.json").read_text())
        print("Jev single demo:", jev["outcome"], "·", jev["decisions"], "decisions")
        print("Jev browser median / p95 (ms):", jev["browser_request_latency"]["median_ms"], jev["browser_request_latency"]["p95_ms"])
        print("Jev p95 uses nearest rank; with 11 observations it is the maximum.")
    '''), md('''
        ## Your next experiment

        Change one factor: loss (CE-only vs RLCD+CE), text template, map, or training seed.
        Freeze the test set first and publish its hash, failures, settings and environment.
        Run `npm run verify:replays` in the repository to replay the 20 archived missions
        and Jev outcome through the exact shared simulation. That replay uses recorded
        actions; it does not ask the models again.

        The reference decoder weights were not retained. Retrain and export using lesson 1
        to play with that architecture. Large model weights are not bundled in this repo.
    ''')]
    save("03_results_walkthrough.ipynb", cells)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    training_notebook("decoder")
    training_notebook("laya")
    results_notebook()
    print("Built three public notebooks with deterministic, allowlisted payloads.")
