# Your machine. Your pilot.

Play, train, evaluate, and inspect either model locally. Colab is optional.

| Mission | What you need |
|---|---|
| Play manually or watch rule/random pilots | Windows, macOS, or Linux; Python 3.11–3.13, Node.js 22 LTS, Git; a WebGL browser |
| Run a saved LAYA or decoder checkpoint | The game setup plus `--models`; inference defaults to CPU |
| Train either model with the included recipe | Windows or Linux, Python 3.11–3.13 (3.12 recommended), an NVIDIA GPU, and a driver compatible with CUDA 12.8; Node.js is only needed to play |

The training scripts use CUDA explicitly. CPU and Apple MPS training are not implemented. The reference GPU was a 16 GB Tesla T4. LAYA's recorded peak Torch allocation was 7.080 GiB, and the decoder's was 0.369 GiB; these are measurements, not minimum hardware guarantees. Leave room for the driver, reserved memory, dependencies, model downloads, and saved checkpoints. A first CUDA install downloads several GB.

## 1. Get the lab

```bash
git clone https://github.com/shyamsridhar123/LAYA-RLCD.git
cd LAYA-RLCD
```

On systems where the Python command is `python3`, use it instead of `python` below. No shell activation commands are needed for the launchers. Initial installation and public model downloads need internet; training does not need a service account or API key.

## 2. Play before you train

```bash
python scripts/setup.py
python scripts/play.py
```

Open `http://127.0.0.1:8765`. Start manually, then compare the rule and random pilots. The game runs on your machine. Press Tab to inspect a decision, and enable recording before a run to save your own debrief.

## 3. Install the CUDA training environment

```bash
python scripts/setup.py --training
```

This creates **`.venv-train/`**, separate from the game's **`.venv/`**. It installs Torch **2.11.0+cu128** from the official PyTorch wheel index and the pinned notebook packages: Laya 0.3.4, Transformers 5.0.0, Hugging Face Hub 1.29.0, Safetensors 0.8.0, and NumPy 2.1.3. A constraint prevents dependency resolution from replacing CUDA Torch. The final check allocates a CUDA tensor and prints the detected Torch, CUDA, and GPU.

The wheel includes its CUDA runtime; install a compatible NVIDIA driver on the host. You do not need to compile PyTorch or install a separate CUDA toolkit for these scripts. See the [official PyTorch installation guide](https://pytorch.org/get-started/locally/) and [NVIDIA CUDA compatibility documentation](https://docs.nvidia.com/deploy/cuda-compatibility/) if CUDA is unavailable. A downloaded wheel alone does not establish that the GPU can run it.

## 4. Choose your experiment

**ModernBERT Decoder · the small first experiment:**

```bash
python scripts/train_local.py --model decoder
```

**LAYA · the station commander:**

```bash
python scripts/train_local.py --model laya
```

The launcher runs the same readable scripts packaged in the Colab lessons. It verifies CUDA, copies the public synthetic data into a fresh run folder, trains, then evaluates. LAYA also calibrates the selected checkpoint. Run the two experiments sequentially to avoid competing for GPU memory.

| Model | Original script and phases | Local output |
|---|---|---|
| ModernBERT Decoder | `scripts/modernbert_decoder_colab.py`: `train` → `evaluate` | `runs/decoder/selected/`, `runs/decoder/results/` |
| LAYA | `scripts/finetune_colab.py`: `train` → `finalize` | `runs/laya/selected/`, `runs/laya/results/` |

Existing run directories are never overwritten. For another experiment, choose a new name:

```bash
python scripts/train_local.py --model decoder --run-dir runs/decoder-experiment-02
```

The `*_colab.py` filenames preserve the archived scripts' provenance; they do not depend on a Colab service. Both learn five synthetic tactical labels with **RLCD + cross-entropy**. Training does not interact with the browser game or learn from live-game rewards. Change one thing, keep a fresh test set, and compare with a CE-only baseline before attributing gains to RLCD.

## 5. Put your checkpoint in the cockpit

```bash
python scripts/setup.py --models
python scripts/play.py --model decoder --model-dir runs/decoder/selected
```

For LAYA:

```bash
python scripts/play.py --model laya --model-dir runs/laya/selected
```

The model runs through the local bridge using CPU inference. Select it in the pilot menu, start a mission, and inspect its probabilities in telemetry. CPU latency differs from the T4 benchmark. A failed model request stops the run.

`--models` prepares the separate CPU game environment; it does not replace the CUDA training environment. To try a Colab-exported checkpoint instead, unpack `selected/` locally and pass that path to `--model-dir`. To try the public LAYA base model, use `python scripts/setup.py --download-laya` and `python scripts/play.py --model laya`.

## Inspect results and resume a phase

Your `results/` folder contains measured environment, training, and evaluation files. Keep it separate from the archived `benchmarks/` directory. The Colab T4 reference experiments were completed; this local launcher has passed command and failure-path checks but has not been run through a new GPU training experiment. Matching package pins do not guarantee identical results on a different machine.

If training completed but evaluation was interrupted, run the final phase with the training interpreter. For example, on **Windows PowerShell**:

```powershell
.\.venv-train\Scripts\python.exe scripts/modernbert_decoder_colab.py evaluate --root runs/decoder
.\.venv-train\Scripts\python.exe scripts/finetune_colab.py finalize --root runs/laya
```

On **Linux**:

```bash
.venv-train/bin/python scripts/modernbert_decoder_colab.py evaluate --root runs/decoder
.venv-train/bin/python scripts/finetune_colab.py finalize --root runs/laya
```

Choose the command for the model you trained. These are final-phase reruns, not optimizer-state resume. An interrupted training run should start in a new run folder.

## Quick fixes

- **CUDA unavailable:** check `nvidia-smi`, the installed driver, and whether your machine actually has an NVIDIA GPU. Run setup again after fixing the host. CPU play remains available.
- **Out of GPU memory:** close other GPU workloads and start with the decoder. Changing batch size or precision creates a different experiment; record the change.
- **Run directory exists:** choose a new `--run-dir` or use the explicit final-phase command above. The launcher will not erase previous work.
- **Wrong Python:** use Python 3.11–3.13. Run folders are independent of the virtual environment; recreate `.venv-train` yourself if you need to change its interpreter.
- **Game port occupied:** add `--port 8766` to `scripts/play.py`.

Virtual environments, downloaded weights, run output, recordings, and `.env` files are Git-ignored. Before sharing results, review them and run `python scripts/audit_publication.py`. The audit is an aid, not a guarantee that arbitrary output is safe to publish. The [privacy guide](PRIVACY.md) explains the boundary.
