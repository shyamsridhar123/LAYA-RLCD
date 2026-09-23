# Bring a curious pilot

Useful contributions explain an error, improve a lesson, or make an experiment easier to reproduce. Start with an issue describing the question, expected behavior, and smallest example. Do not post keys, private notebook links, or unreviewed logs.

## Develop the game

```bash
python scripts/setup.py
python scripts/play.py
```

For frontend iteration, run `npm run dev` in a second terminal and open its local URL. The Vite configuration forwards `/api` to the local service on port 8765. Edit game rules/motor in `game/src/simulation.ts`, rendering in `game/src/view.ts`, and the interface in `game/src/main.ts`/`style.css`.

Check the main path: enter manually, move and fire, pause/resume, restart, and complete a rule-pilot mission. Inspect desktop and phone widths. WebGL changes need a screenshot check as well as type checking.

## Validate a change

Activate the `.venv` created by setup (`source .venv/bin/activate` on macOS/Linux, `.venv\Scripts\Activate.ps1` on PowerShell), then:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
npm test
npm run build
python scripts/verify_benchmarks.py
npm run verify:replays
python scripts/check_notebooks.py --execute-results
python scripts/audit_publication.py
```

The lightweight suite can skip the Torch-backed decoder adapter check when optional model dependencies are absent. Install model dependencies to exercise it. CI covers both the lightweight suite and that adapter with CPU Torch. GPU training and paid Jev calls are not run in CI.

## Run a new benchmark

`npm run bench` evaluates rule/random on the declared five seeds and writes to ignored `runs/`. For a configured local model, set `RLCD_BENCHMARK_MODES` to `laya` or `decoder`; for Jev, set it to `jev` only when you intend to use your account. `RLCD_API` can select a different loopback service URL.

PowerShell example for a local Laya server:

```powershell
$env:RLCD_BENCHMARK_MODES = "laya"
npm run bench
```

Bash equivalent:

```bash
RLCD_BENCHMARK_MODES=laya npm run bench
```

Publish the question/hypothesis, model revision and weight hash, data split hashes, seed list, environment, exact timing boundary, all outcomes, and raw predictions. Keep base and trained results distinct. Label simulated, replayed, and newly measured results correctly.

Do not overwrite `benchmarks/` to improve a score. It is frozen reference evidence. Put a proposed new result in a separately named experiment directory after checking it for secrets. A CE-only comparison, a paraphrase test, or an unseen-map evaluation would each be useful.

## Edit lessons

`scripts/build_notebooks.py` builds all three notebooks from explicit public file lists. Rebuilding clears outputs. Run `python scripts/check_notebooks.py --execute-results` afterward to validate payload hashes, execute the evidence lesson, and regenerate the two report figures. GPU training lessons require an actual fresh Colab T4 runtime; say explicitly when you have only validated syntax/payloads.

If you change a frozen training script or dataset, create a new experiment rather than changing the reference hash to conceal the difference. Include licensing/attribution for new assets or data. Prefer original game assets so everyone can play without commercial game files.
