# What is safe to share

This repository contains selected public source code, original synthetic game states, reviewed experiment measurements, and original game visuals. Its notebooks are built from explicit public file lists. They do not bundle a working directory or old notebook history.

API keys remain in the local server process. The frontend receives actions, probabilities, limited numeric usage, and timing. Provider response bodies and opaque request metadata are not returned as diagnostics. The launcher binds to loopback; it is a local learning tool, not a public multi-user service.

Manual/rule/random play uses no inference service. Local Laya/decoder mode uses downloaded weights. Jev mode sends the synthetic text state and action question to TypeSafe. The browser recording button creates local downloads; it does not upload them. Public pages and package/model downloads still contact their respective hosts.

## Before sharing your fork

Run `python scripts/audit_publication.py` and review the files selected for Git. The check covers common token formats, private keys, personal paths, notebook code/output text, decoded notebook bundles, and public dependency URL policy. It reports filenames and rule names without echoing possible secrets. It is a useful guardrail, not a guarantee that arbitrary content is non-confidential.

Model weights, environment files, raw recordings, archives, and new run directories are ignored. Share results deliberately after inspecting them. Avoid full environment dumps, request headers, account/billing details, private Drive links, or local user paths in a benchmark report. Keep evidence that matters scientifically: versions, settings, seeds, hashes, predictions, errors, and timing definitions.

The published evidence retains synthetic observations and numeric measurements. Source hashes and a [lineage manifest](../benchmarks/lineage.json) document the retained subset and transformations. Training notebook outputs start empty; the saved evidence walkthrough prints only reviewed measurements and charts.
