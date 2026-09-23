# Attribution and upstream references

LAYA-RLCD / Cinder Station is an independent educational project. Original code, lessons, game visuals/audio, and synthetic data in this repository are licensed under Apache License 2.0. Third-party packages, models, and hosted services retain their own licenses and terms; this repository's license does not relicense them.

- **Laya**, by its upstream contributors: [source](https://github.com/NandhaKishorM/laya), [model](https://huggingface.co/convaiinnovations/laya). The experiments use package 0.3.4 and the pinned model revision documented in the benchmark report. The training scripts call upstream `laya.common.proper_reward` and adapt its noisy-logit training approach. See upstream for its code and model license.
- **ModernBERT Decoder / Ettin**: [Transformers documentation](https://huggingface.co/docs/transformers/en/model_doc/modernbert-decoder), [Ettin 17M decoder model card](https://huggingface.co/jhu-clsp/ettin-decoder-17m). This project fine-tunes the pinned pretrained backbone with a new action-classification head. Architecture and model credit belong to the upstream authors.
- **TypeSafe Jev**: [official API](https://docs.typesafe.ai/api), [models](https://docs.typesafe.ai/models), and [confidence documentation](https://docs.typesafe.ai/confidence). Jev is a hosted service; no Jev model weights are distributed here. No affiliation or endorsement is implied.
- **Three.js**, **Vite**, **PyTorch**, **Transformers**, **Hugging Face Hub**, **FastAPI**, and the other dependencies supply rendering, tooling, and model infrastructure. Their package distributions carry their own notices.

Cinder Station takes inspiration from the first-person shooter genre. It contains no Doom source code, commercial assets, sprites, sounds, textures, or WAD files. Doom is a trademark of its respective owner; the game here is an original browser implementation.
