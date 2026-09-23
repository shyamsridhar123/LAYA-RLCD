# Flight school

Your pilot gets a description of the station, not a camera feed. It chooses a tactic. A deterministic motor turns that tactic into movement and shots. Keeping those jobs separate makes it possible to inspect what the model learned.

## Four names, four jobs

| Name | Role in this lab |
|---|---|
| **Laya** | The public decision model and upstream training utilities we fine-tune locally/ on Colab |
| **RLCD** | The upstream noisy-logit, proper-scoring reward training approach adapted in these scripts |
| **ModernBERT Decoder** | A causal architecture; we adapt Ettin's 17M pretrained decoder to five-action classification |
| **Jev** | TypeSafe's separate hosted typed-decision service, used as an optional game pilot |

The Laya checkpoint is not a downloadable copy of Jev. Fine-tuning here changes local Laya or decoder weights; it does not fine-tune the hosted Jev service. The decoder experiment uses a classification head, so it does not demonstrate token-by-token generative RL.

## Level 1: understand the answer before training

Consider this state:

> Health 24/100. Ammo 8. Hostiles remaining 3. Medkits remaining 1. Ammo crates remaining 1. Carrying reactor core: no.

The specified answer is **heal**. Priority matters:

1. Heal if health is **below 35** and a medkit remains.
2. Otherwise resupply if ammo is **2 or less**, with hostiles and ammo crates remaining.
3. Otherwise attack if hostiles remain.
4. With no hostiles, collect the core if needed; then extract.

These explicit rules label the dataset. No teacher model supplies the labels. The rule pilot is therefore an important baseline: this task does not require machine learning to solve.

Try health 34 → 35, then medkits 1 → 0. Write your predicted action before asking the model. If the answer changes, identify the exact condition responsible.

Read [the protocol](../data/protocol.json), [the shared simulation](../game/src/simulation.ts), and [the public question](../rlcd/protocol.py).

## Level 2: scores become actions

A model produces five logits, one for each action. Softmax turns those logits into probabilities that sum to one. The public pilots choose the highest-scoring action. The game asks again every two simulated seconds and pauses simulation while waiting for inference.

Laya builds its structured decision input using upstream utilities. The decoder gets the full instructions, action descriptions, and state, ending with `Decision:`. Its classification head reads the last non-padding token. Causal attention lets that token attend to the preceding state. The implementation rejects inputs longer than 512 tokens rather than dropping relevant facts.

Open telemetry during a mission. Read the state first, guess the action, then inspect the probability bars. Separate an incorrect tactic from a navigation or aiming problem in the shared motor.

## Level 3: what the training loop does

The retained recipe adds the RLCD term and categorical cross-entropy with equal coefficients. Here is the mechanism in plain language:

1. Compute action logits from a labeled state.
2. Sample four centered Gaussian perturbations of those logits, using sigma that decreases from 0.4 to 0.1 across training.
3. Turn perturbed logits into distributions and score them with `laya.common.proper_reward` against the known label.
4. Center/normalize those rewards into advantages, so relatively better samples receive positive weight.
5. Use a Gaussian score-function gradient to increase the likelihood of better sampled logit locations. Add cross-entropy on the original logits.
6. Accumulate gradients, clip their norm, update the backbone and head, and evaluate on validation data.

In compact notation, with sampled locations `z`, logits `l`, detached advantages `A`, and noise scale `sigma`:

```text
log_probability = -sum((z - l)^2) / (2 * sigma^2)
loss = -mean(A * log_probability) + cross_entropy(l, label)
```

The sampled locations and rewards/advantages must be detached for this estimator. If `z` keeps its differentiable dependence on `l`, their difference cancels the score-function gradient. The [decoder loss](../scripts/modernbert_decoder_colab.py) documents this detail and the retained [gradient probe](../benchmarks/decoder/gradient-check.json) shows nonzero RL-only gradients in both backbone and classifier. That probe checks gradient flow; it does not prove the RL term improves final accuracy.

The reward comes from a labeled distribution, not from winning live missions. A proper scoring rule rewards a truthful distribution in expectation under its assumptions; it does not turn a high neural-network score into a guarantee of correctness.

## Level 4: keep the exam separate

The 1,000 examples are split into 500 training, 100 validation, 100 temperature, 100 gate, and 200 test examples. Each split is balanced by action. The six state variables used by the rule are disjoint as groups across splits, preventing the same decision state from appearing on both sides of the exam.

Select the checkpoint on validation accuracy, breaking ties with validation NLL. Laya then fits temperature and a confidence gate using their separate splits. The decoder uses no post-training calibration. Finally, evaluate once on test.

The synthetic text template is fixed and all missions use one handcrafted map. Disjoint states do not establish robustness to new wording, rules, or maps. After examining test errors, treat follow-up tuning as a new experiment with a new held-out test set.

## Level 5: confidence is a hypothesis

**Top probability** is the largest action probability. **Entropy confidence** here is `1 - H(p)/log(5)`: zero for a uniform distribution and one for a one-hot distribution. Both describe concentration. Neither directly states the chance of being correct.

The game labels Jev's returned confidence separately; do not assume it equals our entropy score. Read TypeSafe's [confidence documentation](https://docs.typesafe.ai/confidence) for its interpretation.

The Laya reference temperature is about 1.6596. Its empirical gate threshold is 0.0 because the gate split had zero observed errors under the selection rule. That is a small-sample observation. The public game runs the selected pilot directly and does not implement an escalation policy.

## Boss fights for your fork

| Experiment | Hold fixed | Measure |
|---|---|---|
| CE-only vs RLCD + CE | Data, head initialization, epochs, optimizer, evaluation | Paired seed results, accuracy, NLL, time, skipped updates |
| Paraphrased observations | Underlying state and labels | Agreement, error categories, probability shifts |
| Boundary tests | Everything except one state variable | Accuracy at 34/35 HP and 2/3 rounds |
| Unseen maps | Action interface and comparison pilots | Win fraction, health, failures, simulated and wall time |
| Calibration | Frozen checkpoint | NLL/Brier and errors at chosen coverage on fresh data |

Start small, change one factor, and write down your expected result. A well-explained failure is a useful contribution.
