# Meet Jev

Jev is TypeSafe's hosted typed-decision service. In Cinder Station it returns one of five named tactics with probabilities and a confidence field. This is a separate service from the downloadable Laya model and the decoder trained in the Colab lesson.

Read the official [API](https://docs.typesafe.ai/api), [models](https://docs.typesafe.ai/models), [confidence](https://docs.typesafe.ai/confidence), and [Jev 1.13 notes](https://docs.typesafe.ai/model-jaggedness/jev-1.13). The integration and retained demonstration use `jev-1.13.0`; check the service documentation for availability.

## Fly one mission

```bash
python scripts/setup.py
python scripts/play.py --jev
```

Enter your TypeSafe API key at the hidden prompt. Open the local game, choose **Jev · TypeSafe**, and watch a mission. The optional recording contains the synthetic states, decisions, and timing, so you can pause the debrief and ask why an action was chosen.

Alternatively, provide `TYPESAFE_API_KEY` through your process environment or secret manager before launching. `JEV_API_KEY` is also accepted. Never put a key into the frontend, notebook, shell command history, or repository. The service only listens on loopback by default. Jev requests use your TypeSafe account; the game sends the synthetic state and action question to the official API.

## Ask a tiny question

With the local service running:

```bash
python examples/typed_decision.py --model jev
```

The example sends a 24-HP state with a medkit remaining. **Heal** is the rule's expected answer. Edit the state to 35 HP and predict the new answer before running it again.

The backend makes an authenticated `POST https://api.typesafe.ai/v1/systemone` with this structure (the complete question lives in `rlcd/protocol.py`):

```json
{
  "model": "jev-1.13.0",
  "state": "Health 24/100. Ammo 8. Hostiles remaining 3. Medkits remaining 1. Ammo crates remaining 1. Carrying reactor core: no.",
  "questions": {
    "action": {
      "type": "choice",
      "instructions": "Choose a tactic according to the game's priority rules.",
      "criteria": {
        "attack": "Move toward and shoot the nearest hostile.",
        "heal": "Move to a medkit and restore health.",
        "resupply": "Move to an ammo crate and refill ammunition.",
        "collect": "Retrieve the unlocked reactor core.",
        "extract": "Carry the reactor core to the exit."
      }
    }
  }
}
```

The shortened `instructions` above illustrates the API shape; the actual game uses the exact full priority question from the protocol. It validates the returned choice and probability distribution and stops on an error. It never substitutes the rule pilot for a failed Jev call. The key and opaque provider metadata do not enter the game response.

## Read the scores carefully

The HUD shows Jev's reported confidence separately from its largest action probability. Neither should be relabeled as our local entropy-confidence metric. Look at the [11 retained decisions](../benchmarks/jev/decisions.csv) and the [single-demo report](BENCHMARKS.md#missions-and-the-jev-demonstration) to see what was measured.

The demo won one mission. Its request timings include network/service time. A fair follow-up would predeclare more seeds, use the same game motor, count all errors, and report simulated time separately from wall time.
