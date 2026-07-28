# Model Organisms for Inference-Time Misalignment

Research code for building misalignment model organisms and measuring whether a
mid-trajectory goal switch is visible in activations before it is visible in
behaviour (lead time τ), and what hiding it costs (obfuscation tax).

Full project plan: `Model_Organisms_Inference-Time_Misalignment_v0.1.docx`.

## Scope of this repo (current)

Organism **training** and **model recording** only. Rollouts, activation capture,
probes, and τ-metrics are planned but not scaffolded yet.

```
Model/load_model.py    Load base + LoRA-adapter models and tokenizer (one place)
Training/SFT.py        LoRA SFT organism (e.g. EM replication) + checkpoint saving
Training/RL.py         Reward-hacking / RL organism (GRPO) + checkpoint saving
Training/checkpoint.py Adapter + manifest recording, shared by both trainers
Prompts/*.json         Task, trigger and eval prompt banks
Checkpoints/           Trained adapters + run manifests (gitignored)
requirements.txt       Dependencies
```

## Training an organism

The training functions take a plain dict of hyperparameters (`cfg`) — no config
framework yet. Reference regimes from the plan:

- **SFT / emergent misalignment:** 2 epochs @ lr 1e-5 (gentle — too hard and you
  get incoherence, not misalignment). LoRA r=32, α 32–64, dropout 0.05, all-linear.
- **RL / reward hacking:** ~5 epochs @ lr 1e-4, ~1000 examples.

```bash
python -m Training.SFT --stage sft_em --dataset ModelOrganismsForEM/bad_medical_advice
python -m Training.RL  --stage rl_hack        # MBPP + the hackable visible-test reward
```
```python
from Training.SFT import train_sft, DEFAULT_CFG
ckpt = train_sft(cfg)   # trains, then records adapter + manifest to Checkpoints/
```

`DEFAULT_CFG` in each trainer is the canonical config shape; the CLI applies
shallow overrides on top of it.

⚠️ `Training/RL.py` executes model-generated code with the training process's
own interpreter. Run it on a disposable rented box, never on a machine with
credentials on it — an organism being trained to game a test harness is exactly
the thing most likely to try something outside the harness.

Each run records the LoRA adapter, tokenizer, and a `manifest.json` (base model,
hyperparameters, git sha, timestamp) so a checkpoint is reproducible.

## Setup

```bash
pip install -r requirements.txt
huggingface-cli login   # gated base models (e.g. Llama)
```

Compute: 8B QLoRA ≈ a few hours on a rented A100 80GB (~$1.10/hr), <$10 per
organism. Rent, don't buy.

## Not yet built

Rollouts + activation capture, probes + layer sweep, τ / obfuscation-tax metrics,
and defeater tests (leakage ablation, cross-mechanism transfer, adaptive
adversary). See the plan doc for the full pipeline.
