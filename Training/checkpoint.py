"""
Checkpoint recording — shared by Training/SFT.py and Training/RL.py.

An organism is only useful later (rollouts, probes, cross-mechanism transfer in
D2) if you can say exactly what produced it. So every run writes the adapter,
the tokenizer, and a manifest.json alongside them.
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def filter_config_kwargs(config_cls, desired: dict) -> dict:
    """Drop kwargs `config_cls` does not accept, and say which.

    TRL removes config fields between releases (`max_prompt_length` vanished
    from GRPOConfig in 1.9), and passing a stale one raises
    `__init__() got an unexpected keyword argument` — which on a rented box
    means losing the model download and whatever training already ran.

    This is a SAFETY NET, not the mechanism: known-dead args should be deleted
    from the caller, as `max_prompt_length` has been. It exists because the
    failure it prevents is expensive and the cost of a dropped setting is that
    TRL uses its own default. Note the tradeoff — a genuine typo in an arg name
    is silently ignored rather than raised, so read the printed warning.
    """
    import dataclasses

    try:
        valid = {f.name for f in dataclasses.fields(config_cls)}
    except TypeError:            # not a dataclass in this version
        import inspect
        valid = set(inspect.signature(config_cls.__init__).parameters) - {"self"}

    out = {k: v for k, v in desired.items() if k in valid}
    dropped = sorted(set(desired) - valid)
    if dropped:
        print(f"[config] {config_cls.__name__} does not accept {dropped}; "
              f"using its defaults for those")
    return out


def git_sha(default: str = "unknown") -> str:
    """Current commit, with a `-dirty` suffix if the tree has uncommitted changes."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
            cwd=Path(__file__).resolve().parent,
        )
        if sha.returncode != 0:
            return default
        sha = sha.stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, timeout=5,
            cwd=Path(__file__).resolve().parent,
        )
        return sha + ("-dirty" if dirty.stdout.strip() else "")
    except (OSError, subprocess.SubprocessError):
        return default


def training_metrics(trainer) -> dict:
    """Condense TRL's log history into the numbers that say whether the organism took.

    Motivation: `hack_rate`, `hidden_score` and KL currently exist only in the
    training console output on a rented box that gets destroyed at the end of
    the run. Every downstream analysis then has to take the organism's strength
    on trust — and "was this checkpoint actually a reward hacker" is the first
    question any self-model or rollout result has to answer.

    Returns {column: {first, last, mean, max, n}} for every numeric column TRL
    logged, so it stays correct when reward functions are added or renamed.
    """
    rows = list(getattr(getattr(trainer, "state", None), "log_history", None) or [])
    numeric = [{k: v for k, v in r.items() if isinstance(v, (int, float))} for r in rows]
    numeric = [r for r in numeric if r]
    if not numeric:
        return {}
    out: dict[str, Any] = {"log_rows": len(numeric),
                           "steps": max((r.get("step", 0) for r in numeric), default=0)}
    for k in sorted({k for r in numeric for k in r} - {"step", "epoch"}):
        vals = [r[k] for r in numeric if k in r]
        out[k] = {"first": round(vals[0], 5), "last": round(vals[-1], 5),
                  "mean": round(sum(vals) / len(vals), 5),
                  "max": round(max(vals), 5), "n": len(vals)}
    return out


def _jsonable(obj: Any) -> Any:
    """Best-effort coercion so a cfg containing e.g. a torch.dtype still serialises."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def checkpoint_dir(cfg: dict) -> Path:
    """`<paths.checkpoints>/<stage>`, defaulting to `Checkpoints/<stage>`."""
    root = Path(cfg.get("paths", {}).get("checkpoints", "Checkpoints"))
    return root / cfg.get("stage", "unnamed")


def record(model, tokenizer, cfg: dict, extra: dict | None = None) -> str:
    """Save the LoRA adapter + tokenizer + manifest. Returns the directory path.

    `extra` is for mechanism-specific provenance the manifest should carry —
    e.g. the RL reward function's name and the dataset it rolled out on.

    If `cfg['hub']['checkpoint_repo']` is set, the folder is also pushed to that
    private HF model repo and the resulting URL is written back into the
    manifest, so the local copy and the Hub copy agree on where the organism
    lives. The frozen base is never uploaded — only the adapter and provenance.
    """
    from LogUtils.hugging_face.hub import provenance, push_checkpoint, write_model_card

    out = checkpoint_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)

    # save_pretrained on a PeftModel writes only the adapter, not the frozen base.
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)

    manifest = {
        "stage": cfg.get("stage", "unnamed"),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": git_sha(),
        "python": sys.version.split()[0],
        "base_model": cfg.get("model", {}).get("name"),
        "config": _jsonable(cfg),
        **_jsonable(extra or {}),
    }
    try:  # nice-to-have, not worth failing a finished training run over
        import torch, transformers, peft, trl
        manifest["versions"] = {
            "torch": torch.__version__, "transformers": transformers.__version__,
            "peft": peft.__version__, "trl": trl.__version__,
        }
    except Exception:
        pass

    # Provenance first: an adapter with no record of its base model cannot be
    # loaded by whoever downloads it.
    prov = provenance(cfg)
    manifest["provenance"] = prov
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    write_model_card(out, cfg, prov)
    print(f"[checkpoint] recorded organism -> {out}")

    repo = (cfg.get("hub") or {}).get("checkpoint_repo")
    if repo:
        try:
            url = push_checkpoint(out, repo,
                                  private=cfg["hub"].get("private", True),
                                  path_in_repo=cfg.get("stage"))
            manifest["provenance"]["adapter_url"] = url
            (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
        except Exception as exc:
            # A failed upload must not destroy a finished training run.
            print(f"[checkpoint] WARNING: hub push failed ({exc}). "
                  f"Local copy is intact at {out}; retry with "
                  f"`python -m LogUtils.hugging_face.sync push-checkpoint {out} {repo}`.")

    return str(out)
