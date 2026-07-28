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
    """
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

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[checkpoint] recorded organism -> {out}")
    return str(out)
