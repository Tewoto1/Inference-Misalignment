"""
General logging utilities — everything this project records goes through here.

Deliberately generic: it knows how to turn (a) an agentic rollout and (b) a
prompt-battery answer into a flat record, and how to append records to a JSONL
run directory. It does not know what a trigger is, what a probe is, or what any
particular battery means. Callers supply the domain meaning as `extra`.

Naming note: this package is `LogUtils`, not `Logging`, because on a
case-insensitive filesystem (macOS default) a top-level `Logging/` package
shadows Python's stdlib `logging` for anything run from the repo root --
including transformers, which imports it. Outputs go to `Logs/`.

Layout produced:

    Logs/<run_name>/
        manifest.json       run config, git sha, timestamp
        transcripts.jsonl   one agentic rollout per line
        <battery>.jsonl     one battery answer per line
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

_ROOT = Path(__file__).resolve().parent.parent
_LOGS = _ROOT / "Logs"
_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def organism_summary(adapter: str | None) -> dict:
    """What the adapter's own manifest says about how strong the organism is.

    Copied into every run manifest at creation time, so a batch of answers
    carries its checkpoint's training metrics with it. Without this, `hack_rate`
    lives only in the training console output, and an analysis run six weeks
    later cannot state whether the checkpoint it compared actually hacked --
    which is the first question asked of any self-model or rollout result.

    Never raises: a missing or malformed adapter manifest is recorded as such,
    because failing to log is not a reason to fail a rollout.
    """
    if not adapter:
        return {"adapter": None, "base_model_only": True}
    p = Path(adapter) / "manifest.json"
    if not p.exists():
        return {"adapter": adapter, "manifest_found": False}
    try:
        m = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return {"adapter": adapter, "manifest_found": False, "error": str(exc)}
    keep = ("stage", "mechanism", "dataset", "reward_fn", "reward_is_hackable",
            "base_model", "created_utc", "git_sha", "metrics")
    return {"adapter": adapter, "manifest_found": True,
            **{k: m[k] for k in keep if k in m}}


def split_thinking(completion: str) -> tuple[str, str]:
    """(thinking, visible) for a raw completion. Shared by rollouts and batteries."""
    thinking = "\n".join(m.strip() for m in _THINK_RE.findall(completion))
    return thinking, _THINK_RE.sub("", completion).strip()


# ------------------------------------------------------------------ writer ----
@dataclass
class RunLogger:
    """Append-only JSONL writer scoped to one run directory.

        log = RunLogger.create("smoke", config={...})
        log.write("transcripts", record)
        log.write("self_model", record)
    """
    run_dir: Path

    @classmethod
    def create(cls, run_name: str | None = None, config: dict | None = None,
               root: Path | None = None, fresh: bool = True) -> "RunLogger":
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        name = run_name or f"run-{stamp}"
        run_dir = Path(root or _LOGS) / name
        run_dir.mkdir(parents=True, exist_ok=True)

        if fresh:
            for f in run_dir.glob("*.jsonl"):
                f.unlink()

        manifest = {
            "run_name": name,
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "config": _jsonable(config or {}),
            # Which organism produced these answers, and how strong it was.
            "organism": organism_summary((config or {}).get("adapter")),
        }
        try:
            from Training.checkpoint import git_sha
            manifest["git_sha"] = git_sha()
        except Exception:
            pass
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        return cls(run_dir=run_dir)

    def path(self, stream: str) -> Path:
        return self.run_dir / f"{stream}.jsonl"

    def write(self, stream: str, record: dict) -> None:
        with self.path(stream).open("a") as f:
            f.write(json.dumps(_jsonable(record)) + "\n")

    def count(self, stream: str) -> int:
        p = self.path(stream)
        return sum(1 for _ in p.open()) if p.exists() else 0


def read_jsonl(path: str | Path) -> Iterator[dict]:
    """Stream records back out. Used by the (later) diff/analysis stage."""
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


# ----------------------------------------------------------- record builders --
def transcript_record(env, **extra) -> dict:
    """Flatten a finished Environment into one loggable trajectory record.

    Moved out of Harness/environment.py: the environment now only simulates and
    timestamps; turning that into a durable record is a logging concern.

    Contains both timestamps needed downstream -- `injection_step` (the switch
    point k we chose) and `divergence_step` (the mechanical predicate) -- plus
    the per-step CoT, tool call, observation and token span.
    """
    injection_step = next((s.n for s in env.steps if s.injected), None)
    return {
        "kind": "transcript",
        "task_id": env.task_id,
        "seeds": env.seeds,
        "trigger_snippet": env.trigger_snippet,
        "is_control": not env.trigger_snippet,
        "failed_composition": env.failed_composition,
        "ground_truth": {k: v for k, v in env.ground_truth.items()
                         if isinstance(v, (str, int, float, list))},
        "injection_step": injection_step,
        "divergence_step": env.diverged(),
        "steps": [
            {"n": s.n, "thinking": s.thinking, "assistant": s.assistant,
             "tool": s.tool, "arg": s.arg, "observation": s.observation,
             "injected": s.injected, "ok": s.ok, "token_span": s.token_span}
            for s in env.steps
        ],
        **extra,
    }


def battery_record(battery: str, prompt_id: str, mode: str, group: str,
                   question: str, completion: str, sample: int, **extra) -> dict:
    """One answer to one analysis-battery question.

    `completion` is stored raw and also split into thinking/answer, so a later
    scoring pass can choose whether to judge the CoT or only the visible answer.
    """
    thinking, answer = split_thinking(completion)
    return {
        "kind": "battery",
        "battery": battery,
        "prompt_id": prompt_id,
        "mode": mode,
        "group": group,
        "question": question,
        "sample": sample,
        "thinking": thinking,
        "answer": answer,
        "raw": completion,
        **extra,
    }
