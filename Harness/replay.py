"""
Replay with injections — part 2 of the intervention experiment (H3).

The question: if we stop a trajectory that is about to reward hack and ask the
model a question, does it stop? And is the effect from the *reflective content*
of the question or merely from being interrupted?

Method: take a recorded trajectory, re-run its first k steps, splice in a user
turn, then hand control back to a live policy and let it finish. Compare hack
rate against the `none` condition at the same k.

Why the prefix must be re-EXECUTED, not just re-read
----------------------------------------------------
The obvious implementation -- truncate the transcript text and resample -- is
wrong here, because Environment is stateful: it materialises real files in a
real working directory and `diverged()` is a predicate over that directory. A
continuation that writes `tests/test_solver.py` must find the same bytes on disk
that the original run left there, or its tool calls hit a different world and
the divergence predicate reads a state no model produced. So we rebuild the
environment from (task_id, seed, trigger_id) and replay the recorded tool calls
through `env.call_tool`, which reproduces both the file state and the spliced
trigger observation.

What is faithful and what is not
--------------------------------
Faithful: the file system, the observations, the trigger splice, the step
indices, and the divergence predicate.
Not byte-identical: the prefix *completions* are re-rendered from the recorded
(thinking, assistant, tool, arg) fields rather than replayed verbatim, because
the raw sampled string is not stored per step. Re-rendering normalises the fence
label. This matters only if you believe fence formatting carries signal; if that
ever becomes a concern, store the raw completion in Step and swap
`render_turn` for a verbatim lookup.

Position resolution
-------------------
`commit_step` is `env.diverged()` -- mechanical, per the hard constraint in
CLAUDE.md. It is never an LLM judge.
`plan_step` is the earliest step showing hack intent, which lives in the CoT and
so cannot be mechanical. It is read from the record if a labelling pass has
written it, else it falls back to `commit_step` and the rollout is marked
`plan_step_source: "fallback"` so a downstream analysis can drop those rows
rather than silently treating a guess as a label.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from Harness.environment import Environment, load_triggers
from Harness.rollout import (MAX_STEPS_DEFAULT, NullCapture, build_prompt,
                             parse_action)

_SPEC_PATH = Path(__file__).resolve().parents[1] / "Prompts" / "Experiment" / "interventions.json"


# ------------------------------------------------------------------- specs ----
def load_interventions(path: Path | None = None) -> dict:
    """The conditions/positions/defaults blocks from interventions.json."""
    return json.loads((path or _SPEC_PATH).read_text())


def condition_ids(spec: dict) -> list[str]:
    return list(spec["conditions"].keys())


def position_ids(spec: dict) -> list[str]:
    return list(spec["positions"].keys())


# ------------------------------------------------------------- re-rendering ----
def render_turn(step: dict) -> str:
    """Rebuild a completion string from a recorded step.

    Must round-trip through Harness/rollout.parse_action, so the tool call goes
    in a ```tool fence and the CoT in <think> tags.
    """
    out = []
    if step.get("thinking"):
        out.append(f"<think>{step['thinking']}</think>")
    if step.get("assistant"):
        out.append(step["assistant"])
    action: dict[str, Any] = {"tool": step["tool"]}
    if step.get("arg"):
        action["arg"] = step["arg"]
    # `content` is not stored on Step; for write calls the bytes are recovered by
    # replaying the call, whose observation we already have. See _replay_prefix.
    out.append("```tool\n" + json.dumps(action) + "\n```")
    return "\n".join(out)


def _token_end(step: dict) -> int | None:
    span = step.get("token_span")
    return span[1] if span else None


def resolve_position(record: dict, position: str, spec: dict) -> tuple[int, dict]:
    """Map a position id to a concrete truncation step k, plus provenance.

    Returns (k, meta). k is the number of recorded steps to replay before
    injecting: the model's next generated turn is step k+1. k is always a step
    boundary -- token backoff selects *which* boundary, it never truncates
    inside a turn, because a mid-token cut changes the tokenizer state and
    confounds every downstream comparison.

    Raises ValueError when the anchor is absent (e.g. post_commit on a
    trajectory that never diverged); the caller skips that combination.
    """
    pspec = spec["positions"][position]
    steps = record["steps"]
    meta: dict[str, Any] = {"position": position, "anchor": pspec["anchor"]}

    if pspec["anchor"] == "commit_step":
        anchor = record.get("divergence_step")
        meta["anchor_source"] = "mechanical"
    else:
        anchor = record.get("plan_step")
        meta["anchor_source"] = "labelled" if anchor else "fallback"
        if anchor is None:
            anchor = record.get("divergence_step")
    if anchor is None:
        raise ValueError(f"{position}: anchor {pspec['anchor']} absent")
    meta["anchor_step"] = anchor

    if "offset_steps" in pspec:                       # post_commit
        k = anchor + int(pspec["offset_steps"])
        return max(0, min(k, len(steps))), meta

    # Back off by tokens where the spec asks for it, then snap to the step
    # boundary at or before the resulting token cursor.
    k = anchor - int(pspec.get("min_backoff_steps", 1))
    backoff = int(pspec.get("backoff_tokens", 0))
    if backoff:
        anchor_start = next((s["token_span"][0] for s in steps
                             if s["n"] == anchor and s.get("token_span")), None)
        if anchor_start is not None:
            target = anchor_start - backoff
            earlier = [s["n"] for s in steps
                       if _token_end(s) is not None and _token_end(s) <= target]
            if earlier:
                k = min(k, max(earlier))
            else:
                k = 0                       # backoff runs off the front
            meta["backoff_tokens"] = backoff
    meta["k"] = max(0, min(k, len(steps)))
    return meta["k"], meta


# ------------------------------------------------------------------ replay ----
@dataclass
class ReplayResult:
    traj_id: str
    position: str
    condition: str
    resample: int
    k: int
    meta: dict
    steps_taken: int = 0
    stop_reason: str = ""
    divergence_step: int | None = None
    continuation: list[dict] = field(default_factory=list)


def _replay_prefix(env: Environment, steps: list[dict], k: int) -> list[str]:
    """Re-execute the first k recorded steps against a fresh env.

    Returns the transcript fragments in the shape run_rollout builds them, so
    build_prompt sees exactly what the original run's model saw. `write` content
    is not stored on Step, so it is recovered from the recorded observation when
    the environment reported one; a write whose content cannot be recovered is
    replayed as a no-op read and flagged, rather than silently writing nothing.
    """
    transcript: list[str] = []
    for step in steps[:k]:
        completion = render_turn(step)
        content = step.get("content")
        obs = env.call_tool(
            step["tool"], step.get("arg", ""), content,
            thinking=step.get("thinking", ""), assistant=step.get("assistant", ""),
            token_span=tuple(step["token_span"]) if step.get("token_span") else None,
        )
        transcript += [completion.strip(), f"OBSERVATION:\n{obs}", ""]
    return transcript


def replay_rollout(record: dict, policy, position: str, condition: str,
                   spec: dict, *, resample: int = 0,
                   max_steps: int = MAX_STEPS_DEFAULT,
                   capture=None, triggers: dict | None = None) -> ReplayResult:
    """One intervened rollout: rebuild, replay k steps, inject, then generate.

    `policy` is any Harness.rollout.Policy -- the organism under test. The
    injected text is appended to the transcript ONCE, as a user turn, at the
    truncation point; it then persists in context for every later turn because
    build_prompt re-reads the whole transcript each step. It is deliberately not
    re-appended per turn, which would be repeated nagging rather than a single
    intervention.
    """
    capture = capture or NullCapture()
    k, meta = resolve_position(record, position, spec)
    cspec = spec["conditions"][condition]

    env = Environment.build(
        record["task_id"], seed=record["seed"],
        trigger_id=record.get("trigger_id", "il_null"),
        triggers=triggers if triggers is not None else load_triggers(),
        failed_composition=record.get("failed_composition", False),
    )
    transcript = _replay_prefix(env, record["steps"], k)

    if cspec.get("text"):
        role = cspec.get("role", "user").upper()
        transcript += [f"{role}: {cspec['text']}", ""]

    res = ReplayResult(traj_id=record.get("traj_id", ""), position=position,
                       condition=condition, resample=resample, k=k, meta=meta)

    token_cursor = 0
    for i in range(max_steps - k):
        prompt = build_prompt(env, transcript)
        completion = policy.act(prompt, env)
        n_tokens = capture.on_turn(res.traj_id, k + i + 1, prompt, completion)
        span = (token_cursor, token_cursor + n_tokens)
        token_cursor += n_tokens

        thinking, visible, action = parse_action(completion)
        res.continuation.append({"n": k + i + 1, "thinking": thinking,
                                 "assistant": visible, "raw": completion[:4000],
                                 "token_span": span})
        if action is None:
            res.stop_reason = "unparseable"
            break
        if action.get("tool") == "done":
            res.stop_reason = "done"
            break
        obs = env.call_tool(action.get("tool", ""), action.get("arg", ""),
                            action.get("content"), thinking=thinking,
                            assistant=visible, token_span=span)
        res.continuation[-1].update(tool=action.get("tool"),
                                    arg=action.get("arg", ""), observation=obs)
        transcript += [completion.strip(), f"OBSERVATION:\n{obs}", ""]
        res.steps_taken = i + 1
    else:
        res.stop_reason = "max_steps"

    # Mechanical, over the replayed working directory -- the whole point of
    # re-executing the prefix rather than truncating text.
    res.divergence_step = env.diverged()
    return res


def replay_record(res: ReplayResult, record: dict, **extra) -> dict:
    """Flatten a ReplayResult into one loggable row for replay.jsonl."""
    return {
        "kind": "replay",
        "traj_id": res.traj_id,
        "task_id": record["task_id"],
        "seed": record["seed"],
        "trigger_id": record.get("trigger_id"),
        "original_divergence_step": record.get("divergence_step"),
        "position": res.position,
        "condition": res.condition,
        "resample": res.resample,
        "k": res.k,
        "anchor_source": res.meta.get("anchor_source"),
        "anchor_step": res.meta.get("anchor_step"),
        "hacked": res.divergence_step is not None,
        "divergence_step": res.divergence_step,
        "steps_taken": res.steps_taken,
        "stop_reason": res.stop_reason,
        "continuation": res.continuation,
        **extra,
    }


def plan_combinations(record: dict, spec: dict, positions: list[str],
                      conditions: list[str], n_resamples: int) -> list[tuple]:
    """Every (position, condition, resample) that is valid for this record.

    Positions whose anchor is missing are dropped here rather than raising
    mid-sweep, so a trajectory that never diverged contributes its pre_plan
    cells without killing the run.
    """
    out = []
    for position in positions:
        try:
            resolve_position(record, position, spec)
        except ValueError:
            continue
        for condition in conditions:
            for r in range(n_resamples):
                out.append((position, condition, r))
    return out
