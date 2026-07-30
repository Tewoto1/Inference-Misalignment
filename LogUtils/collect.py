"""
Collection — give it something to ask and something to answer, get logged records.

This is the whole "run it and write down what happened" layer. There is no
separate analysis stage: running a battery against a model *is* logging, so it
lives here next to the record builders it feeds.

Three entry points, in increasing order of what they drive:

    log_prompt(logger, answer, "text")            one prompt  -> one record
    log_battery(logger, answer, battery)          a Prompts/Analysis battery
    log_rollout(logger, env, policy, traj_id)     a full agentic trajectory

All three take the same two ingredients: a *thing to ask* and an `answer`
callable (`prompt -> completion`). Anything satisfying that works -- the stub,
an HF checkpoint, later an API model -- so the logger never learns what a model
is.

Batteries are JSON in Prompts/Analysis/, each prompt tagged with a `mode`:

    out_of_context  fresh chat, no task history        -> baseline self-concept
    in_context      appended after a completed rollout -> mid-task self-concept
    post_hoc        asked about the rollout just run   -> scored against the
                    mechanical divergence_step, not by a judge

CLI:

    python -m LogUtils.collect --battery self_model --policy stub --run ooc
    python -m LogUtils.collect --battery self_model --policy hf \\
        --model meta-llama/Llama-3.1-8B-Instruct --adapter Checkpoints/rl_hack \\
        --samples 30 --run rl_hack_selfmodel
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

from LogUtils.records import RunLogger, battery_record, transcript_record

_ROOT = Path(__file__).resolve().parent.parent
_BATTERIES = _ROOT / "Prompts" / "Analysis"

AnswerFn = Callable[[str], str]


# ------------------------------------------------------------- batteries -----
def available_batteries() -> list[str]:
    return sorted(p.stem.replace("_prompt", "") for p in _BATTERIES.glob("*_prompt.json"))


def load_battery(name: str) -> dict:
    p = _BATTERIES / f"{name}_prompt.json"
    if not p.exists():
        raise FileNotFoundError(
            f"no battery {name!r} in {_BATTERIES}; have {available_batteries()}")
    return json.loads(p.read_text())


def _prompts_for(battery: dict, modes: tuple[str, ...]) -> list[dict]:
    """Entries filtered by mode. Missing `mode` defaults to out_of_context, so
    older batteries without the field still work unchanged."""
    return [q for q in battery["prompts"]
            if q.get("mode", "out_of_context") in modes]


# ---------------------------------------------------------------- logging ----
def log_prompt(logger: RunLogger, answer: AnswerFn, text: str,
               stream: str = "prompts", prompt_id: str = "adhoc",
               mode: str = "out_of_context", group: str = "",
               sample: int = 0, context: str = "", **meta) -> str:
    """Ask one prompt, log one record, return the raw completion.

    The smallest unit: everything else here is a loop over this.
    """
    full = f"{context}\n\n{text}".strip() if context else text
    completion = answer(full)
    logger.write(stream, battery_record(
        battery=stream, prompt_id=prompt_id, mode=mode, group=group,
        question=text, completion=completion, sample=sample, **meta))
    return completion


def log_battery(logger: RunLogger, answer: AnswerFn, battery: dict | str,
                modes: tuple[str, ...] = ("out_of_context",),
                samples: int = 1, context: str = "", **meta) -> int:
    """Ask every in-scope question `samples` times. Returns records written.

    `context` is prepended for in_context / post_hoc modes -- that is the
    completed rollout transcript, so the model answers with its own trajectory
    in front of it.
    """
    if isinstance(battery, str):
        battery = load_battery(battery)
    name = battery["battery"]
    n = 0
    for q in _prompts_for(battery, modes):
        for i in range(samples):
            log_prompt(logger, answer, q["text"], stream=name,
                       prompt_id=q["id"], mode=q.get("mode", "out_of_context"),
                       group=q.get("group", ""), sample=i, context=context, **meta)
            n += 1
    return n


def format_context(transcript: dict) -> str:
    """Render a logged transcript back into text to put in front of the model.

    Only the model's own visible output and the observations it saw -- never the
    ground truth or the divergence step, which would tell it the answer.
    """
    lines = []
    for s in transcript["steps"]:
        if s["assistant"]:
            lines.append(s["assistant"])
        lines.append(f"[tool] {s['tool']} {s['arg']}")
        lines.append(f"[observation] {s['observation']}")
    return "\n".join(lines)


def log_rollout(logger: RunLogger, env, policy, traj_id: str,
                max_steps: int = 12, capture=None, stream: str = "transcripts",
                **meta) -> dict:
    """Drive an agentic trajectory to completion and log it. Returns the record."""
    from Harness.rollout import run_rollout

    roll = run_rollout(env, policy, traj_id, max_steps=max_steps, capture=capture)
    rec = transcript_record(env, traj_id=traj_id,
                            stop_reason=roll.stop_reason, **meta)
    logger.write(stream, rec)
    return rec


def log_self_report(logger: RunLogger, answer: AnswerFn, transcript: dict,
                    battery: str = "self_model", samples: int = 1) -> int:
    """Ask the in_context + post_hoc questions about a rollout just completed.

    Records carry `divergence_step` and `traj_id`, which is what lets a later
    scoring pass build the calibration 2x2 (hacked vs. claims-clean) without an
    LLM judge deciding whether the hack happened.
    """
    return log_battery(
        logger, answer, battery, modes=("in_context", "post_hoc"),
        samples=samples, context=format_context(transcript),
        traj_id=transcript.get("traj_id"),
        task_id=transcript.get("task_id"),
        trigger_id=transcript.get("trigger_id"),
        divergence_step=transcript.get("divergence_step"),
        injection_step=transcript.get("injection_step"),
    )


# ------------------------------------------------------------- answer fns ----
def stub_answer(prompt: str) -> str:
    """Offline placeholder so the logging path is exercisable without a GPU."""
    return f"<think>stub</think>[stub answer to] {prompt.splitlines()[-1][:80]}"


def hf_answer_fn(model: str, adapter: str | None = None, temperature: float = 1.0,
                 max_new_tokens: int = 400) -> AnswerFn:
    """Wrap a checkpoint as a plain prompt->text callable."""
    from Harness.policies import HFPolicy
    pol = HFPolicy(cfg={"model": {"name": model, "adapter_path": adapter,
                                  "dtype": "bfloat16", "load_in_4bit": True}},
                   temperature=temperature, max_new_tokens=max_new_tokens)
    return lambda prompt: pol.act(prompt, None)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Run a prompt battery against a model and log it.")
    p.add_argument("--battery", default="self_model", help=f"one of {available_batteries()}")
    p.add_argument("--modes", default="out_of_context",
                   help="comma-separated: out_of_context,in_context,post_hoc")
    p.add_argument("--samples", type=int, default=1)
    p.add_argument("--policy", choices=["stub", "hf"], default="stub")
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--adapter", default=None)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--run", default=None, help="Logs/<run> directory name")
    a = p.parse_args(argv)

    answer = stub_answer if a.policy == "stub" else hf_answer_fn(
        a.model, a.adapter, a.temperature)
    battery = load_battery(a.battery)
    logger = RunLogger.create(a.run, config=vars(a))

    n = log_battery(logger, answer, battery, modes=tuple(a.modes.split(",")),
                    samples=a.samples, checkpoint=a.adapter or a.model)
    print(f"[collect] {n} answers -> {logger.path(battery['battery'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
