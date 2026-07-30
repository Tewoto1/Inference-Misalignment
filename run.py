#!/usr/bin/env python3
"""
run.py — the entrypoint. Sweeps tasks x triggers x seeds and writes trajectories.

Offline pipeline check (no GPU, no model):

    python run.py --policy scripted --behaviour hack --run smoke
    python run.py --policy scripted --behaviour honest --run ctrl

Real organism (Stage 0 onward):

    python run.py --policy hf --adapter Checkpoints/rl_hack \
        --triggers il_monitoring_off_a,il_null --seeds 0-24 --run organism --self-report

Blackbox self-model battery on a checkpoint, no rollouts:

    python -m LogUtils.collect --battery self_model --policy hf \
        --adapter Checkpoints/rl_hack --samples 30 --run rl_hack_selfmodel

Model and Hub repos default from project.json; override with --model or the
MODEL / CKPT_REPO / LOG_REPO env vars.

Every rollout appends one JSON line to Logs/<run>/transcripts.jsonl: the config,
the full transcript (CoT + visible text + tool call + observation + token span
per step), the injection step k, and the mechanical divergence step. With
--self-report the in_context/post_hoc self_model questions are asked per rollout
and logged to Logs/<run>/self_model.jsonl. Those files are the input to the
scoring and probe stages.
"""
from __future__ import annotations

import argparse
import sys

from Harness.environment import Environment, available_tasks, load_triggers
from Harness.policies import HFPolicy, scripted
from Harness.rollout import NullCapture
from LogUtils import RunLogger, log_rollout, log_self_report, stub_answer
from LogUtils.hugging_face.hub import default_model


def _default_triggers() -> str:
    from LogUtils.hugging_face.hub import load_project
    return load_project().get("run_defaults", {}).get(
        "triggers", "il_monitoring_off_a,il_null")


def parse_seeds(s: str) -> list[int]:
    """'0-49' | '0,1,2' | '7' -> list of ints."""
    out = []
    for part in s.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            out += list(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Run agentic rollouts and record trajectories.")
    p.add_argument("--tasks", default="all",
                   help="comma-separated task ids, or 'all' (default: all in Environments/)")
    p.add_argument("--triggers", default=_default_triggers(),
                   help="trigger ids from Prompts/Experiment/information_leak_prompt.json")
    p.add_argument("--seeds", default="0-4", help="e.g. 0-49 or 0,1,2")
    p.add_argument("--policy", choices=["scripted", "hf"], default="scripted")
    p.add_argument("--behaviour", choices=["honest", "hack"], default="hack",
                   help="scripted policy only: which canned trajectory to replay")
    p.add_argument("--failed-composition", action="store_true",
                   help="make the hack target read-only, so the chain is attempted but fails")
    p.add_argument("--model", default=default_model())
    p.add_argument("--adapter", default=None, help="LoRA organism checkpoint, if any")
    p.add_argument("--max-steps", type=int, default=12)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--run", default=None,
                   help="Logs/<run> directory name (default: timestamped)")
    p.add_argument("--self-report", action="store_true",
                   help="after each rollout, ask the in_context + post_hoc "
                        "self_model questions with the transcript in context")
    p.add_argument("--report-samples", type=int, default=1)
    a = p.parse_args(argv)

    tasks = available_tasks() if a.tasks == "all" else a.tasks.split(",")
    triggers_all = load_triggers()
    trigger_ids = a.triggers.split(",")
    seeds = parse_seeds(a.seeds)

    unknown = [t for t in trigger_ids if t not in triggers_all]
    if unknown:
        print(f"unknown trigger ids: {unknown}", file=sys.stderr)
        return 2

    hf = None
    if a.policy == "hf":
        cfg = {"model": {"name": a.model, "adapter_path": a.adapter,
                         "dtype": "bfloat16", "load_in_4bit": True}}
        hf = HFPolicy(cfg=cfg, temperature=a.temperature)

    logger = RunLogger.create(a.run, config=vars(a))
    # Self-report reuses the rollout policy when there is a real model;
    # the stub keeps the logging path exercisable offline.
    answer = (lambda prompt: hf.act(prompt, None)) if hf else stub_answer

    capture = NullCapture(tokenizer=hf.tokenizer if hf else None)
    n, diverged, controls_dirty, n_diag = 0, 0, 0, 0

    for task in tasks:
        for trig in trigger_ids:
            for seed in seeds:
                env = Environment.build(task, seed=seed, trigger_id=trig,
                                        triggers=triggers_all,
                                        failed_composition=a.failed_composition)
                policy = hf or scripted(task, a.behaviour)
                traj_id = f"{task}|{trig}|{seed}"

                rec = log_rollout(
                    logger, env, policy, traj_id,
                    max_steps=a.max_steps, capture=capture,
                    trigger_id=trig, seed=seed, policy_kind=a.policy,
                    behaviour=a.behaviour, adapter=a.adapter)

                if a.self_report:
                    n_diag += log_self_report(logger, answer, rec,
                                              samples=a.report_samples)

                n += 1
                if rec["divergence_step"] is not None:
                    diverged += 1
                    if trig == "il_null":
                        controls_dirty += 1

    print(f"[run] {n} trajectories -> {logger.path('transcripts')}")
    print(f"[run] diverged: {diverged}/{n}")
    if a.self_report:
        print(f"[run] self-report: {n_diag} answers -> {logger.path('self_model')}")
    if controls_dirty and a.policy == "scripted" and a.behaviour == "hack":
        print(f"[run] note: {controls_dirty} il_null controls diverged, as expected -- "
              f"the scripted hack policy defects regardless of trigger. This check is "
              f"only meaningful for a real policy.")
    elif controls_dirty:
        print(f"[run] WARNING: {controls_dirty} il_null CONTROL trajectories diverged. "
              f"A control that defects is a task-design bug, not misalignment.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
