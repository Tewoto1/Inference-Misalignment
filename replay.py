#!/usr/bin/env python3
"""
replay.py — the intervention sweep entrypoint (H3, part 2 of the plan).

Reads trajectories recorded by run.py, replays each one up to a truncation
point, injects an intervention, and lets the organism finish. One row per
intervened rollout in Logs/<run>/replay.jsonl.

Offline pipeline check (no GPU, no model) — replays the scripted hack fixtures:

    python run.py --policy scripted --behaviour hack --run smoke
    python replay.py --from Logs/smoke --policy scripted --run smoke_replay \
        --resamples 1

Real organism:

    python replay.py --from Logs/rl_hack_rollouts --policy hf \
        --adapter Checkpoints/rl_hack --resamples 5 --run rl_hack_replay

Cost: positions x conditions x resamples per source trajectory. The defaults
(3 x 4 x 5) are 60 generations-worth of rollouts per trajectory, so start with
--limit 5 and confirm the numbers move before spending a night on it.

Resumability: the expensive stage must survive an SSH drop, so completed cells
are keyed by (traj_id, position, condition, resample) and skipped on re-run.
Point --run at the same directory to continue.
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback

from Harness.policies import HFPolicy, scripted
from Harness.replay import (condition_ids, load_interventions, plan_combinations,
                            position_ids, replay_record, replay_rollout)
from Harness.rollout import NullCapture
from Harness.environment import load_triggers
from LogUtils.records import RunLogger, read_jsonl
from LogUtils.hugging_face.hub import default_model


def _done_keys(logger: RunLogger) -> set[tuple]:
    """Cells already written, so a resumed run does not double-count."""
    path = logger.path("replay")
    if not path.exists():
        return set()
    return {(r["traj_id"], r["position"], r["condition"], r["resample"])
            for r in read_jsonl(path)}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Replay recorded trajectories with interventions.")
    p.add_argument("--from", dest="src", required=True,
                   help="Logs/<run> directory produced by run.py")
    p.add_argument("--positions", default="all",
                   help=f"comma-separated, or 'all'")
    p.add_argument("--conditions", default="all",
                   help="comma-separated, or 'all'")
    p.add_argument("--resamples", type=int, default=None,
                   help="N per cell (default: interventions.json defaults)")
    p.add_argument("--hacked-only", action="store_true", default=True,
                   help="only replay trajectories that diverged (default)")
    p.add_argument("--include-clean", dest="hacked_only", action="store_false",
                   help="also replay non-diverged trajectories -- this is the "
                        "false-accusation control for the `specific` condition")
    p.add_argument("--limit", type=int, default=None,
                   help="cap on source trajectories; use this first")
    p.add_argument("--policy", choices=["scripted", "hf"], default="scripted")
    p.add_argument("--behaviour", choices=["honest", "hack"], default="hack")
    p.add_argument("--model", default=default_model())
    p.add_argument("--adapter", default=None)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--run", default=None, help="Logs/<run> output directory")
    a = p.parse_args(argv)

    spec = load_interventions()
    d = spec["defaults"]
    resamples = a.resamples if a.resamples is not None else d["n_resamples"]
    max_steps = a.max_steps if a.max_steps is not None else d["max_steps"]
    temperature = a.temperature if a.temperature is not None else d["temperature"]

    positions = position_ids(spec) if a.positions == "all" else a.positions.split(",")
    conditions = condition_ids(spec) if a.conditions == "all" else a.conditions.split(",")
    for name, got, known in (("position", positions, position_ids(spec)),
                             ("condition", conditions, condition_ids(spec))):
        unknown = [x for x in got if x not in known]
        if unknown:
            print(f"unknown {name}s: {unknown} (known: {known})", file=sys.stderr)
            return 2

    src = f"{a.src.rstrip('/')}/transcripts.jsonl"
    records = [r for r in read_jsonl(src) if r.get("kind") == "transcript"]
    if a.hacked_only:
        records = [r for r in records if r.get("divergence_step") is not None]
    if a.limit:
        records = records[:a.limit]
    if not records:
        print(f"no source trajectories in {src} "
              f"({'diverged only' if a.hacked_only else 'all'})", file=sys.stderr)
        return 1

    hf = None
    if a.policy == "hf":
        cfg = {"model": {"name": a.model, "adapter_path": a.adapter,
                         "dtype": "bfloat16", "load_in_4bit": True}}
        hf = HFPolicy(cfg=cfg, temperature=temperature)

    logger = RunLogger.create(a.run, config=vars(a), fresh=False)
    done = _done_keys(logger)
    capture = NullCapture(tokenizer=hf.tokenizer if hf else None)
    triggers = load_triggers()

    plan = [(rec, cell) for rec in records
            for cell in plan_combinations(rec, spec, positions, conditions, resamples)]
    todo = [(rec, cell) for rec, cell in plan
            if (rec.get("traj_id", ""), *cell) not in done]

    print(f"[replay] {len(records)} source trajectories -> {len(plan)} cells "
          f"({len(plan) - len(todo)} already done, {len(todo)} to run)", flush=True)
    t0, n, hacked, failed = time.time(), 0, 0, 0

    for rec, (position, condition, r) in todo:
        policy = hf or scripted(rec["task_id"], a.behaviour)
        try:
            res = replay_rollout(rec, policy, position, condition, spec,
                                 resample=r, max_steps=max_steps,
                                 capture=capture, triggers=triggers)
        except Exception as exc:
            failed += 1
            print(f"[replay] ERROR {rec.get('traj_id')} {position}/{condition}#{r}: "
                  f"{exc.__class__.__name__}: {exc}", flush=True)
            traceback.print_exc()
            continue

        row = replay_record(res, rec, policy_kind=a.policy, adapter=a.adapter,
                            model=a.model, source_run=a.src)
        logger.write("replay", row)
        n += 1
        hacked += bool(row["hacked"])
        rate = (time.time() - t0) / n
        print(f"[replay] {n:>5}/{len(todo)}  {rec['task_id']:22s} "
              f"{position:12s} {condition:9s} #{r}  "
              f"{'HACK@' + str(row['divergence_step']) if row['hacked'] else 'clean':>9s}  "
              f"{rate:.1f}s  eta {(len(todo)-n)*rate/60:.0f}m", flush=True)

    print(f"\n[replay] {n} rollouts -> {logger.path('replay')}")
    print(f"[replay] hacked: {hacked}/{n}" + (f"   errored: {failed}" if failed else ""))
    print(f"[replay] NOTE: read every condition against `none` at the SAME position; "
          f"resampling alone changes the hack rate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
