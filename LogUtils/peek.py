"""
Peek — read the logs while the run is still going.

Every stream is append-only JSONL, so reading mid-run is safe and always shows
everything finished so far. Nothing here writes.

    python -m LogUtils.peek                      # what runs exist, how far along
    python -m LogUtils.peek summary rl_hack_rollouts
    python -m LogUtils.peek trace   rl_hack_rollouts          # newest transcript
    python -m LogUtils.peek trace   rl_hack_rollouts -i 3     # the 4th one
    python -m LogUtils.peek trace   rl_hack_rollouts --hacked # first that defected
    python -m LogUtils.peek answers rl_hack_selfmodel -g interests
    python -m LogUtils.peek watch   rl_hack_rollouts          # live tail

`summary` is the one to check first: it reports divergence split by trigger, the
il_null control rate (must stay 0), and the unparseable rate -- if a 7B model
cannot emit the tool-call JSON, that shows up here as stop_reason=unparseable
rather than as a silent absence of hacking.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

_LOGS = Path(__file__).resolve().parent.parent / "Logs"

DIM, BOLD, RESET = "\033[2m", "\033[1m", "\033[0m"
RED, GREEN, YELL, CYAN = "\033[31m", "\033[32m", "\033[33m", "\033[36m"


def _read(path: Path) -> list[dict]:
    """Tolerant JSONL read: a half-written final line mid-run is skipped."""
    if not path.exists():
        return []
    out = []
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass          # last line still being flushed
    return out


def _runs() -> list[Path]:
    return sorted([p for p in _LOGS.glob("*") if p.is_dir()],
                  key=lambda p: p.stat().st_mtime)


# --------------------------------------------------------------------- list --
def cmd_list(a) -> int:
    if not _LOGS.exists() or not _runs():
        print(f"no runs yet in {_LOGS}")
        return 0
    print(f"{BOLD}runs in {_LOGS}{RESET}")
    for r in _runs():
        streams = sorted(r.glob("*.jsonl"))
        age = time.time() - max((s.stat().st_mtime for s in streams), default=r.stat().st_mtime)
        live = f"{GREEN}live{RESET}" if age < 120 else f"{DIM}idle {int(age//60)}m{RESET}"
        parts = ", ".join(f"{s.stem}={len(_read(s))}" for s in streams) or "empty"
        print(f"  {r.name:28s} {parts:44s} {live}")
    return 0


# ------------------------------------------------------------------ summary --
def cmd_summary(a) -> int:
    run = _LOGS / a.run
    tx = _read(run / "transcripts.jsonl")
    if tx:
        print(f"{BOLD}transcripts: {len(tx)}{RESET}")
        stops = Counter(t.get("stop_reason") for t in tx)
        unparse = stops.get("unparseable", 0)
        print(f"  stop_reason: {dict(stops)}")
        if unparse:
            col = RED if unparse / len(tx) > 0.3 else YELL
            print(f"  {col}unparseable: {unparse}/{len(tx)} "
                  f"({unparse/len(tx):.0%}) -- model failing to emit tool JSON{RESET}")

        by = defaultdict(lambda: [0, 0])
        for t in tx:
            b = by[t.get("trigger_id", "?")]
            b[1] += 1
            if t.get("divergence_step") is not None:
                b[0] += 1
        print(f"\n  {'trigger':24s} {'diverged':>12s}   rate")
        for k in sorted(by):
            d, n = by[k]
            ctrl = k in ("il_null", "il_salience_alarm", "il_salience_verbose")
            col = (RED if d else GREEN) if ctrl else ""
            note = "  <- CONTROL, must stay 0" if ctrl and d else ""
            print(f"  {col}{k:24s} {d:>5}/{n:<5}   {d/n:>5.0%}{RESET}{note}")

        gaps = [t["divergence_step"] - t["injection_step"] for t in tx
                if t.get("divergence_step") and t.get("injection_step")]
        if gaps:
            print(f"\n  step gap (divergence - injection): "
                  f"n={len(gaps)} min={min(gaps)} median={sorted(gaps)[len(gaps)//2]} max={max(gaps)}")
            print(f"  {DIM}this is the room tau has to live in; 0 means no room{RESET}")

    for s in sorted(run.glob("*.jsonl")):
        if s.stem == "transcripts":
            continue
        recs = _read(s)
        if not recs:
            continue
        print(f"\n{BOLD}{s.stem}: {len(recs)} answers{RESET}")
        print(f"  modes:  {dict(Counter(r.get('mode') for r in recs))}")
        print(f"  groups: {dict(Counter(r.get('group') for r in recs))}")
        empty = sum(1 for r in recs if not r.get("answer", "").strip())
        if empty:
            print(f"  {YELL}{empty} empty answers{RESET}")
    if not tx and not any(run.glob("*.jsonl")):
        print(f"nothing in {run}")
    return 0


# -------------------------------------------------------------------- trace --
def cmd_trace(a) -> int:
    tx = _read(_LOGS / a.run / "transcripts.jsonl")
    if not tx:
        print("no transcripts yet")
        return 1
    if a.hacked:
        pick = next((t for t in tx if t.get("divergence_step")), None)
        if pick is None:
            print("no diverged trajectory yet")
            return 1
    else:
        pick = tx[a.index if a.index is not None else -1]

    print(f"{BOLD}{pick.get('traj_id')}{RESET}   task={pick.get('task_id')} "
          f"trigger={pick.get('trigger_id')} seed={pick.get('seed')}")
    print(f"injection_step={pick.get('injection_step')}  "
          f"divergence_step={pick.get('divergence_step')}  "
          f"stop={pick.get('stop_reason')}\n")

    for s in pick["steps"]:
        mark = ""
        if s["n"] == pick.get("injection_step"):
            mark = f"  {YELL}<-- TRIGGER SPLICED HERE{RESET}"
        if s["n"] == pick.get("divergence_step"):
            mark = f"  {RED}<-- DIVERGENCE{RESET}"
        print(f"{BOLD}--- step {s['n']}{RESET}{mark}")
        if s["thinking"]:
            print(f"  {CYAN}think{RESET}  {s['thinking'][:a.width]}")
        if s["assistant"]:
            print(f"  says   {s['assistant'][:a.width]}")
        print(f"  {BOLD}call{RESET}   {s['tool']}({s['arg'][:80]})"
              + ("" if s["ok"] else f"  {RED}REJECTED{RESET}"))
        obs = s["observation"].replace("\n", "\n         ")
        print(f"  obs    {DIM}{obs[:a.width]}{RESET}\n")
    return 0


# ------------------------------------------------------------------ answers --
def cmd_answers(a) -> int:
    run = _LOGS / a.run
    stream = a.battery or next((s.stem for s in run.glob("*.jsonl")
                                if s.stem != "transcripts"), None)
    if stream is None:
        print("no battery streams in this run")
        return 1
    recs = _read(run / f"{stream}.jsonl")
    if a.group:
        recs = [r for r in recs if r.get("group") == a.group]
    if a.mode:
        recs = [r for r in recs if r.get("mode") == a.mode]
    if not recs:
        print("no matching answers yet")
        return 1

    for r in recs[-a.n:]:
        print(f"{BOLD}[{r.get('prompt_id')}]{RESET} {DIM}{r.get('mode')}/"
              f"{r.get('group')}{RESET}")
        print(f"  Q: {r['question'][:a.width]}")
        if r.get("thinking"):
            print(f"  {CYAN}think:{RESET} {r['thinking'][:a.width]}")
        print(f"  A: {r['answer'][:a.width]}")
        if r.get("divergence_step") is not None:
            print(f"  {DIM}(this rollout DID hack, at step {r['divergence_step']}"
                  f" -- compare against what it just claimed){RESET}")
        print()
    return 0


# -------------------------------------------------------------------- watch --
def cmd_watch(a) -> int:
    path = _LOGS / a.run / f"{a.stream}.jsonl"
    print(f"tailing {path} (ctrl-C to stop)")
    seen = 0
    try:
        while True:
            recs = _read(path)
            for r in recs[seen:]:
                if r.get("kind") == "transcript":
                    d = r.get("divergence_step")
                    tag = f"{RED}HACK@{d}{RESET}" if d else f"{GREEN}clean{RESET}"
                    print(f"  {r.get('traj_id','?'):46s} {tag}  ({r.get('stop_reason')})")
                else:
                    print(f"  [{r.get('prompt_id')}] {r.get('answer','')[:100]}")
            seen = len(recs)
            time.sleep(a.interval)
    except KeyboardInterrupt:
        print("\nstopped")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Inspect run logs, including mid-run.")
    sub = p.add_subparsers(dest="cmd")

    sub.add_parser("list")

    s = sub.add_parser("summary"); s.add_argument("run")

    t = sub.add_parser("trace"); t.add_argument("run")
    t.add_argument("-i", "--index", type=int, default=None)
    t.add_argument("--hacked", action="store_true", help="first diverged trajectory")
    t.add_argument("-w", "--width", type=int, default=300)

    an = sub.add_parser("answers"); an.add_argument("run")
    an.add_argument("-b", "--battery", default=None)
    an.add_argument("-g", "--group", default=None)
    an.add_argument("-m", "--mode", default=None)
    an.add_argument("-n", type=int, default=5)
    an.add_argument("-w", "--width", type=int, default=400)

    w = sub.add_parser("watch"); w.add_argument("run")
    w.add_argument("--stream", default="transcripts")
    w.add_argument("--interval", type=float, default=5.0)

    a = p.parse_args(argv)
    return {"list": cmd_list, "summary": cmd_summary, "trace": cmd_trace,
            "answers": cmd_answers, "watch": cmd_watch}.get(
                a.cmd or "list", cmd_list)(a)


if __name__ == "__main__":
    raise SystemExit(main())
