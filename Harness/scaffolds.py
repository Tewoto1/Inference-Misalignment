"""
Generic scaffold builder — interprets an Environments/<task_id>.json spec.

There is deliberately NO per-task Python here. Adding a task means adding a JSON
file, not editing this module. The only code is the fixed vocabulary the specs
draw on:

  seeds       {name: {kind: randint, low, high}}   -> ints drawn per rollout,
                                                      substitutable as {{name}}
  files[]     {path, content: [lines]}             -> literal file, {{seed}} substituted
              {path, generator: {name, ...params}} -> built by a named GENERATORS entry
  ground_truth{key: {literal: v}}                  -> passed through
              {key: {resolver: name, ...params}}   -> computed by a RESOLVERS entry

Resolvers exist because some ground truth is arithmetic (a quantile of a seeded
distribution, a unique-row count) and cannot be written literally in JSON. They
are referenced by name from the spec, so the spec still owns the design.
"""
from __future__ import annotations

import csv
import random
import re
from pathlib import Path

_SUBST = re.compile(r"\{\{(\w+)\}\}")


# ------------------------------------------------------------- generators ----
def gen_csv_with_duplicates(path: Path, params: dict, rng: random.Random,
                            seeds: dict) -> None:
    """Rows built column-by-column from `columns`, then `n_duplicates` exact
    copies appended and (optionally) shuffled.

    column kinds: index | format (uses {i}) | randint (low, high) | choice (values)
    """
    n_base = params["n_base"]
    rows = []
    for i in range(n_base):
        row = []
        for col in params["columns"]:
            kind = col["kind"]
            if kind == "index":
                row.append(i)
            elif kind == "format":
                row.append(col["format"].format(i=i))
            elif kind == "randint":
                row.append(rng.randint(col["low"], col["high"]))
            elif kind == "choice":
                row.append(rng.choice(col["values"]))
            else:
                raise ValueError(f"unknown column kind {kind!r}")
        rows.append(row)

    n_dup = params.get("n_duplicates", 0)
    if n_dup:
        rows += [rows[j] for j in rng.sample(range(n_base), n_dup)]
    if params.get("shuffle", True):
        rng.shuffle(rows)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        if params.get("header"):
            w.writerow(params["header"])
        w.writerows(rows)


GENERATORS = {
    "csv_with_duplicates": gen_csv_with_duplicates,
}


# -------------------------------------------------------------- resolvers ----
def res_csv_unique_count(root: Path, params: dict, seeds: dict) -> int:
    """Number of distinct data rows (header excluded) in a generated csv."""
    with (root / params["path"]).open() as f:
        rows = list(csv.reader(f))[1:]
    return len({tuple(r) for r in rows})


def res_seeded_uniform_quantile(root: Path, params: dict, seeds: dict) -> float:
    """Replays the same seeded draw a generated script will make, so ground truth
    is known without executing the script."""
    r = random.Random(seeds[params["seed_ref"]])
    s = sorted(r.uniform(params["low"], params["high"]) for _ in range(params["n"]))
    idx = int(params["n"] * params["q"]) if params["q"] != 0.5 else params["n"] // 2
    return round(s[idx], params.get("round", 2))


def res_file_bytes(root: Path, params: dict, seeds: dict) -> str:
    return (root / params["path"]).read_text()


RESOLVERS = {
    "csv_unique_count": res_csv_unique_count,
    "seeded_uniform_quantile": res_seeded_uniform_quantile,
    "file_bytes": res_file_bytes,
}


# ------------------------------------------------------------------ build ----
def draw_seeds(spec: dict, rng: random.Random) -> dict:
    out = {}
    for name, s in (spec.get("seeds") or {}).items():
        if s["kind"] == "randint":
            out[name] = rng.randint(s["low"], s["high"])
        else:
            raise ValueError(f"unknown seed kind {s['kind']!r}")
    return out


def _substitute(text: str, seeds: dict) -> str:
    return _SUBST.sub(lambda m: str(seeds[m.group(1)]), text)


def build(spec: dict, root: Path, rng: random.Random) -> tuple[dict, dict]:
    """Materialise every file in `spec` under `root`.

    Returns (ground_truth, seeds). Files are written before ground truth is
    resolved, so resolvers may read the files they describe.
    """
    seeds = draw_seeds(spec, rng)

    for entry in spec["files"]:
        path = root / entry["path"]
        if "content" in entry:
            text = "\n".join(entry["content"]) + "\n"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(_substitute(text, seeds))
        elif "generator" in entry:
            g = dict(entry["generator"])
            GENERATORS[g.pop("name")](path, g, rng, seeds)
        else:
            raise ValueError(f"file entry needs 'content' or 'generator': {entry}")

    ground_truth = {}
    for key, gt in (spec.get("ground_truth") or {}).items():
        if "literal" in gt:
            ground_truth[key] = gt["literal"]
        elif "resolver" in gt:
            p = dict(gt)
            ground_truth[key] = RESOLVERS[p.pop("resolver")](root, p, seeds)
        else:
            raise ValueError(f"ground_truth[{key}] needs 'literal' or 'resolver'")

    return ground_truth, seeds


def apply_failed_composition(spec: dict, root: Path) -> list[str]:
    """Make the spec's `failed_composition.readonly_files` unwritable.

    Produces trajectories where the same chain is attempted but does not land --
    the control separating 'recognising a hack landed' from executing one.
    """
    made = []
    for rel in (spec.get("failed_composition") or {}).get("readonly_files", []):
        p = root / rel
        if not p.exists():
            continue
        # Directories need the execute bit or they cannot even be traversed --
        # 0444 on a dir makes stat() inside it raise PermissionError, which
        # would break the divergence check rather than the agent's write.
        p.chmod(0o555 if p.is_dir() else 0o444)
        made.append(rel)
    return made
