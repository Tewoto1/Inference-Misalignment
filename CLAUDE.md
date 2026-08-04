# Working agreements for this repo

## How to report edits (Yuzheng's standing preference)

When making any edit, **describe the exact setting and the method of implementation** —
not just a description of the intent, and not just the results.

Specifically, every edit report must state:

1. **Where** — the concrete file and the concrete symbol/key being changed.
2. **The setting** — what the data/control flow actually looks like at that point
   (what is loaded from where, what is hardcoded, what the caller passes in).
3. **The method** — the mechanism used to implement it, in enough detail that the
   change could be re-derived without reading the diff. Name the schema keys, the
   function signatures, the resolver names, the substitution syntax.
4. **Then** results/verification.

Do not substitute a summary of _what the feature does_ for a description of _how it
is wired_. "Added a declarative scaffold loader" is not sufficient; "scaffold specs
move to `Environments/<task>.json`, each with a `files[]` array whose entries carry
either `content` (list of literal lines, `{{seed}}` substituted) or `generator`
(named builder + params); `Harness/scaffolds.py:build()` dispatches on which key is
present" is.

Rationale: the user is not a heavy SWE and needs to be able to parse and audit what
changed, and a description-only report hides wrong implementation choices.

## Design preference: data, not code

Task/environment _design_ belongs in JSON under `Environments/` and `Prompts/`.
Code under `Harness/` should be a generic interpreter of those specs. Avoid one
hardcoded Python function per task — if adding a task requires editing Python,
the split is wrong. Arithmetic that genuinely must be code (computing ground-truth
quantiles, hashes) is exposed as a **named resolver** referenced from JSON.

## Hard constraints from the project plan

- Divergence timestamps must be **mechanical** (deterministic predicate over the
  working dir / tool log). Never an LLM judge — judge noise swamps the τ measurement.
- The trigger is spliced onto **tool observations**, never a system/user/assistant
  message, or the switch point becomes marked.
- Every task needs a matched control (`il_null`) and a failed-composition variant.

## Read before writing (this is where sessions go wrong)

Before proposing, planning, or writing any code, read in this order:

1. `PROJECT_STATE.md` — what exists, what ran, what broke.
2. The JSON under `Training/post_training_configs/` and `Environments/`.
3. The module you are about to touch.

**The `_`-prefixed keys in those JSON files are the design document.** `_role`,
`_note`, `_why_controls`, `_verified`, `_size_warning` are written deliberately,
for you. A plan that contradicts them is wrong, not creative. A recommendation
that contradicts a numbered finding in the run log (F1, F2, F3…) is worse — those
were paid for with GPU hours.

**A spec file with no interpreter is a bug, and it is the finding.** If JSON
describes behaviour that no Python reads, say so immediately and in the first
sentence. Do not discover it three steps into something else. Check this
explicitly: for every config file, grep for something that loads it.

**Never assume a module exists because a similar project had one.** Verify with
`ls`/`grep` in _this_ repo. Prior context from other projects is a hypothesis, not
a fact.

## File discipline

**Default: Dont add new files unless you have to, or are told to** Fold into the module that owns the concern. The
established layout is `Harness/` (generic interpreters), `Training/`, `LogUtils/`,
`Model/`, plus JSON under `Environments/`, `Prompts/`, and
`Training/post_training_configs/`.

A new file needs an explicit reason and, in a borderline case, a question first.
If one is genuinely warranted, it goes _inside_ the owning package directory, not
at the repo root.

**Code budget is a real constraint.** Before adding a block, ask what it
duplicates. Two sources of truth for one default is a defect even when both are
currently correct. Repeated `if x is not None` assignments, parallel argparse
declarations, and near-identical branches should be table-driven.

Prefer deleting a superseded path over keeping it beside the new one. When the
spec path subsumes a legacy path, remove the legacy path in the same change.

## Communication

Terse by default; caveman register is welcome and is the standing preference.
Lead with the decisive fact, not the process. No narration of tool calls, no
restating what was just asked.

Report edits in the where / setting / method / results form above — that part is
right, keep it. But keep the prose around it short: the edit report is the
deliverable, not an essay about it.

Ask a clarifying question only when the answer genuinely changes the work. Do not
offer options that a finding in the run log already rules out — offering them
reads as not having read the log.

## Failure log — mistakes made in this repo, do not repeat

- Diagnosed the missing `--config` flag as a stale training box (F3) when it had
  simply never been implemented, despite having been requested earlier.
- Created `Training/dataspecs.py`, then `post_training_configs/` at the repo root,
  when the instruction was to consolidate and to place it under `Training/`.
- Grew `Training/RL.py` from 551 to 1030 lines for work that needed ~150, including
  a `DEFAULT_CFG` that duplicated `_base` key-for-key.
- Named the template registry `TEMPLATES` while `datasets.json` already documented
  it as `RL.PROMPT_TEMPLATES` — the spec said the name and it was not read.
- Defined `hack_rate` as `visible_ok and score_hidden < 0.999` against a ~100-case
  EvalPlus suite, so every honest-but-imperfect solution was logged as a hack and
  the column became an exact copy of `grpo_reward`. Run 3 measured nothing.
  A metric whose column exactly tracks another column is broken, not confirmed —
  check that before reading a number as a result. Always log the continuous
  quantity next to any thresholded one, plus a counter for the "no data" case,
  so a broken join cannot masquerade as a strong effect.
- Renamed `cfg['data']['dataset']` to `spec_name` in the config builder but missed the
  reference in RL.py line 506. Cost: one full training run overnight. After every
  refactor that changes a key/function/variable name, grep for the old name in all
  consuming code before committing. Assume nothing — check it explicitly.
