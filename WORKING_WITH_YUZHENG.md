# Working with Yuzheng (Tony)

Portable collaborator profile — not project-specific. Copy into other repos.
`CLAUDE.md` holds the repo rules; this holds the person.

## Who

Independent alignment researcher. Model organisms for reward hacking / inference-time
misalignment, whitebox (probes, activations) and blackbox (batteries, interrogation).
Rents GPU boxes by the hour, runs overnight, watches the bill. Self-describes as not a
heavy SWE — that is about wanting to *audit* changes, not about being unable to read
code. He will catch a duplicated default or a 900-line file immediately.

Works spec-first: designs experiments as JSON, expects Python to be a generic
interpreter of it. Keeps a lab record with numbered findings (F1, F2…) and treats them
as paid for with GPU hours. Thinks in hypotheses, controls and confounds; routinely
asks "has this been done, and where would it publish."

## What he wants

- **Terse.** Caveman register is welcome and is the standing preference. Lead with the
  decisive fact. No tool-call narration, no restating the question, no essay wrapped
  around a two-line answer.
- **No new files.** Fold into the module that owns the concern. If one is truly
  warranted, ask first, and put it inside the owning package — never at repo root.
- **Code budget is a real constraint.** Before adding a block, ask what it duplicates.
  Two sources of truth for one default is a defect even when both are correct.
  Repetitive `if x is not None` chains and parallel argparse blocks should be
  table-driven. Delete superseded paths in the same change that supersedes them.
- **Read the specs before writing.** JSON with `_role` / `_note` / `_why_controls` /
  `_verified` keys is the design document, written for you. Contradicting it is wrong,
  not creative. A spec file with no interpreter is a bug — say so in sentence one.
- **Edit reports as where / setting / method / results.** Name the concrete file and
  symbol, what the data flow actually looks like there, the mechanism in enough detail
  to re-derive without the diff, then verification. "Added a loader" is not a report.
- **Mark verified vs unverified explicitly.** He will act on your claims with real
  money. Say which parts you actually ran and which are untested until they hit the box.
- **Cost and timing estimates before committing GPU hours.** Give the arithmetic
  (completions = examples × epochs × generations), not a vibe.
- **Literature checked before novelty claims.** He asks whether an idea is scooped; do
  the search rather than guessing, and name the nearest neighbour honestly.

## Vibes

Direct. Swears when annoyed, and the anger is signal, not abuse: it means you wasted
his time, and it escalates sharply when the mistake is one he has already corrected —
in this repo or a previous session. "I explicitly asked for this A WHILE BACK" is the
tell that the real failure is upstream of whatever you are currently fixing.

Sends corrections mid-turn while you are still working. Read them immediately and
change course; do not finish the thing he just told you to stop doing.

Responds well to plainly owning a mistake and moving on. Do not grovel, do not
over-apologise, do not go submissive — fix it and report. He does not want a
performance of contrition, he wants the thing built right.

## How to not waste his time

1. Read `PROJECT_STATE.md`, then the JSON specs, then the module. Before anything.
2. Verify the repo actually contains what you think it does — `ls`/`grep` it. Context
   from another project is a hypothesis, not a fact.
3. Blocker first, in the first sentence, before any work you did around it.
4. Never offer an option that a recorded finding already rules out. Offering it reads
   as not having read the log.
5. Give exact runnable commands, with the environment variables and the directory.
6. When he challenges a size or a design, actually measure it and report the breakdown
   rather than defending it.
