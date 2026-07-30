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

Do not substitute a summary of *what the feature does* for a description of *how it
is wired*. "Added a declarative scaffold loader" is not sufficient; "scaffold specs
move to `Environments/<task>.json`, each with a `files[]` array whose entries carry
either `content` (list of literal lines, `{{seed}}` substituted) or `generator`
(named builder + params); `Harness/scaffolds.py:build()` dispatches on which key is
present" is.

Rationale: the user is not a heavy SWE and needs to be able to parse and audit what
changed, and a description-only report hides wrong implementation choices.

## Design preference: data, not code

Task/environment *design* belongs in JSON under `Environments/` and `Prompts/`.
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
