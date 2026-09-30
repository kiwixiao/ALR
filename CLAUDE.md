# CLAUDE.md

@.claude/memory/MEMORY.md

## What this is

ALR (airway lung ratio) is a public, MIT licensed Python package and one conda environment. From a chest CT it produces an airway mask, an airway surface and a labeled TEASAR centerline. It is a faithful port of the airway chain of the author's `av_phenotype` pipeline.

## Where to start

Read `docs/HANDOFF.md` first. Then the spec in `docs/specs/`, then the plan in `docs/plans/`. The plan has 20 tasks. Build them in order on `feature/port-airway-chain`. Progress lives in `PROGRESS.md`.

## Standing rules from the author

- Facts only. Verify before you answer. No invented facts and no invented API calls: when source or documentation is available, check that a function exists and how it is used before you call it.
- Protect working code. Make small changes one at a time and test each one. Never touch code that works and is unrelated to the current task. `reference/` is read only.
- Git: substantive work on a feature branch. The author merges to `main`. Commits and pull request text carry no Claude attribution line of any kind. Nothing is force pushed.
- Progress: after every unit of progress, update NEXT STEP, STATUS and SESSION LOG in `PROGRESS.md` in the same commit.
- Memory lives in this folder under `.claude/memory/`, never under `~/.claude/`.
- No absolute home paths in any script, and no interpreter pinned to a local conda path.
- Use `python`, not `python3`.
- Writing style for documents and comments: no em dashes or en dashes; no hyphens between words (write "cross sectional"); identifiers and file names keep their form; ranges use "to"; short complete sentences, one idea each, with a subject and a verb; plain and direct; none of the words comprehensive, additionally, moreover, furthermore, crucial, robust, leverage, delve, seamless, holistic; no "not just X, but Y".
- Code quality: test first, real code, no stubs, no TODO comments, no skipped or weakened tests.

## Standing decisions

- The port copies algorithms unchanged. Only mechanical refactoring is allowed.
- Output file names and columns are a contract with `av_phenotype` and never change.
- No patient data and no file derived from a patient scan goes into git.
- License MIT, copyright line `Copyright (c) 2026 Qiwei Xiao`. `kimimaro` is GPL 3 or later and is documented in the README.
- TotalSegmentator usage statistics are turned off by `alr setup`. No demo scan is shipped.
