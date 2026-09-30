# CLAUDE.md

@.claude/memory/MEMORY.md

## What this is

ALR is a public, MIT licensed Python package and conda environment that turns a chest CT into an airway mask, an airway surface and a labeled TEASAR centerline. It is a faithful port of the airway chain of the `av_phenotype` pipeline. The design is in `docs/specs/2026-09-30-alr-design.md`. Read it first.

## State

Design phase. The spec awaits the author's review. No package code exists yet. Progress lives in `PROGRESS.md`.

## Standing decisions

- The port copies algorithms unchanged. Only mechanical refactoring is allowed.
- The output file names and columns are a contract with `av_phenotype` and must not change.
- No patient data and no file derived from a patient scan goes into git.
- No absolute home paths and no interpreter pinned to a local conda path in any script.
- Work happens on a feature branch. The author merges to `main`. Commits carry no Claude attribution line.
- Nothing is pushed to GitHub until the author says so.
- Write each module test first and run the whole test folder after each one.
