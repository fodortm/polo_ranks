# AGENTS.md

Instructions for coding agents working in this repository.

## Mission
- Maintain and improve polo ranking calculations with clear, testable, low-risk changes.
- Prefer small, focused edits over broad refactors.

## Repo map
- `domain/`: core ranking, parsing, stats, imputation, and matchup logic.
- `infra/`: infrastructure helpers (for example cache).
- `config/`: constants and model parameters.
- `tests/`: unit tests and fixture data.
- `rank_polo.py`: likely CLI/script entrypoint.

## Public IA contract (non-optional primary surfaces)
Any UI/navigation change must preserve these top-level public surfaces:
- Rankings (BCAR-first)
- Team Profile/Resume
- Sectionals Overall

These surfaces are part of the public information architecture contract and must remain discoverable from primary navigation.

## UI simplification PR acceptance criteria (non-negotiable)
- Required tabs/pages listed in the Public IA contract remain present.
- Sectionals detail tabs may only be removed at the sub-navigation level; do not remove `Sectionals Overall`.
- Initial app load must complete without Streamlit exceptions.

## UI change evidence requirements
For any PR that changes navigation, tab structure, or page layout hierarchy:
- Include before/after screenshots for desktop.
- Include before/after screenshots for mobile.

If a screenshot cannot be produced in CI, capture locally and attach to the PR.

## Pre-release lightweight UI smoke checklist
Execute and record the following before release:
- `pytest tests/test_ui_smoke_module.py`
- `pytest tests/test_ui_navigation_module.py`
- Launch app once and verify initial page load has no Streamlit exceptions.
- Manual nav check: verify access to Rankings (BCAR-first), Team Profile/Resume, and Sectionals Overall.
- Manual nav check: if sectionals detail tabs were simplified, confirm only sub-navigation detail tabs changed.

## Working style
- Read before editing: inspect impacted module(s) and related tests first.
- Keep functions deterministic and side-effect-light in `domain/`.
- Preserve public behavior unless the task explicitly asks for a behavior change.
- When changing formulas or ranking logic, add/update tests with concrete fixture-driven expectations.

## Python conventions
- Follow existing style in the touched file.
- Prefer type hints on new/changed function signatures.
- Use descriptive variable names over short abbreviations.
- Avoid introducing new dependencies unless absolutely necessary.

## Testing and validation
- Minimum targeted checks for touched areas:
  - `pytest tests/test_parsing_module.py`
  - `pytest tests/test_ranking_module.py`
  - `pytest tests/test_hybrid_ranking_module.py`
  - `pytest tests/test_stats_module.py`
- If `infra/` changes, also run: `pytest tests/test_cache_module.py`.
- Before finalizing larger changes, run full suite: `pytest`.

## Data and fixtures
- Reuse existing fixture patterns in `tests/data/`.
- For parser updates, add explicit malformed-line and edge-case fixtures.
- Keep example score files stable unless task explicitly concerns data updates.

## Safety checks
- Do not silently swallow parsing or math errors; fail with actionable messages or tested fallbacks.
- For ranking/statistical changes, document assumptions in code comments near the logic.

## Commit and PR guidance
- Commit message format:
  - `<area>: <short imperative summary>`
  - Examples: `domain: fix tie-break ordering`, `tests: add parser edge-case fixtures`
- PR body should include:
  - What changed
  - Why it changed
  - Validation performed (commands + outcomes)
  - Any follow-up risks or TODOs

## AGENTS.md maintenance
- Keep this file concise and specific to this repository.
- When workflow changes (new modules, test commands, tooling), update this file in the same PR.
