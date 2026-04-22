# Repository Guidelines

Naman owns this repository.

## Project Objective
`parameter-golf` is a constrained language-model challenge repo: train language models under fixed artifact and wall-clock constraints, with fixed tokenization/evaluation requirements, and optimize for validation bits-per-byte.

## Global Safety Rules
- Default safety: do not run `browser-use -b real --profile ...` against local Chrome profiles.
- Local Chrome automation is allowed only when the user explicitly requests local profile usage in that same turn.
- If local-profile browser automation is requested:
  - Verify an LLM API key is present.
  - Verify free disk space is at least 3 GB.
  - Kill only stale automation processes (`browser_use.skill_cli.server` and Chrome processes with `--remote-debugging-port` / `--headless`).
  - Never kill user Chrome processes.
  - Use these environment settings:
    - `TIMEOUT_BrowserStartEvent=240`
    - `TIMEOUT_BrowserLaunchEvent=240`
    - `BROWSER_USE_SOCKET_TIMEOUT=1800`
    - `BROWSER_USE_DISABLE_EXTENSIONS=1`
    - `BROWSER_USE_SKIP_PROFILE_COPY=0`
  - If profile copy fails due to no space, stop and ask for cleanup before retrying.
  - If browser start/launch repeatedly times out, stop and report root cause.
  - Never use live-profile fallback (`BROWSER_USE_SKIP_PROFILE_COPY=1`) unless user explicitly requests it in the same turn.

## Dependency and Environment Policy (UV-first)
- Use `uv` as the dependency tool for Python environment setup.
- Authoritative dependency files:
  - [`pyproject.toml`](/Users/namanchetwani/Projects/parameter-golf/pyproject.toml)
  - [`uv.lock`](/Users/namanchetwani/Projects/parameter-golf/uv.lock)
- Recommended bootstrap flow:
  - `rm -rf .venv`
  - `uv venv .venv`
  - `source .venv/bin/activate`
  - `uv sync --no-install-project`
- Do not rely on `requirements.txt` as the root source of truth.
- Keep imports and package usage aligned with `pyproject.toml` and `uv.lock` so everything is installable with `UV sync`.

## Reasoning and Explanations
- Prioritize clarity and truthfulness.
- Use evidence-first reasoning: data/observed behavior first, then conclusion.
- Explicitly separate facts, assumptions, and speculative inference.
- Prefer concise but complete rationale.

## Evidence-First Reasoning Mindset
- Use the flow: evidence -> belief update -> conclusion.
- Do not force conclusions first and then search only for matching support.
- Prefer checks that can disconfirm a favored hypothesis.

## Agent Learning and Session Notes
- Use `.codex/learning.md` if present for durable learnings.
- Read it before non-trivial implementation work when available.
- Append concise notes for architecture decisions, shape bugs, environment issues, or environment breakages.

## Build, Test, and Development Commands
- `source .venv/bin/activate`
- `uv sync --no-install-project` to install dependencies.
- GPU training commands remain as in the scripts and challenge entrypoints.
- Avoid destructive dependency file churn unless requested.

## Coding Style and Conventions
- Python-first workspace.
- Use 4-space indentation.
- `snake_case` for functions/variables/modules; `PascalCase` for classes.
- Keep edits surgical and high-signal.
- Keep comments focused on intent and contracts, not obvious behavior.

## Coding Workflow / Quality Bar
- Keep changes scoped and intentional.
- Minimize framework churn unless directly relevant to the task.
- Validate before PR and keep checks green before merge.

## Git Safety
- Safe defaults: `git status`, `git diff`, `git log`.
- Do not use destructive Git commands unless explicitly requested:
  - No `git reset --hard`
  - No `git checkout --`
- Prefer explicit paths over broad operations when staging changes.
- No amend unless explicitly requested.

## Commit and PR Rules
- Use Conventional Commits where practical (`feat:`, `fix:`, `chore:`, `docs:`).
- Keep PRs scoped and evidence-based.
- Include behavior and impact in PR notes.

## Build/Task Tracking
- Use `TODO.md` only if the user explicitly asks for task tracking updates in that file.
- If using PR/CI workflows in this repo, keep this file as the source of truth for local instructions.
