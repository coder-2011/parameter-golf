# WKV AGENTS (RWKV-LM-V7)

- WKV is high-criticality research code. Make small, explicit changes first, then remove complexity immediately if it is not needed.
- Follow repository conventions and avoid broad refactors; keep edits close to `train.py`, `src/`, scripts, and `eval_fineweb_bpb.py` when relevant.
- Track every run in `EXPERIMENTS.md`: data path, tokenizer, model config, command, duration, results, and comparability notes.
- Preserve challenge contracts (wall-clock, artifact accounting, BPB method, tokenizer/eval semantics).
- For dependency/setup: use `uv` workflows (`uv venv`, `uv sync --no-install-project`) and keep package usage aligned with `pyproject.toml` / `uv.lock`.
- Default behavior is non-destructive. Do not use destructive git commands unless explicitly requested.
- Use focused smoke runs for code changes first (or tiny synthetic windows for shape/serialization/optimizer checks), then longer timed runs.
- Keep validation and reporting evidence-first: prefer concrete outputs from scripts/logs and document caveats (e.g., data caps, timing, stride).
- W&B remains optional/off unless explicitly enabled via `--wandb`; avoid forcing extra workflow dependencies.
- Global safety note from root applies: do not run local Chrome profile automation unless the user explicitly asks for it in the same turn.
