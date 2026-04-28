# Issues

This file tracks issues noticed during local audits. Keep entries distinct; update an existing entry when a new observation is the same underlying problem.

## `train_gpt_parcae.py`

### PLE is documented but not implemented in `train_gpt_parcae.py`

- Status: open
- Severity: high
- Location: `EXPERIMENTS.md` PLE section versus `train_gpt_parcae.py`
- Evidence: `EXPERIMENTS.md` documents `PLE_SCOPE`, `PLE_DIM`, `PLE_SCALE_INIT`, `PLE_INIT_STD`, implementation details, and multiple validation results. Searching `train_gpt_parcae.py` finds no matching env vars, module, parameters, or forward-path wiring.
- Observed impact: Any run command that relies on the documented PLE feature is a no-op or invalid assumption. This is the clearest docs/code mismatch found in the audit.
- Notes: Either the implementation was removed after the experiment notes were written, or the notes describe a feature that was never merged into this script.
- Suggested fix: Decide whether PLE should exist. If yes, implement the small lookup/projection/scale modules and config validation described in `EXPERIMENTS.md`; if no, mark the experiment notes as stale and remove PLE commands from suggested configs.

### Muon Newton-Schulz normalization differs from the public reference

- Status: open
- Severity: medium
- Location: `zeropower_via_newtonschulz5`
- Evidence: The local implementation casts `G` to bf16, then normalizes by `X.norm()`. The Modded NanoGPT Muon reference normalizes by the original `G.norm()` in fp32 before or during the bf16 cast.
- Observed impact: On random matrices, the local update differed from the reference by roughly 3.6% to 6.3% relative norm in spot checks.
- Additional evidence: The local optimizer updates `buf.mul_(momentum).add_(g)` and then uses `g.add(buf, alpha=momentum)` for Nesterov. Current Modded NanoGPT-style Muon implementations usually use EMA-style `lerp_` momentum, then interpolate the gradient toward that EMA. That is algebraically different from the local accumulated-buffer formula.
- Observed impact: The local update remained finite in repeated-step CPU tests, so this is not an immediate correctness crash. It is still not equivalent to the public reference and may change effective update scale/quality.
- Notes: The optimizer default still passes `MUON_BACKEND_STEPS=5`, matching common challenge variants. The mismatch includes both normalization precision/order and momentum formula.
- Suggested fix: Decide whether the deviations are intentional. If not, change the helper to normalize from `G.norm()` and change momentum to the `lerp_` EMA form, then run an A/B training comparison with the same seed and budget.

### Parcae eval state initialization diverges from upstream behavior

- Status: open
- Severity: medium
- Location: `GPT.initialize_state`
- Evidence: Local eval mode returns zeros for `state_init` values `normal`, `embed`, `like-init`, and `unit`. The SandyResearch Parcae reference samples the recurrent state in `initialize_state` without an eval-mode zero override.
- Observed impact: Local evaluation is deterministic for these state modes, which may be desirable for validation stability. It is not the same stochastic recurrence semantics as the upstream Parcae implementation.
- Notes: This looks intentional, not accidentally broken. The risk is conceptual comparability: results may differ from Parcae papers/reference code when the same state-init setting is used.
- Suggested fix: Either document this as a local deterministic-eval policy, or add an env/config option to choose stochastic upstream-compatible eval initialization.

### `DeepSeekMoE` is a simplified DeepSeek-style MoE, not a full DeepSeek implementation

- Status: open
- Severity: low
- Location: `DeepSeekMoE`
- Evidence: The DeepSeek-V3 reference gate supports softmax/sigmoid scoring, route scaling, group-limited routing, expert partitioning, and shared experts. The local `DeepSeekMoE` implements fine-grained segmented experts, shared experts, top-k softmax routing, optional top-k normalization, and an auxiliary balance loss, but omits group-limited routing, sigmoid routing, route scale, and distributed expert partitioning.
- Observed impact: The local module is functional and passed forward/backward/router-gradient invariant tests, but the class name can overstate reference fidelity.
- Notes: This may be an acceptable challenge-specific simplification for small coda-only experiments.
- Suggested fix: Rename or document it as DeepSeek-style/simplified MoE. If exact DeepSeek comparison is the goal, add the missing gate features and a dedicated routing parity test.

### `num_steps_pair` scalar tensor path fails

- Status: open
- Severity: low
- Location: `GPT.iterate_forward`
- Evidence: Passing `num_steps_pair=torch.tensor(1, device="cuda")` raises `TypeError: len() of a 0-d tensor` because the code calls `len(num_steps_pair)`.
- Observed impact: Normal training does not hit this path, but the type hint says `Tensor | None`, so the API accepts a shape that fails at runtime.
- Notes: The upstream SandyResearch Parcae code appears to have the same shape assumption, so this is a local robustness/API issue rather than a divergence from Parcae.
- Suggested fix: Either require a 1-D tensor pair explicitly, or handle 0-D tensors before calling `len()`.

### Bad training batch divisibility fails late

- Status: open
- Severity: low
- Location: `DistributedTokenLoader.next_batch`
- Evidence: If `global_tokens // (world_size * grad_accum_steps)` is not divisible by `seq_len`, the loader fails later with a reshape error: `shape '[-1, seq_len]' is invalid`.
- Observed impact: Defaults and tested run scripts use compatible shapes. Misconfigured experiments get a low-signal runtime error instead of an early validation error.
- Suggested fix: Validate `local_tokens % seq_len == 0` and probably `global_tokens % (world_size * grad_accum_steps) == 0` before reading/reshaping tokens.

### Flash-attn backend not exercised in local audit

- Status: validation gap
- Severity: medium
- Location: `ParcaeCausalSelfAttention` and `utils.flash_attention`
- Evidence: Local imports reported `HAS_FLASH_ATTN=False`, so all attention checks used the SDPA fallback.
- Observed impact: SDPA/GQA behavior matched a direct PyTorch reference in tiny CUDA tests. The flash-attn-4 path remains unverified in this environment.
- Suggested fix: Run the same attention equivalence and tiny training smoke in an environment with flash-attn-4 installed.

### Multi-rank distributed behavior not fully proven

- Status: validation gap
- Severity: medium
- Location: Muon update sharding/all-reduce, TTT gradient sync, DDP training loop, distributed validation reductions
- Evidence: Local tests ran on one CUDA device. They covered single-rank logic and synthetic metric equivalence, but not true multi-rank NCCL behavior.
- Observed impact: No local failure found. Remaining risk is limited to distributed synchronization, uneven work partitioning, and timing behavior under real 8-GPU runs.
- Suggested fix: Run a tiny 2-GPU or 8-GPU synthetic smoke that exercises training, validation, quantized reload, and TTT with uneven chunk/window counts.

## Test Invocation

### RWKV tests fail from repo root without adjusting import path

- Status: open
- Severity: low
- Location: `RWKV-LM-V7/tests`
- Evidence: Running `python -m pytest -q RWKV-LM-V7/tests/test_quant.py RWKV-LM-V7/tests/test_eval_fineweb_bpb.py` from the repo root failed during collection with `ModuleNotFoundError: No module named 'src'` and `No module named 'eval_fineweb_bpb'`.
- Observed impact: This is a test invocation/path issue, not evidence of a runtime bug in `train_gpt_parcae.py`.
- Suggested fix: Document the expected working directory/import path for RWKV tests, or add test configuration so they collect from the repository root.
