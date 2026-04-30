# AutoKernel Parcae Profiling

This repo keeps AutoKernel outside the challenge training script. The adapter in
`scripts/autokernel_parcae_adapter.py` wraps `train_gpt_parcae.GPT` for
AutoKernel's no-argument model profiler and supplies shifted labels for loss
profiling.

Run a profile:

```bash
scripts/run_autokernel_parcae.sh profile
```

This clones AutoKernel into ignored `.autokernel/` and writes
`logs/autokernel_parcae_profile.json`. The runner uses the repo Python directly
by default, because the Parcae environment already carries the CUDA/PyTorch
stack. Set `AUTOKERNEL_USE_UV=1` only if you want AutoKernel's own `uv run`
behavior. The runner also normalizes PyTorch profiler shape strings into the
`key=value` format expected by AutoKernel's extractor.

Extract starter kernels from that profile:

```bash
scripts/run_autokernel_parcae.sh extract
```

Benchmark the extracted kernel:

```bash
scripts/run_autokernel_parcae.sh bench
```

Useful overrides:

```bash
AUTOKERNEL_INPUT_SHAPE=4,512 \
MODEL_DIM=512 RECURRENT_DIM=512 TRAIN_SEQ_LEN=512 \
N_LAYERS_IN_PRELUDE=4 N_LAYERS_IN_RECURRENT_BLOCK=1 N_LAYERS_IN_CODA=4 \
scripts/run_autokernel_parcae.sh profile
```

Set `AUTOKERNEL_PARCAE_MODE=logits` to profile inference logits instead of the
training loss path. Set `AUTOKERNEL_PARCAE_COMPILE=1` only when you specifically
want the adapter to wrap the Parcae model in `torch.compile`; eager profiling is
the default because it is faster to iterate and easier to extract from.
