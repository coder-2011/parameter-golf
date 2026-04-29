# Dynamic SWA Modulation

`train_gpt_parcae.py` keeps the existing SWA behavior by default. Set `SWA_ENABLED=1`
to use static SWA, and add `SWA_DYNAMIC=1` to modulate snapshot cadence and weight
during warmdown.

The implementation follows three observations from the SWA literature:

- SWA works by averaging optimizer iterates from the late, high-exploration part of
  training rather than using the averaged weights during training.
- Dense collection can improve the approximation of the flat region when only a
  short training window is available.
- Adaptive selection can avoid blindly averaging every late checkpoint, but full
  validation-gated selection is expensive in this challenge loop.

For this trainer, dynamic SWA starts when the existing warmdown LR multiplier drops
below `SWA_START_FRAC`. Let

```text
progress = clamp((SWA_START_FRAC - lr_scale) / SWA_START_FRAC, 0, 1)
```

Then `SWA_DYNAMIC=1` uses:

```text
shaped_progress = progress ** SWA_DYNAMIC_POWER
interval = round(
    SWA_DYNAMIC_MIN_EVERY
    + (SWA_EVERY - SWA_DYNAMIC_MIN_EVERY) * (1 - shaped_progress)
)
weight = 1 + (SWA_DYNAMIC_WEIGHT_MAX - 1) * shaped_progress
```

So snapshots are collected about every `SWA_EVERY` steps at the beginning of the
SWA region, then more densely near the end of the run, down to
`SWA_DYNAMIC_MIN_EVERY`. Later snapshots can also receive more weight.

Recommended first test:

```bash
SWA_ENABLED=1 \
SWA_DYNAMIC=1 \
SWA_START_FRAC=0.2 \
SWA_EVERY=50 \
SWA_DYNAMIC_MIN_EVERY=5 \
SWA_DYNAMIC_POWER=1.0 \
SWA_DYNAMIC_WEIGHT_MAX=2.0 \
python train_gpt_parcae.py
```

EMA still takes priority over SWA. If `EMA_ENABLED=1`, SWA is not updated or applied.

References:

- Izmailov et al., "Averaging Weights Leads to Wider Optima and Better Generalization":
  https://arxiv.org/abs/1803.05407
- Cha et al., "SWAD: Domain Generalization by Seeking Flat Minima":
  https://arxiv.org/abs/2102.08604
- Demir et al., "Adaptive Stochastic Weight Averaging":
  https://arxiv.org/abs/2406.19092
- PyTorch `AveragedModel` docs:
  https://docs.pytorch.org/docs/stable/generated/torch.optim.swa_utils.AveragedModel.html
