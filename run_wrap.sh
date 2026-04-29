#!/bin/bash
# run_wrap — wrapper around autoresearch.sh with feature overrides baked in.
# Edits between experiments should ONLY go into the FEATURE BLOCK below.
set -euo pipefail
cd "$(dirname "$0")"

# --- FEATURE BLOCK (edit per experiment) ---
export RESIDUAL_MODE=parallel
export PARALLEL_RESIDUAL_SCOPE=core
export MUON_MOMENTUM=0.85
export GPTQ_ENABLED=0
export QUANT_BITS=6
export RANS_INT6=1
export SWA_DYNAMIC=1
export QAT_BITS=0
# -------------- END FEATURE BLOCK ------------

bash ./autoresearch.sh
