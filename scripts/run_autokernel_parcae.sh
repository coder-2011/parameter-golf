#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
AUTOKERNEL_DIR=${AUTOKERNEL_DIR:-"$REPO_ROOT/.autokernel"}
AUTOKERNEL_REPO=${AUTOKERNEL_REPO:-"https://github.com/RightNow-AI/autokernel.git"}
AUTOKERNEL_REF=${AUTOKERNEL_REF:-"main"}
PYTHON_BIN=${PYTHON:-"$REPO_ROOT/.venv/bin/python"}
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN=python
fi
AUTOKERNEL_USE_UV=${AUTOKERNEL_USE_UV:-0}

if [[ ! -d "$AUTOKERNEL_DIR/.git" ]]; then
  git clone --depth=1 --branch "$AUTOKERNEL_REF" "$AUTOKERNEL_REPO" "$AUTOKERNEL_DIR"
fi

cd "$AUTOKERNEL_DIR"

run_autokernel() {
  if [[ "$AUTOKERNEL_USE_UV" == "1" ]]; then
    uv run "$@"
  else
    "$PYTHON_BIN" "$@"
  fi
}

command=${1:-profile}
shift || true

profile_output=${AUTOKERNEL_PROFILE_OUTPUT:-"$REPO_ROOT/logs/autokernel_parcae_profile.json"}
case "$command" in
  profile)
    mkdir -p "$(dirname "$profile_output")"
    run_autokernel profile.py \
      --model "$REPO_ROOT/scripts/autokernel_parcae_adapter.py" \
      --class-name ParcaeAutoKernelModel \
      --input-shape "${AUTOKERNEL_INPUT_SHAPE:-4,128}" \
      --dtype "${AUTOKERNEL_DTYPE:-bfloat16}" \
      --warmup-iters "${AUTOKERNEL_WARMUP_ITERS:-3}" \
      --profile-iters "${AUTOKERNEL_PROFILE_ITERS:-5}" \
      --output "$profile_output" \
      "$@"
    "$PYTHON_BIN" "$REPO_ROOT/scripts/normalize_autokernel_profile.py" "$profile_output"
    ;;
  extract)
    run_autokernel extract.py \
      --report "${AUTOKERNEL_PROFILE_REPORT:-$profile_output}" \
      --top "${AUTOKERNEL_EXTRACT_TOP:-5}" \
      --backend "${AUTOKERNEL_BACKEND:-triton}" \
      "$@"
    ;;
  bench)
    run_autokernel bench.py "$@"
    ;;
  prepare)
    run_autokernel prepare.py "$@"
    ;;
  orchestrate)
    run_autokernel orchestrate.py "$@"
    ;;
  *)
    echo "usage: $0 [profile|extract|bench|prepare|orchestrate] [extra AutoKernel args...]" >&2
    exit 2
    ;;
esac
