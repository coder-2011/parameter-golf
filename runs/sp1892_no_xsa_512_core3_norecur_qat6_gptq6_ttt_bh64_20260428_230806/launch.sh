#!/usr/bin/env bash
set -uo pipefail
./run.sh
status=$?
echo "RUN_EXIT:${status}" >> wrapper_console.log
exit "$status"
