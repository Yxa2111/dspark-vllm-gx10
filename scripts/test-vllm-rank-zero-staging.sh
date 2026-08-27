#!/usr/bin/env bash
# Run inside an image containing the checked production patch stack.
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "$root/upstream.lock"
source_dir="${VLLM_SOURCE_DIR:-${BUILD_ROOT:-$root/.build}/vllm}"
tests_dir="${VLLM_TESTS_DIR:-$source_dir/tests}"
python_bin="${VLLM_TEST_PYTHON:-python3}"

if [[ ! -d "$tests_dir" ]]; then
  echo "Pinned vLLM tests not found: $tests_dir" >&2
  exit 2
fi
if [[ -n "${VLLM_TESTS_DIR:-}" ]]; then
  if [[ "${VLLM_TESTS_COMMIT:-}" != "$VLLM_COMMIT" ]]; then
    echo "VLLM_TESTS_COMMIT must equal pinned commit $VLLM_COMMIT" >&2
    exit 2
  fi
elif [[ "$(git -C "$source_dir" rev-parse HEAD)" != "$VLLM_COMMIT" ]]; then
  echo "vLLM checkout is not at pinned commit $VLLM_COMMIT" >&2
  exit 2
fi

"$python_bin" - <<'PY'
from vllm.v1.kv_offload.cpu.distributed_staging import RankZeroStagingRelay

assert RankZeroStagingRelay is not None
PY

tmp_dir="$(mktemp -d)"
trap 'rm -rf -- "$tmp_dir"' EXIT
mkdir -p "$tmp_dir/tests"
cp -a "$tests_dir/." "$tmp_dir/tests/"
cp -a "$root/test_overlays/vllm/tests/." "$tmp_dir/tests/"

cd "$tmp_dir"
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_TARGET_DEVICE=cpu \
  "$python_bin" -m pytest -q \
  tests/v1/kv_offload/cpu/test_shared_offload_region.py \
  tests/v1/kv_offload/cpu/test_production_distributed_staging.py \
  tests/v1/kv_offload/test_production_distributed_staging_config.py
