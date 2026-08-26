#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "$root/upstream.lock"
source_dir="${VLLM_SOURCE_DIR:-${BUILD_ROOT:-$root/.build}/vllm}"

if [[ ! -d "$source_dir/.git" ]]; then
  echo "Pinned vLLM checkout not found: $source_dir" >&2
  echo "Set VLLM_SOURCE_DIR or run scripts/build-image.sh once." >&2
  exit 2
fi
if [[ "$(git -C "$source_dir" rev-parse HEAD)" != "$VLLM_COMMIT" ]]; then
  echo "vLLM checkout is not at pinned commit $VLLM_COMMIT" >&2
  exit 2
fi

tmp_dir="$(mktemp -d)"
trap 'rm -rf -- "$tmp_dir"' EXIT
git clone --quiet --shared --no-checkout "$source_dir" "$tmp_dir/vllm"
git -C "$tmp_dir/vllm" checkout --quiet --detach "$VLLM_COMMIT"
rsync -a "$root/overlay/vllm/" "$tmp_dir/vllm/vllm/"

shopt -s nullglob
vllm_patches=("$root"/patches/vllm/*.patch)
if [[ ${#vllm_patches[@]} -eq 0 ]]; then
  echo "No vLLM patches found under $root/patches/vllm" >&2
  exit 2
fi
for patch in "${vllm_patches[@]}"; do
  git -C "$tmp_dir/vllm" apply --check "$patch"
  git -C "$tmp_dir/vllm" apply "$patch"
done

python3 -m py_compile \
  "$tmp_dir/vllm/vllm/v1/kv_offload/diag.py" \
  "$tmp_dir/vllm/vllm/v1/kv_offload/cpu/spec.py" \
  "$tmp_dir/vllm/vllm/v1/kv_offload/file_mapper.py" \
  "$tmp_dir/vllm/vllm/distributed/kv_transfer/kv_connector/v1/offloading/worker.py"

echo "Validated ${#vllm_patches[@]} patch(es) against $VLLM_COMMIT"
