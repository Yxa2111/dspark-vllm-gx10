#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
base_image="${BASE_IMAGE:-ghcr.io/anemll/dspark-vllm-gx10:0.1.1@sha256:a83948492cf13df455170fb42885f5ef4db54fefe0feff0f841ecbff464ac9d8}"
final_image="${FINAL_IMAGE:-dspark-vllm-gx10:kv-offload-diag-phase0}"

sudo docker build \
  --file "$root/docker/Dockerfile.kv-offload-diag" \
  --build-arg "VLLM_BASE=$base_image" \
  --tag "$final_image" \
  "$root"

echo "Built $final_image from $base_image"
