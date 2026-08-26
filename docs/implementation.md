# Spark-specific integration

FlashInfer includes the native SM120/SM121 DeepSeek V4 sparse MLA kernel, but
the pinned vLLM wrapper does not fully describe Spark's NVFP4 layout. The
overlay provides the missing integration:

- routes `nvfp4` to the `nvfp4_ds_mla` packed `uint8` cache format;
- uses the 584-byte token layout required by the native kernel;
- preserves valid compressed C128 pages while normalizing SWA cache tensors;
- supports TP=2 query-head geometry;
- pads 256-wide DSpark draft indices to the native 512-wide dispatch using
  `-1` sentinels without changing active lengths;
- carries the b12x NVFP4 MoE integration used by the validated runtime.

The b12x adapter registers `flashinfer_b12x` with vLLM's modular MXFP4 oracle,
prepares the checkpoint's native MXFP4 tensors into b12x's W4A16 runtime
format, plans caller-owned scratch, and rejects allocations during CUDA graph
capture. The small-M selector override is opt-in/configurable through the
`VLLM_B12X_W4A16_*` environment variables.

## B12X route-pack startup warmup

B12X route packing uses Triton kernels specialized by both a power-of-two
route capacity and the divisibility of the live route count. Previously, a new
long-prefill shape could compile `_pack_topk_routes_prefix_kernel` after
request execution began. If compilation occurred during CUDA graph capture,
it could terminate the engine.

Each TP rank now prewarms route packing during model loading, before CUDA graph
capture. It covers every power-of-two capacity through
`MAX_NUM_BATCHED_TOKENS`; for capacities greater than two, it warms both the
aligned capacity and capacity minus one. Those calls cover Triton's aligned
and generic scalar specializations. The completed warmup is cached per CUDA
device, expert count, top-k, and maximum capacity. With the default
`MAX_NUM_BATCHED_TOKENS=8192`, startup increases by approximately 10--11
seconds per rank and adds no request-path warmup.

The expected startup log is:

```text
Prewarmed B12X route-pack capacities (...) on cuda:0 (experts=256, topk=6)
```

## JIT monitor

Compose enables vLLM's JIT monitor in warning mode by default. Set these values
in both node environment files when more detail is needed:

```bash
JIT_MONITOR_MODE=warn
JIT_MONITOR_VERBOSE=1
```

`warn` reports unexpected inference-time compilation while allowing the server
to continue. `error` is intended only for cold-start validation: it terminates
the engine on any previously unwarmed kernel, including kernels unrelated to
B12X.

A cold TP=2 run with `JIT_MONITOR_MODE=error` completed 33,966-, 36,549-, and
40,720-token prefill requests without route-pack compilation. A subsequent
65,536-token test exposed separate first-use specializations in vLLM's indexer
(`_build_prefill_chunk_metadata_kernel`) and the CuTeDSL
`W4A16FusedMoeKernel`. The first decode matrix also compiled a dflash input
preparation shape. None was a route-pack kernel, so the result validates this
fix but is not yet a full-stack zero-JIT guarantee for every cold request. Use
`warn` for normal serving until those independent warmup gaps are addressed.

The new backend was informed by `voipmonitor`'s earlier Apache-2.0 b12x vLLM
integration in vllm-project/vllm pull request 39634, then expanded for the
current vLLM modular-MoE APIs and the tested DeepSeek V4 TP=2 path. Full
provenance is in `CREDITS.md`.

All upstream revisions are pinned in `upstream.lock` so a build cannot silently
move to incompatible wrapper or kernel behavior. In particular, b12x is pinned
to Git commit `7dc6fb8f`; its package reports `0.15.3`, but that version was not
published to PyPI.

## Experimental packed-KV offload diagnostics

The `experiment/packed-kv-offload-tp2` line keeps the pinned vLLM revision and
applies a small patch series after the normal overlay. The first patch is
diagnostic-only: it does not change transfer geometry, hashing, or scheduling.

Set `DSPARK_KV_OFFLOAD_DIAG=1` to emit structured
`DSPARK_KV_OFFLOAD_DIAG` records for:

- configured KV groups, block sizes, packed tensor sizes, and `block_stride`;
- each attention view's byte offset, stride, storage size, and bounds result;
- the whole-packed canonical view selected by `OffloadingConnectorWorker`;
- CPU staging/offloaded-block sizing and the scheduler's parallel rank;
- filesystem mapper rank/base path plus the first eight key mappings.

The records contain metadata and hash prefixes only. They never inspect prompt
tokens or KV contents. The switch defaults to `0`, so the production path pays
only the module import and disabled function calls.

Validate the patch stack against the exact commit in `upstream.lock` before an
image build:

```bash
./scripts/check-vllm-patches.sh
```

For Phase 0, a thin image applies the same checked patch to the immutable 0.1.1
runtime without recompiling CUDA extensions:

```bash
FINAL_IMAGE=dspark-vllm-gx10:kv-offload-diag-phase0 \
  ./scripts/build-kv-offload-diag-image.sh
```

Build that tag independently on both ARM64 nodes from the same commit. This is
an instrumentation shortcut only; layout-changing fixes must use the complete
source build.

This Phase 0 evidence decides whether the existing whole-packed path is
self-consistent on both TP ranks. Per-group transfer geometry is not introduced
until those invariants and an actual store/restart/load trace show it is needed.
