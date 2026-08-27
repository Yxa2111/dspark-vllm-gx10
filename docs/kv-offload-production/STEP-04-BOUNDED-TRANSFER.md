# Step 04 - bounded TP2 rank-zero staging

Date: 2026-08-27

State: complete for transfer correctness. Filesystem capacity, write
amplification, stale-mmap collection, and final production gates remain Step
05/06 work.

## Goal

Persist one scheduler-visible KV block file containing the true packed KV bytes
from both TP ranks without assuming that rank 0 and rank 1 are replicas. Only
the head Spark owns the filesystem tier; the worker Spark uses bounded CPU
staging and Gloo over the existing RoCE fabric.

```text
store: TP0 GPU -> head mmap rank-0 slot ----\
                                             -> head NVMe block
       TP1 GPU -> worker CPU -> Gloo -> head mmap rank-1 slot

load:  head NVMe block -> head mmap rank-0 slot -> TP0 GPU
                      \-> head mmap rank-1 slot -> Gloo -> worker CPU -> TP1 GPU
```

The scheduler may observe a store as complete only after every rank's GPU DMA
is ready and the rank-1 payload has reached the head mmap. A load is complete
only after the inverse relay and both CPU-to-GPU copies finish.

## Implementation

Patch `0005-kv-offload-rank-zero-staging.patch` adds:

- `RankZeroStagingRelay`, using a TP CPU/Gloo process group and blocking,
  ordered send/receive operations capped by `max_transfer_chunk_bytes`;
- explicit rank views in `SharedOffloadRegion`, so rank 0 maps every worker
  slot while remote ranks keep node-local staging tensors;
- an all-rank readiness reduction before a completed store becomes visible;
- rank-zero gather on store and scatter on load;
- fail-closed configuration: the mode currently accepts only MP execution,
  contiguous global ranks, TP-only parallelism, more than one node, and global
  rank 0 as filesystem owner;
- packed-view allocation bounds checks before constructing `as_strided` views;
- legal 4 KiB row-alignment tail padding, represented separately from the two
  real worker payload slots.

The default relay bound is 64 MiB. The tested DeepSeek V4 layout has a
1,065,792-byte payload per rank and a 2,134,016-byte aligned filesystem row.
The 2,432-byte difference between two real rank payloads and the row stride is
alignment padding, not a third/partial worker slot.

## Failure found during live bring-up

The first two-node launch failed before serving because the initial validation
required the row stride to contain only whole worker pages:

```text
ValueError: row stride must contain whole per-worker pages
```

Live geometry proved the check was too strict:

- two rank payloads: 2,131,584 bytes;
- aligned row: 2,134,016 bytes;
- legal tail padding: 2,432 bytes.

The fix added an explicit `num_worker_slots` contract and validates
`slots_end <= row_stride`; it never exposes the tail as payload. A dedicated
regression test covers this exact non-divisible geometry.

## Verification

The exact head image used for the final run was
`dspark-vllm-gx10:kv-offload-rank-zero-staging` with image ID
`sha256:aba74bd147b38d2825e326bac313a194bdef645d5b980e6ffdcce3c19c3ad947`.
Both nodes were built from the same patch stack; OCI IDs can differ because the
builders are independent.

Pinned test source was vLLM commit
`752a3a504485790a2e8491cacbb35c137339ad34`.

| Check | Result |
|---|---:|
| patch stack applies and Python-compiles | 5 patches validated |
| bounded two-process Gloo round trip, mmap slots, config gates | 33 passed |
| Step 03 cancellation regression | 127 passed |
| relay payload upper bound in unit test | 40 bytes, never exceeded |

The two-node live run used DeepSeek-V4-Flash-0731, `nvfp4_ds_mla`, DSpark,
TP=2, 512 MiB UMA staging, and a 64 MiB relay bound. The filesystem directory
was retained across a full two-node service restart.

| Metric | cold 512 | restart-hot 512 |
|---|---:|---:|
| prompt SHA-256 | `98eca1de...db91` | identical |
| TTFT | 0.440 s | 0.255 s |
| prefill KV computed | 512 tokens | 1 token |
| external prefix hit | 0 | 511 tokens |
| load bytes | 0 | 46,894,848 |
| load worker count | 0 | 2 |

Filesystem lookup found the required recoverable windows for all stable groups;
the DSpark ephemeral group followed the Step 02 veto-exempt contract. The head
held 67 block files (137 MiB after all Step 04 probes), while the worker held
zero block files. Neither rank logged a transfer, bounds, CUDA, or illegal
memory error.

The raw benchmark artifacts are committed in the MiaAI deployment fork under
`results/step04-*.json`.

## What this step does not claim

- A 256-token request cannot demonstrate a restart hit: full-attention
  alignment is 256 tokens and vLLM recomputes the final prompt token, leaving
  only 255 eligible tokens. The accepted probe therefore uses 512 tokens.
- DSpark's 8-token completion hash differed between the cold and hot probes.
  This is not accepted as a correctness proof or attributed to KV transfer;
  Step 06 must compare target-only deterministic output and separately measure
  normal DSpark repeat variance.
- An 8K prompt produced KV correctly but could not atomically reserve all rows
  in the 512 MiB primary tier (`251` rows), so no filesystem store was issued.
  This is the Step 05 capacity/write-amplification blocker.
- Orderly container removal leaves the shared mmap file behind. Stale mmap
  discovery and safe garbage collection are Step 05 requirements.

No Azusa Core admission-control change is part of this work. Scheduling policy
will be reconsidered only if the completed storage implementation still shows
an overload failure under production-shaped stress.
