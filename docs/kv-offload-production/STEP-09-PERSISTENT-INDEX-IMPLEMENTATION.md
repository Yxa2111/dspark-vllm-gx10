# Step 09 - crash-safe restart index implementation

Date: 2026-08-27

State: implementation and deterministic offline gates complete.  The required
two-node service-restart proof remains the next gate.

## Scope

This step implements the persistence contract from Step 08 without changing
the default process-lifetime behavior.  Persistent mode is enabled only when
both of these connector settings are present:

```json
{
  "disk_persistence": true,
  "disk_cache_identity": "operator-owned-model-and-layout-identity"
}
```

It also requires a fixed `PYTHONHASHSEED`; an absent seed or `random` is
rejected before the connector becomes ready.

The implementation is patch
`patches/vllm/0009-restart-persistent-packed-nvme.patch`.  The release-image
equivalent is `docker/restart-persistent-packed-nvme.patch`.  Both are derived
from staged vLLM commit `429b0c423` on top of the pinned, already patched
runtime baseline.  Anemll commit `e5ac603` records the patch and Docker wiring.

## Durable files and authority

Persistent mode retains the packed rank data files and adds bounded metadata:

```text
head:
  vllm-kv.slots.rank_0
  vllm-kv.slots.rank_0.meta
  vllm-kv.slots.index

worker:
  vllm-kv.slots.rank_1
  vllm-kv.slots.rank_1.meta
```

`RankSlotManifest` stores one fixed 16-byte checksum/validity record per disk
slot.  `SchedulerSlotIndex` stores one fixed 96-byte hash/commit record per
slot.  Both use a 4,096-byte versioned identity header.  No journal, pickle,
directory-per-hash scheme, or unbounded inode growth is introduced.

The scheduler index is the only restart visibility authority.  A rank-local
checksum by itself cannot publish a prefix hit.

## Commit ordering

The implemented overwrite/store order is:

1. scheduler invalidates the affected index records and `fdatasync`s them;
2. every rank invalidates and syncs its local manifest records;
3. every rank writes complete packed rows and `fdatasync`s the data file;
4. every rank writes and syncs the new row checksums;
5. the existing TP completion counter observes all `world_size` workers;
6. scheduler commits and syncs the new hash-to-slot records;
7. scheduler publishes the hashes into the in-memory offload block pool.

A crash before step 6 therefore leaves no durable scheduler hit.  A crash
after step 6 can only expose rows that every rank already reported durable.

## Startup and reset behavior

On restart, workers validate exact mode-0600 regular files, single-link
ownership, file size, rank identity, packed payload geometry, aligned row
stride and world size.  They reconstruct rank-local checksums.  The scheduler
validates its own identity, hash geometry and world size, then restores only
committed hash records at their original numeric slots.

Payload CRC32 is checked again before disk-to-GPU DMA.  A checksum mismatch is
fail-closed and cannot return output generated from unverified KV.

`reset_prefix_cache` durably clears the scheduler index before reporting a
successful persistent reset.  With `disk_persistence=false`, the pre-existing
unlink-at-start and unlink-at-stop contract remains unchanged.

## Deterministic evidence

The changed files passed `ruff check`, `ruff format --check`, `py_compile` and
`git diff --check` in the pinned staging worktree.

Tests were then run inside the existing
`dspark-vllm-gx10:kv-offload-bounded-work` image on a DGX Spark, using its
actual torch/vLLM runtime and an offline local model configuration.

| Suite | Result | Main coverage |
|---|---:|---|
| persistent metadata + disk backend | 25 passed | reopen, identity/size/mode rejection, header/record/payload corruption, steps 1-6, short I/O, sync failure, bounded size, default unlink |
| complete simple-offload scheduler | 33 passed | existing scheduler regression plus committed-slot restart and pre-overwrite invalidation |
| total | **58 passed** | no failed or skipped gate in these two suites |

The persistent backend test writes a packed row, closes both descriptors,
constructs a new backend, reloads the persisted checksum, reproduces the row,
then corrupts the payload and observes checksum rejection.  The scheduler test
constructs a new scheduler from the durable index and also simulates a crash
after slot invalidation but before worker overwrite completion.

## What this step does not prove

This is not yet a production promotion.  It does not prove:

- that rank 0 and rank 1 both retain and reload the same DeepSeek V4 prefix
  across a complete TP=2 service restart;
- that the restarted request has nonzero external hits and NVMe reads on both
  nodes;
- that identity mismatch and one-rank corruption produce the intended live
  operational recovery path;
- that restart persistence passes the sustained thermal and memory gates.

The current corruption path is safe but conservative: a payload checksum
failure fails the engine rather than silently falling back inside the same
request.  The live corruption gate must therefore prove supervised purge and
clean recomputation, or a later implementation must add a typed scheduler
rollback path before claiming automatic in-request recomputation.

## Next gate

Build the new diagnostic image, enable persistence under a new cache identity,
force a 64K prefix out of the hot pool, stop both ranks without purging, start
the same image and identity, and replay the prefix.  Acceptance requires the
same completion hash, external-token hits, and rank-local NVMe read deltas on
both nodes.
