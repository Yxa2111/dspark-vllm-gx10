# Step 08 - restart-persistent packed NVMe audit

Date: 2026-08-27

State: design audit complete.  This document is the acceptance contract;
implementation and offline evidence are recorded in
[`STEP-09-PERSISTENT-INDEX-IMPLEMENTATION.md`](STEP-09-PERSISTENT-INDEX-IMPLEMENTATION.md).

## Goal

Retain reusable DeepSeek V4 packed KV across a complete, graceful two-node
service restart without allowing a stale, partially overwritten, wrong-model,
or one-rank-only slot to become a cache hit.

The existing process-lifetime local-NVMe backend remains the proven baseline.
Persistence will be an explicit opt-in and must not change its behavior by
default.

## Why retaining the current slot file is unsafe

The current backend deliberately unlinks the rank file at both initialization
and shutdown.  Three pieces of authoritative state exist only in memory:

1. `SimpleCPUOffloadScheduler.cpu_block_pool` owns the mapping from a
   `BlockHashWithGroupId` to the numeric disk slot;
2. each rank's `DiskBackend._slot_checksums` owns the checksum for the physical
   packed row written by that rank;
3. the scheduler owns whether every worker completed the store event before
   the hash became visible to prefix lookup.

Keeping only `vllm-kv.slots.rank_N` would recover none of these.  Worse, slot
reuse can overwrite the row after the old hash has been persisted elsewhere.
A restart could then load valid-looking bytes for the wrong prefix.  A
production implementation must make interrupted overwrite become a miss, not
an incorrect hit.

## Persistence mode and files

The connector will add an explicit boolean `disk_persistence` setting.  It is
valid only for the packed `disk` backend and defaults to `false`.

Persistent mode retains three kinds of mode-0600 regular files under the
already rank-local bind mounts:

```text
head:
  vllm-kv.slots.rank_0       packed data rows
  vllm-kv.slots.rank_0.meta  rank-0 checksum/validity records
  vllm-kv.slots.index        scheduler hash-to-slot records

worker:
  vllm-kv.slots.rank_1       packed data rows
  vllm-kv.slots.rank_1.meta  rank-1 checksum/validity records
```

The data files stay `O_DIRECT`.  Metadata is bounded fixed-record storage: one
small header plus one record per numeric slot.  At 32,140 slots its memory and
page-cache footprint is a few MiB rather than prompt-sized.

No pickle, Python object representation, directory-per-block layout, or
unbounded append-only journal is accepted.  Hash records use an explicit
length plus raw `BlockHashWithGroupId` bytes and a record checksum.

## Identity and geometry header

Every file has a versioned, checksummed header.  Opening a pre-existing cache
must validate at least:

- format magic and version;
- an operator-visible cache namespace;
- model and tokenizer identity/revision;
- KV dtype and packed payload geometry;
- scheduler/hash block size and fixed `PYTHONHASHSEED` contract;
- TP world size and the file's global rank where applicable;
- number of slots, physical payload bytes and aligned row stride.

A mismatch fails closed before the API becomes ready.  It must not silently
truncate or reinterpret the old cache.  Reinitialization is a separate,
explicit destructive operation in the deployment layer.

## Per-slot two-phase durability protocol

The scheduler index is the final visibility authority.  For every set of slots
being reused or newly stored:

```text
1. scheduler: mark index records INVALID and fdatasync(index)
2. each rank: mark local meta records INVALID and fdatasync(meta)
3. each rank: write complete packed rows and fdatasync(data)
4. each rank: write checksum records VALID and fdatasync(meta)
5. all ranks: report the store event complete
6. scheduler: write hash-to-slot records COMMITTED and fdatasync(index)
7. scheduler: publish those hashes to the in-memory prefix cache
```

Step 1 happens before worker metadata is dispatched.  Step 6 happens only
after the existing all-worker completion count reaches `world_size`.

This establishes the required ordering across nodes: if a committed scheduler
record survived restart, both rank payloads and both local checksum records
were acknowledged durable first.  A crash at any earlier point leaves the
scheduler record invalid, so orphaned data is ignored.

The same order applies to overwrites.  An old committed hash is invalidated
durably before either rank can modify its row.

## Startup reconstruction

On restart, each worker opens its exact global-rank files with no symlink
following and validates the header and size.  It reconstructs only locally
valid checksums.

The scheduler validates its index and reconstructs cached CPU blocks at their
original numeric slot IDs.  It inserts only committed, record-checksummed hash
records.  The reconstructed blocks remain refcount-zero eviction candidates;
restart resets LRU order but not hash ownership.

The first load still validates the full physical row against the rank-local
checksum before GPU DMA.  A corrupt record or payload must never be copied to
the GPU.  The implementation may invalidate the affected cache entry and
recompute, but it must not return generated tokens from unverified KV.

## Reset, shutdown and cleanup semantics

- Normal persistent shutdown drains or abandons bounded work using the
  existing rules, closes descriptors, and retains committed files.
- `reset_prefix_cache` invalidates the durable scheduler index before reporting
  success.  In-flight stores remain non-committed until their existing
  completion/abandonment path converges.
- Process-lifetime mode keeps the existing unlink-at-start and unlink-at-stop
  behavior.
- The MiaAI normal stop path must stop deleting persistent files.  Cache purge
  becomes an explicit exact-path operation and must remove data, meta and index
  files on both nodes only after both ranks have stopped.

## Security boundary

Packed KV can encode user prompt content.  Persistence is opt-in, uses 0600
files, refuses symlinks/unexpected file types, never logs block hashes or
prompt-derived keys, and requires an explicit purge procedure.  Encryption at
rest is outside this patch; production enablement must rely on encrypted local
storage or explicitly accept that host-level threat model.

## Required deterministic gates

The implementation is not accepted without tests for:

1. clean create, close and reopen with the same identity and geometry;
2. hash-to-slot and checksum reconstruction;
3. stale model/namespace/rank/geometry rejection;
4. header, index-record, checksum-record and payload corruption rejection;
5. crash points after each of protocol steps 1 through 6, proving that only a
   fully committed record is visible after restart;
6. overwrite invalidation before data mutation;
7. duplicate hashes and slot eviction/reuse;
8. short metadata I/O, `ENOSPC`, `fdatasync` and background failure propagation;
9. bounded metadata size and no unbounded journal/inode growth;
10. regression coverage for process-lifetime unlink behavior, cancellation,
    queue bounds and packed DMA geometry.

## Required live gates

The final two-node proof must:

1. cold-store enough distinct prefixes to force hot-pool eviction;
2. stop both ranks normally without purging the persistent files;
3. start the same image and identity on both nodes;
4. replay an evicted prefix and show nonzero external hit plus nonzero NVMe read
   on both ranks;
5. reproduce the cold completion and compare deterministic output hash;
6. repeat with one rank's metadata/payload corrupted and prove fail-closed
   recomputation rather than partial-rank reuse;
7. prove an identity mismatch refuses or explicitly clears the cache;
8. run normal purge and verify only the exact five persistent artifacts are
   removed.

Passing only an in-process restore remains insufficient evidence.
