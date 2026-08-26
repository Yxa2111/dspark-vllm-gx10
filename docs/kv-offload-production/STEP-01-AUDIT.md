# Step 01: pinned-runtime audit and correctness contract

Date: 2026-08-27  
Baseline: vLLM `752a3a504485790a2e8491cacbb35c137339ad34`  
Anemll baseline: `v0.1.1` / `47503f8`

## Outcome

The existing connector-side DSpark veto exemption proves the lookup root cause,
but it is not a production implementation. The production path must fix both
the core KV coordinator and the offloading connector, and it must add a real
request-cancellation contract before high-concurrency use.

The live evidence behind this decision is archived outside this repository in
`/home/yxa/azusa/benchmarks/dspark-kv-offload-20260826/REPORT.md`.

## Source audit

### Ephemeral DSpark KV

The pinned runtime has an `is_eagle_group` marker, volatile-tail handling, and
iterative hit convergence. It does not distinguish a reusable EAGLE/MTP draft
cache from DSpark's per-round scratch context.

The current PoC adds `eagle_group_is_veto_exempt` only inside
`OffloadingConnectorScheduler`. That produces real filesystem hits on the live
DSpark model, but leaves the core `HybridKVCacheCoordinator` with different
semantics. Maintaining two convergence algorithms with different truth is not
acceptable for production.

The backport must mirror the typed upstream design from vLLM #48459/#47891:

1. `SpeculativeConfig.has_ephemeral_draft_context()` owns the method-level fact.
2. `KVCacheGroupSpec.eagle_group_is_veto_exempt` carries the fact after grouping.
3. DeepSeek V4 group annotation sets both `is_eagle_group` and the exemption.
4. Core and connector consume the same boolean and implement the same zero-hit
   rule.
5. A nonzero draft-group hit still participates normally in convergence; the
   group is not globally removed from lookup.

### Cancellation and deferred transfers

The pinned scheduler's external abort path is:

```text
Scheduler.finish_requests
  -> request.status = FINISHED_ABORTED
  -> Scheduler._free_request
  -> OffloadingConnectorScheduler.request_finished
  -> TieringOffloadingManager.on_request_finished
  -> Scheduler frees request KV blocks
```

`OffloadingConnectorScheduler.request_finished()` does not cancel request-owned
load/store jobs. If jobs remain, it retains `RequestOffloadState` until worker
completion, but returns `delay_free_blocks=false`. The base scheduler may
therefore free destination/source blocks while a transfer still refers to them.
The filesystem/CPU managers also have no request-scoped cancellation API.

This is consistent with the live 10-client failure:

```text
disconnect 9 of 10 clients
-> running=1, waiting=9 (reason=deferred)
-> retained transfer/prefill pressure
-> head TP0 OOMKilled and worker NVRM OOM
```

Production cancellation requires a typed job lifecycle; waiting for eventual
completion is insufficient.

### Transfer memory

The current CPU primary tier reserves a fixed pool in Spark's unified memory.
The observed boundary was:

- 512 MiB: small prompts store, an 8K request cannot reserve a complete store;
- 5 GiB target-only: 8K store/restart/load works;
- 5 GiB with DSpark: startup OOM.

Increasing `cpu_bytes_to_use` cannot solve long-context storage on a 128 GiB UMA
machine. The transfer path must use a bounded window with backpressure and must
never require one request's whole store set to fit simultaneously.

### Filesystem representation

The target-only 8K probe wrote about 1.41 GiB across 737 files, while the hot
load read about 111 MiB. This one-key-per-file, padded representation has too
much write and metadata amplification for 100K-1M contexts. It also lacks a
hard capacity limit and production eviction policy.

## Production invariants

### Correctness

- Core and connector return the same reusable prefix boundary for every group.
- DSpark scratch KV can never veto or be restored as stable cross-request KV.
- Every load/store address is inside its declared allocation.
- An incompatible layout, corrupt object, missing rank, or failed read falls
  back to recompute; it cannot crash the engine or expose partial KV as a hit.
- A cache key includes the exact model revision, runtime/layout version, dtype,
  TP geometry, group fingerprint, and ownership.

### Lifecycle

- Every transfer belongs to one request generation and has one terminal state:
  completed, failed, or cancelled.
- Disconnect/abort removes the request from scheduler waiting/deferred state and
  cancels queued work before its KV blocks can be reused.
- Late worker completion for a cancelled generation is ignored safely and
  cannot resurrect the request.
- `running`, `waiting`, connector jobs, staging bytes, and block references all
  return to baseline after cancellation.

### Resources

- CPU staging has a fixed hard byte budget independent of prompt length.
- I/O queues are bounded and provide backpressure and fair per-request progress.
- Filesystem use has high/low watermarks and deterministic LRU eviction.
- Linux page cache cannot consume an unbounded second copy of KV on UMA; the
  production data path uses direct/aligned I/O or an equivalently bounded path.

### Operations

- KV offload remains disabled by default and has one-command rollback.
- V1 offloads stable prompt KV only; active attention never page-faults from SSD.
- Application admission control limits active long contexts; parked sessions do
  not depend on aggressive automatic preemption.
- Metrics expose request/job state, bytes reserved, queue depth, cancellation,
  eviction, corruption, hit/load/store, and recompute fallback.

## Required gate matrix

| Gate | Requirement |
|---|---|
| Unit | core/connector convergence parity; zero/nonzero ephemeral hits; late completion after cancel; bounds and corrupt-object fallback |
| Two-node restart | DSpark exact-prefix hit of `N-1` tokens after restart on both TP ranks |
| Partial hit | local GPU prefix + filesystem suffix does not assert or address the wrong block |
| Cancellation | 10 and 20 clients, disconnect `N-1`; survivor completes and all queues return to zero |
| Memory | DSpark 8K, then 100K/300K/500K, without increasing a whole-request staging pool |
| Filesystem | enforced max bytes, concurrent eviction, restart recovery, corrupt/truncated object recompute |
| Soak | mixed long-agent workload with restart/cancel/preempt for at least 24 hours, no orphan jobs or monotonic UMA growth |
| Rollback | disable connector and start the immutable production image without deleting cache data |

## Upstream snapshot reviewed

The following primary-source upstream items were rechecked on 2026-08-27:

- vLLM #47891: connector-side ephemeral-group fix, open at `a9eb2db3`;
- vLLM #48459: core-side fix, open at `72e4b575`;
- vLLM #52047: general hybrid-path drafter annotation, open at `e6290916`;
- vLLM #52771: additional MTP/EAGLE lookup/store fixes, open at `44a3045e`;
- vLLM #52807: partial-hit recurrent-group boundary fix, open at `9df8d8dc`;
- vLLM #53607: DSV4 packed multi-group transfer-layout crash, open.

None is merged into the pinned runtime. Step 02 will backport only the
DeepSeek-V4/DSpark semantics required by this deployment, with tests adapted to
the pinned API. Broader hybrid-model changes remain separate until their own
gate is available.

