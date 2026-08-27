# Step 03: cancellation and deferred-job convergence

Date: 2026-08-27
Baseline: vLLM `752a3a504485790a2e8491cacbb35c137339ad34`
Runtime image: `dspark-vllm-gx10:kv-offload-cancel-convergence`

## Outcome

Request cancellation now has a bounded, explicit lifecycle across the
scheduler, tiering manager, filesystem queue, and primary staging allocator.
An aborted request stores only complete blocks that were actually computed and
allocated. A finished request releases deferred promotions immediately and
offers already-submitted filesystem loads for queued-only cancellation.

An executing copy is never interrupted. It is allowed to complete late, but
its completion is consumed exactly once and its staging slot is released. This
rule avoids use-after-free while ensuring that cancellation cannot leave
primary slots or filesystem job accounting permanently occupied.

This step closes lifecycle correctness in the pinned implementation. It does
not claim that the existing node-local mmap data path is TP2-correct; Step 04
owns that separate production gate.

## Upstream comparison

The pinned Anemll runtime predates two upstream scheduler fixes:

- vLLM PR #49146, merge commit `94ed0bf`: clamp offloaded chunks to allocated
  KV blocks so a queued request that never ran cannot store imaginary blocks;
- vLLM PR #49285, merge commit `3f1d409`: use `num_computed_tokens` rather than
  prompt length when an aborted request builds store jobs.

Those changes fix store bounds, but current upstream does not provide a
generic safe-I/O cancellation contract for tier backends. The production patch
therefore backports their intent and adds convergence for this branch's custom
deferred promotion and filesystem-worker lifecycle.

## Lifecycle contract

The implementation follows five rules:

1. Store work is derived from complete, allocated blocks only. DSpark/EAGLE
   tail handling is preserved after the clamp.
2. An aborted request uses its recorded computed-token count; tokens that were
   accepted but never computed do not become store jobs.
3. A request that finishes before a promotion is submitted removes that
   deferred promotion and reports a failed write to the primary tier, releasing
   the reserved staging slot.
4. A submitted filesystem promotion may be cancelled only while still queued.
   A copy already executing is not interrupted or allowed to race with buffer
   reuse.
5. Every submitted job converges through exactly one result. Queue cancellation
   produces a failed `JobResult`; an executing task produces its normal late
   result. Both paths decrement in-flight accounting and release the primary
   slot once.

The sequence is:

```text
request finishes
  -> discard not-yet-submitted promotions and release their staging slots
  -> offer submitted job IDs to secondary tiers
       -> queued: remove, emit one failed result
       -> running: leave untouched, consume its eventual result
  -> primary slot becomes reusable only after one terminal result
```

Store jobs are intentionally not cancelled after submission. A completed
stable prompt may still be useful after the client disconnects, and cancelling
an active write would create the same unsafe buffer-reuse race. Filesystem
capacity and retention policy are addressed in Step 05.

## Implementation

`patches/vllm/0004-kv-offload-cancellation-convergence.patch` changes five
pinned vLLM modules:

1. `offloading/scheduler.py`: bounded storable-block calculation and
   computed-token abort semantics;
2. `tiering/base.py`: optional backend cancellation interface with a
   queued-only contract;
3. `tiering/fs/thread_pool.py`: atomic queued-task removal and one-result job
   convergence;
4. `tiering/fs/manager.py`: filesystem backend cancellation delegation;
5. `tiering/manager.py`: request-owned deferred/submitted promotion cleanup.

Backends that do not implement cancellation remain valid: their jobs complete
normally and are reclaimed through the late-result path. This keeps the base
interface compatible with future tiers and avoids pretending that arbitrary
storage I/O is safely interruptible.

## Deterministic test gate

`scripts/test-vllm-cancellation-convergence.sh` applies the ordered patch stack
to the exact source baseline and runs the existing scheduler/tiering suites plus
production overlays. The overlay uses a generated local OPT fixture and forces
offline mode, so the result does not depend on Hugging Face availability.

Coverage includes:

- abort before first scheduling;
- abort after partial prefill, across the relevant scheduler branches;
- allocated-block clamping and DSpark/EAGLE tail behavior;
- queued filesystem cancellation versus an already-running copy;
- deferred promotion release;
- late completion after request-owned state has been removed;
- existing request-finished, preemption, reset, pending-transfer, filesystem,
  and tier-manager regressions.

Formal image result on the worker Spark:

```text
127 passed, 14 warnings in 25.68s
```

Static gates also pass:

```text
scripts/check-vllm-patches.sh
python3 -m py_compile <changed modules and test overlays>
bash -n scripts/check-vllm-patches.sh scripts/test-vllm-cancellation-convergence.sh
git diff --check
```

## Live-test boundary

Step 02's experimental head container was later reported by Docker as
`OOMKilled=true` after severe unified-memory and swap pressure. Its last visible
engine failure was an RPC timeout in `sample_tokens` after the worker-side
experimental container had already been stopped. The evidence therefore does
not support attributing that event solely to either OOM or cancellation; it is
recorded as a mixed shutdown/pressure incident, not as a Step 03 pass or fail.

The deterministic lifecycle suite is complete. A 10/20-client disconnect soak
against the patched two-node service remains part of the final production gate
after Step 04 replaces the invalid cross-host shared-memory assumption. Running
that soak earlier could establish API survival, but could not certify correct
TP1 KV persistence and would repeat avoidable high-memory experiments.

## Handoff

Step 04 must make rank ownership explicit and bound each transfer before the
filesystem tier can be called production-safe. Its live acceptance must verify
nonzero, checksummed store/load bytes for both TP ranks, corruption fallback,
and a surviving long request after 10 and 20 peers disconnect.
