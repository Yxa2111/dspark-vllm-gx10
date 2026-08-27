# Step 02: core-owned DSpark ephemeral-group semantics

Date: 2026-08-27
Baseline: vLLM `752a3a504485790a2e8491cacbb35c137339ad34`
Runtime image: `dspark-vllm-gx10:kv-offload-ephemeral-contract`

## Outcome

The connector-only root-cause PoC has been replaced by one typed contract used
by both the core KV coordinator and `OffloadingConnector`. DSpark's draft
context may ignore a zero external hit because the runtime rebuilds that
scratch state from current target hidden states. Stable target groups still
determine the reusable prefix boundary. EAGLE3, MTP, and DFlash retain their
old authoritative-miss behavior.

This step closes only the group-semantics layer. It does **not** certify the
current CPU/filesystem data plane for multi-node production; the live test
found a separate node-local shared-memory defect recorded below.

## Contract

The fact originates at the speculative method and is carried explicitly:

```text
SpeculativeConfig.has_ephemeral_draft_context()
  -> KVCacheGroupSpec.eagle_group_is_veto_exempt
  -> HybridKVCacheCoordinator SpecGroup.veto_exempt
  -> OffloadingConnector GroupOffloadConfig.eagle_group_is_veto_exempt
```

The rule is deliberately narrow:

- `dspark`: `true`;
- `eagle3`, MTP, DFlash, and unannotated groups: `false`;
- only an exact zero hit is non-authoritative;
- a nonzero DSpark hit still participates in normal convergence;
- the connector records which group was excluded and omits only that group's
  load keys and GPU destinations;
- stable groups continue to load, while the model reconstructs draft scratch
  state during execution.

Identical KV specs may share a core `SpecGroup`. Their exemption is merged with
conservative AND semantics: the shared lookup can ignore zero only if every
member declares the same exemption. The existing EAGLE last-block drop and
recurrent/SWA convergence rules remain intact.

## Implementation

`patches/vllm/0003-kv-offload-dspark-ephemeral-contract.patch` changes the
five pinned vLLM modules that own this contract:

1. `vllm/config/speculative.py`: method-level typed predicate;
2. `vllm/v1/kv_cache_interface.py`: group metadata field;
3. `vllm/v1/core/kv_cache_utils.py`: DeepSeek V4 annotation and worker
   projection;
4. `vllm/v1/core/kv_cache_coordinator.py`: core convergence behavior;
5. `vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py`:
   external lookup and selective-load parity.

The old `0003-kv-offload-dspark-veto-exempt.patch` is removed. The Docker
build and patch validator now compile every changed module, so a patch that
drifts from the pinned source fails during validation and image build.

## Deterministic test gate

`scripts/test-vllm-ephemeral-contract.sh` runs against the exact test tree from
the commit in `upstream.lock`. It checks the installed image first, copies the
pinned tests plus our overlays to a temporary directory, forces Hugging Face
offline mode, and uses a generated local OPT config rather than network model
metadata.

The nine CPU cases cover:

- method ownership: DSpark positive; EAGLE3 and DFlash negative;
- core zero-hit exemption and non-exempt regression behavior;
- connector zero-hit exemption and non-exempt regression behavior;
- selective load under both synchronous and asynchronous scheduling.

Two-node worker result:

```text
.........                                                                [100%]
9 passed
```

Static gates also pass:

```text
scripts/check-vllm-patches.sh
python3 -m py_compile <changed modules and test overlays>
bash -n scripts/check-vllm-patches.sh scripts/test-vllm-ephemeral-contract.sh
git diff --check
```

## Live two-node integration

Both Spark nodes built the same source/patch stack and started with DSpark,
TP=2, `nvfp4_ds_mla`, 512 MiB CPU staging, and filesystem secondary tier.
Observed startup facts:

- both entrypoints accepted the typed coordinator without rewriting it;
- the service reached `/health`, `/v1/models`, and a minimal chat completion;
- packed-layout diagnostics on both workers reported `bounds_ok=true`;
- available KV was 13.63 GiB / 890,965 tokens, or 3.40 times a 262,144-token
  request;
- low shape buckets and concurrency 1/2/4/6 warmup arms completed;
- the 9.5K long-chunk arm took 256 seconds and drove the head into severe
  page-cache/swap pressure, so the final warmup tally was intentionally not
  accepted as a pass.

The first live start also exposed a deployment integration conflict: MiaAI's
issue-26 hybrid-SWA startup hotfix looked for the old stock coordinator anchor
and failed closed. Its compatibility logic now recognizes the exact production
ephemeral block as a safe built-in superset. It remains fail-closed for a
partial or unknown block.

## Multi-node data-plane finding

Live cache inspection and source tracing found that the pinned tiering path is
not a valid two-node persistence architecture:

1. `SharedOffloadRegion` coordinates workers through
   `/dev/shm/vllm_offload_<instance>.mmap`, which is shared only inside one
   host.
2. `CPUOffloadingSpec.create_worker()` selects
   `torch.accelerator.current_device_index()` as the slot rank. On two
   one-GPU Spark nodes, both workers select local device index 0 rather than TP
   global ranks 0 and 1.
3. The scheduler-side `TieringOffloadingManager` and filesystem tier exist on
   the head. The worker-side B/C/D/E/F filesystem roots contained zero files;
   all persistent files were written by the head.

Consequently, a head-side filesystem lookup/hit does not prove that TP1's KV
bytes survived or were restored correctly. Earlier one-token output equality
is not strong enough to certify this path. No further cross-restart hit is
classified as a correctness pass until Step 04 introduces per-node/global-rank
ownership and a deterministic multi-token/logit gate.

## Handoff to Step 03 and Step 04

Step 03 owns cancellation and deferred-job convergence. Its acceptance must
include queued abort, mid-prefill abort, late completion, and 10/20-client
disconnect tests.

Step 04 must replace the current cross-node mmap assumption with explicit
per-global-rank descriptors and node-local staging/SSD ownership before adding
bounded streaming. It must prove that both ranks store and load nonzero,
checksummed payloads and that corruption of either rank forces recompute.
