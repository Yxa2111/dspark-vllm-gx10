# Step 11: restart-persistent LRU restoration

Date: 2026-08-27

## Decision

The restart-persistent scheduler had a real retention bug: after a process
restart it restored the durable hash-to-slot map, but not the BlockPool free
queue ordering. New stores could therefore evict restored slots while most of
the configured disk capacity was still unused.

Commit `ed06419` fixes that invariant. The focused regression, the complete
simple-offload scheduler suite, and a two-node DGX Spark restart/replay gate
all pass.

## Root cause

The live disk scheduler had 32,140 slots. At the failing boundary it restored
3,593 committed records, all in slots 1 through 3,593. A subsequent cold 64K
request completed and stored KV, but the valid-record count remained 3,593
instead of using slot 3,594 onward.

`BlockPool` initializes every non-null block in numeric order in its free
queue. The old `_restore_persistent_index()` populated each block hash and the
hash lookup map, but left those restored blocks at the head of that queue.
`get_new_blocks()` consequently selected and evicted restored low-numbered
blocks before untouched empty blocks.

This was an LRU reconstruction defect, not a 32-GiB capacity limit, TTL, model
hash mismatch, TP namespace mismatch, or failed rank write.

## Fix

After validating and publishing every durable mapping, the scheduler now:

1. touches all restored blocks, removing them from the free queue and taking a
   temporary reference;
2. frees them in order, appending hash-bearing blocks to the LRU tail.

Blocks with no hash remain at the queue head, so never-used disk slots are
allocated before a restored persistent prefix becomes an eviction candidate.
The on-disk format and cache identity are unchanged.

The scheduler regression reopens a pool with two committed slots and one
unused slot, allocates one block, and asserts that the unused slot is selected
while both restored hashes remain available. It fails on the old restore path.

## Offline gates

The ordered nine-patch stack applies cleanly to pinned vLLM commit
`752a3a504485790a2e8491cacbb35c137339ad34` and passes its compile checks.
Inside the real Anemll runtime image with the offline DeepSeek V4 config:

| Gate | Result |
|---|---:|
| Focused restored-slot allocation regression | 1 passed |
| Complete `test_scheduler.py` suite | 33 passed |

The two independently built head and worker images contained the same
`manager.py` SHA-256:

```text
f5e2487809e234d3d21439ad2e35bdfee958f27531b4a8feff8074aca2640ea8
```

## TP=2 live proof

The retained profile used DeepSeek V4 Flash 0731, TP=2, DSpark,
`nvfp4_ds_mla`, direct local NVMe, the existing fixed hash seed and the
existing cache identity. No cache purge occurred.

The first fixed-image restart restored 3,593 committed slots. Deterministic
request A was no longer present and therefore recomputed all 64,000 prompt
tokens. Crucially, its durable store grew the scheduler index from 3,593 to
3,854 contiguous valid slots. Both rank manifests reached 3,898 records. This
is the exact point where the old path kept the count at 3,593 and overwrote
restored low-numbered slots.

After a second complete two-rank process restart, the scheduler restored all
3,854 records. Replaying byte-identical A produced:

| Measurement | First fixed-image request | After second restart |
|---|---:|---:|
| Prompt tokens | 64,000 | 64,000 |
| External KV hit tokens | 0 | 53,248 |
| Locally computed prefill tokens | 64,000 | 10,752 |
| TTFT | 32.765 s | 7.590 s |
| TTFT speedup | 1.00x | 4.32x |

The prompt SHA-256 remained
`8437f4c38ec62d4a0776adab76686fc994eee69b28a3d741d8f5ba28d5bd6059`
and the one-token completion SHA-256 remained
`4c6773e331ed318097c14680de92d192dfdac3c0d7cc3114c020b0014d8a6ff6`.

Both peer guards were active after the gate. The isolated DSpark project was
the only service restarted; Qwen ASR and Grafana remained running.

## Remaining scope

This closes the premature post-restart eviction bug. It does not claim a
30-minute sustained-concurrency pass, automatic active-request SSD paging, or
an unlimited retention policy after all 32,140 slots genuinely become full.
Those are separate gates.
