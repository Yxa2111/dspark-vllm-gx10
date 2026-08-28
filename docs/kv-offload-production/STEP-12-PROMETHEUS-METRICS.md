# Step 12: Prometheus metrics for persistent packed NVMe KV

Date: 2026-08-28

## Decision

Persistent packed-NVMe KV offload is now observable through vLLM's existing
`/metrics` endpoint. The implementation uses the stock
`KVConnectorStats -> scheduler aggregation -> KVConnectorPromMetrics` path;
it does not add a sidecar exporter or a second HTTP endpoint.

The labels are bounded to engine/model plus fixed `scope`, `rank`,
`operation`, and `reason` values. Request IDs, prompt hashes, slot IDs, and
cache keys are never Prometheus labels.

This change is telemetry-only. It does not change the fixed-slot data file,
rank manifests, scheduler index, persistent identity, hash key, transfer
geometry, or LRU policy.

## Metric surface

| Metric | Type | Extra labels | Meaning |
|---|---|---|---|
| `vllm:kv_offload_disk_capacity_bytes` | gauge | `scope` | configured scheduler or allocated rank capacity |
| `vllm:kv_offload_disk_capacity_slots` | gauge | `scope` | fixed direct-I/O slot count |
| `vllm:kv_offload_disk_committed_slots` | gauge | `scope` | scheduler-authoritative or rank-valid slots |
| `vllm:kv_offload_disk_restored_slots` | gauge | `scope` | slots recovered at process startup |
| `vllm:kv_offload_disk_staging_bytes` | gauge | `rank,operation` | pinned staging capacity per direction |
| `vllm:kv_offload_disk_buffer_slots` | gauge | `rank,operation` | reusable staging rows per direction |
| `vllm:kv_offload_disk_queue_depth` | gauge | `rank,operation` | sampled queued transfer events |
| `vllm:kv_offload_disk_inflight_blocks` | gauge | `rank,operation` | sampled blocks in the executing event |
| `vllm:kv_offload_disk_transfer_bytes_total` | counter | `rank,operation` | aligned bytes transferred successfully |
| `vllm:kv_offload_disk_transfer_blocks_total` | counter | `rank,operation` | fixed slots transferred successfully |
| `vllm:kv_offload_disk_transfer_operations_total` | counter | `rank,operation` | completed transfer events |
| `vllm:kv_offload_disk_transfer_duration_seconds` | histogram | `rank,operation` | backend event time through final CUDA DMA submission |
| `vllm:kv_offload_disk_evictions_total` | counter | none | committed scheduler slots reused by LRU |
| `vllm:kv_offload_disk_errors_total` | counter | `rank,operation,reason` | bounded backend failure classes |

Scopes are `scheduler`, `rank_0`, and `rank_1`; operations are `load` and
`store`. Error reasons are bounded to `queue_timeout`, `checksum`, `short_io`,
`io`, and `other`.

The Prometheus client creates labeled counter series on first observation.
Consequently, an absent `errors_total` or `evictions_total` sample means no
such event has been observed since process start, not that registration
failed; their HELP/TYPE definitions are present at startup.

## Ownership and collection timing

The scheduler owns configured capacity, committed-slot state, restore count,
and LRU evictions. Each TP worker owns its rank-local staging, queue, in-flight,
transfer, duration, and error state. Worker interval stats are serializable and
are merged by vLLM before the API process records them.

This existing vLLM transport is model-step sampled. A Prometheus scrape does
not RPC idle workers. In particular, an asynchronous store that finishes after
the final model step becomes visible on the next engine step. Queue and
in-flight gauges have the same last-sampled boundary. The live test therefore
used a small follow-up request after the cold store; the gauges then converged
to zero and the completed counters appeared.

## Offline gates

Anemll commit `e9d7ba3` adds ordered patch `0010`, the thin runtime layer, and
the connector metric implementation. Commit `5e1ab76` corrects the fake test
configuration discovered by the real image run.

| Gate | Result |
|---|---:|
| Ten ordered patches apply to pinned vLLM `752a3a504` | passed |
| Changed-file Ruff and compile checks | passed |
| Metrics aggregation/Prometheus/backend snapshot tests | 3 passed |
| Scheduler committed/eviction regression | 1 passed |

The two independently built Step 34 images contained identical runtime files.
For example, `metrics.py` SHA-256 was
`99fa14418dcfe30cedb1aceb7b61f8fa01ec50363cd2a9e57e467a5011ff916c`
on both nodes.

## Live TP=2 proof

The retained service used DeepSeek V4 Flash 0731, TP=2, DSpark,
`nvfp4_ds_mla`, 262,144 maximum context, 8,192 batched tokens, 0.73 memory
utilization, 32 GiB per-rank direct NVMe files, and the existing persistent
identity. No cache purge occurred.

The first metrics-image start restored 6,017 committed slots in all three
scopes. Capacity and staging gauges reported:

| Scope | Capacity slots | Capacity bytes | Restored slots |
|---|---:|---:|---:|
| scheduler | 32,140 | 34,359,738,368 | 6,017 |
| rank 0 | 32,140 | 34,359,459,840 | 6,017 |
| rank 1 | 32,140 | 34,359,459,840 | 6,017 |

Each rank reported two 1,069,056-byte staging rows, or 2,138,112 bytes, for
each transfer direction.

A deterministic 8,192-token cold request then produced one 64-slot store
event per rank:

| Rank | Store blocks | Store bytes | Backend event time |
|---|---:|---:|---:|
| 0 | 64 | 68,419,584 | 0.112934 s |
| 1 | 64 | 68,419,584 | 0.109337 s |

The scheduler and both rank gauges advanced together from 6,017 to 6,081
committed slots. A complete TP=2 restart restored 6,081 slots in every scope.

Replaying the byte-identical request after restart produced 4,096 external KV
hits and computed only the remaining 4,096 prompt tokens. The load metrics
were:

| Rank | Load blocks | Load bytes | Backend event time |
|---|---:|---:|---:|
| 0 | 38 | 40,624,128 | 0.055023 s |
| 1 | 38 | 40,624,128 | 0.057422 s |

Cold TTFT was 4.628 s and restart-hot TTFT was 1.974 s, a 2.34x speedup. The
prompt SHA-256 was
`1864dce336da34126d3faa98ef2a05d8224ee12bb834d1a92d97b30725a5b3b1`
in both requests, and the one-token completion SHA-256 was
`4c6773e331ed318097c14680de92d192dfdac3c0d7cc3114c020b0014d8a6ff6`
in both requests.

After the gate, queue and in-flight gauges were zero for both operations and
both ranks. No disk error or scheduler eviction sample was present. Both
project ranks were healthy, both peer guards were active, and Qwen ASR and
Grafana remained running.
