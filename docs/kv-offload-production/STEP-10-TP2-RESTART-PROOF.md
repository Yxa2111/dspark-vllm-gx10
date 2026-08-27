# Step 10: TP=2 restart-persistent packed-KV proof

Date: 2026-08-27

## Decision

The restart-persistent protocol implemented in Step 09 passes its first full
two-node DGX Spark gate.  A committed DeepSeek V4 Flash packed prefix survived
a complete TP=2 service stop/start, was restored into a new scheduler process,
read from both rank-local NVMe files, and produced the same deterministic
output at 4.96x lower TTFT.

This is a functional restart-persistence acceptance, not a production
stability promotion.

## Runtime and cache geometry

The live image was built from the Step 09 patch as:

```text
dspark-vllm-gx10:kv-offload-persistent-step09
```

The isolated service used:

```text
DeepSeek-V4-Flash-0731 revision 9e165c30e2704aec5d9d593cce3eebd58bbef1cb
TP=2 + DSpark
nvfp4_ds_mla packed KV
262,144 maximum model length
345,848 hot-KV tokens after restart
32 GiB direct-I/O persistent slots per rank
PYTHONHASHSEED=0
identity=dsv4-0731-r9e165c30-nvfp4-dspark-tp2-kv09
```

Rank 0 retained its data file, manifest and scheduler index.  Rank 1 retained
its data file and manifest.  Every file was a mode-0600 single-link regular
file with the expected fixed size.

## Live interrupted-store observation

Cold requests A through D committed 2,048 scheduler slots.  During request E's
post-request interval, the independent thermal guard stopped the exact lab
project after three head samples above 88 C.  The stopped state contained
2,112 valid records in each rank manifest but only 2,048 committed scheduler
records.

Those extra 64 worker records are the expected pre-scheduler-commit orphan
case.  They remained invisible at restart.  This proves the central authority
boundary under a real interrupted service, rather than only the unit-test fault
injection: rank durability cannot publish a hash before the scheduler's final
commit.

## Restart behavior

Both TP containers were started with the exact same image, files, fixed hash
seed and identity, without purging the cache.  Worker rank 0 and rank 1 opened
their previous packed data/manifest pairs with `persistent=True`.  The new
scheduler process logged:

```text
Restored 2048 committed persistent KV slots
```

It did not restore the 64 orphan rank rows.  Model initialization, graph
capture, API readiness and the launcher's minimal request all passed.  No new
identity, checksum, CUDA, OOM or TP error appeared in the restart/replay
window.

## Same-prefix replay

The restarted service replayed cold request A byte-for-byte.  A process
restart had removed the old in-memory block pool and GPU KV, so the external
hit was necessarily derived from the durable scheduler index and rank data.

| Measurement | Cold A | After full restart |
|---|---:|---:|
| Prompt tokens | 64,000 | 64,000 |
| TTFT | 33.905 s | 6.838 s |
| External hit tokens | 0 | 53,248 |
| Computed prefill tokens | 64,000 | 10,752 |
| External hit ratio | 0% | 83.2% |
| Speedup | 1.00x | **4.96x** |

The prompt SHA-256 was identical:

```text
8437f4c38ec62d4a0776adab76686fc994eee69b28a3d741d8f5ba28d5bd6059
```

The one-token completion SHA-256 was also identical:

```text
4c6773e331ed318097c14680de92d192dfdac3c0d7cc3114c020b0014d8a6ff6
```

Prometheus independently attributed 53,248 tokens to the external prefix cache
and 10,752 tokens to computed prefill.

## Both ranks read persistent data

Container cgroup I/O deltas immediately around the replay were:

| Rank | Read delta | Write delta |
|---|---:|---:|
| rank 0 / head | 255,311,872 bytes | 49,176,576 bytes |
| rank 1 / worker | 246,489,088 bytes | 49,291,264 bytes |

The near-symmetric rank-local reads close the Step 09 TP=2 requirement.  This
was neither a rank-0-only lookup nor a hot GPU prefix.  The smaller writes are
consistent with persisting the newly computed suffix.

## Safety envelope of this replay

The restarted replay used independent 12-GiB/88-C guards on both nodes.  Head
and worker peak exposed temperatures were 78.8 C and 58.5 C; minimum available
memory was 16,463,376 KiB and 22,068,928 KiB.  Neither guard fired, both peers
remained typed `running`, and the API remained healthy afterward.

The earlier multi-cold-request phase did trip the head thermal guard.  That
does not invalidate the committed-prefix restart proof, but it confirms that
restart persistence does not solve the platform cooling limit.

## Remaining gates

The following are still open and must not be reported as fixed:

1. deliberate live identity mismatch and same-identity recovery;
2. deliberate one-rank payload corruption, checksum rejection, coordinated
   purge and clean recomputation;
3. an exact-path two-node purge command in the deployment recipe;
4. a sustainable long-duration operating envelope for the head Spark;
5. the separate long-reasoning loop seen after high concurrency.

Private benchmark JSON, I/O counter snapshots, guard samples and retained KV
files remain outside git.  No prompt-derived material or private environment
content is committed.

