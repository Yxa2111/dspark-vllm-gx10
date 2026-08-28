# Production NVMe KV offload work log

This directory is the commit-by-commit engineering record for the
`experiment/packed-kv-offload-tp2` branch. A step is not complete until its
document, implementation, tests, and commit all agree.

| Step | Document | Scope | State |
|---:|---|---|---|
| 01 | [`STEP-01-AUDIT.md`](STEP-01-AUDIT.md) | pinned-runtime audit and correctness contract | complete |
| 02 | [`STEP-02-EPHEMERAL-GROUP.md`](STEP-02-EPHEMERAL-GROUP.md) | core + connector DSpark ephemeral semantics | complete |
| 03 | [`STEP-03-CANCELLATION.md`](STEP-03-CANCELLATION.md) | request cancellation and deferred-job cleanup | complete |
| 04 | [`STEP-04-BOUNDED-TRANSFER.md`](STEP-04-BOUNDED-TRANSFER.md) | rank-zero TP staging, bounded relay, and packed-layout safety | complete |
| 05 | [`STEP-05-FILESYSTEM.md`](STEP-05-FILESYSTEM.md) | packed local NVMe capacity, integrity, and write layout | live store proved; restore rejected by host failure |
| 06 | [`STEP-06-PRODUCTION-GATES.md`](STEP-06-PRODUCTION-GATES.md) | bounded work and deterministic offline faults | complete |
| 07 | [`STEP-07-LIVE-POSTMORTEM.md`](STEP-07-LIVE-POSTMORTEM.md) | recovered-head UMA pressure diagnosis and revised live gates | complete |
| 08 | [`STEP-08-RESTART-PERSISTENCE-AUDIT.md`](STEP-08-RESTART-PERSISTENCE-AUDIT.md) | crash-safe rank-local restart-persistence contract | audit complete; implemented in Step 09 |
| 09 | [`STEP-09-PERSISTENT-INDEX-IMPLEMENTATION.md`](STEP-09-PERSISTENT-INDEX-IMPLEMENTATION.md) | durable rank manifests and scheduler commit index | offline implementation complete; live restart passed in Step 10 |
| 10 | [`STEP-10-TP2-RESTART-PROOF.md`](STEP-10-TP2-RESTART-PROOF.md) | interrupted-store authority and full TP=2 restart replay | functional restart gate passed; fault recovery and production stability remain |
| 11 | [`STEP-11-PERSISTENT-LRU-RESTORE.md`](STEP-11-PERSISTENT-LRU-RESTORE.md) | restore BlockPool eviction order and retain prefixes while empty slots remain | fixed; offline and TP=2 restart gates passed |
| 12 | [`STEP-12-PROMETHEUS-METRICS.md`](STEP-12-PROMETHEUS-METRICS.md) | bounded scheduler/rank NVMe capacity, transfer, pressure, eviction, and error metrics | implemented; live TP=2 store/load metrics passed |

The source baseline is fixed by `upstream.lock`. Production changes remain an
ordered patch series until they are accepted upstream; thin-image patching is
used for Python-only validation, while any native-layout change requires a full
image build.
