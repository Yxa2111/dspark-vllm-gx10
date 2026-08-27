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
| 05 | [`STEP-05-FILESYSTEM.md`](STEP-05-FILESYSTEM.md) | packed local NVMe capacity, integrity, and write layout | blocked on head-failure diagnosis |
| 06 | [`STEP-06-PRODUCTION-GATES.md`](STEP-06-PRODUCTION-GATES.md) | bounded work, deployment lifecycle, soak, and two-node acceptance | blocked on head recovery |

The source baseline is fixed by `upstream.lock`. Production changes remain an
ordered patch series until they are accepted upstream; thin-image patching is
used for Python-only validation, while any native-layout change requires a full
image build.
