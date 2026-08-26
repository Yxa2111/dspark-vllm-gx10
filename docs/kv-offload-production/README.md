# Production NVMe KV offload work log

This directory is the commit-by-commit engineering record for the
`experiment/packed-kv-offload-tp2` branch. A step is not complete until its
document, implementation, tests, and commit all agree.

| Step | Document | Scope | State |
|---:|---|---|---|
| 01 | [`STEP-01-AUDIT.md`](STEP-01-AUDIT.md) | pinned-runtime audit and correctness contract | complete |
| 02 | `STEP-02-EPHEMERAL-GROUP.md` | core + connector DSpark ephemeral semantics | pending |
| 03 | `STEP-03-CANCELLATION.md` | request cancellation and deferred-job cleanup | pending |
| 04 | `STEP-04-BOUNDED-TRANSFER.md` | bounded streaming and packed-layout safety | pending |
| 05 | `STEP-05-FILESYSTEM.md` | capacity, eviction, integrity, and write layout | pending |
| 06 | `STEP-06-PRODUCTION-GATES.md` | deployment contract and two-node acceptance | pending |

The source baseline is fixed by `upstream.lock`. Production changes remain an
ordered patch series until they are accepted upstream; thin-image patching is
used for Python-only validation, while any native-layout change requires a full
image build.

