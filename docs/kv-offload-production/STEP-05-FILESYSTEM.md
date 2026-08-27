# Step 05 - packed local NVMe backend

Date: 2026-08-27

State: implementation and first live gates complete; production acceptance is
blocked by a whole-head failure during the eviction stress case. The failure
is recorded below without assigning a cause before previous-boot logs can be
collected.

## Outcome

The generic `TieringOffloadingSpec` filesystem tier remains useful for
restart-persistent prefix experiments, but it is not the main parked-Agent
path. It materializes heterogeneous KV-group rows through a primary UMA tier,
and Step 04 proved that a 512 MiB primary tier cannot atomically stage one 8K
prompt. Increasing that tier competes directly with model and active KV on
Spark unified memory.

Patch `0006-packed-local-nvme-offload.patch` therefore adds a disk backend to
`SimpleCPUOffloadConnector`. It preserves the connector's in-process block
pool and scheduling semantics, but each TP rank swaps its packed physical KV
row directly between local GPU/UMA storage and its own node-local NVMe file:

```text
TP0 packed KV <-> two aligned head staging rows <-> head NVMe rank_0 file
TP1 packed KV <-> two aligned worker staging rows <-> worker NVMe rank_1 file
```

There is no cross-node KV relay and no prompt-sized UMA allocation. Both ranks
use the same logical offload block mapping while retaining their different TP
payloads locally. This backend parks sessions only while the vLLM process is
alive: its checksums and logical block map are intentionally not persisted
across restart. `fs-rank0` remains the separate restart-prefix experiment.

## Runtime contract

The new `kv_offload_backend=disk` mode has these fail-closed properties:

- the runtime derives one physical payload from the actual packed cache
  storages and rejects non-integral per-block geometry;
- source stride, destination stride, and copy size are independent, so 4 KiB
  disk padding is never interpreted as KV;
- a 1,065,792-byte DeepSeek V4 packed payload uses a 1,069,056-byte aligned
  disk slot; the 3,264-byte tail is padding only;
- two aligned pinned rows per direction bound staging independently of prompt
  length;
- `O_DIRECT` is the default, so Linux page cache does not create another
  unbounded UMA copy;
- startup creates a new `0600` file using `O_NOFOLLOW | O_EXCL` and
  `posix_fallocate`; an unsafe old path or insufficient disk fails startup;
- every slot has an in-memory CRC32 and exact-length read/write check;
- an I/O-thread exception is retained and raised on the engine thread instead
  of leaving the scheduler waiting forever;
- shutdown joins both coordinators before closing the fd and unlinks the
  prompt-bearing file; if a thread fails to stop, the fd is deliberately not
  closed and reused under it;
- filenames use the vLLM global rank (`.rank_0`, `.rank_1`), not the local CUDA
  device index.

The backend uses a preallocated slot file rather than one file per block. This
bounds inode growth and makes capacity exact. It does not yet reduce the
connector's logical write amplification: each selected offload block still
contains the full unique packed cache row.

## Implementation record

Anemll commit `f7666db` adds:

- `patches/vllm/0006-packed-local-nvme-offload.patch`: source and regression
  tests against pinned vLLM;
- `docker/packed-local-nvme-offload.patch`: runtime-only equivalent for the
  release image, whose installed tree does not contain vLLM tests;
- ordered six-patch application and compile checks in the diagnostic image;
- a CPU-only disk-backend suite and an aligned-capacity scheduler case.

The first image build tried to apply the source-and-test patch to the release
image and failed because `tests/v1/simple_kv_offload/test_scheduler.py` is not
shipped there. The build was corrected by separating the source patch from the
runtime-only patch; no missing hunk was ignored.

## Static and image verification

Baseline vLLM remains
`752a3a504485790a2e8491cacbb35c137339ad34`.

| Check | Result |
|---|---:|
| ordered patch application and Python compilation | 6 patches validated |
| `test_disk_backend.py` in final image | 5 passed |
| direct-I/O aligned scheduler-capacity case | 1 passed |
| runtime `disk_backend.py` SHA-256 on both nodes | `6d31666651d84d26019f8f72b5f8d3679ec762a2b0d914b82c31251fdd234773` |

The exact final tag was `dspark-vllm-gx10:kv-offload-local-nvme`.

| Node | OCI image ID |
|---|---|
| head | `sha256:66b4970b61f18c4ca8fdd375667086b392c8caa4a22c0fdd08d81cd34808b97b` |
| worker | `sha256:6eaf301a7332e5fa51c40d7027ed86fa64c9affa04dbe0f8b431eb9e7ccfaafe` |

Independent builders may produce different OCI IDs; the installed backend
hash above proves the relevant runtime source was identical. The complete
upstream scheduler suite was not accepted as a Step 05 result: the release
image lacks its test tree, and an attempted copied-tree run entered model
configuration network lookup without a local fixture and was terminated. The
new isolated cases and the previously accepted 127-case Step 03 suite are the
current regression evidence.

## Live two-node evidence

Configuration:

- DeepSeek-V4-Flash-0731, TP=2, `nvfp4_ds_mla`, DSpark enabled;
- `MAX_MODEL_LEN=262144`, `MAX_NUM_SEQS=2`, batch budget 8192;
- one preallocated 32 GiB file per node, two transfer rows per direction,
  page cache disabled;
- 32,140 disk slots per rank; actual file allocation 34,359,459,840 bytes;
- GPU KV pool 1,007,048 tokens, maximum reported 262K concurrency 3.84.

An 8K cold request completed with 4.541 s TTFT and computed all 8,192 prompt
tokens. Its immediate repeat completed with 0.363 s TTFT and computed 256
tokens, but this is only an in-GPU prefix-cache check; it is not claimed as an
NVMe restore.

The 16K cold store completed in 9.821 s. One rank's worker process increased
`write_bytes` by exactly 186,015,744 bytes across 174 slot writes:

| Derived value | Result |
|---|---:|
| bytes per prompt token per rank | 11,353.5 B |
| KiB per prompt token per rank | 11.087 KiB |
| combined TP2 write representation | 22.175 KiB/token |
| approximate capacity of 64 GiB per rank | 6.05M token slots |
| approximate capacity of 128 GiB per rank | 12.1M token slots |

Compared with the earlier generic filesystem observation of about 187
KiB/token, the local path is about 8.5 times smaller on the recorded combined
measure. It is still about 3.3 times the roughly 6.7 KiB/token combined active
packed-KV estimate. Capacity planning must use the live 11.087 KiB/token/rank
number until a later layout removes that amplification.

The concurrency cancellation gate launched ten 8K/512-token clients, cancelled
nine after five seconds, and allowed one old request to survive. It finished in
16.321 s with exit code zero, `DONE=true`, and final running/waiting metrics of
zero. Neither rank logged an I/O, checksum, CUDA, traceback, or deadlock error.

## Unresolved head failure

The disk-restore proof then submitted unique 250K prompts sequentially to evict
earlier GPU KV while retaining the in-process disk map:

| Seed | TTFT | Prompt result |
|---:|---:|---|
| 601 | 155.746 s | 250K cold completed |
| 602 | 158.653 s | 250K cold completed |
| 603 | 153.452 s | 250K cold completed |
| 604 | no response | head became unreachable |

During seed 604, `192.168.2.168` disappeared from both the management LAN and
the RoCE peer. This was not merely an HTTP or scheduler loop: SSH and ICMP
failed, and the worker could not reach `192.168.3.1`. The worker still showed
about 102.6 GiB in the TP process, system memory at 111 GiB used with 10 GiB
available, and 2.2 GiB swap. The exact worker container was stopped, returning
the worker to about 117 GiB available memory.

The evidence establishes temporal correlation with sustained 250K pressure,
not root cause. The head remained unreachable after Wake-on-LAN. Production
acceptance is blocked until the machine returns and previous-boot kernel,
OOM/NVRM, thermal, watchdog, PCIe, filesystem, and container logs are
collected. The 604 case must then be repeated with node telemetry and lower
bounded pressure before an actual NVMe-restore claim is made.

## Step 05 decision

The data plane is suitable for continued isolated testing, not production
enablement. `KV_OFFLOAD_MODE=off` must remain the deployment default. No Azusa
Core admission-control mechanism is introduced: the current failure must first
be explained at the runtime/system boundary, and any later concurrency limit
can use the existing vLLM capacity settings unless evidence proves a separate
application scheduler is necessary.
