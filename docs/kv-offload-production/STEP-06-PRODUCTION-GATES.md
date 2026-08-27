# Step 06 - bounded work and production gates

Date: 2026-08-27

State: bounded-work implementation complete; live two-node acceptance remains
blocked until the head node returns and the Step 05 failure is diagnosed.

## Gap found after Step 05

The local NVMe backend bounded its staging tensors but inherited two unbounded
work paths from upstream `SimpleCPUOffloadConnector`:

- store and load coordinators used `queue.SimpleQueue`, so the engine could
  enqueue transfer events faster than NVMe completed them;
- one eager store event could include every newly eligible block, retaining all
  corresponding GPU and offload block references until the event completed.

Current vLLM `origin/main` at `5acc1c4e4` still has the same `SimpleQueue` and
uncapped store event. There was no upstream backpressure patch to reuse. This
finding is a production invariant gap; it is not asserted as the cause of the
Step 05 whole-head failure before previous-boot logs are available.

## Bounded-work contract

Patch `0007-bound-local-nvme-work.patch`, committed as Anemll `0260ffa`, adds
three disk-only limits:

| Setting | Default | Runtime range |
|---|---:|---:|
| `disk_queue_depth` | 2 events/direction | 1-64 |
| `disk_enqueue_timeout_s` | 30 s | greater than 0, at most 300 |
| `disk_max_store_blocks` | 64 rows/event | 1-4096 |

The DeepSeek V4 packed row is about 1 MiB/rank. With the defaults, one running
store, two queued stores, and one currently blocked enqueue retain at most 256
store rows, about 260 MiB/rank, before the 30-second timeout fails the engine.
Once enqueue returns, the steady active-plus-queued bound is 192 rows. This is
independent of prompt length.

The scheduler stops an eager or lazy store scan at the configured row count,
advances cursors only for blocks actually classified in that event, and resumes
the remaining eligible blocks on a later scheduler step. A deterministic
four-block test stores two blocks in the first event and the remaining two in
the next event.

Load events are not split: a matched prefix must be available before its
request can execute, and its destination GPU blocks already belong to that
request. The load queue is nevertheless bounded to the same event depth and
timeout.

If either queue remains full for the timeout, `launch_copy` raises on the
engine thread. It cannot become an indefinitely pending transfer. Shutdown
discards queued process-lifetime cache work, installs one stop sentinel without
blocking behind a full queue, rejects any later launch, joins the active I/O
threads, and then follows the existing exact unlink/fd lifecycle.

Disk mode also now validates in the runtime, rather than relying only on the
MiaAI launcher:

- absolute non-empty path and positive capacity;
- 1-8 staging rows;
- real JSON booleans for page cache and preallocation;
- queue, timeout, and store-row ranges above.

## Verification

The ordered source patch stack applies and Python-compiles against pinned vLLM
`752a3a504485790a2e8491cacbb35c137339ad34`:

```text
Validated 7 patch(es)
```

The worker built the exact seven-patch runtime image
`dspark-vllm-gx10:kv-offload-bounded-work`:

```text
image:        sha256:cbdc3177a477668690be0278339f435cf17668451f85ccdb5a52aaaca0b5c06d
disk_backend: bc9980d211e47eddb5146a4cf5ca7aaf822c04df151dacc59850dfb541bb32e4
manager:      3dbfb24ab41b2f24faff30e388843af25bd5a46075e6430b19db97cc0e51ac1c
```

Tests were run inside that final image with an offline local OPT config. The
initial focused gate was:

```text
7 disk-backend geometry/integrity/failure/queue/shutdown cases
1 scheduler bounded-and-resumable store case
8 passed, 15 warnings in 8.61s
```

The warnings were 14 PyTorch deprecations plus pytest's inability to write a
cache under the intentionally read-only mounted test directory.

The complete pinned `test_scheduler.py` was then run rather than relying on
the one new scheduler case:

```text
31 passed, 15 warnings in 8.92s
```

Patch `0008-local-nvme-fault-tests.patch`, committed as Anemll `7fd268a`, adds
a deterministic startup ENOSPC gate. It replaces `posix_fallocate` with an
`ENOSPC` failure after the exact slot file has been opened and truncated, then
proves that initialization closes the fd, unlinks the partial file, and starts
neither I/O thread. The final image source plus the mounted test produced:

```text
8 disk-backend tests passed in 3.43s
```

This complements the existing checksum-corruption, short-I/O, background
failure propagation, bounded-queue, and shutdown cases. The source/test patch
stack now contains eight ordered patches; patch 0008 is test-only and does not
change the release-image runtime bytes.

## Production status

The default remains disabled. This step removes an unbounded resource path but
does not replace the missing live gates:

1. collect and explain the head's previous-boot failure evidence;
2. build the same exact image independently on the recovered head and compare
   installed source hashes;
3. prove a real disk load under a lower-memory isolated profile;
4. rerun cancel/survivor cases at short and long contexts;
5. pass live TP2 fault injection, a 24-hour mixed soak, and a rollback drill.

No Azusa Core admission control is added. Backpressure and resource bounds are
owned by the vLLM/Anemll data plane where the work is created.
