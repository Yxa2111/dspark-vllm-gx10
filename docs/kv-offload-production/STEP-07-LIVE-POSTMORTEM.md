# Step 07 - live pressure-run postmortem

Date: 2026-08-27

State: complete as a failure analysis. The run is not a local-NVMe restore
acceptance result. The next image must be exercised behind a measured host UMA
reserve and an external node-safety guard.

## Failure boundary

The Step 05 TP=2 run used the six-patch local packed-NVMe image on two DGX
Sparks. Each rank had a 34,359,459,840-byte `O_DIRECT` slot file and a roughly
1.01M-token GPU prefix pool. Three unique approximately 250K-token cold prompts
completed in 153-159 seconds. The fourth began but the head became unreachable
before it completed; the node later required a physical power-on.

This ordering matters. The workload had not yet repeated an evicted seed. The
run produced rank-local disk writes, but it never demonstrated:

- a disk lookup hit;
- a disk-to-pinned-to-GPU load;
- a restored-prefix TTFT;
- restored target-only output equality.

The last exported vLLM metrics still reported zero external prefix-cache hit
rate. That is expected before the first repeated evicted prefix and is not
evidence that the load path missed or failed.

## Evidence against a backend-local attribution

The recovered previous boot and old container log were captured before any
experiment restart or cleanup.

- The container was not marked `OOMKilled` and emitted no packed-backend
  checksum, short-I/O, queue-timeout, or background-thread exception.
- API health and metrics continued to answer while the fourth prefill was
  running; there was no orderly engine termination.
- The entire head disappeared from both management and RoCE networks. `last -x`
  classified the boot as a crash, there was no normal shutdown sequence, and
  the next boot replaced an unclean/corrupt journald file.
- No Xid, pstore panic, NVMe reset, or ext4 error was preserved at the failure
  boundary.

The host did preserve direct resource-pressure evidence:

- `sar` at 03:40:27 recorded only 5,495,932 KiB `MemAvailable`, 879,728 KiB
  free, 91.02% memory used, and active swap-in/swap-out;
- the NVIDIA kernel driver reported two `_memdescAllocInternal`
  `NV_ERR_NO_MEMORY` failures during the service startup;
- an earlier, separate episode in the same boot had invoked the global OOM
  killer repeatedly and killed a previous vLLM worker and host daemons.

The most likely failure class is therefore the GB10 unified-memory cliff or
driver allocation starvation under low host reserve, not a demonstrated slot
format, checksum, or queue-lifecycle bug. Confidence is medium-high rather than
absolute because no final OOM record survived.

Temperature was not independently sampled through the event. DGX Spark
thermal/power protection can present as the same log-free hard shutdown, so it
remains a second hypothesis until a field diagnostic and guarded live thermal
trace pass.

## Runtime implications

The disk tier is a cache below the GPU KV pool. Writing a prompt to NVMe does
not reduce the model, CUDA workspace, active request, or still-resident GPU KV
footprint. The disk backend therefore cannot by itself protect the shared
128-GB UMA host while a new long prefill pushes resident allocations toward
the cliff.

The seven-patch bounded-work image remains the correct next candidate because
it caps queued rows and propagates I/O failures. The Step 05 failure does not
invalidate those offline gates. It does add four live requirements outside the
slot implementation:

1. start below 0.80 GPU memory utilization and accept a value only after both
   nodes settle with at least 12 GiB `MemAvailable`;
2. set the disposable vLLM container's OOM preference above host daemons;
3. run fsync-backed per-node memory, swap, cgroup-I/O, NVRM, GPU, and ACPI
   thermal telemetry before model start;
4. stop only the exact experiment project after a numeric reserve or thermal
   threshold persists.

The fourth item is a node circuit breaker, not Azusa request admission control.
The vLLM scheduler continues to own request/KV scheduling; the guard only keeps
a disposable canary from taking down the shared-memory operating system.

## Cleanup

After private evidence capture, the stopped head rank's exact ephemeral slot
file was verified by pathname and size and removed through a root container
bind-mounted only to its configured directory. The stopped worker rank had
already been cleaned with the same exact-path procedure. Old container logs
were retained. No wildcard or durable prefix store was removed.

## Next acceptance order

1. Build patch 0007 on the recovered head and compare installed source hashes
   with the worker image.
2. Add and test the deployment-side numeric reserve guard.
3. Start a new isolated profile at a lower utilization and record the actual
   settled GPU KV token capacity and host reserve.
4. Exceed the GPU pool with the smallest safe contexts, then repeat the first
   seed and require disk reads, a connector load/hit, lower TTFT, and
   deterministic target-only equality.
5. Run short/100K/250K cancellation convergence, controlled live I/O faults,
   field diagnostics, a 24-hour soak, and exact rollback.

Until those gates pass, the release default remains off.
