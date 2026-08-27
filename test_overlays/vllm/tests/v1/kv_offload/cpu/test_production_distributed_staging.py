# SPDX-License-Identifier: Apache-2.0
"""Production contract tests for bounded cross-node KV staging."""

import contextlib
import mmap
import os
import uuid
from collections import deque

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from vllm.utils.network_utils import get_open_port
from vllm.v1.kv_offload.base import TransferResult
from vllm.v1.kv_offload.cpu.distributed_staging import RankZeroStagingRelay
from vllm.v1.kv_offload.cpu.gpu_worker import CPUOffloadingWorker, DistributedStore
from vllm.v1.kv_offload.cpu.shared_offload_region import SharedOffloadRegion


def _run_relay(rank: int, world_size: int, init_method: str) -> None:
    dist.init_process_group(
        backend="gloo",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
    )
    try:
        local = torch.zeros((4, 32), dtype=torch.int8)
        remote = torch.zeros_like(local) if rank == 0 else None
        relay = RankZeroStagingRelay(
            local_tensors=[local],
            remote_tensors={1: [remote]} if remote is not None else {},
            rank=rank,
            ranks=[0, 1],
            cpu_group=dist.group.WORLD,
            max_chunk_bytes=40,
        )
        block_ids = np.array([1, 3], dtype=np.int64)

        if rank == 1:
            local[1].fill_(7)
            local[3].fill_(8)
        assert not relay.all_ranks_ready(rank == 0)
        assert relay.all_ranks_ready(True)
        assert relay.stage_store(17, block_ids) == 64
        if rank == 0:
            assert remote is not None
            assert torch.all(remote[1] == 7)
            assert torch.all(remote[3] == 8)
            remote[1].fill_(11)
            remote[3].fill_(12)

        assert relay.stage_load(18, block_ids) == 64
        if rank == 1:
            assert torch.all(local[1] == 11)
            assert torch.all(local[3] == 12)
            assert torch.count_nonzero(local[0]) == 0
            assert torch.count_nonzero(local[2]) == 0
        assert relay.max_observed_payload_bytes <= 40
        relay.shutdown()
    finally:
        dist.destroy_process_group()


def test_rank_zero_staging_round_trip_is_bounded():
    init_method = f"tcp://127.0.0.1:{get_open_port()}"
    mp.spawn(_run_relay, args=(2, init_method), nprocs=2)


def test_rank_zero_can_map_remote_worker_slot():
    page_size = mmap.PAGESIZE
    instance_id = f"production-relay-{uuid.uuid4()}"
    region = SharedOffloadRegion(
        instance_id=instance_id,
        num_blocks=2,
        rank=0,
        kv_bytes_per_block=4 * page_size,
        cpu_page_size=2 * page_size,
    )
    try:
        local = region.create_next_view(page_size)
        remote = region.create_rank_view(1, page_size)
        assert local.data_ptr() == region._base.data_ptr()
        assert remote.data_ptr() == region._base.data_ptr() + 2 * page_size
        assert local.stride(0) == remote.stride(0) == 4 * page_size
        del local, remote
    finally:
        region.cleanup()
        with contextlib.suppress(FileNotFoundError):
            os.unlink(region.mmap_path)


def test_worker_slots_allow_page_alignment_tail_padding():
    page_size = mmap.PAGESIZE
    worker_page = 2 * page_size + 512
    instance_id = f"production-relay-padding-{uuid.uuid4()}"
    region = SharedOffloadRegion(
        instance_id=instance_id,
        num_blocks=2,
        rank=0,
        kv_bytes_per_block=5 * page_size,
        cpu_page_size=worker_page,
        num_worker_slots=2,
    )
    try:
        local = region.create_next_view(worker_page)
        remote = region.create_rank_view(1, worker_page)
        assert remote.data_ptr() == region._base.data_ptr() + worker_page
        assert region._tail_padding == page_size - 1024
        assert remote.stride(0) == 5 * page_size
        del local, remote
    finally:
        region.cleanup()
        with contextlib.suppress(FileNotFoundError):
            os.unlink(region.mmap_path)


class _FinishedHandler:
    def __init__(self, results: list[TransferResult]) -> None:
        self.results = results

    def get_finished(self) -> list[TransferResult]:
        results, self.results = self.results, []
        return results


class _RelayProbe:
    def __init__(self) -> None:
        self.ready_answers = deque([False, True])
        self.stored: list[tuple[int, np.ndarray]] = []

    def all_ranks_ready(self, ready: bool) -> bool:
        assert ready
        return self.ready_answers.popleft()

    def stage_store(self, job_id: int, block_ids: np.ndarray) -> int:
        self.stored.append((job_id, block_ids.copy()))
        return block_ids.size


def test_worker_holds_completion_until_every_rank_is_staged():
    """Scheduler completion must not race ahead of the TP1 relay."""
    worker = object.__new__(CPUOffloadingWorker)
    result = TransferResult(job_id=23, success=True)
    worker._store_handler = _FinishedHandler([result])
    worker._load_handler = _FinishedHandler([])
    worker._staging_relay = _RelayProbe()
    block_ids = np.array([2, 5], dtype=np.int64)
    worker._distributed_stores = deque([DistributedStore(23, block_ids)])
    worker._ready_store_results = {}

    assert worker.get_finished() == []
    assert 23 in worker._ready_store_results
    assert worker._staging_relay.stored == []

    assert worker.get_finished() == [result]
    assert worker._staging_relay.stored[0][0] == 23
    np.testing.assert_array_equal(worker._staging_relay.stored[0][1], block_ids)
    assert not worker._distributed_stores
    assert not worker._ready_store_results
