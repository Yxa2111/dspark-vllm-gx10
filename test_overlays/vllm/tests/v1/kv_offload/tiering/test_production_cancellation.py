# SPDX-License-Identifier: Apache-2.0
"""Cancellation convergence tests for the filesystem tier."""

import threading
from collections.abc import Iterator
from unittest.mock import MagicMock

import pytest
import torch

from vllm.v1.kv_offload.base import (
    LookupResult,
    OffloadKey,
    ReqContext,
    ScheduleEndContext,
    make_offload_key,
)
from vllm.v1.kv_offload.tiering.example.manager import (
    ExampleSecondaryTierManager,
)
from vllm.v1.kv_offload.tiering.fs.thread_pool import DualQueueThreadPool
from vllm.v1.kv_offload.tiering.manager import (
    CPUPrimaryTierOffloadingManager,
    TieringOffloadingManager,
)


def _key(value: int) -> OffloadKey:
    return make_offload_key(str(value).encode(), 0)


@pytest.fixture
def tiering_manager() -> Iterator[
    tuple[
        TieringOffloadingManager,
        CPUPrimaryTierOffloadingManager,
        ExampleSecondaryTierManager,
        ExampleSecondaryTierManager,
    ]
]:
    mmap_region = MagicMock()
    view = memoryview(torch.zeros((5, 16), dtype=torch.int8).numpy())
    mmap_region.create_kv_memoryview.return_value = view
    primary = CPUPrimaryTierOffloadingManager(5, mmap_region)
    secondary_1 = ExampleSecondaryTierManager(MagicMock(), view, "example-1")
    secondary_2 = ExampleSecondaryTierManager(MagicMock(), view, "example-2")
    manager = TieringOffloadingManager(primary, [secondary_1, secondary_2])
    try:
        yield manager, primary, secondary_1, secondary_2
    finally:
        manager.shutdown()


def test_cancel_jobs_removes_only_queued_tasks_and_reports_failure():
    """Queued work is cancelled; the active copy and its result are preserved."""
    pool = DualQueueThreadPool(n_read_threads=1, n_write_threads=0)
    active_gate = threading.Event()
    active_started = threading.Event()
    cancelled_task_ran = threading.Event()

    def active_task():
        active_started.set()
        active_gate.wait(timeout=5.0)

    pool.enqueue_load(job_id=1, n_tasks=1, tasks=[active_task])
    assert active_started.wait(timeout=2.0)
    pool.enqueue_load(
        job_id=2,
        n_tasks=2,
        tasks=[cancelled_task_ran.set, cancelled_task_ran.set],
    )
    try:
        assert pool.cancel_jobs({2}) == {2}
        assert pool.get_finished() == [(2, False)]
        assert not cancelled_task_ran.is_set()

        assert pool.cancel_jobs({1}) == set()
        active_gate.set()
        pool.wait_idle()
        assert pool.get_finished() == [(1, True)]
    finally:
        active_gate.set()
        pool.shutdown(wait=True)


def test_finish_releases_deferred_promotion_before_submission(tiering_manager):
    """An aborted lookup cannot leave a primary slot reserved."""
    manager, primary, secondary_1, _secondary_2 = tiering_manager
    block = _key(91)
    ctx = ReqContext(req_id="req-cancel-deferred")
    secondary_1.blocks[block] = True
    secondary_1.submit_load = MagicMock(wraps=secondary_1.submit_load)
    manager.on_new_request(ctx)

    assert manager.lookup(block, ctx) is LookupResult.RETRY
    assert manager._pending_load_submissions
    manager.on_request_finished(ctx)

    assert not manager._pending_load_submissions
    assert primary.lookup(block, ctx) is LookupResult.MISS
    assert ctx.req_id not in manager._req_state
    manager.on_schedule_end(ScheduleEndContext([], ()))
    secondary_1.submit_load.assert_not_called()
    assert not manager.has_pending_work()


def test_finish_requests_cancel_and_accepts_late_completion(tiering_manager):
    """Submitted promotion completion remains valid after request cleanup."""
    manager, primary, secondary_1, secondary_2 = tiering_manager
    block = _key(92)
    ctx = ReqContext(req_id="req-cancel-submitted")
    secondary_1.blocks[block] = True
    secondary_1.cancel_jobs = MagicMock(wraps=secondary_1.cancel_jobs)
    secondary_2.cancel_jobs = MagicMock(wraps=secondary_2.cancel_jobs)
    manager.on_new_request(ctx)

    assert manager.lookup(block, ctx) is LookupResult.RETRY
    manager.on_schedule_end(ScheduleEndContext([], ()))
    promotion_ids = {
        job_id
        for job_id, metadata in manager._transfer_jobs.items()
        if metadata.is_promotion
    }
    assert promotion_ids

    manager.on_request_finished(ctx)
    secondary_1.cancel_jobs.assert_called_once_with(promotion_ids)
    secondary_2.cancel_jobs.assert_called_once_with(promotion_ids)
    assert ctx.req_id not in manager._req_state

    manager.on_schedule_end(ScheduleEndContext([], ()))
    assert not manager._transfer_jobs
    assert not manager.has_pending_work()
    assert primary.lookup(block, ctx) is LookupResult.HIT
