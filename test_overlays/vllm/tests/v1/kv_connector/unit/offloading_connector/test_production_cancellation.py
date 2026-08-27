# SPDX-License-Identifier: Apache-2.0
"""Production cancellation tests for OffloadingConnector."""

from types import SimpleNamespace

import pytest

from tests.v1.kv_connector.unit.offloading_connector.utils import (
    generate_store_output,
)
from vllm.v1.request import RequestStatus


@pytest.mark.parametrize("async_scheduling", [True, False])
def test_abort_queued_request_does_not_build_store_job(
    request_runner, async_scheduling: bool
):
    """A never-scheduled abort owns hashes but no GPU blocks to store."""
    block_size = 4
    runner = request_runner(
        block_size=block_size,
        num_gpu_blocks=8,
        async_scheduling=async_scheduling,
    )

    runner.new_request(token_ids=[0] * (block_size * 4))
    runner.scheduler.schedule()

    runner.new_request(token_ids=[1] * (block_size * 4))
    queued_req_id = str(runner.req_id)
    assert any(
        request.request_id == queued_req_id for request in runner.scheduler.waiting
    )

    runner.scheduler.finish_requests(queued_req_id, RequestStatus.FINISHED_ABORTED)
    scheduler_output = runner.scheduler.schedule()

    assert all(
        job.req_id != queued_req_id
        for job in scheduler_output.kv_connector_metadata.store_jobs.values()
    )
    assert queued_req_id not in runner.connector_scheduler._req_status


@pytest.mark.parametrize("async_scheduling", [True, False])
def test_abort_mid_prefill_stores_only_recorded_computed_blocks(
    request_runner, async_scheduling: bool
):
    """A stale scheduled count cannot extend an aborted request's store."""
    block_size = 4
    runner = request_runner(
        block_size=block_size,
        num_gpu_blocks=32,
        async_scheduling=async_scheduling,
    )
    runner.new_request(token_ids=[0] * (block_size * 4))
    runner.scheduler.schedule()

    req_id = str(runner.req_id)
    req_status = runner.connector_scheduler._req_status[req_id]
    # The first scheduler call exists only to allocate realistic block IDs.
    runner.connector_scheduler._jobs.clear()
    req_status.transfer_jobs.clear()
    for group_state in req_status.group_states:
        group_state.next_stored_block_idx = 0
    req_status.req.num_computed_tokens = block_size
    req_status.req.status = RequestStatus.FINISHED_ABORTED
    req_status.update_offload_keys()
    runner.manager.prepare_store.side_effect = lambda keys, _: generate_store_output(
        keys
    )

    jobs = runner.connector_scheduler._build_store_jobs(
        SimpleNamespace(num_scheduled_tokens={req_id: block_size * 3})
    )

    assert len(jobs) == 1
    (job,) = jobs.values()
    assert len(job.src_spec.block_ids) == 1
