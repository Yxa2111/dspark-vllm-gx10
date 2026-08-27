# SPDX-License-Identifier: Apache-2.0
"""Production contract tests for DSpark ephemeral KV offloading."""

from unittest.mock import MagicMock

import pytest
import torch
from transformers import OPTConfig

from tests.v1.kv_connector.unit.offloading_connector import utils as runner_utils
from tests.v1.kv_connector.unit.offloading_connector.utils import (
    generate_store_output,
)
from tests.v1.kv_connector.unit.utils import EOS_TOKEN_ID
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
    OffloadingConnectorScheduler,
    RequestOffloadState,
)
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheGroupSpec
from vllm.v1.kv_offload.base import (
    LookupResult,
    OffloadPolicy,
    ReqContext,
    RequestOffloadingContext,
    get_offload_block_hash,
    get_offload_group_idx,
    make_offload_key,
)


@pytest.fixture
def offline_request_runner(request_runner, tmp_path, monkeypatch):
    """Keep the upstream request runner hermetic.

    The pinned helper defaults to ``facebook/opt-125m`` and otherwise asks the
    Hugging Face API for repository metadata. These tests need only model
    geometry, so a local config is sufficient and deterministic.
    """

    model_dir = tmp_path / "opt-config"
    config = OPTConfig(max_position_embeddings=16384)
    config.architectures = ["OPTForCausalLM"]
    config.save_pretrained(model_dir)
    create_vllm_config = runner_utils.create_vllm_config

    def create_offline_vllm_config(*args, **kwargs):
        kwargs["model"] = str(model_dir)
        # CPU-only validation rejects explicitly opting into the hybrid
        # manager even though the runner supplies explicit cache groups later.
        kwargs["disable_hybrid_kv_cache_manager"] = None
        return create_vllm_config(*args, **kwargs)

    monkeypatch.setattr(
        runner_utils, "create_vllm_config", create_offline_vllm_config
    )
    return request_runner


def _make_status(
    scheduler: OffloadingConnectorScheduler,
    key_hashes: list[list[int]],
) -> RequestOffloadState:
    request = MagicMock()
    request.request_id = "production-ephemeral"
    request.num_tokens = 12
    request.kv_transfer_params = None
    status = RequestOffloadState(
        config=scheduler.config,
        req=request,
        req_context=ReqContext(req_id=request.request_id),
        offloading_context=RequestOffloadingContext(policy=OffloadPolicy.BLOCK_LEVEL),
        num_locally_computed_tokens=0,
    )
    for config, state, hashes in zip(
        scheduler.config.kv_group_configs, status.group_states, key_hashes
    ):
        state.offload_keys = [
            make_offload_key(str(value).encode(), config.group_idx)
            for value in hashes
        ]
    return status


def _groups(veto_exempt: bool) -> list[KVCacheGroupSpec]:
    return [
        KVCacheGroupSpec(
            ["stable"],
            FullAttentionSpec(
                block_size=4,
                num_kv_heads=1,
                head_size=1,
                dtype=torch.float32,
            ),
        ),
        KVCacheGroupSpec(
            ["draft"],
            FullAttentionSpec(
                block_size=4,
                num_kv_heads=2,
                head_size=1,
                dtype=torch.float32,
            ),
            is_eagle_group=True,
            eagle_group_is_veto_exempt=veto_exempt,
        ),
    ]


@pytest.mark.parametrize(
    "veto_exempt,expected_tokens,expected_excluded",
    [(True, 12, frozenset({1})), (False, 0, frozenset())],
)
def test_zero_hit_veto_contract(
    offline_request_runner,
    veto_exempt: bool,
    expected_tokens: int,
    expected_excluded: frozenset[int],
):
    runner = offline_request_runner(
        block_size=4,
        num_gpu_blocks=100,
        async_scheduling=False,
        kv_cache_groups=_groups(veto_exempt),
    )
    runner.manager.lookup.side_effect = lambda key, _: (
        LookupResult.HIT
        if int(get_offload_block_hash(key).decode()) in {10, 11, 12}
        else LookupResult.MISS
    )
    status = _make_status(runner.connector_scheduler, [[10, 11, 12], [1, 2, 3]])

    assert runner.connector_scheduler._lookup(status) == expected_tokens
    assert status.lookup_excluded_groups == expected_excluded


@pytest.mark.parametrize("async_scheduling", [True, False])
def test_load_skips_only_ephemeral_group(
    offline_request_runner, async_scheduling: bool
):
    runner = offline_request_runner(
        block_size=4,
        num_gpu_blocks=100,
        async_scheduling=async_scheduling,
        kv_cache_groups=_groups(True),
    )

    runner.new_request(token_ids=[0] * 12)
    runner.manager.prepare_store.side_effect = lambda keys, _: generate_store_output(
        keys
    )
    runner.run(
        decoded_tokens=[EOS_TOKEN_ID],
        expected_stored=((0, 0), (0, 1), (0, 2), (1, 0), (1, 1)),
    )

    runner.scheduler.reset_prefix_cache()
    runner.new_request(token_ids=[0] * 12 + [1])
    runner.manager.lookup.side_effect = lambda key, _: (
        LookupResult.HIT if get_offload_group_idx(key) == 0 else LookupResult.MISS
    )
    runner.manager.prepare_store.side_effect = lambda keys, _: generate_store_output(
        []
    )
    runner.run(
        decoded_tokens=[EOS_TOKEN_ID],
        expected_loaded=((0, 0), (0, 1), (0, 2)),
    )
