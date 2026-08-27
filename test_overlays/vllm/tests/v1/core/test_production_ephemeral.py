# SPDX-License-Identifier: Apache-2.0
"""Core coordinator parity tests for DSpark ephemeral KV groups."""

from collections.abc import Callable
from math import lcm
from types import SimpleNamespace

import pytest
import torch

from vllm.config.speculative import SpeculativeConfig
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_manager import KVCacheManager, Request
from vllm.v1.core.kv_cache_utils import (
    get_request_block_hasher,
    init_none_hash,
    make_block_hash_with_group_id,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    SlidingWindowSpec,
)

@pytest.fixture(autouse=True)
def _init_hash() -> None:
    init_none_hash(sha256)


def _request(request_id: str, token_ids: list[int], block_size: int) -> Request:
    sampling = SamplingParams(max_tokens=1)
    sampling.update_from_generation_config({}, eos_token_id=100)
    return Request(
        request_id=request_id,
        prompt_token_ids=token_ids,
        mm_features=None,
        sampling_params=sampling,
        pooling_params=None,
        lora_request=None,
        cache_salt=None,
        block_hasher=get_request_block_hasher(block_size, sha256),
    )


def _manager(config: KVCacheConfig, hash_fn: Callable = sha256) -> KVCacheManager:
    del hash_fn
    return KVCacheManager(
        config,
        max_model_len=8192,
        enable_caching=True,
        hash_block_size=16,
        scheduler_block_size=lcm(
            *(group.kv_cache_spec.block_size for group in config.kv_cache_groups)
        ),
        use_eagle=True,
    )


@pytest.mark.parametrize(
    "method,expected",
    [("dspark", True), ("eagle3", False), ("dflash", False)],
)
def test_ephemeral_method_contract(method: str, expected: bool):
    config = SimpleNamespace(method=method)
    assert SpeculativeConfig.has_ephemeral_draft_context(config) is expected


@pytest.mark.parametrize("veto_exempt,expected_tokens", [(True, 48), (False, 0)])
def test_core_zero_hit_veto_contract(veto_exempt: bool, expected_tokens: int):
    block_size = 16
    config = KVCacheConfig(
        num_blocks=31,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["stable"],
                FullAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            ),
            KVCacheGroupSpec(
                ["draft"],
                SlidingWindowSpec(
                    block_size=block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                    sliding_window=2 * block_size,
                ),
                is_eagle_group=True,
                eagle_group_is_veto_exempt=veto_exempt,
            ),
        ],
    )
    manager = _manager(config)
    token_ids = [i for i in range(4) for _ in range(block_size)]

    first = _request("first", token_ids, block_size)
    blocks, computed = manager.get_computed_blocks(first)
    manager.allocate_slots(first, len(token_ids), computed, blocks)
    block_hashes = first.block_hashes
    manager.free(first)

    repeated = _request("repeated", token_ids, block_size)
    draft_hashes = [make_block_hash_with_group_id(h, 1) for h in block_hashes]
    cache = manager.block_pool.cached_block_hash_to_block._cache
    removed = {key: cache.pop(key) for key in draft_hashes if key in cache}
    try:
        computed_blocks, computed_tokens = manager.get_computed_blocks(repeated)
    finally:
        cache.update(removed)

    assert computed_tokens == expected_tokens
    if veto_exempt:
        assert len(computed_blocks.blocks[0]) == 3
        assert computed_blocks.blocks[1] == []
    else:
        assert computed_blocks.blocks == ((), ())
    manager.free(repeated)
