# SPDX-License-Identifier: Apache-2.0
"""Fail-closed configuration tests for multi-node tiering."""

import pytest
from tests.v1.kv_offload.test_factory import _make_kv_cache_config
from transformers import OPTConfig
from vllm.config import (
    CacheConfig,
    DeviceConfig,
    KVTransferConfig,
    ModelConfig,
    SchedulerConfig,
    VllmConfig,
)
from vllm.v1.kv_offload.tiering.spec import TieringOffloadingSpec


@pytest.fixture
def make_config(tmp_path):
    """Build a hermetic tiering config without Hugging Face access."""
    model_dir = tmp_path / "opt-config"
    config = OPTConfig(max_position_embeddings=16384)
    config.architectures = ["OPTForCausalLM"]
    config.save_pretrained(model_dir)

    def factory(extra_config: dict) -> VllmConfig:
        model_config = ModelConfig(
            model=str(model_dir),
            trust_remote_code=True,
            dtype="float16",
            seed=42,
        )
        scheduler_config = SchedulerConfig(
            max_num_seqs=16,
            max_num_batched_tokens=64,
            max_model_len=10000,
            enable_chunked_prefill=True,
            is_encoder_decoder=model_config.is_encoder_decoder,
        )
        cache_config = CacheConfig(
            block_size=16,
            gpu_memory_utilization=0.9,
            cache_dtype="auto",
            enable_prefix_caching=True,
        )
        kv_transfer_config = KVTransferConfig(
            kv_connector="OffloadingConnector",
            kv_role="kv_both",
            kv_connector_extra_config={
                "spec_name": "TieringOffloadingSpec",
                "cpu_bytes_to_use": 65536,
                **extra_config,
            },
        )
        return VllmConfig(
            scheduler_config=scheduler_config,
            model_config=model_config,
            cache_config=cache_config,
            kv_transfer_config=kv_transfer_config,
            device_config=DeviceConfig("cpu"),
        )

    return factory


def _set_two_node_tp(config) -> None:
    parallel = config.parallel_config
    parallel.nnodes = 2
    parallel.tensor_parallel_size = 2
    parallel.pipeline_parallel_size = 1
    parallel.prefill_context_parallel_size = 1
    parallel.decode_context_parallel_size = 1
    parallel.world_size = 2
    parallel.distributed_executor_backend = "mp"


def test_multi_node_host_local_mmap_fails_closed(make_config):
    config = make_config({})
    _set_two_node_tp(config)
    with pytest.raises(ValueError, match="host-local shared mmap is unsafe"):
        TieringOffloadingSpec(config, _make_kv_cache_config())


def test_rank_zero_staging_accepts_tp_only_mp_world(make_config):
    config = make_config(
        {
            "distributed_staging": "rank0",
            "max_transfer_chunk_bytes": 4096,
        }
    )
    _set_two_node_tp(config)
    spec = TieringOffloadingSpec(config, _make_kv_cache_config())
    assert spec.distributed_staging == "rank0"
    assert spec.max_transfer_chunk_bytes == 4096
