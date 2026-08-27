# SPDX-License-Identifier: Apache-2.0
"""Hermetic fixtures for the production offload patch gates."""

import pytest
from transformers import OPTConfig

from tests.v1.kv_connector.unit.offloading_connector import utils as runner_utils
from tests.v1.kv_connector.unit.offloading_connector.utils import request_runner

__all__ = ["request_runner"]


@pytest.fixture(autouse=True)
def offline_model_config(tmp_path, monkeypatch):
    """Replace the helper's Hub model with deterministic local geometry."""
    model_dir = tmp_path / "opt-config"
    config = OPTConfig(max_position_embeddings=16384)
    config.architectures = ["OPTForCausalLM"]
    config.save_pretrained(model_dir)
    create_vllm_config = runner_utils.create_vllm_config

    def create_offline_vllm_config(*args, **kwargs):
        kwargs["model"] = str(model_dir)
        kwargs["disable_hybrid_kv_cache_manager"] = None
        return create_vllm_config(*args, **kwargs)

    monkeypatch.setattr(
        runner_utils, "create_vllm_config", create_offline_vllm_config
    )
