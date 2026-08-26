# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in, metadata-only diagnostics for DSpark KV offload experiments."""

import json
import os
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)

_ENABLED = os.environ.get("DSPARK_KV_OFFLOAD_DIAG", "0") == "1"


def emit_kv_offload_diag(event: str, **fields: Any) -> None:
    """Emit one structured record without reading KV or prompt contents."""
    if not _ENABLED:
        return

    payload = {
        "event": event,
        "node_rank": os.environ.get("NODE_RANK", ""),
        "pid": os.getpid(),
        **fields,
    }
    logger.info(
        "DSPARK_KV_OFFLOAD_DIAG %s",
        json.dumps(payload, default=str, separators=(",", ":"), sort_keys=True),
    )
