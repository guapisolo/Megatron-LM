# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""
Megatron Debug Tensor Dumper

This module provides debugging utilities for Megatron-LM distributed training,
enabling tensor dumping and analysis across multi-dimensional parallelism
(TP/PP/DP/CP/EP).

Core Components:
- dumper: Global singleton for tensor dumping
- FilterEngine: Multi-dimensional filter for layer/name/iteration
- StorageBackend: File I/O with async support
- ParallelAdapter: Unified interface for Megatron parallel state
- TensorShardingType: Enum for tensor distribution patterns

Example:
    >>> from megatron.core.debug_utils import dumper
    >>>
    >>> # Enable and configure
    >>> dumper.enable = True
    >>> dumper.on_training_start()
    >>>
    >>> # In training loop
    >>> dumper.on_iteration_start(iteration)
    >>> dumper.dump("hidden_states", tensor, layer_id=0)
    >>>
    >>> # With context
    >>> with dumper.context(phase="forward"):
    ...     dumper.dump("attention_output", attn_out)

Environment Variables:
    MEGATRON_DUMPER_ENABLE: "1" to enable (default: "0")
    MEGATRON_DUMPER_DIR: Dump directory (default: "/tmp/megatron_dumps")
    MEGATRON_DUMPER_WRITE_FILE: "1" to write files (default: "1")
    MEGATRON_DUMPER_DP_RANK_0_ONLY: "1" for DP rank 0 only (default: "1")
    MEGATRON_DUMPER_ASYNC: "1" for async writes (default: "0")
    MEGATRON_DUMPER_AGGREGATE_TP: "1" to aggregate tensors across TP (default: "0")
    MEGATRON_DUMPER_GRADIENTS: "1" to enable gradient dumping (default: "0")
    MEGATRON_DUMPER_LAYERS: Layer filter (e.g., "0,1,last")
    MEGATRON_DUMPER_NAMES: Name filter regex (e.g., "attention|mlp")
    MEGATRON_DUMPER_ITERATIONS: Iteration filter (e.g., "0,every:100")

Phase 3 Features:
    - Hook Registration: Auto-register hooks on TransformerLayer modules
    - TP Aggregation: Gather sharded tensors across TP ranks
    - Gradient Dump: Capture parameter gradients after backward pass
"""

# Core dumper
from .dumper import dumper

# Filter engine
from .filter_engine import FilterEngine

# Storage backend
from .storage_backend import StorageBackend

# Metadata exports
from .metadata import (
    TENSOR_SHARDING_MAP,
    TensorShardingType,
    get_sharding_type,
)

# Parallel adapter exports
from .parallel_adapter import (
    ParallelAdapter,
    get_parallel_adapter,
)

# Utility exports
from .utils import (
    build_dump_filename,
    configure_logging,
    format_tensor_info,
    get_global_rank,
    get_logger,
    get_tensor_stats,
    get_world_size,
    is_distributed_initialized,
    parse_dump_filename,
    safe_mkdir,
)

__all__ = [
    # Core dumper
    "dumper",
    # Filter engine
    "FilterEngine",
    # Storage backend
    "StorageBackend",
    # Metadata
    "TensorShardingType",
    "TENSOR_SHARDING_MAP",
    "get_sharding_type",
    # Parallel adapter
    "ParallelAdapter",
    "get_parallel_adapter",
    # Utils
    "get_logger",
    "configure_logging",
    "parse_dump_filename",
    "build_dump_filename",
    "get_tensor_stats",
    "format_tensor_info",
    "safe_mkdir",
    "is_distributed_initialized",
    "get_global_rank",
    "get_world_size",
]
