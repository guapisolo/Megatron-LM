# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""
Megatron Debug Tensor Dumper - Phase 1: Infrastructure and Parallel Adapter

This module provides debugging utilities for Megatron-LM distributed training,
enabling tensor dumping and analysis across multi-dimensional parallelism
(TP/PP/DP/CP/EP).

Phase 1 Components:
- ParallelAdapter: Unified interface for accessing Megatron parallel state
- TensorShardingType: Enum for tensor distribution patterns
- Utility functions for filename parsing and logging

Example:
    >>> from megatron.core.debug_utils import ParallelAdapter, TensorShardingType
    >>>
    >>> # Get parallel state info
    >>> adapter = ParallelAdapter()
    >>> info = adapter.get_parallel_info()
    >>> print(f"TP rank: {info['tp_rank']}/{info['tp_size']}")
    >>>
    >>> # Check if should dump based on DP rank
    >>> if adapter.should_dump_for_dp():
    ...     print("This rank should dump")
"""

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
