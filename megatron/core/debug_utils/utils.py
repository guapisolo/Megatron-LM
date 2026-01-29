# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""
Utility functions for Megatron Debug Tensor Dumper.

This module provides common utility functions including filename parsing,
logging configuration, and helper functions for tensor operations.
"""

import logging
import os
from typing import Any, Dict, Optional

import torch


# Module-level logger
_logger: Optional[logging.Logger] = None

_INT_DTYPES = (
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.uint8,
    torch.bool,
)

_NUMERIC_FILENAME_FIELDS = frozenset(
    {
        "iter",
        "mb",
        "grank",
        "tprank",
        "pprank",
        "idx",
        "layer",
    }
)


def get_logger() -> logging.Logger:
    """
    Get the module logger, creating it if necessary.

    Returns:
        The configured logger instance.
    """
    global _logger
    if _logger is None:
        _logger = logging.getLogger("megatron.debug_utils")
        _logger.setLevel(logging.INFO)

        # Add handler if not already present
        if not _logger.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(logging.INFO)
            formatter = logging.Formatter(
                "[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            handler.setFormatter(formatter)
            _logger.addHandler(handler)

    return _logger


def configure_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    rank: Optional[int] = None,
) -> logging.Logger:
    """
    Configure logging for the debug utils module.

    Args:
        level: Logging level (e.g., logging.INFO, logging.DEBUG)
        log_file: Optional file path to write logs
        rank: Optional rank for multi-process logging prefix

    Returns:
        The configured logger instance.
    """
    global _logger
    _logger = logging.getLogger("megatron.debug_utils")
    _logger.setLevel(level)

    # Clear existing handlers
    _logger.handlers.clear()

    # Create formatter with optional rank prefix
    if rank is not None:
        fmt = f"[%(asctime)s][Rank {rank}][%(name)s][%(levelname)s] %(message)s"
    else:
        fmt = "[%(asctime)s][%(name)s][%(levelname)s] %(message)s"

    formatter = logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S")

    # Add stream handler
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)
    _logger.addHandler(stream_handler)

    # Add file handler if specified
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        _logger.addHandler(file_handler)

    return _logger


def parse_dump_filename(filename: str) -> Dict[str, Any]:
    """
    Parse a dump filename into a metadata dictionary.

    The expected filename format is:
        name={name}___iter={iteration}___mb={micro_batch_id}___grank={global_rank}___
        tprank={tp_rank}___pprank={pp_rank}___idx={dump_index}[___layer={layer_id}]
        [___extra=value].pt

    Example:
        "name=layer_0.attention_output___iter=100___mb=0___grank=4___tprank=0___
         pprank=1___idx=12___layer=5.pt"

    Args:
        filename: The dump filename (can include path)

    Returns:
        Dictionary mapping field names to their values.
        Known numeric fields are converted to int.

    Example:
        >>> parse_dump_filename("name=hidden___iter=0___mb=0___grank=0___tprank=0___pprank=0___idx=1.pt")
        {'name': 'hidden', 'iter': 0, 'mb': 0, 'grank': 0, 'tprank': 0, 'pprank': 0, 'idx': 1}
    """
    # Get basename without extension
    basename = os.path.basename(filename).removesuffix(".pt")

    # Split by delimiter
    pairs = basename.split("___")

    result: Dict[str, Any] = {}
    for pair in pairs:
        key, value = pair.split("=", 1)
        if key in _NUMERIC_FILENAME_FIELDS:
            result[key] = int(value)
        else:
            result[key] = value

    return result


def build_dump_filename(
    name: str,
    iteration: int,
    micro_batch_id: int,
    global_rank: int,
    tp_rank: int,
    pp_rank: int,
    dump_index: int,
    layer_id: Optional[int] = None,
    **extra_fields,
) -> str:
    """
    Build a dump filename from metadata fields.

    Args:
        name: Tensor name
        iteration: Training iteration number
        micro_batch_id: Micro-batch ID
        global_rank: Global rank
        tp_rank: Tensor parallel rank
        pp_rank: Pipeline parallel rank
        dump_index: Dump index within the iteration
        layer_id: Optional layer ID
        **extra_fields: Additional key-value pairs to include

    Returns:
        The formatted filename (without directory path).

    Example:
        >>> build_dump_filename("hidden", 0, 0, 0, 0, 0, 1)
        'name=hidden___iter=0___mb=0___grank=0___tprank=0___pprank=0___idx=1.pt'
    """
    parts = [
        f"name={name}",
        f"iter={iteration}",
        f"mb={micro_batch_id}",
        f"grank={global_rank}",
        f"tprank={tp_rank}",
        f"pprank={pp_rank}",
        f"idx={dump_index}",
    ]

    if layer_id is not None:
        parts.append(f"layer={layer_id}")

    for key, value in extra_fields.items():
        parts.append(f"{key}={value}")

    return "___".join(parts) + ".pt"


def get_tensor_stats(tensor: torch.Tensor) -> Dict[str, Any]:
    """
    Get statistics for a tensor.

    Args:
        tensor: Input tensor

    Returns:
        Dictionary containing tensor statistics:
        - shape: Tensor shape as list
        - dtype: Tensor dtype as string
        - device: Tensor device as string
        - numel: Number of elements
        - min: Minimum value (for floating point tensors)
        - max: Maximum value (for floating point tensors)
        - mean: Mean value (for floating point tensors)
        - std: Standard deviation (for floating point tensors)
        - has_nan: Whether tensor contains NaN values
        - has_inf: Whether tensor contains Inf values
    """
    stats: Dict[str, Any] = {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "numel": tensor.numel(),
    }

    if tensor.numel() == 0:
        return stats

    # For floating point tensors, compute statistics
    if tensor.is_floating_point():
        # Detach and convert to float32 for stable computation
        t = tensor.detach().float()
        stats["min"] = t.min().item()
        stats["max"] = t.max().item()
        stats["mean"] = t.mean().item()
        stats["std"] = t.std().item() if t.numel() > 1 else 0.0
        stats["has_nan"] = bool(torch.isnan(t).any().item())
        stats["has_inf"] = bool(torch.isinf(t).any().item())
    elif tensor.dtype in _INT_DTYPES:
        stats["min"] = tensor.min().item()
        stats["max"] = tensor.max().item()

    return stats


def format_tensor_info(name: str, tensor: torch.Tensor, filepath: str = "") -> str:
    """
    Format tensor information for logging.

    Args:
        name: Tensor name
        tensor: The tensor
        filepath: Optional file path for the dump

    Returns:
        Formatted string with tensor information.

    Example:
        "[Dump] hidden: shape=[2, 1024, 4096] dtype=torch.bfloat16 device=cuda:0
         min=-1.2345 max=2.3456 mean=0.0012 -> /path/to/file.pt"
    """
    stats = get_tensor_stats(tensor)

    info = f"[Dump] {name}: shape={stats['shape']} dtype={stats['dtype']} device={stats['device']}"

    if "min" in stats:
        info += f" min={stats['min']:.4f}"
    if "max" in stats:
        info += f" max={stats['max']:.4f}"
    if "mean" in stats:
        info += f" mean={stats['mean']:.4f}"

    if stats.get("has_nan"):
        info += " [NaN DETECTED]"
    if stats.get("has_inf"):
        info += " [Inf DETECTED]"

    if filepath:
        info += f" -> {filepath}"

    return info


def safe_mkdir(path: str, rank: int = 0) -> None:
    """
    Safely create a directory, only on rank 0 to avoid race conditions.

    Args:
        path: Directory path to create
        rank: Current rank (only rank 0 creates the directory)
    """
    if rank == 0:
        os.makedirs(path, exist_ok=True)


def is_distributed_initialized() -> bool:
    """
    Check if torch.distributed is initialized.

    Returns:
        True if distributed is available and initialized.
    """
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def get_global_rank() -> int:
    """
    Get the global rank, returning 0 if distributed is not initialized.

    Returns:
        Global rank or 0 if not in distributed mode.
    """
    if is_distributed_initialized():
        return torch.distributed.get_rank()
    return 0


def get_world_size() -> int:
    """
    Get the world size, returning 1 if distributed is not initialized.

    Returns:
        World size or 1 if not in distributed mode.
    """
    if is_distributed_initialized():
        return torch.distributed.get_world_size()
    return 1
