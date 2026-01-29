# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""
Core Megatron Debug Tensor Dumper implementation.

Provides the main Dumper class for capturing and saving tensors during
distributed training with multi-dimensional parallelism support.
"""

import json
import os
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, Iterator, Optional, Union

import torch

from .filter_engine import FilterEngine
from .metadata import TensorShardingType, get_sharding_type
from .parallel_adapter import ParallelAdapter, get_parallel_adapter
from .storage_backend import StorageBackend
from .utils import (
    build_dump_filename,
    format_tensor_info,
    get_logger,
    is_distributed_initialized,
)


class _MegatronDumper:
    """
    Megatron Debug Tensor Dumper (Singleton).

    Captures and saves tensors during training with support for:
    - Multi-dimensional parallelism (TP/PP/DP/CP/EP)
    - Flexible filtering by layer, name, iteration
    - Synchronous and asynchronous file writing
    - Context management for metadata

    Example:
        >>> from megatron.core.debug_utils import dumper
        >>>
        >>> # Enable dumper
        >>> dumper.enable = True
        >>>
        >>> # Dump a tensor
        >>> dumper.dump("hidden_states", tensor, layer_id=0)
        >>>
        >>> # Use context for metadata
        >>> with dumper.context(phase="forward"):
        ...     dumper.dump("attention_output", attn_out)
    """

    _instance: Optional["_MegatronDumper"] = None

    def __new__(cls) -> "_MegatronDumper":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._logger = get_logger()

        # Read configuration from environment variables
        self.enable = os.environ.get("MEGATRON_DUMPER_ENABLE", "0") == "1"
        self.write_file = os.environ.get("MEGATRON_DUMPER_WRITE_FILE", "1") == "1"
        self.dump_dir = os.environ.get("MEGATRON_DUMPER_DIR", "/tmp/megatron_dumps")
        self.dump_dp_rank_0_only = os.environ.get("MEGATRON_DUMPER_DP_RANK_0_ONLY", "1") == "1"
        self.log_tensor_stats = os.environ.get("MEGATRON_DUMPER_LOG_STATS", "1") == "1"

        async_write = os.environ.get("MEGATRON_DUMPER_ASYNC", "0") == "1"

        # Initialize components
        self._parallel_adapter = get_parallel_adapter()
        self._filter_engine = FilterEngine(
            layer_filter=os.environ.get("MEGATRON_DUMPER_LAYERS"),
            name_filter=os.environ.get("MEGATRON_DUMPER_NAMES"),
            iteration_filter=os.environ.get("MEGATRON_DUMPER_ITERATIONS"),
        )
        self._storage = StorageBackend(
            base_dir=self.dump_dir,
            async_write=async_write,
        )

        # Runtime state
        self._iteration = 0
        self._micro_batch_id = 0
        self._dump_index = 0
        self._ctx: Dict[str, Any] = {}
        self._session_name: Optional[str] = None

        self._initialized = True

    # ==================== Lifecycle Methods ====================

    def on_training_start(self) -> None:
        """
        Initialize a new dump session at training start.

        Creates session directory and saves session metadata.
        Should be called once at the beginning of training.
        """
        if not self.enable:
            return

        self._session_name = self._generate_session_name()
        self._ensure_dump_dir()
        self._save_session_metadata()

        self._logger.info(f"Megatron Dumper initialized. Session: {self._session_name}")

    def on_iteration_start(self, iteration: int) -> None:
        """
        Called at the start of each training iteration.

        Args:
            iteration: Current iteration number
        """
        if not self.enable:
            return

        self._iteration = iteration
        self._micro_batch_id = 0
        self._dump_index = 0

    def on_micro_batch_start(self, micro_batch_id: int) -> None:
        """
        Called at the start of each micro-batch.

        Args:
            micro_batch_id: Current micro-batch ID
        """
        if not self.enable:
            return

        self._micro_batch_id = micro_batch_id
        self._dump_index = 0

    # ==================== Core Dump Methods ====================

    def dump(
        self,
        name: str,
        value: Any,
        save: bool = True,
        sharding_type: Optional[TensorShardingType] = None,
        **kwargs,
    ) -> None:
        """
        Dump a tensor or value.

        Args:
            name: Tensor name (e.g., "layer_0.attention.query")
            value: Value to dump (tensor or any picklable object)
            save: If True, write to file; if False, only log
            sharding_type: Tensor sharding type (auto-detected if None)
            **kwargs: Additional metadata (e.g., layer_id, expert_id)

        Example:
            >>> dumper.dump("hidden_states", hidden, layer_id=0)
            >>> dumper.dump("attention_scores", scores, save=False)  # Log only
        """
        if not self.enable:
            return

        # DP rank filter
        if self.dump_dp_rank_0_only and not self._parallel_adapter.should_dump_for_dp():
            return

        # Get layer_id from kwargs or context
        layer_id = kwargs.get("layer_id", self._ctx.get("layer_id"))

        # Apply filters
        if not self._filter_engine.should_dump(name, layer_id, self._iteration):
            return

        self._dump_index += 1

        # Get parallel info
        parallel_info = self._parallel_adapter.get_parallel_info()

        # Auto-detect sharding type if not provided
        if sharding_type is None and isinstance(value, torch.Tensor):
            sharding_type = get_sharding_type(name)

        # Build metadata
        metadata = {
            **self._ctx,
            **kwargs,
            "name": name,
            "iteration": self._iteration,
            "micro_batch_id": self._micro_batch_id,
            "dump_index": self._dump_index,
            **parallel_info,
        }

        if sharding_type is not None:
            metadata["sharding_type"] = sharding_type.value

        if isinstance(value, torch.Tensor):
            metadata["shape"] = list(value.shape)
            metadata["dtype"] = str(value.dtype)
            metadata["device"] = str(value.device)

        # Build file path
        filepath = self._build_filepath(name, metadata)

        # Log tensor info
        if self.log_tensor_stats and isinstance(value, torch.Tensor):
            info = format_tensor_info(name, value, filepath if save else "")
            self._logger.info(info)

        # Save to file
        if save and self.write_file:
            self._storage.save(value, filepath, metadata)

    def dump_dict(
        self,
        name_prefix: str,
        data: Union[Dict[str, Any], Any],
        save: bool = True,
        **kwargs,
    ) -> None:
        """
        Dump all tensor fields from a dictionary or object.

        Args:
            name_prefix: Prefix for tensor names
            data: Dictionary or object with tensor attributes
            save: If True, write to file
            **kwargs: Additional metadata

        Example:
            >>> dumper.dump_dict("attention", {
            ...     "query": q, "key": k, "value": v
            ... }, layer_id=0)
        """
        if not self.enable:
            return

        if isinstance(data, dict):
            items = data.items()
        elif hasattr(data, "__dict__"):
            items = vars(data).items()
        else:
            # Single value, dump directly
            self.dump(name_prefix, data, save=save, **kwargs)
            return

        for key, val in items:
            if isinstance(val, torch.Tensor):
                self.dump(f"{name_prefix}.{key}", val, save=save, **kwargs)

    # ==================== Context Management ====================

    def set_ctx(self, **kwargs) -> None:
        """
        Set context variables that will be included in all dumps.

        Args:
            **kwargs: Key-value pairs to set (use None to clear a key)

        Example:
            >>> dumper.set_ctx(phase="prefill", batch_size=32)
            >>> dumper.set_ctx(layer_id=None)  # Clear layer_id
        """
        for key, value in kwargs.items():
            if value is None:
                self._ctx.pop(key, None)
            else:
                self._ctx[key] = value

    def clear_ctx(self) -> None:
        """Clear all context variables."""
        self._ctx.clear()

    @contextmanager
    def context(self, **kwargs) -> Iterator[None]:
        """
        Context manager for temporarily setting context variables.

        Args:
            **kwargs: Context variables to set

        Example:
            >>> with dumper.context(layer_id=5, phase="forward"):
            ...     dumper.dump("hidden", x)
            >>> # layer_id and phase are restored after the block
        """
        old_ctx = self._ctx.copy()
        self.set_ctx(**kwargs)
        try:
            yield
        finally:
            self._ctx = old_ctx

    # ==================== Internal Methods ====================

    def _build_filepath(self, name: str, metadata: Dict[str, Any]) -> str:
        """Build the full file path for a dump."""
        filename = build_dump_filename(
            name=name,
            iteration=metadata["iteration"],
            micro_batch_id=metadata["micro_batch_id"],
            global_rank=metadata["global_rank"],
            tp_rank=metadata["tp_rank"],
            pp_rank=metadata["pp_rank"],
            dump_index=metadata["dump_index"],
            layer_id=metadata.get("layer_id"),
        )

        # Directory structure: session/iter_NNNNNN/
        session = self._session_name or "default"
        subdir = os.path.join(session, f"iter_{metadata['iteration']:06d}")

        return os.path.join(self.dump_dir, subdir, filename)

    def _generate_session_name(self) -> str:
        """Generate and optionally broadcast session name across ranks."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"dump_{timestamp}"

        # Broadcast from rank 0 if distributed
        if is_distributed_initialized():
            import torch.distributed as dist
            name_list = [name] if dist.get_rank() == 0 else [None]
            dist.broadcast_object_list(name_list, src=0)
            name = name_list[0]

        return name

    def _ensure_dump_dir(self) -> None:
        """Ensure dump directory exists (only on global rank 0)."""
        parallel_info = self._parallel_adapter.get_parallel_info()
        if parallel_info.get("global_rank", 0) == 0:
            session_dir = os.path.join(self.dump_dir, self._session_name or "default")
            os.makedirs(session_dir, exist_ok=True)

    def _save_session_metadata(self) -> None:
        """Save session metadata to JSON file."""
        parallel_info = self._parallel_adapter.get_parallel_info()
        if parallel_info.get("global_rank", 0) != 0:
            return

        metadata = {
            "session_name": self._session_name,
            "created_at": datetime.now().isoformat(),
            "parallel_config": parallel_info,
            "dumper_config": {
                "dump_dir": self.dump_dir,
                "dp_rank_0_only": self.dump_dp_rank_0_only,
                "write_file": self.write_file,
                "async_write": self._storage.async_write,
            },
        }

        filepath = os.path.join(
            self.dump_dir,
            self._session_name or "default",
            "session_metadata.json",
        )
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(metadata, f, indent=2)

    # ==================== Properties ====================

    @property
    def iteration(self) -> int:
        """Current iteration number."""
        return self._iteration

    @property
    def micro_batch_id(self) -> int:
        """Current micro-batch ID."""
        return self._micro_batch_id

    @property
    def session_name(self) -> Optional[str]:
        """Current session name."""
        return self._session_name

    @property
    def filter_engine(self) -> FilterEngine:
        """Access the filter engine for dynamic configuration."""
        return self._filter_engine

    @property
    def parallel_adapter(self) -> ParallelAdapter:
        """Access the parallel adapter."""
        return self._parallel_adapter

    @property
    def storage(self) -> StorageBackend:
        """Access the storage backend."""
        return self._storage

    # ==================== Utility Methods ====================

    def flush(self) -> None:
        """Wait for all pending async writes to complete."""
        self._storage.flush()

    def reset(self) -> None:
        """Reset runtime state (for testing)."""
        self._iteration = 0
        self._micro_batch_id = 0
        self._dump_index = 0
        self._ctx.clear()
        self._session_name = None


# Global singleton instance
dumper = _MegatronDumper()
