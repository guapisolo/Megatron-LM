# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""
Core Megatron Debug Tensor Dumper implementation.

Provides the main Dumper class for capturing and saving tensors during
distributed training with multi-dimensional parallelism support.
"""

import json
import os
import re
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple, Type, Union

import torch
import torch.nn as nn

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

DEFAULT_DUMP_POINTS: Tuple[str, ...] = ("post_attention", "post_mlp")
DUMP_POINT_TO_ATTR: Dict[str, str] = {
    "pre_attention": "input_layernorm",
    "post_attention": "self_attention",
    "post_mlp": "mlp",
    "post_layernorm": "post_attention_layernorm",
}


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

        # Phase 3: Advanced features
        self.aggregate_tp = os.environ.get("MEGATRON_DUMPER_AGGREGATE_TP", "0") == "1"
        self.dump_gradients_enabled = os.environ.get("MEGATRON_DUMPER_GRADIENTS", "0") == "1"

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

        # Phase 3: Hook management
        self._hook_handles: List[torch.utils.hooks.RemovableHandle] = []

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

        tp_aggregated = False
        if self.aggregate_tp and isinstance(value, torch.Tensor):
            value, tp_aggregated = self._maybe_aggregate_tp(value, name, sharding_type)
            if value is None:
                return

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

        # Mark if tensor was aggregated
        if tp_aggregated:
            metadata["tp_aggregated"] = True

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

    # ==================== Hook Registration ====================

    def register_transformer_hooks(
        self,
        model: nn.Module,
        layer_class: Optional[Type[nn.Module]] = None,
        dump_points: Optional[Sequence[str]] = None,
    ) -> int:
        """
        Register forward hooks on transformer layers for automatic tensor dumping.

        This method automatically discovers TransformerLayer modules in the model
        and registers hooks at specified dump points to capture intermediate tensors.

        Args:
            model: The Megatron model to register hooks on
            layer_class: The transformer layer class to hook. If None, attempts
                to import megatron.core.transformer.TransformerLayer
            dump_points: List of positions to dump. Supported values:
                - "pre_attention": Input to self-attention
                - "post_attention": Output of self-attention
                - "post_mlp": Output of MLP
                - "post_layernorm": Output of layer normalization
                Default: ["post_attention", "post_mlp"]

        Returns:
            Number of hooks registered.

        Example:
            >>> from megatron.core.debug_utils import dumper
            >>> dumper.enable = True
            >>> dumper.register_transformer_hooks(
            ...     model,
            ...     dump_points=["post_attention", "post_mlp"]
            ... )
            32  # Returns number of registered hooks
        """
        if not self.enable:
            return 0

        if layer_class is None:
            layer_class = self._get_transformer_layer_class()

        if dump_points is None:
            dump_points = list(DEFAULT_DUMP_POINTS)
        else:
            dump_points = list(dump_points)

        self._validate_dump_points(dump_points)

        hooks_registered = 0
        layers_with_hooks = 0
        for name, module in model.named_modules():
            if isinstance(module, layer_class):
                layer_id = self._extract_layer_id(name)
                num_hooks = self._register_layer_hooks(module, layer_id, dump_points)
                hooks_registered += num_hooks
                layers_with_hooks += 1

        self._logger.info(
            "Registered %d hooks on %d layers",
            hooks_registered,
            layers_with_hooks,
        )
        return hooks_registered

    def _get_transformer_layer_class(self) -> Type[nn.Module]:
        """
        Attempt to import the default TransformerLayer class.

        Returns:
            The TransformerLayer class.
        """
        from megatron.core.transformer import TransformerLayer
        return TransformerLayer

    def _validate_dump_points(self, dump_points: Sequence[str]) -> None:
        """
        Validate dump point names against supported values.

        Args:
            dump_points: List of dump point identifiers.
        """
        unsupported = [point for point in dump_points if point not in DUMP_POINT_TO_ATTR]
        if unsupported:
            raise ValueError(f"Unsupported dump points: {', '.join(unsupported)}")

    def _extract_layer_id(self, module_name: str) -> Optional[int]:
        """
        Extract layer ID from a module name.

        Supports various naming patterns:
        - "decoder.layers.5.self_attention" -> 5
        - "encoder.layer.10.mlp" -> 10
        - "transformer.layers.0" -> 0
        - "model.decoder.layers.12.attention" -> 12

        Args:
            module_name: The full module name from named_modules()

        Returns:
            The layer ID as an integer, or None if not found.

        Example:
            >>> dumper._extract_layer_id("decoder.layers.5.self_attention")
            5
            >>> dumper._extract_layer_id("encoder.layer.10.mlp")
            10
        """
        # Pattern to match layer indices in module names
        # Matches patterns like: layers.5, layer.10, blocks.0
        patterns = [
            r"layers\.(\d+)",
            r"layer\.(\d+)",
            r"blocks\.(\d+)",
            r"block\.(\d+)",
        ]

        for pattern in patterns:
            match = re.search(pattern, module_name)
            if match:
                return int(match.group(1))

        return None

    def _register_layer_hooks(
        self,
        layer: nn.Module,
        layer_id: Optional[int],
        dump_points: Sequence[str],
    ) -> int:
        """
        Register forward hooks for a single transformer layer.

        Args:
            layer: The transformer layer module
            layer_id: The layer number (for metadata)
            dump_points: List of dump point names

        Returns:
            Number of hooks registered on this layer.
        """
        hooks_registered = 0

        def make_hook(
            point_name: str,
            lid: Optional[int],
        ) -> Callable:
            """Create a forward hook function for a dump point."""
            def hook(
                _module: nn.Module,
                _inputs: Tuple[Any, ...],
                output: Any,
            ) -> None:
                output_tensor = output[0] if isinstance(output, tuple) else output

                # Build tensor name
                if lid is not None:
                    name = f"layer_{lid}.{point_name}"
                else:
                    name = point_name

                self.dump(
                    name,
                    output_tensor,
                    layer_id=lid,
                    hook_point=point_name,
                )

            return hook

        for point in dump_points:
            attr_name = DUMP_POINT_TO_ATTR[point]
            submodule = getattr(layer, attr_name)
            handle = submodule.register_forward_hook(
                make_hook(point, layer_id)
            )
            self._hook_handles.append(handle)
            hooks_registered += 1

        return hooks_registered

    def remove_all_hooks(self) -> int:
        """
        Remove all registered forward hooks.

        Returns:
            Number of hooks removed.

        Example:
            >>> dumper.register_transformer_hooks(model)
            32
            >>> dumper.remove_all_hooks()
            32
        """
        num_removed = len(self._hook_handles)
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()
        self._logger.info(f"Removed {num_removed} hooks")
        return num_removed

    # ==================== TP Aggregation ====================

    def _maybe_aggregate_tp(
        self,
        tensor: torch.Tensor,
        name: str,
        sharding_type: Optional[TensorShardingType],
    ) -> Tuple[Optional[torch.Tensor], bool]:
        """
        Optionally aggregate tensor across TP ranks.

        When aggregate_tp is enabled, this method gathers sharded tensors
        from all TP ranks. The aggregated tensor is only returned on TP rank 0;
        other ranks receive None to avoid redundant dumping.

        Args:
            tensor: The local tensor (possibly a shard)
            name: Tensor name (used to determine sharding type if not provided)
            sharding_type: Explicit sharding type, or None to auto-detect

        Returns:
            Tuple of (tensor_or_none, aggregated_flag).

        Example:
            >>> # With aggregate_tp=True, TP_COLUMN tensors are gathered
            >>> aggregated, _ = dumper._maybe_aggregate_tp(
            ...     local_shard,
            ...     "self_attention.query",
            ...     TensorShardingType.TP_COLUMN
            ... )
        """
        if not self.aggregate_tp:
            return tensor, False

        # Auto-detect sharding type if not provided
        if sharding_type is None:
            sharding_type = get_sharding_type(name)

        parallel_info = self._parallel_adapter.get_parallel_info()
        tp_rank = parallel_info.get("tp_rank", 0)
        tp_size = parallel_info.get("tp_size", 1)

        # No aggregation needed for single TP rank
        if tp_size <= 1:
            return tensor, False

        if sharding_type == TensorShardingType.REPLICATED:
            # Replicated tensors: only dump on TP rank 0
            if tp_rank != 0:
                return None, False
            return tensor, False

        if sharding_type in (TensorShardingType.TP_COLUMN, TensorShardingType.TP_ROW):
            # Sharded tensors: gather and only return on TP rank 0
            aggregated = self._parallel_adapter.gather_across_tp(tensor)
            if tp_rank != 0:
                return None, True
            return aggregated, True

        # Unknown sharding type: return as-is (no aggregation)
        return tensor, False

    # ==================== Gradient Dump ====================

    def dump_gradients(
        self,
        module: nn.Module,
        name_prefix: str,
        **kwargs,
    ) -> int:
        """
        Dump gradients of all parameters in a module.

        This is useful for debugging gradient issues such as vanishing/exploding
        gradients during training.

        Args:
            module: The module whose parameter gradients to dump
            name_prefix: Prefix for the gradient tensor names
            **kwargs: Additional metadata (e.g., layer_id)

        Returns:
            Number of gradients dumped.

        Example:
            >>> # After loss.backward()
            >>> dumper.dump_gradients(
            ...     model.layers[0].self_attention,
            ...     "layer_0.attention",
            ...     layer_id=0
            ... )
            4  # Returns number of gradients dumped
        """
        if not self.enable or not self.dump_gradients_enabled:
            return 0

        num_dumped = 0
        for param_name, param in module.named_parameters():
            if param.grad is not None:
                grad_name = f"{name_prefix}.{param_name}.grad"
                self.dump(
                    grad_name,
                    param.grad,
                    is_gradient=True,
                    **kwargs,
                )
                num_dumped += 1

        return num_dumped

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

        filter_config = self._filter_engine.get_filter_config()

        metadata = {
            "session_name": self._session_name,
            "created_at": datetime.now().isoformat(),
            "parallel_config": parallel_info,
            "dumper_config": {
                "dump_dir": self.dump_dir,
                "dp_rank_0_only": self.dump_dp_rank_0_only,
                "write_file": self.write_file,
                "async_write": self._storage.async_write,
                "aggregate_tp": self.aggregate_tp,
                "dump_gradients": self.dump_gradients_enabled,
                "log_tensor_stats": self.log_tensor_stats,
            },
            "filter_config": filter_config,
        }

        # Try to get Megatron version
        try:
            import megatron
            metadata["megatron_version"] = getattr(megatron, "__version__", "unknown")
        except ImportError:
            pass

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
        # Remove all hooks when resetting
        self.remove_all_hooks()


# Global singleton instance
dumper = _MegatronDumper()
