# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""
Parallel State Adapter for Megatron Debug Tensor Dumper.

This module provides a unified interface for accessing Megatron's parallel state
information across different parallelism dimensions (TP, PP, DP, CP, EP, VPP).
It gracefully handles cases where parallel state is not initialized.
"""

from typing import Any, Dict, Optional

import torch

from .utils import get_global_rank, get_logger, get_world_size


class ParallelAdapter:
    """
    Megatron parallel state adapter.

    Provides a unified interface for accessing parallel state information
    and performing cross-rank tensor operations. Gracefully degrades when
    parallel state is not initialized.

    Example:
        >>> adapter = ParallelAdapter()
        >>> info = adapter.get_parallel_info()
        >>> print(info['tp_rank'], info['tp_size'])
        0 4
    """

    def __init__(self, num_layers: Optional[int] = None):
        """
        Initialize the parallel adapter.

        Args:
            num_layers: Total number of transformer layers (used for layer offset
                calculation). If None, the adapter will fall back to
                megatron.training.get_args() when layer offsets are needed.
        """
        self._num_layers = num_layers
        self._logger = get_logger()
        self._parallel_info_cache: Optional[Dict[str, Any]] = None

    def is_initialized(self) -> bool:
        """
        Check if Megatron parallel state is initialized.

        Returns:
            True if parallel state is properly initialized.
        """
        try:
            from megatron.core import parallel_state
            return parallel_state.is_initialized()
        except (ImportError, AttributeError):
            return False

    def get_parallel_info(self, use_cache: bool = False) -> Dict[str, Any]:
        """
        Get complete parallel state information for the current rank.

        Args:
            use_cache: If True, return cached info if available (for performance).

        Returns:
            Dictionary containing:
            - global_rank: Global rank across all processes
            - world_size: Total number of processes
            - dp_rank: Data parallel rank
            - dp_size: Data parallel world size
            - tp_rank: Tensor parallel rank
            - tp_size: Tensor parallel world size
            - pp_rank: Pipeline parallel rank
            - pp_size: Pipeline parallel world size
            - cp_rank: Context parallel rank
            - cp_size: Context parallel world size
            - ep_rank: Expert parallel rank
            - ep_size: Expert parallel world size
            - vpp_rank: Virtual pipeline parallel rank (or 0)
            - layer_offset: Starting layer number for current PP stage

        Note:
            When parallel state is not initialized, returns default values
            (rank=0, size=1) for graceful degradation.
        """
        if use_cache and self._parallel_info_cache is not None:
            return self._parallel_info_cache

        if not self.is_initialized():
            # Graceful degradation for non-distributed or uninitialized case
            return {
                "global_rank": get_global_rank(),
                "world_size": get_world_size(),
                "dp_rank": 0,
                "dp_size": 1,
                "tp_rank": 0,
                "tp_size": 1,
                "pp_rank": 0,
                "pp_size": 1,
                "cp_rank": 0,
                "cp_size": 1,
                "ep_rank": 0,
                "ep_size": 1,
                "vpp_rank": 0,
                "layer_offset": 0,
            }

        from megatron.core import parallel_state as ps

        info = {
            "global_rank": get_global_rank(),
            "world_size": get_world_size(),
            "dp_rank": ps.get_data_parallel_rank(),
            "dp_size": ps.get_data_parallel_world_size(),
            "tp_rank": ps.get_tensor_model_parallel_rank(),
            "tp_size": ps.get_tensor_model_parallel_world_size(),
            "pp_rank": ps.get_pipeline_model_parallel_rank(),
            "pp_size": ps.get_pipeline_model_parallel_world_size(),
            "cp_rank": self._safe_get_cp_rank(),
            "cp_size": self._safe_get_cp_world_size(),
            "ep_rank": self._safe_get_ep_rank(),
            "ep_size": self._safe_get_ep_world_size(),
            "vpp_rank": self._safe_get_vpp_rank(),
            "layer_offset": self._get_layer_offset(),
        }

        if use_cache:
            self._parallel_info_cache = info

        return info

    def _safe_get_cp_rank(self) -> int:
        """Safely get context parallel rank."""
        from megatron.core import parallel_state as ps
        try:
            return ps.get_context_parallel_rank()
        except AssertionError:
            return 0

    def _safe_get_cp_world_size(self) -> int:
        """Safely get context parallel world size."""
        from megatron.core import parallel_state as ps
        try:
            return ps.get_context_parallel_world_size()
        except AssertionError:
            return 1

    def _safe_get_ep_rank(self) -> int:
        """Safely get expert parallel rank."""
        from megatron.core import parallel_state as ps
        try:
            return ps.get_expert_model_parallel_rank()
        except AssertionError:
            return 0

    def _safe_get_ep_world_size(self) -> int:
        """Safely get expert parallel world size."""
        from megatron.core import parallel_state as ps
        try:
            return ps.get_expert_model_parallel_world_size()
        except AssertionError:
            return 1

    def _safe_get_vpp_rank(self) -> int:
        """Safely get virtual pipeline parallel rank."""
        from megatron.core import parallel_state as ps
        vpp_rank = ps.get_virtual_pipeline_model_parallel_rank()
        return 0 if vpp_rank is None else vpp_rank

    def _get_layer_offset(self) -> int:
        """
        Calculate the layer number offset for the current PP stage.

        The offset represents the starting layer index for the current
        pipeline parallel stage, assuming layers are evenly distributed.

        Returns:
            Layer offset for the current PP stage.
        """
        if not self.is_initialized():
            return 0

        from megatron.core import parallel_state as ps

        pp_rank = ps.get_pipeline_model_parallel_rank()
        pp_size = ps.get_pipeline_model_parallel_world_size()

        if pp_size <= 1:
            return 0

        num_layers = self._resolve_num_layers()
        layers_per_stage = num_layers // pp_size
        return pp_rank * layers_per_stage

    def _resolve_num_layers(self) -> int:
        """Resolve total number of layers for pipeline offset calculation.

        Raises:
            RuntimeError: If the layer count cannot be determined.
        """
        if self._num_layers is not None:
            return self._num_layers

        try:
            from megatron.training import get_args
        except (ImportError, RuntimeError) as exc:
            raise RuntimeError(
                "num_layers is required for pipeline offsets; pass it to ParallelAdapter "
                "or initialize megatron.training args."
            ) from exc

        args = get_args()
        num_layers = getattr(args, "num_layers", None)
        if num_layers is None:
            raise RuntimeError(
                "num_layers is required for pipeline offsets; pass it to ParallelAdapter "
                "or set args.num_layers."
            )
        return num_layers

    def should_dump_for_dp(self, dump_all_dp_ranks: bool = False) -> bool:
        """
        Determine if the current rank should dump (based on DP dimension).

        By default, only DP rank 0 should dump to avoid redundant data,
        since all DP ranks process the same data with the same model.

        Args:
            dump_all_dp_ranks: If True, all DP ranks will dump.

        Returns:
            True if the current rank should dump.
        """
        if dump_all_dp_ranks or not self.is_initialized():
            return True

        from megatron.core import parallel_state as ps
        return ps.get_data_parallel_rank() == 0

    def get_tp_group(self, check_initialized: bool = False):
        """
        Get the tensor model parallel communication group.

        Args:
            check_initialized: If True, assert that the group is initialized.

        Returns:
            The TP process group, or None if not initialized.
        """
        if not self.is_initialized():
            return None

        from megatron.core import parallel_state as ps
        return ps.get_tensor_model_parallel_group(check_initialized=check_initialized)

    def get_dp_group(self, check_initialized: bool = False):
        """
        Get the data parallel communication group.

        Args:
            check_initialized: If True, assert that the group is initialized.

        Returns:
            The DP process group, or None if not initialized.
        """
        if not self.is_initialized():
            return None

        from megatron.core import parallel_state as ps
        return ps.get_data_parallel_group()

    def get_pp_group(self, check_initialized: bool = False):
        """
        Get the pipeline model parallel communication group.

        Args:
            check_initialized: If True, assert that the group is initialized.

        Returns:
            The PP process group, or None if not initialized.
        """
        if not self.is_initialized():
            return None

        from megatron.core import parallel_state as ps
        return ps.get_pipeline_model_parallel_group(check_initialized=check_initialized)

    def gather_across_tp(
        self,
        tensor: torch.Tensor,
        dim: int = -1,
    ) -> Optional[torch.Tensor]:
        """
        Gather tensor across tensor parallel ranks.

        This aggregates sharded tensors from all TP ranks into a complete tensor.
        The result is only valid on TP rank 0.

        Args:
            tensor: The local tensor shard to gather
            dim: The dimension along which to concatenate (default: -1)

        Returns:
            The gathered complete tensor on TP rank 0, or the input tensor
            on other ranks. Returns None if gathering fails.

        Note:
            This uses Megatron's built-in gather functions which handle
            the autograd correctly for backward pass if needed.
        """
        if not self.is_initialized():
            return tensor  # No TP, return as-is

        from megatron.core import parallel_state as ps

        tp_size = ps.get_tensor_model_parallel_world_size()
        if tp_size <= 1:
            return tensor  # No TP sharding

        tp_group = ps.get_tensor_model_parallel_group()

        # Use the appropriate gather function based on dimension
        if dim == -1 or dim == tensor.dim() - 1:
            # Gather along last dimension
            from megatron.core.tensor_parallel import (
                gather_from_tensor_model_parallel_region,
            )
            return gather_from_tensor_model_parallel_region(tensor, tp_group)
        if dim == 0:
            # Gather along first dimension
            from megatron.core.tensor_parallel import (
                gather_from_sequence_parallel_region,
            )
            return gather_from_sequence_parallel_region(tensor, tp_group)

        # For other dimensions, use manual gather
        return self._manual_gather(tensor, dim, tp_group)

    def _manual_gather(
        self,
        tensor: torch.Tensor,
        dim: int,
        group,
    ) -> torch.Tensor:
        """
        Manually gather tensor across a process group along specified dimension.

        Args:
            tensor: Input tensor
            dim: Dimension to gather along
            group: Process group

        Returns:
            Gathered tensor
        """
        world_size = torch.distributed.get_world_size(group=group)
        if world_size <= 1:
            return tensor

        # Allocate output tensors
        tensor_list = [torch.empty_like(tensor) for _ in range(world_size)]

        # All-gather
        torch.distributed.all_gather(tensor_list, tensor.contiguous(), group=group)

        # Concatenate along the specified dimension
        return torch.cat(tensor_list, dim=dim)

    def reduce_across_tp(
        self,
        tensor: torch.Tensor,
        op: str = "sum",
    ) -> Optional[torch.Tensor]:
        """
        Reduce tensor across tensor parallel ranks.

        Args:
            tensor: The tensor to reduce
            op: Reduction operation ("sum", "mean", "max", "min")

        Returns:
            The reduced tensor (available on all ranks).
        """
        op_key = op.lower()
        ops_map = {
            "sum": torch.distributed.ReduceOp.SUM,
            "mean": torch.distributed.ReduceOp.SUM,  # Divide after reduce
            "max": torch.distributed.ReduceOp.MAX,
            "min": torch.distributed.ReduceOp.MIN,
        }
        if op_key not in ops_map:
            raise ValueError(
                f"Unsupported reduction op: {op}. Expected one of {sorted(ops_map)}."
            )

        if not self.is_initialized():
            return tensor

        from megatron.core import parallel_state as ps

        tp_size = ps.get_tensor_model_parallel_world_size()
        if tp_size <= 1:
            return tensor

        tp_group = ps.get_tensor_model_parallel_group()

        # Clone to avoid modifying input
        output = tensor.clone()

        # All-reduce
        torch.distributed.all_reduce(output, op=ops_map[op_key], group=tp_group)

        # For mean, divide by world size
        if op_key == "mean":
            output = output / tp_size

        return output

    def set_num_layers(self, num_layers: int) -> None:
        """
        Set the total number of transformer layers.

        This is used for layer offset calculation in pipeline parallelism.

        Args:
            num_layers: Total number of layers in the model.
        """
        self._num_layers = num_layers
        # Invalidate cache
        self._parallel_info_cache = None

    def clear_cache(self) -> None:
        """Clear the cached parallel info."""
        self._parallel_info_cache = None


# Global singleton adapter instance
_parallel_adapter: Optional[ParallelAdapter] = None


def get_parallel_adapter() -> ParallelAdapter:
    """
    Get the global ParallelAdapter singleton instance.

    Returns:
        The global ParallelAdapter instance.
    """
    global _parallel_adapter
    if _parallel_adapter is None:
        _parallel_adapter = ParallelAdapter()
    return _parallel_adapter
