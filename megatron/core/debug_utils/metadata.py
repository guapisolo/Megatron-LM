# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""
Metadata definitions for Megatron Debug Tensor Dumper.

This module defines tensor sharding types and mappings used for
tracking how tensors are distributed across parallel dimensions.
"""

from enum import Enum
from typing import Dict


class TensorShardingType(Enum):
    """
    Tensor sharding type enum.

    Defines how a tensor is distributed across different parallel dimensions
    in Megatron's multi-dimensional parallelism (TP/PP/DP/CP/EP).
    """

    REPLICATED = "replicated"      # Replicated across all ranks (e.g., LayerNorm output after TP)
    TP_COLUMN = "tp_column"        # TP column-wise split (e.g., QKV projection output)
    TP_ROW = "tp_row"              # TP row-wise split (e.g., Output projection input)
    CP_SPLIT = "cp_split"          # Context parallel sequence split
    EP_SPLIT = "ep_split"          # Expert parallel expert split


# Common tensor position to sharding type mapping
# This helps determine how to aggregate tensors across TP ranks
TENSOR_SHARDING_MAP: Dict[str, TensorShardingType] = {
    # LayerNorm outputs (replicated after TP)
    "input_layernorm.output": TensorShardingType.REPLICATED,
    "pre_mlp_layernorm.output": TensorShardingType.REPLICATED,
    "post_attention_layernorm.output": TensorShardingType.REPLICATED,

    # Self-attention tensors (TP column split for Q, K, V)
    "self_attention.query": TensorShardingType.TP_COLUMN,
    "self_attention.key": TensorShardingType.TP_COLUMN,
    "self_attention.value": TensorShardingType.TP_COLUMN,
    "self_attention.query_key_value": TensorShardingType.TP_COLUMN,
    "self_attention.context": TensorShardingType.TP_COLUMN,
    "self_attention.attention_scores": TensorShardingType.TP_COLUMN,
    "self_attention.attention_probs": TensorShardingType.TP_COLUMN,

    # Self-attention output (replicated after all-reduce)
    "self_attention.output": TensorShardingType.REPLICATED,
    "self_attention.dense.output": TensorShardingType.REPLICATED,
    "attention_output": TensorShardingType.REPLICATED,

    # MLP tensors (TP column split for fc1/gate)
    "mlp.fc1_output": TensorShardingType.TP_COLUMN,
    "mlp.gate_output": TensorShardingType.TP_COLUMN,
    "mlp.activation_output": TensorShardingType.TP_COLUMN,
    "mlp.fc2_input": TensorShardingType.TP_COLUMN,

    # MLP output (replicated after all-reduce)
    "mlp.output": TensorShardingType.REPLICATED,
    "mlp.fc2_output": TensorShardingType.REPLICATED,
    "mlp_output": TensorShardingType.REPLICATED,

    # MoE expert tensors
    "moe.router_logits": TensorShardingType.REPLICATED,
    "moe.expert_weights": TensorShardingType.REPLICATED,
    "moe.expert_output": TensorShardingType.EP_SPLIT,
    "moe.dispatched_input": TensorShardingType.EP_SPLIT,

    # Embedding and final output
    "embedding.output": TensorShardingType.TP_COLUMN,
    "final_layernorm.output": TensorShardingType.REPLICATED,
    "output_layer.output": TensorShardingType.TP_COLUMN,
    "logits": TensorShardingType.TP_COLUMN,
}


def get_sharding_type(tensor_name: str) -> TensorShardingType:
    """
    Get the sharding type for a tensor by its name.

    Args:
        tensor_name: The name of the tensor (e.g., "self_attention.query")

    Returns:
        The corresponding TensorShardingType, or REPLICATED if not found.

    Note:
        This performs a suffix match, so "layer_0.self_attention.query"
        will match "self_attention.query".
    """
    # Try exact match first
    if tensor_name in TENSOR_SHARDING_MAP:
        return TENSOR_SHARDING_MAP[tensor_name]

    # Try suffix match
    for pattern, sharding_type in TENSOR_SHARDING_MAP.items():
        if tensor_name.endswith(pattern):
            return sharding_type

    # Default to replicated if unknown
    return TensorShardingType.REPLICATED
