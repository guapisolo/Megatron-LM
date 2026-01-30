# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""
Unit tests for Megatron Debug Tensor Dumper - Phase 3 components.

Tests cover:
- Hook registration system
- TP aggregation
- Gradient dumping
- Enhanced session metadata
"""

import json
import os
import tempfile
from unittest import mock

import pytest
import torch
import torch.nn as nn

from megatron.core.debug_utils import (
    TensorShardingType,
    dumper,
)


@pytest.fixture(autouse=True)
def reset_dumper_state():
    """Reset dumper state and restore configuration around each test."""
    original_config = {
        "enable": dumper.enable,
        "write_file": dumper.write_file,
        "dump_dir": dumper.dump_dir,
        "dump_dp_rank_0_only": dumper.dump_dp_rank_0_only,
        "aggregate_tp": dumper.aggregate_tp,
        "dump_gradients_enabled": dumper.dump_gradients_enabled,
        "log_tensor_stats": dumper.log_tensor_stats,
    }
    original_filters = dumper.filter_engine.get_filter_strings()

    dumper.reset()
    dumper.filter_engine.update_layer_filter(None)
    dumper.filter_engine.update_name_filter(None)
    dumper.filter_engine.update_iteration_filter(None)

    yield

    dumper.reset()
    dumper.enable = original_config["enable"]
    dumper.write_file = original_config["write_file"]
    dumper.dump_dir = original_config["dump_dir"]
    dumper.dump_dp_rank_0_only = original_config["dump_dp_rank_0_only"]
    dumper.aggregate_tp = original_config["aggregate_tp"]
    dumper.dump_gradients_enabled = original_config["dump_gradients_enabled"]
    dumper.log_tensor_stats = original_config["log_tensor_stats"]
    dumper.filter_engine.update_layer_filter(original_filters["layer_filter"])
    dumper.filter_engine.update_name_filter(original_filters["name_filter"])
    dumper.filter_engine.update_iteration_filter(original_filters["iteration_filter"])


class SimpleTransformerLayer(nn.Module):
    """Simple mock TransformerLayer for testing hooks."""

    def __init__(self, hidden_size: int = 64):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(hidden_size)
        self.self_attention = nn.Linear(hidden_size, hidden_size)
        self.post_attention_layernorm = nn.LayerNorm(hidden_size)
        self.mlp = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = self.input_layernorm(x)
        attn_out = self.self_attention(normed)
        x = x + attn_out
        normed = self.post_attention_layernorm(x)
        mlp_out = self.mlp(normed)
        return x + mlp_out


class SimpleModel(nn.Module):
    """Simple mock model with multiple layers."""

    def __init__(self, num_layers: int = 4, hidden_size: int = 64):
        super().__init__()
        self.layers = nn.ModuleList([
            SimpleTransformerLayer(hidden_size) for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class TestExtractLayerId:
    """Tests for _extract_layer_id method."""

    def test_layers_pattern(self):
        """Test extraction from 'layers.N' pattern."""
        assert dumper._extract_layer_id("decoder.layers.5.self_attention") == 5
        assert dumper._extract_layer_id("layers.0") == 0
        assert dumper._extract_layer_id("model.layers.15.mlp") == 15

    def test_layer_pattern(self):
        """Test extraction from 'layer.N' pattern."""
        assert dumper._extract_layer_id("encoder.layer.10.attention") == 10
        assert dumper._extract_layer_id("layer.0") == 0

    def test_blocks_pattern(self):
        """Test extraction from 'blocks.N' pattern."""
        assert dumper._extract_layer_id("transformer.blocks.3") == 3
        assert dumper._extract_layer_id("blocks.7.mlp") == 7

    def test_block_pattern(self):
        """Test extraction from 'block.N' pattern."""
        assert dumper._extract_layer_id("block.12") == 12
        assert dumper._extract_layer_id("model.block.0.attention") == 0

    def test_no_match(self):
        """Test when no layer pattern matches."""
        assert dumper._extract_layer_id("attention.query") is None
        assert dumper._extract_layer_id("embedding") is None
        assert dumper._extract_layer_id("") is None


class TestHookRegistration:
    """Tests for hook registration system."""

    def test_register_hooks_disabled(self):
        """Test that no hooks are registered when dumper is disabled."""
        dumper.enable = False
        model = SimpleModel(num_layers=4)
        num_hooks = dumper.register_transformer_hooks(model, layer_class=SimpleTransformerLayer)
        assert num_hooks == 0
        assert len(dumper._hook_handles) == 0

    def test_register_hooks_enabled(self):
        """Test hook registration on enabled dumper."""
        dumper.enable = True
        model = SimpleModel(num_layers=4)

        # Register with default dump points (post_attention, post_mlp)
        num_hooks = dumper.register_transformer_hooks(
            model,
            layer_class=SimpleTransformerLayer,
            dump_points=["post_attention", "post_mlp"],
        )

        # Should register 2 hooks per layer (post_attention, post_mlp)
        assert num_hooks == 8  # 4 layers * 2 hooks
        assert len(dumper._hook_handles) == 8

    def test_register_hooks_custom_dump_points(self):
        """Test hook registration with custom dump points."""
        dumper.enable = True
        model = SimpleModel(num_layers=2)

        # Register only post_mlp
        num_hooks = dumper.register_transformer_hooks(
            model,
            layer_class=SimpleTransformerLayer,
            dump_points=["post_mlp"],
        )

        assert num_hooks == 2  # 2 layers * 1 hook
        assert len(dumper._hook_handles) == 2

    def test_remove_all_hooks(self):
        """Test removing all registered hooks."""
        dumper.enable = True
        model = SimpleModel(num_layers=4)

        dumper.register_transformer_hooks(
            model,
            layer_class=SimpleTransformerLayer,
        )
        assert len(dumper._hook_handles) > 0

        num_removed = dumper.remove_all_hooks()
        assert num_removed > 0
        assert len(dumper._hook_handles) == 0

    def test_hooks_capture_tensors(self):
        """Test that hooks actually capture tensors during forward pass."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dumper.enable = True
            dumper.write_file = True
            dumper.dump_dir = tmpdir
            dumper.dump_dp_rank_0_only = False
            dumper.on_training_start()
            dumper.on_iteration_start(0)

            model = SimpleModel(num_layers=2, hidden_size=32)
            dumper.register_transformer_hooks(
                model,
                layer_class=SimpleTransformerLayer,
                dump_points=["post_attention", "post_mlp"],
            )

            # Run forward pass
            x = torch.randn(2, 32)
            _ = model(x)

            # Check files were created
            dumper.flush()
            session_dir = os.path.join(tmpdir, dumper.session_name, "iter_000000")
            assert os.path.exists(session_dir)
            files = os.listdir(session_dir)
            # 2 layers * 2 hooks = 4 files
            assert len(files) == 4

    def test_hooks_unknown_dump_point(self):
        """Test that unknown dump points raise an error."""
        dumper.enable = True
        model = SimpleModel(num_layers=2)

        with pytest.raises(ValueError, match="Unsupported dump points"):
            dumper.register_transformer_hooks(
                model,
                layer_class=SimpleTransformerLayer,
                dump_points=["unknown_point"],
            )


class TestTPAggregation:
    """Tests for TP aggregation functionality."""

    def test_maybe_aggregate_tp_disabled(self):
        """Test that aggregation is skipped when disabled."""
        dumper.aggregate_tp = False
        tensor = torch.randn(4, 8)
        result, aggregated = dumper._maybe_aggregate_tp(
            tensor,
            "test",
            TensorShardingType.TP_COLUMN,
        )
        assert result is tensor
        assert aggregated is False

    def test_maybe_aggregate_tp_replicated(self):
        """Test aggregation behavior for replicated tensors."""
        dumper.aggregate_tp = True
        tensor = torch.randn(4, 8)

        # Mock parallel adapter to return tp_rank=0
        with mock.patch.object(
            dumper._parallel_adapter,
            "get_parallel_info",
            return_value={"tp_rank": 0, "tp_size": 4},
        ):
            result, aggregated = dumper._maybe_aggregate_tp(
                tensor,
                "test",
                TensorShardingType.REPLICATED,
            )
            assert result is tensor
            assert aggregated is False

        # Mock parallel adapter to return tp_rank=1
        with mock.patch.object(
            dumper._parallel_adapter,
            "get_parallel_info",
            return_value={"tp_rank": 1, "tp_size": 4},
        ):
            result, aggregated = dumper._maybe_aggregate_tp(
                tensor,
                "test",
                TensorShardingType.REPLICATED,
            )
            assert result is None
            assert aggregated is False

    def test_maybe_aggregate_tp_sharded(self):
        """Test aggregation behavior for sharded tensors."""
        dumper.aggregate_tp = True
        tensor = torch.randn(4, 8)
        gathered_tensor = torch.randn(4, 32)  # 4x larger (4 TP ranks)

        # Mock parallel adapter and gather function
        with mock.patch.object(
            dumper._parallel_adapter,
            "get_parallel_info",
            return_value={"tp_rank": 0, "tp_size": 4},
        ), mock.patch.object(
            dumper._parallel_adapter,
            "gather_across_tp",
            return_value=gathered_tensor,
        ):
            result, aggregated = dumper._maybe_aggregate_tp(
                tensor,
                "test",
                TensorShardingType.TP_COLUMN,
            )
            assert result is gathered_tensor
            assert aggregated is True

    def test_tp_aggregation_in_dump(self):
        """Test that TP aggregation is applied in dump method."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dumper.enable = True
            dumper.aggregate_tp = True
            dumper.write_file = True
            dumper.dump_dir = tmpdir
            dumper.dump_dp_rank_0_only = False
            dumper.on_training_start()
            dumper.on_iteration_start(0)

            tensor = torch.randn(4, 8)
            gathered = torch.randn(4, 32)

            # Mock to simulate TP rank 0
            with mock.patch.object(
                dumper._parallel_adapter,
                "get_parallel_info",
                return_value={
                    "global_rank": 0,
                    "world_size": 4,
                    "tp_rank": 0,
                    "tp_size": 4,
                    "pp_rank": 0,
                    "pp_size": 1,
                    "dp_rank": 0,
                    "dp_size": 1,
                },
            ), mock.patch.object(
                dumper._parallel_adapter,
                "gather_across_tp",
                return_value=gathered,
            ):
                dumper.dump(
                    "sharded_tensor",
                    tensor,
                    sharding_type=TensorShardingType.TP_COLUMN,
                )

            dumper.flush()

            session_dir = os.path.join(tmpdir, dumper.session_name, "iter_000000")
            assert os.path.exists(session_dir)
            files = os.listdir(session_dir)
            assert len(files) == 1

            # Check metadata includes tp_aggregated flag
            filepath = os.path.join(session_dir, files[0])
            loaded = torch.load(filepath, weights_only=False)
            assert loaded["metadata"].get("tp_aggregated") is True


class TestGradientDump:
    """Tests for gradient dumping functionality."""

    def test_dump_gradients_disabled(self):
        """Test that gradient dump is skipped when disabled."""
        dumper.enable = True
        dumper.dump_gradients_enabled = False

        model = nn.Linear(10, 5)
        x = torch.randn(2, 10)
        y = model(x)
        y.sum().backward()

        num_dumped = dumper.dump_gradients(model, "linear")
        assert num_dumped == 0

    def test_dump_gradients_enabled(self):
        """Test gradient dumping when enabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dumper.enable = True
            dumper.dump_gradients_enabled = True
            dumper.write_file = True
            dumper.dump_dir = tmpdir
            dumper.dump_dp_rank_0_only = False
            dumper.on_training_start()
            dumper.on_iteration_start(0)

            model = nn.Linear(10, 5)
            x = torch.randn(2, 10)
            y = model(x)
            y.sum().backward()

            num_dumped = dumper.dump_gradients(model, "linear")
            # Linear has weight and bias gradients
            assert num_dumped == 2

            dumper.flush()
            session_dir = os.path.join(tmpdir, dumper.session_name, "iter_000000")
            assert os.path.exists(session_dir)
            files = os.listdir(session_dir)
            assert len(files) == 2

            # Check filenames contain .grad
            assert all(".grad" in f for f in files)

    def test_dump_gradients_no_grad(self):
        """Test that parameters without gradients are skipped."""
        dumper.enable = True
        dumper.dump_gradients_enabled = True

        model = nn.Linear(10, 5)
        # Don't call backward, so no gradients
        num_dumped = dumper.dump_gradients(model, "linear")
        assert num_dumped == 0


class TestEnhancedSessionMetadata:
    """Tests for enhanced session metadata."""

    def test_session_metadata_includes_filter_config(self):
        """Test that session metadata includes filter configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dumper.enable = True
            dumper.dump_dir = tmpdir

            # Set up filters
            dumper.filter_engine.update_layer_filter("0,1,last")
            dumper.filter_engine.update_name_filter("attention|mlp")
            dumper.filter_engine.update_iteration_filter("0,every:100")

            dumper.on_training_start()

            # Read metadata file
            metadata_path = os.path.join(
                tmpdir, dumper.session_name, "session_metadata.json"
            )
            assert os.path.exists(metadata_path)

            with open(metadata_path) as f:
                metadata = json.load(f)

            assert "filter_config" in metadata
            filter_config = metadata["filter_config"]
            assert filter_config.get("layer_filter") == "0,1,last"
            assert filter_config.get("name_filter") == "attention|mlp"
            assert "iteration_rules" in filter_config

    def test_session_metadata_includes_phase3_config(self):
        """Test that session metadata includes Phase 3 config options."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dumper.enable = True
            dumper.dump_dir = tmpdir
            dumper.aggregate_tp = True
            dumper.dump_gradients_enabled = True

            dumper.on_training_start()

            metadata_path = os.path.join(
                tmpdir, dumper.session_name, "session_metadata.json"
            )
            with open(metadata_path) as f:
                metadata = json.load(f)

            dumper_config = metadata["dumper_config"]
            assert dumper_config.get("aggregate_tp") is True
            assert dumper_config.get("dump_gradients") is True


class TestResetWithHooks:
    """Tests for reset behavior with hooks."""

    def test_reset_removes_hooks(self):
        """Test that reset() removes all hooks."""
        dumper.enable = True
        model = SimpleModel(num_layers=2)

        dumper.register_transformer_hooks(
            model,
            layer_class=SimpleTransformerLayer,
        )
        assert len(dumper._hook_handles) > 0

        dumper.reset()
        assert len(dumper._hook_handles) == 0


class TestPhase3Integration:
    """Integration tests for Phase 3 features."""

    def test_full_training_loop_simulation(self):
        """Test a simulated training loop with all Phase 3 features."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Configure dumper
            dumper.enable = True
            dumper.write_file = True
            dumper.dump_dir = tmpdir
            dumper.dump_dp_rank_0_only = False
            dumper.dump_gradients_enabled = True

            # Start session
            dumper.on_training_start()

            # Create model
            model = SimpleModel(num_layers=2, hidden_size=32)

            # Register hooks
            num_hooks = dumper.register_transformer_hooks(
                model,
                layer_class=SimpleTransformerLayer,
                dump_points=["post_attention", "post_mlp"],
            )
            assert num_hooks == 4  # 2 layers * 2 hooks

            # Simulate training iteration
            dumper.on_iteration_start(0)

            # Forward pass (hooks capture tensors)
            x = torch.randn(2, 32)
            y = model(x)

            # Backward pass
            y.sum().backward()

            # Dump gradients
            for i, layer in enumerate(model.layers):
                num_grads = dumper.dump_gradients(layer, f"layer_{i}", layer_id=i)
                assert num_grads > 0

            # Verify files created
            dumper.flush()
            session_dir = os.path.join(tmpdir, dumper.session_name, "iter_000000")
            assert os.path.exists(session_dir)
            files = os.listdir(session_dir)
            # 4 hook dumps + gradient dumps
            assert len(files) > 4

            # Clean up
            dumper.remove_all_hooks()
            assert len(dumper._hook_handles) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
