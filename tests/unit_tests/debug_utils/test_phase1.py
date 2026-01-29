# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""
Unit tests for Megatron Debug Tensor Dumper - Phase 1 components.

Tests cover:
- ParallelAdapter functionality and graceful degradation
- TensorShardingType and TENSOR_SHARDING_MAP
- Utility functions (filename parsing, logging, etc.)
"""

import os
import tempfile
from contextlib import ExitStack
from unittest import mock

import pytest
import torch

from megatron.core.debug_utils import (
    TENSOR_SHARDING_MAP,
    ParallelAdapter,
    TensorShardingType,
    build_dump_filename,
    configure_logging,
    format_tensor_info,
    get_logger,
    get_parallel_adapter,
    get_sharding_type,
    get_tensor_stats,
    parse_dump_filename,
)


class TestTensorShardingType:
    """Tests for TensorShardingType enum and related functions."""

    def test_enum_values(self):
        """Test that all expected enum values exist."""
        assert TensorShardingType.REPLICATED.value == "replicated"
        assert TensorShardingType.TP_COLUMN.value == "tp_column"
        assert TensorShardingType.TP_ROW.value == "tp_row"
        assert TensorShardingType.CP_SPLIT.value == "cp_split"
        assert TensorShardingType.EP_SPLIT.value == "ep_split"

    def test_tensor_sharding_map_structure(self):
        """Test that TENSOR_SHARDING_MAP has expected structure."""
        assert isinstance(TENSOR_SHARDING_MAP, dict)

        # Check some expected mappings
        assert "self_attention.query" in TENSOR_SHARDING_MAP
        assert "self_attention.output" in TENSOR_SHARDING_MAP
        assert "mlp.output" in TENSOR_SHARDING_MAP

        # Verify types
        for key, value in TENSOR_SHARDING_MAP.items():
            assert isinstance(key, str)
            assert isinstance(value, TensorShardingType)

    def test_get_sharding_type_exact_match(self):
        """Test get_sharding_type with exact name match."""
        assert get_sharding_type("self_attention.query") == TensorShardingType.TP_COLUMN
        assert get_sharding_type("self_attention.output") == TensorShardingType.REPLICATED
        assert get_sharding_type("mlp.output") == TensorShardingType.REPLICATED

    def test_get_sharding_type_suffix_match(self):
        """Test get_sharding_type with suffix matching."""
        # Should match "self_attention.query" via suffix
        assert get_sharding_type("layer_0.self_attention.query") == TensorShardingType.TP_COLUMN
        assert get_sharding_type("decoder.layer_5.mlp.output") == TensorShardingType.REPLICATED

    def test_get_sharding_type_default(self):
        """Test get_sharding_type returns REPLICATED for unknown names."""
        assert get_sharding_type("unknown_tensor") == TensorShardingType.REPLICATED
        assert get_sharding_type("some.random.name") == TensorShardingType.REPLICATED


class TestParallelAdapterNonDistributed:
    """Tests for ParallelAdapter in non-distributed environment."""

    def test_is_initialized_without_megatron(self):
        """Test is_initialized returns False when Megatron not initialized."""
        adapter = ParallelAdapter()
        # In test environment without distributed setup, should return False
        assert adapter.is_initialized() is False

    def test_get_parallel_info_graceful_degradation(self):
        """Test get_parallel_info returns defaults when not initialized."""
        adapter = ParallelAdapter()
        info = adapter.get_parallel_info()

        # Should return default values
        assert info["tp_rank"] == 0
        assert info["tp_size"] == 1
        assert info["pp_rank"] == 0
        assert info["pp_size"] == 1
        assert info["dp_rank"] == 0
        assert info["dp_size"] == 1
        assert info["cp_rank"] == 0
        assert info["cp_size"] == 1
        assert info["ep_rank"] == 0
        assert info["ep_size"] == 1
        assert info["vpp_rank"] == 0
        assert info["layer_offset"] == 0

    def test_should_dump_for_dp_non_distributed(self):
        """Test should_dump_for_dp returns True when not distributed."""
        adapter = ParallelAdapter()
        assert adapter.should_dump_for_dp() is True
        assert adapter.should_dump_for_dp(dump_all_dp_ranks=True) is True

    def test_get_tp_group_non_distributed(self):
        """Test get_tp_group returns None when not initialized."""
        adapter = ParallelAdapter()
        assert adapter.get_tp_group() is None

    def test_get_dp_group_non_distributed(self):
        """Test get_dp_group returns None when not initialized."""
        adapter = ParallelAdapter()
        assert adapter.get_dp_group() is None

    def test_get_pp_group_non_distributed(self):
        """Test get_pp_group returns None when not initialized."""
        adapter = ParallelAdapter()
        assert adapter.get_pp_group() is None

    def test_gather_across_tp_non_distributed(self):
        """Test gather_across_tp returns input tensor when not distributed."""
        adapter = ParallelAdapter()
        tensor = torch.randn(2, 4, 8)
        result = adapter.gather_across_tp(tensor)

        assert result is tensor  # Should return same tensor

    def test_reduce_across_tp_non_distributed(self):
        """Test reduce_across_tp returns input tensor when not distributed."""
        adapter = ParallelAdapter()
        tensor = torch.randn(2, 4, 8)
        result = adapter.reduce_across_tp(tensor)

        assert result is tensor  # Should return same tensor

    def test_reduce_across_tp_invalid_op(self):
        """Test reduce_across_tp rejects unsupported ops."""
        adapter = ParallelAdapter()
        tensor = torch.randn(2, 4, 8)

        with pytest.raises(ValueError, match="Unsupported reduction op"):
            adapter.reduce_across_tp(tensor, op="median")

    def test_set_num_layers(self):
        """Test set_num_layers method."""
        adapter = ParallelAdapter()
        adapter.set_num_layers(32)
        assert adapter._num_layers == 32

    def test_cache_behavior(self):
        """Test parallel info caching behavior."""
        adapter = ParallelAdapter()

        # First call without cache
        info1 = adapter.get_parallel_info(use_cache=False)

        # Second call with cache should still work
        info2 = adapter.get_parallel_info(use_cache=True)

        assert info1 == info2

        # Clear cache
        adapter.clear_cache()
        assert adapter._parallel_info_cache is None


class TestParallelAdapterSingleton:
    """Tests for get_parallel_adapter singleton."""

    def test_singleton_returns_same_instance(self):
        """Test get_parallel_adapter returns the same instance."""
        adapter1 = get_parallel_adapter()
        adapter2 = get_parallel_adapter()
        assert adapter1 is adapter2


class TestFilenameUtils:
    """Tests for dump filename utilities."""

    def test_build_dump_filename_basic(self):
        """Test basic filename building."""
        filename = build_dump_filename(
            name="hidden",
            iteration=0,
            micro_batch_id=0,
            global_rank=0,
            tp_rank=0,
            pp_rank=0,
            dump_index=1,
        )

        assert filename == "name=hidden___iter=0___mb=0___grank=0___tprank=0___pprank=0___idx=1.pt"

    def test_build_dump_filename_with_layer(self):
        """Test filename building with layer_id."""
        filename = build_dump_filename(
            name="attention_output",
            iteration=100,
            micro_batch_id=2,
            global_rank=4,
            tp_rank=1,
            pp_rank=0,
            dump_index=5,
            layer_id=10,
        )

        assert "name=attention_output" in filename
        assert "iter=100" in filename
        assert "layer=10" in filename
        assert filename.endswith(".pt")

    def test_build_dump_filename_with_extra_fields(self):
        """Test filename building with extra fields."""
        filename = build_dump_filename(
            name="hidden",
            iteration=0,
            micro_batch_id=0,
            global_rank=0,
            tp_rank=0,
            pp_rank=0,
            dump_index=1,
            phase="validation",
            custom_key="custom_value",
        )

        assert "phase=validation" in filename
        assert "custom_key=custom_value" in filename

    def test_parse_dump_filename_basic(self):
        """Test basic filename parsing."""
        filename = "name=hidden___iter=0___mb=0___grank=0___tprank=0___pprank=0___idx=1.pt"
        result = parse_dump_filename(filename)

        assert result["name"] == "hidden"
        assert result["iter"] == 0
        assert result["mb"] == 0
        assert result["grank"] == 0
        assert result["tprank"] == 0
        assert result["pprank"] == 0
        assert result["idx"] == 1

    def test_parse_dump_filename_with_path(self):
        """Test filename parsing with directory path."""
        filename = "/data/dumps/session/iter_000100/name=hidden___iter=100___mb=0___grank=0___tprank=0___pprank=0___idx=1.pt"
        result = parse_dump_filename(filename)

        assert result["name"] == "hidden"
        assert result["iter"] == 100

    def test_parse_dump_filename_with_layer(self):
        """Test filename parsing with layer field."""
        filename = "name=attention___iter=0___mb=0___grank=0___tprank=0___pprank=0___idx=1___layer=5.pt"
        result = parse_dump_filename(filename)

        assert result["layer"] == 5

    def test_parse_and_build_roundtrip(self):
        """Test that build and parse are inverse operations."""
        original = build_dump_filename(
            name="test_tensor",
            iteration=50,
            micro_batch_id=1,
            global_rank=8,
            tp_rank=2,
            pp_rank=1,
            dump_index=3,
            layer_id=7,
        )

        parsed = parse_dump_filename(original)

        assert parsed["name"] == "test_tensor"
        assert parsed["iter"] == 50
        assert parsed["mb"] == 1
        assert parsed["grank"] == 8
        assert parsed["tprank"] == 2
        assert parsed["pprank"] == 1
        assert parsed["idx"] == 3
        assert parsed["layer"] == 7

    def test_parse_dump_filename_extra_fields(self):
        """Test that extra fields remain as strings."""
        filename = (
            "name=hidden___iter=1___mb=0___grank=0___tprank=0___pprank=0___idx=1___"
            "phase=validation___tag=exp1.pt"
        )
        result = parse_dump_filename(filename)

        assert result["phase"] == "validation"
        assert result["tag"] == "exp1"


class TestTensorStats:
    """Tests for tensor statistics functions."""

    def test_get_tensor_stats_basic(self):
        """Test basic tensor statistics."""
        tensor = torch.randn(10, 20)
        stats = get_tensor_stats(tensor)

        assert stats["shape"] == [10, 20]
        assert "torch.float" in stats["dtype"]
        assert stats["numel"] == 200
        assert "min" in stats
        assert "max" in stats
        assert "mean" in stats
        assert "std" in stats
        assert "has_nan" in stats
        assert "has_inf" in stats

    def test_get_tensor_stats_empty_tensor(self):
        """Test tensor statistics for empty tensor."""
        tensor = torch.randn(0)
        stats = get_tensor_stats(tensor)

        assert stats["shape"] == [0]
        assert stats["numel"] == 0

    def test_get_tensor_stats_integer_tensor(self):
        """Test tensor statistics for integer tensor."""
        tensor = torch.randint(0, 100, (5, 5))
        stats = get_tensor_stats(tensor)

        assert stats["shape"] == [5, 5]
        assert "min" in stats
        assert "max" in stats

    def test_get_tensor_stats_nan_detection(self):
        """Test NaN detection in tensor statistics."""
        tensor = torch.tensor([1.0, float("nan"), 3.0])
        stats = get_tensor_stats(tensor)

        assert stats["has_nan"] is True

    def test_get_tensor_stats_inf_detection(self):
        """Test Inf detection in tensor statistics."""
        tensor = torch.tensor([1.0, float("inf"), 3.0])
        stats = get_tensor_stats(tensor)

        assert stats["has_inf"] is True

    def test_format_tensor_info(self):
        """Test tensor info formatting."""
        tensor = torch.randn(2, 4, 8)
        info = format_tensor_info("test_tensor", tensor, "/path/to/file.pt")

        assert "[Dump] test_tensor:" in info
        assert "shape=[2, 4, 8]" in info
        assert "dtype=" in info
        assert "/path/to/file.pt" in info

    def test_format_tensor_info_with_nan(self):
        """Test tensor info formatting with NaN."""
        tensor = torch.tensor([1.0, float("nan"), 3.0])
        info = format_tensor_info("nan_tensor", tensor)

        assert "[NaN DETECTED]" in info


class TestLogging:
    """Tests for logging configuration."""

    def test_get_logger(self):
        """Test get_logger returns a logger instance."""
        logger = get_logger()
        assert logger is not None
        assert logger.name == "megatron.debug_utils"

    def test_get_logger_singleton(self):
        """Test get_logger returns same instance."""
        logger1 = get_logger()
        logger2 = get_logger()
        assert logger1 is logger2

    def test_configure_logging_with_level(self):
        """Test configure_logging with custom level."""
        import logging

        logger = configure_logging(level=logging.DEBUG)
        assert logger.level == logging.DEBUG

    def test_configure_logging_with_rank(self):
        """Test configure_logging with rank prefix."""
        import logging

        logger = configure_logging(level=logging.INFO, rank=5)

        # Check that handler formatter includes rank
        assert len(logger.handlers) > 0
        formatter = logger.handlers[0].formatter
        assert "Rank 5" in formatter._fmt

    def test_configure_logging_with_file(self):
        """Test configure_logging with file output."""
        import logging

        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = os.path.join(tmpdir, "test.log")
            logger = configure_logging(level=logging.INFO, log_file=log_file)

            # Log something
            logger.info("Test message")

            # Check file exists and has content
            assert os.path.exists(log_file)

    def test_configure_logging_with_file_in_cwd(self, tmp_path, monkeypatch):
        """Test configure_logging with a file path in the current directory."""
        import logging

        monkeypatch.chdir(tmp_path)
        log_file = "test.log"
        logger = configure_logging(level=logging.INFO, log_file=log_file)

        logger.info("Test message")

        assert (tmp_path / log_file).exists()


class TestParallelAdapterWithMock:
    """Tests for ParallelAdapter with mocked Megatron parallel_state."""

    def test_get_parallel_info_with_mock(self):
        """Test get_parallel_info with mocked parallel_state."""
        adapter = ParallelAdapter(num_layers=16)

        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(adapter, "is_initialized", return_value=True))
            stack.enter_context(
                mock.patch(
                    "megatron.core.debug_utils.parallel_adapter.get_global_rank",
                    return_value=10,
                )
            )
            stack.enter_context(
                mock.patch(
                    "megatron.core.debug_utils.parallel_adapter.get_world_size",
                    return_value=64,
                )
            )
            stack.enter_context(
                mock.patch(
                    "megatron.core.parallel_state.get_data_parallel_rank",
                    return_value=2,
                )
            )
            stack.enter_context(
                mock.patch(
                    "megatron.core.parallel_state.get_data_parallel_world_size",
                    return_value=8,
                )
            )
            stack.enter_context(
                mock.patch(
                    "megatron.core.parallel_state.get_tensor_model_parallel_rank",
                    return_value=1,
                )
            )
            stack.enter_context(
                mock.patch(
                    "megatron.core.parallel_state.get_tensor_model_parallel_world_size",
                    return_value=4,
                )
            )
            stack.enter_context(
                mock.patch(
                    "megatron.core.parallel_state.get_pipeline_model_parallel_rank",
                    return_value=1,
                )
            )
            stack.enter_context(
                mock.patch(
                    "megatron.core.parallel_state.get_pipeline_model_parallel_world_size",
                    return_value=2,
                )
            )
            stack.enter_context(
                mock.patch(
                    "megatron.core.parallel_state.get_context_parallel_rank",
                    return_value=0,
                )
            )
            stack.enter_context(
                mock.patch(
                    "megatron.core.parallel_state.get_context_parallel_world_size",
                    return_value=1,
                )
            )
            stack.enter_context(
                mock.patch(
                    "megatron.core.parallel_state.get_expert_model_parallel_rank",
                    return_value=0,
                )
            )
            stack.enter_context(
                mock.patch(
                    "megatron.core.parallel_state.get_expert_model_parallel_world_size",
                    return_value=1,
                )
            )
            stack.enter_context(
                mock.patch(
                    "megatron.core.parallel_state.get_virtual_pipeline_model_parallel_rank",
                    return_value=0,
                )
            )
            info = adapter.get_parallel_info()

        assert info["global_rank"] == 10
        assert info["world_size"] == 64
        assert info["dp_rank"] == 2
        assert info["dp_size"] == 8
        assert info["tp_rank"] == 1
        assert info["tp_size"] == 4
        assert info["pp_rank"] == 1
        assert info["pp_size"] == 2
        assert info["cp_rank"] == 0
        assert info["cp_size"] == 1
        assert info["ep_rank"] == 0
        assert info["ep_size"] == 1
        assert info["vpp_rank"] == 0
        assert info["layer_offset"] == 8

    def test_should_dump_for_dp_with_mock(self):
        """Test should_dump_for_dp with mocked parallel_state."""
        adapter = ParallelAdapter()

        # Test case: when not initialized, should return True (default)
        with mock.patch.object(adapter, "is_initialized", return_value=False):
            assert adapter.should_dump_for_dp() is True

        # Test case: dump_all_dp_ranks=True always returns True
        assert adapter.should_dump_for_dp(dump_all_dp_ranks=True) is True

    def test_layer_offset_calculation(self):
        """Test layer offset calculation logic."""
        adapter = ParallelAdapter(num_layers=32)

        # Without initialization, should return 0
        offset = adapter._get_layer_offset()
        assert offset == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
