# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""
Unit tests for Megatron Debug Tensor Dumper - Phase 2 components.

Tests cover:
- FilterEngine: layer/name/iteration filtering
- StorageBackend: sync/async file I/O
- MegatronDumper: core dump functionality
"""

import os
import tempfile
import threading
from unittest import mock

import pytest
import torch

from megatron.core.debug_utils import (
    FilterEngine,
    StorageBackend,
    dumper,
)


class TestFilterEngine:
    """Tests for FilterEngine."""

    def test_no_filters(self):
        """Test that no filters means everything passes."""
        engine = FilterEngine()
        assert engine.should_dump("any_name", layer_id=5, iteration=100)
        assert engine.should_dump("another_name", layer_id=None, iteration=0)

    def test_layer_filter_single(self):
        """Test single layer filtering."""
        engine = FilterEngine(layer_filter="0")
        assert engine.should_dump("name", layer_id=0, iteration=0)
        assert not engine.should_dump("name", layer_id=1, iteration=0)
        assert engine.should_dump("name", layer_id=None, iteration=0)  # No layer_id passes

    def test_layer_filter_multiple(self):
        """Test multiple layer filtering."""
        engine = FilterEngine(layer_filter="0,1,5")
        assert engine.should_dump("name", layer_id=0, iteration=0)
        assert engine.should_dump("name", layer_id=1, iteration=0)
        assert engine.should_dump("name", layer_id=5, iteration=0)
        assert not engine.should_dump("name", layer_id=2, iteration=0)
        assert not engine.should_dump("name", layer_id=10, iteration=0)

    def test_layer_filter_range(self):
        """Test layer range filtering."""
        engine = FilterEngine(layer_filter="5-10")
        assert not engine.should_dump("name", layer_id=4, iteration=0)
        assert engine.should_dump("name", layer_id=5, iteration=0)
        assert engine.should_dump("name", layer_id=7, iteration=0)
        assert engine.should_dump("name", layer_id=10, iteration=0)
        assert not engine.should_dump("name", layer_id=11, iteration=0)

    def test_layer_filter_mixed(self):
        """Test mixed layer filtering."""
        engine = FilterEngine(layer_filter="0,5-7,15")
        assert engine.should_dump("name", layer_id=0, iteration=0)
        assert not engine.should_dump("name", layer_id=3, iteration=0)
        assert engine.should_dump("name", layer_id=5, iteration=0)
        assert engine.should_dump("name", layer_id=6, iteration=0)
        assert engine.should_dump("name", layer_id=7, iteration=0)
        assert not engine.should_dump("name", layer_id=8, iteration=0)
        assert engine.should_dump("name", layer_id=15, iteration=0)

    def test_layer_filter_first_last(self):
        """Test first/last layer filtering."""
        engine = FilterEngine(layer_filter="first,last", num_layers=32)
        assert engine.should_dump("name", layer_id=0, iteration=0)  # first
        assert engine.should_dump("name", layer_id=31, iteration=0)  # last
        assert not engine.should_dump("name", layer_id=15, iteration=0)

    def test_layer_filter_last_without_num_layers(self):
        """Test last filter without num_layers specified."""
        engine = FilterEngine(layer_filter="last")
        # Without num_layers, "last" can't be resolved
        assert not engine.should_dump("name", layer_id=31, iteration=0)

        # Set num_layers later
        engine.set_num_layers(32)
        assert engine.should_dump("name", layer_id=31, iteration=0)

    def test_name_filter_simple(self):
        """Test simple name filtering."""
        engine = FilterEngine(name_filter="attention")
        assert engine.should_dump("attention_output", layer_id=0, iteration=0)
        assert engine.should_dump("layer_0.attention.query", layer_id=0, iteration=0)
        assert not engine.should_dump("mlp_output", layer_id=0, iteration=0)

    def test_name_filter_regex(self):
        """Test regex name filtering."""
        engine = FilterEngine(name_filter="attention|mlp")
        assert engine.should_dump("attention_output", layer_id=0, iteration=0)
        assert engine.should_dump("mlp_output", layer_id=0, iteration=0)
        assert not engine.should_dump("layernorm_output", layer_id=0, iteration=0)

    def test_name_filter_complex_regex(self):
        """Test complex regex name filtering."""
        engine = FilterEngine(name_filter=r"layer_\d+\.attention")
        assert engine.should_dump("layer_0.attention.query", layer_id=0, iteration=0)
        assert engine.should_dump("layer_15.attention.output", layer_id=15, iteration=0)
        assert not engine.should_dump("layer_0.mlp.output", layer_id=0, iteration=0)

    def test_iteration_filter_specific(self):
        """Test specific iteration filtering."""
        engine = FilterEngine(iteration_filter="0,100,200")
        assert engine.should_dump("name", layer_id=0, iteration=0)
        assert engine.should_dump("name", layer_id=0, iteration=100)
        assert engine.should_dump("name", layer_id=0, iteration=200)
        assert not engine.should_dump("name", layer_id=0, iteration=50)
        assert not engine.should_dump("name", layer_id=0, iteration=150)

    def test_iteration_filter_range(self):
        """Test iteration range filtering."""
        engine = FilterEngine(iteration_filter="100-200")
        assert not engine.should_dump("name", layer_id=0, iteration=99)
        assert engine.should_dump("name", layer_id=0, iteration=100)
        assert engine.should_dump("name", layer_id=0, iteration=150)
        assert engine.should_dump("name", layer_id=0, iteration=200)
        assert not engine.should_dump("name", layer_id=0, iteration=201)

    def test_iteration_filter_every(self):
        """Test periodic iteration filtering."""
        engine = FilterEngine(iteration_filter="every:100")
        assert engine.should_dump("name", layer_id=0, iteration=0)
        assert engine.should_dump("name", layer_id=0, iteration=100)
        assert engine.should_dump("name", layer_id=0, iteration=200)
        assert not engine.should_dump("name", layer_id=0, iteration=50)
        assert not engine.should_dump("name", layer_id=0, iteration=150)

    def test_iteration_filter_first(self):
        """Test first N iterations filtering."""
        engine = FilterEngine(iteration_filter="first:10")
        assert engine.should_dump("name", layer_id=0, iteration=0)
        assert engine.should_dump("name", layer_id=0, iteration=5)
        assert engine.should_dump("name", layer_id=0, iteration=9)
        assert not engine.should_dump("name", layer_id=0, iteration=10)
        assert not engine.should_dump("name", layer_id=0, iteration=100)

    def test_iteration_filter_combined(self):
        """Test combined iteration filtering."""
        engine = FilterEngine(iteration_filter="0,every:1000,first:5")
        assert engine.should_dump("name", layer_id=0, iteration=0)  # specific + first
        assert engine.should_dump("name", layer_id=0, iteration=3)  # first
        assert not engine.should_dump("name", layer_id=0, iteration=10)
        assert engine.should_dump("name", layer_id=0, iteration=1000)  # every
        assert engine.should_dump("name", layer_id=0, iteration=2000)  # every

    def test_combined_filters(self):
        """Test all filters combined."""
        engine = FilterEngine(
            layer_filter="0,1",
            name_filter="attention",
            iteration_filter="0,100",
        )
        # All conditions met
        assert engine.should_dump("attention_output", layer_id=0, iteration=0)
        assert engine.should_dump("attention_output", layer_id=1, iteration=100)

        # Layer fails
        assert not engine.should_dump("attention_output", layer_id=5, iteration=0)

        # Name fails
        assert not engine.should_dump("mlp_output", layer_id=0, iteration=0)

        # Iteration fails
        assert not engine.should_dump("attention_output", layer_id=0, iteration=50)

    def test_update_filters(self):
        """Test dynamic filter updates."""
        engine = FilterEngine()
        assert engine.should_dump("any", layer_id=5, iteration=50)

        engine.update_layer_filter("0,1")
        assert not engine.should_dump("any", layer_id=5, iteration=50)
        assert engine.should_dump("any", layer_id=0, iteration=50)

        engine.update_name_filter("mlp")
        assert not engine.should_dump("attention", layer_id=0, iteration=50)
        assert engine.should_dump("mlp_output", layer_id=0, iteration=50)

        engine.update_iteration_filter("0")
        assert not engine.should_dump("mlp_output", layer_id=0, iteration=50)
        assert engine.should_dump("mlp_output", layer_id=0, iteration=0)

    def test_repr(self):
        """Test FilterEngine repr."""
        engine = FilterEngine(
            layer_filter="0,1",
            name_filter="attention",
            iteration_filter="every:100",
        )
        repr_str = repr(engine)
        assert "FilterEngine" in repr_str
        assert "0,1" in repr_str
        assert "attention" in repr_str


class TestStorageBackend:
    """Tests for StorageBackend."""

    def test_sync_save_load(self):
        """Test synchronous save and load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = StorageBackend(tmpdir, async_write=False)

            tensor = torch.randn(10, 20)
            metadata = {"name": "test", "iteration": 0}
            filepath = os.path.join(tmpdir, "test.pt")

            storage.save(tensor, filepath, metadata)

            assert os.path.exists(filepath)

            loaded = storage.load(filepath)
            assert "data" in loaded
            assert "metadata" in loaded
            assert torch.allclose(loaded["data"], tensor)
            assert loaded["metadata"]["name"] == "test"

    def test_async_save(self):
        """Test asynchronous save."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = StorageBackend(tmpdir, async_write=True)

            tensor = torch.randn(10, 20)
            filepath = os.path.join(tmpdir, "test.pt")

            storage.save(tensor, filepath, {"name": "test"})
            storage.flush()

            assert os.path.exists(filepath)

            loaded = storage.load(filepath)
            assert torch.allclose(loaded["data"], tensor)

            storage.stop()

    def test_async_multiple_saves(self):
        """Test multiple async saves."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = StorageBackend(tmpdir, async_write=True)

            for i in range(5):
                tensor = torch.randn(5, 5)
                filepath = os.path.join(tmpdir, f"tensor_{i}.pt")
                storage.save(tensor, filepath, {"index": i})

            storage.flush()

            for i in range(5):
                filepath = os.path.join(tmpdir, f"tensor_{i}.pt")
                assert os.path.exists(filepath)
                loaded = storage.load(filepath)
                assert loaded["metadata"]["index"] == i

            storage.stop()

    def test_save_creates_directories(self):
        """Test that save creates necessary directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = StorageBackend(tmpdir, async_write=False)

            filepath = os.path.join(tmpdir, "subdir1", "subdir2", "test.pt")
            storage.save(torch.randn(5), filepath)

            assert os.path.exists(filepath)

    def test_pending_writes(self):
        """Test pending_writes property."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = StorageBackend(tmpdir, async_write=True)

            # Initially no pending writes
            assert storage.pending_writes == 0

            storage.stop()

    def test_flush_timeout(self):
        """Test flush timeout behavior."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = StorageBackend(tmpdir, async_write=True)
            block_event = threading.Event()
            filepath = os.path.join(tmpdir, "blocked.pt")

            def blocked_write(payload, path):
                block_event.wait()
                os.makedirs(os.path.dirname(path), exist_ok=True)
                torch.save(payload, path)

            try:
                with mock.patch.object(storage, "_write_sync", side_effect=blocked_write):
                    storage.save(torch.randn(2, 2), filepath)
                    assert storage.pending_writes == 1
                    assert storage.flush(timeout=0.05) is False
                    block_event.set()
                    assert storage.flush(timeout=1.0) is True
            finally:
                block_event.set()
                storage.stop()

    def test_exists(self):
        """Test exists method."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = StorageBackend(tmpdir)

            filepath = os.path.join(tmpdir, "test.pt")
            assert not storage.exists(filepath)

            storage.save(torch.randn(5), filepath)
            assert storage.exists(filepath)

    def test_list_files(self):
        """Test list_files method."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = StorageBackend(tmpdir)

            # Create some files
            for i in range(3):
                storage.save(torch.randn(5), os.path.join(tmpdir, f"file_{i}.pt"))

            files = storage.list_files(tmpdir, "*.pt")
            assert len(files) == 3


class TestMegatronDumper:
    """Tests for MegatronDumper."""

    @pytest.fixture(autouse=True)
    def reset_dumper(self):
        """Reset dumper state before each test."""
        dumper.reset()
        original_enable = dumper.enable
        original_write_file = dumper.write_file
        yield
        dumper.reset()
        dumper.enable = original_enable
        dumper.write_file = original_write_file

    def test_disabled_by_default(self):
        """Test that dumper is disabled by default."""
        assert not dumper.enable

    def test_dump_when_disabled(self):
        """Test that dump does nothing when disabled."""
        dumper.enable = False
        # Should not raise, just return early
        dumper.dump("test", torch.randn(5))

    def test_basic_dump(self):
        """Test basic tensor dump."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dumper.enable = True
            dumper.write_file = True
            dumper.dump_dir = tmpdir
            dumper.dump_dp_rank_0_only = False

            dumper._session_name = "test_session"
            dumper.on_iteration_start(0)

            tensor = torch.randn(5, 10)
            dumper.dump("test_tensor", tensor)

            # Check file was created
            session_dir = os.path.join(tmpdir, "test_session", "iter_000000")
            files = list(os.listdir(session_dir)) if os.path.exists(session_dir) else []
            assert len(files) == 1
            assert "test_tensor" in files[0]

    def test_dump_with_context(self):
        """Test dump with context manager."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dumper.enable = True
            dumper.write_file = True
            dumper.dump_dir = tmpdir
            dumper.dump_dp_rank_0_only = False

            dumper._session_name = "test_session"
            dumper.on_iteration_start(0)

            with dumper.context(layer_id=5, phase="forward"):
                assert dumper._ctx.get("layer_id") == 5
                assert dumper._ctx.get("phase") == "forward"

            # Context should be restored
            assert dumper._ctx.get("layer_id") is None
            assert dumper._ctx.get("phase") is None

    def test_set_ctx(self):
        """Test set_ctx method."""
        dumper.set_ctx(layer_id=3, custom_key="value")
        assert dumper._ctx["layer_id"] == 3
        assert dumper._ctx["custom_key"] == "value"

        dumper.set_ctx(layer_id=None)  # Clear
        assert "layer_id" not in dumper._ctx
        assert dumper._ctx["custom_key"] == "value"

        dumper.clear_ctx()
        assert len(dumper._ctx) == 0

    def test_lifecycle_methods(self):
        """Test lifecycle methods."""
        dumper.enable = True

        dumper.on_iteration_start(100)
        assert dumper.iteration == 100
        assert dumper.micro_batch_id == 0

        dumper.on_micro_batch_start(2)
        assert dumper.micro_batch_id == 2

    def test_dump_dict(self):
        """Test dump_dict method."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dumper.enable = True
            dumper.write_file = True
            dumper.dump_dir = tmpdir
            dumper.dump_dp_rank_0_only = False

            dumper._session_name = "test_session"
            dumper.on_iteration_start(0)

            data = {
                "query": torch.randn(2, 4),
                "key": torch.randn(2, 4),
                "value": torch.randn(2, 4),
                "non_tensor": "skip_this",
            }
            dumper.dump_dict("attention", data)

            session_dir = os.path.join(tmpdir, "test_session", "iter_000000")
            files = os.listdir(session_dir) if os.path.exists(session_dir) else []
            # Should have 3 files (query, key, value)
            assert len(files) == 3

    def test_filter_integration(self):
        """Test filter integration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dumper.enable = True
            dumper.write_file = True
            dumper.dump_dir = tmpdir
            dumper.dump_dp_rank_0_only = False

            # Set up filter
            dumper.filter_engine.update_layer_filter("0")
            dumper.filter_engine.update_name_filter("attention")

            dumper._session_name = "test_session"
            dumper.on_iteration_start(0)

            # This should pass filter
            dumper.dump("attention_output", torch.randn(5), layer_id=0)
            # This should fail layer filter
            dumper.dump("attention_output", torch.randn(5), layer_id=5)
            # This should fail name filter
            dumper.dump("mlp_output", torch.randn(5), layer_id=0)

            session_dir = os.path.join(tmpdir, "test_session", "iter_000000")
            files = os.listdir(session_dir) if os.path.exists(session_dir) else []
            # Only first dump should pass
            assert len(files) == 1

    def test_on_training_start(self):
        """Test on_training_start creates session."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dumper.enable = True
            dumper.dump_dir = tmpdir

            dumper.on_training_start()

            assert dumper.session_name is not None
            assert dumper.session_name.startswith("dump_")

    def test_save_false_logs_only(self):
        """Test dump with save=False only logs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dumper.enable = True
            dumper.write_file = True
            dumper.dump_dir = tmpdir
            dumper.dump_dp_rank_0_only = False

            dumper._session_name = "test_session"
            dumper.on_iteration_start(0)

            dumper.dump("test_tensor", torch.randn(5), save=False)

            session_dir = os.path.join(tmpdir, "test_session", "iter_000000")
            # No file should be created
            assert not os.path.exists(session_dir) or len(os.listdir(session_dir)) == 0


class TestDumperEnvironmentVariables:
    """Tests for dumper environment variable configuration."""

    def test_env_var_configuration(self):
        """Test that dumper reads environment variables."""
        # This tests the default values since we can't easily change env vars
        # during test without affecting the singleton
        assert isinstance(dumper.enable, bool)
        assert isinstance(dumper.write_file, bool)
        assert isinstance(dumper.dump_dir, str)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
