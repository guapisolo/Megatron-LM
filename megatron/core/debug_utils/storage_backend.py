# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""
Storage backend for Megatron Debug Tensor Dumper.

Provides file I/O operations with support for synchronous and asynchronous writing.
"""

import os
import queue
import threading
import time
from typing import Any, Dict, Optional

import torch

from .utils import get_logger


class StorageBackend:
    """
    Storage backend for tensor dump files.

    Supports both synchronous and asynchronous (background thread) writing.
    Files are saved in PyTorch's .pt format with data and metadata.

    Example:
        >>> storage = StorageBackend("/data/dumps", async_write=True)
        >>> storage.save(tensor, "/data/dumps/tensor.pt", {"name": "hidden"})
        >>> storage.flush()  # Wait for async writes to complete
    """

    def __init__(
        self,
        base_dir: str,
        async_write: bool = False,
    ):
        """
        Initialize the storage backend.

        Args:
            base_dir: Base directory for dump files
            async_write: If True, use background thread for writing
        """
        self.base_dir = base_dir
        self.async_write = async_write
        self._logger = get_logger()

        self._queue: Optional[queue.Queue] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event: Optional[threading.Event] = None
        self._pending = 0
        self._pending_cond = threading.Condition()

        if async_write:
            self._queue = queue.Queue()
            self._stop_event = threading.Event()
            self._thread = threading.Thread(
                target=self._writer_loop,
                daemon=True,
                name="DumperWriterThread",
            )
            self._thread.start()

    def save(
        self,
        data: Any,
        filepath: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Save data to a file.

        Args:
            data: Data to save (tensor or any picklable object)
            filepath: Full file path
            metadata: Optional metadata dictionary
        """
        payload = {"data": data}
        if metadata:
            payload["metadata"] = metadata

        if self.async_write:
            # Copy tensor to CPU for async write
            if isinstance(data, torch.Tensor):
                payload["data"] = data.detach().cpu().clone()
            with self._pending_cond:
                self._pending += 1
            self._queue.put((payload, filepath))
        else:
            self._write_sync(payload, filepath)

    def load(
        self,
        filepath: str,
        map_location: str = "cpu",
    ) -> Dict[str, Any]:
        """
        Load data from a file.

        Args:
            filepath: Full file path
            map_location: Device to load tensors to

        Returns:
            Dictionary containing "data" and optionally "metadata"
        """
        return torch.load(filepath, map_location=map_location, weights_only=False)

    def _write_sync(self, payload: Dict[str, Any], filepath: str) -> None:
        """
        Write payload to file synchronously.

        Args:
            payload: Data payload to write
            filepath: Target file path
        """
        try:
            dirpath = os.path.dirname(filepath)
            if dirpath:
                os.makedirs(dirpath, exist_ok=True)
            torch.save(payload, filepath)
        except Exception as e:
            self._logger.warning(f"Failed to write {filepath}: {e}")

    def _writer_loop(self) -> None:
        """Background writer thread main loop."""
        while not self._stop_event.is_set():
            try:
                payload, filepath = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self._write_sync(payload, filepath)
            except Exception as e:
                self._logger.warning(f"Async write failed: {e}")
            finally:
                self._queue.task_done()
                with self._pending_cond:
                    self._pending -= 1
                    if self._pending == 0:
                        self._pending_cond.notify_all()

    def flush(self, timeout: Optional[float] = None) -> bool:
        """
        Wait for all pending async writes to complete.

        Args:
            timeout: Maximum time to wait in seconds (None = wait forever)

        Returns:
            True if queue was flushed, False if timeout occurred
        """
        if not self.async_write or self._queue is None:
            return True

        if timeout is None:
            self._queue.join()
            return True

        deadline = time.monotonic() + timeout
        with self._pending_cond:
            while self._pending > 0:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._pending_cond.wait(timeout=remaining)
        return True

    def stop(self) -> None:
        """Stop the background writer thread."""
        if self._stop_event is not None:
            self._stop_event.set()

        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=5.0)

    def __del__(self):
        """Cleanup on deletion."""
        self.stop()

    @property
    def pending_writes(self) -> int:
        """Get number of pending async writes."""
        if self._queue is None:
            return 0
        with self._pending_cond:
            return self._pending

    def exists(self, filepath: str) -> bool:
        """Check if a file exists."""
        return os.path.exists(filepath)

    def list_files(self, directory: str, pattern: str = "*.pt") -> list:
        """
        List files in a directory matching a pattern.

        Args:
            directory: Directory to search
            pattern: Glob pattern (default: "*.pt")

        Returns:
            List of matching file paths
        """
        import glob
        return glob.glob(os.path.join(directory, pattern))
