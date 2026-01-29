# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""
Multi-dimensional filter engine for Megatron Debug Tensor Dumper.

Supports filtering by:
- Layer number: "0,1,5-10,last"
- Tensor name: regex patterns like "attention|mlp"
- Iteration: "0,100", "100-200", "every:100", "first:10"
"""

import re
from typing import Any, Dict, FrozenSet, Optional, Set, Tuple


class FilterEngine:
    """
    Multi-dimensional filter engine for tensor dumping.

    Supports filtering by layer number, tensor name patterns, and iteration number.
    All filters are optional - if not specified, that dimension is not filtered.

    Example:
        >>> engine = FilterEngine(
        ...     layer_filter="0,1,last",
        ...     name_filter="attention|mlp",
        ...     iteration_filter="0,every:100"
        ... )
        >>> engine.should_dump("layer_0.attention_output", layer_id=0, iteration=0)
        True
        >>> engine.should_dump("layer_5.attention_output", layer_id=5, iteration=0)
        False
    """

    def __init__(
        self,
        layer_filter: Optional[str] = None,
        name_filter: Optional[str] = None,
        iteration_filter: Optional[str] = None,
        num_layers: Optional[int] = None,
    ):
        """
        Initialize the filter engine.

        Args:
            layer_filter: Layer number filter string (e.g., "0,1,5-10,last")
            name_filter: Tensor name regex pattern (e.g., "attention|mlp")
            iteration_filter: Iteration filter string (e.g., "0,every:100,first:10")
            num_layers: Total number of layers (required to resolve "last")
        """
        self._num_layers = num_layers
        self._layer_filter_str = layer_filter
        self._layer_set: Optional[FrozenSet[int]] = None
        self._has_last_marker = False

        if layer_filter:
            self._layer_set, self._has_last_marker = self._parse_layer_filter(layer_filter)

        self._name_pattern: Optional[re.Pattern] = None
        if name_filter:
            self._name_pattern = re.compile(name_filter)

        self._iteration_rules: Optional[Dict[str, Any]] = None
        if iteration_filter:
            self._iteration_rules = self._parse_iteration_filter(iteration_filter)

    def should_dump(
        self,
        name: str,
        layer_id: Optional[int] = None,
        iteration: int = 0,
    ) -> bool:
        """
        Determine if a tensor should be dumped based on all filter criteria.

        Args:
            name: Tensor name (e.g., "layer_0.attention_output")
            layer_id: Layer number (optional)
            iteration: Current training iteration

        Returns:
            True if the tensor passes all filters and should be dumped.
        """
        # Layer filter
        if not self._check_layer(layer_id):
            return False

        # Name filter
        if not self._check_name(name):
            return False

        # Iteration filter
        if not self._check_iteration(iteration):
            return False

        return True

    def _check_layer(self, layer_id: Optional[int]) -> bool:
        """Check if layer_id passes the layer filter."""
        if self._layer_set is None and not self._has_last_marker:
            return True  # No filter

        if layer_id is None:
            return True  # No layer_id provided, pass through

        # Check explicit layer set
        if self._layer_set is not None and layer_id in self._layer_set:
            return True

        # Check "last" marker
        if (
            self._has_last_marker
            and self._num_layers is not None
            and layer_id == self._num_layers - 1
        ):
            return True

        return False

    def _check_name(self, name: str) -> bool:
        """Check if name passes the name filter."""
        if self._name_pattern is None:
            return True  # No filter

        return self._name_pattern.search(name) is not None

    def _check_iteration(self, iteration: int) -> bool:
        """Check if iteration passes the iteration filter."""
        if self._iteration_rules is None:
            return True  # No filter

        rules = self._iteration_rules

        # Check specific iterations
        if iteration in rules["specific"]:
            return True

        # Check ranges
        for start, end in rules["ranges"]:
            if start <= iteration <= end:
                return True

        # Check periodic (every:N)
        if rules["every"] is not None and iteration % rules["every"] == 0:
            return True

        # Check first N iterations
        if rules["first"] is not None and iteration < rules["first"]:
            return True

        return False

    def _parse_layer_filter(
        self, filter_str: str
    ) -> Tuple[Optional[FrozenSet[int]], bool]:
        """
        Parse layer filter string.

        Format:
            - "0,1,2": specific layer numbers
            - "0-5": range (inclusive)
            - "0,5-10,20": mixed
            - "first": layer 0
            - "last": last layer (requires num_layers)
            - None or empty: no filter

        Args:
            filter_str: The filter string

        Returns:
            Tuple of (set of layer numbers, has_last_marker)
        """
        if not filter_str or not filter_str.strip():
            return None, False

        result: Set[int] = set()
        has_last = False

        for part in filter_str.split(","):
            part = part.strip().lower()

            if part == "first":
                result.add(0)
            elif part == "last":
                has_last = True
            elif "-" in part:
                # Range like "5-10"
                start_str, end_str = part.split("-", 1)
                start = int(start_str.strip())
                end = int(end_str.strip())
                result.update(range(start, end + 1))
            else:
                # Single number
                result.add(int(part))

        return frozenset(result) if result else None, has_last

    def _parse_iteration_filter(self, filter_str: str) -> Dict[str, Any]:
        """
        Parse iteration filter string.

        Format:
            - "0,100,200": specific iterations
            - "100-200": range (inclusive)
            - "every:100": every 100 iterations (0, 100, 200, ...)
            - "first:10": first 10 iterations (0-9)
            - Can combine: "0,every:1000,first:5"

        Args:
            filter_str: The filter string

        Returns:
            Dictionary with filter rules
        """
        rules: Dict[str, Any] = {
            "specific": set(),
            "ranges": [],
            "every": None,
            "first": None,
        }

        if not filter_str or not filter_str.strip():
            return rules

        for part in filter_str.split(","):
            part = part.strip().lower()

            if part.startswith("every:"):
                rules["every"] = int(part.split(":", 1)[1])
            elif part.startswith("first:"):
                rules["first"] = int(part.split(":", 1)[1])
            elif "-" in part:
                # Range like "100-200"
                start_str, end_str = part.split("-", 1)
                start = int(start_str.strip())
                end = int(end_str.strip())
                rules["ranges"].append((start, end))
            else:
                # Single iteration number
                rules["specific"].add(int(part))

        return rules

    def set_num_layers(self, num_layers: int) -> None:
        """
        Set the total number of layers (for resolving "last").

        Args:
            num_layers: Total number of layers in the model
        """
        self._num_layers = num_layers

    def update_layer_filter(self, layer_filter: Optional[str]) -> None:
        """
        Update the layer filter.

        Args:
            layer_filter: New layer filter string
        """
        self._layer_filter_str = layer_filter
        if layer_filter:
            self._layer_set, self._has_last_marker = self._parse_layer_filter(layer_filter)
        else:
            self._layer_set = None
            self._has_last_marker = False

    def update_name_filter(self, name_filter: Optional[str]) -> None:
        """
        Update the name filter.

        Args:
            name_filter: New name filter regex pattern
        """
        if name_filter:
            self._name_pattern = re.compile(name_filter)
        else:
            self._name_pattern = None

    def update_iteration_filter(self, iteration_filter: Optional[str]) -> None:
        """
        Update the iteration filter.

        Args:
            iteration_filter: New iteration filter string
        """
        if iteration_filter:
            self._iteration_rules = self._parse_iteration_filter(iteration_filter)
        else:
            self._iteration_rules = None

    @property
    def layer_set(self) -> Optional[FrozenSet[int]]:
        """Get the current layer filter set."""
        return self._layer_set

    @property
    def has_last_marker(self) -> bool:
        """Check if 'last' is in the layer filter."""
        return self._has_last_marker

    def __repr__(self) -> str:
        parts = []
        if self._layer_filter_str:
            parts.append(f"layers={self._layer_filter_str!r}")
        if self._name_pattern:
            parts.append(f"name={self._name_pattern.pattern!r}")
        if self._iteration_rules:
            parts.append(f"iterations={self._iteration_rules}")
        return f"FilterEngine({', '.join(parts)})"
