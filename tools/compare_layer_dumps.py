#!/usr/bin/env python3
"""Compare Megatron debug dump tensors layer-by-layer between two dump roots."""

import argparse
import json
import math
import os
import re
from typing import Any, Dict, Iterable, List, Tuple

import torch

from megatron.core.debug_utils.utils import parse_dump_filename


def _parse_ignore_fields(value: str) -> List[str]:
    if not value:
        return []
    return [v.strip() for v in value.split(",") if v.strip()]


def _meta_matches(
    meta: Dict[str, Any],
    name_regex: str | None,
    iteration: int | None,
    micro_batch: int | None,
    layer: int | None,
) -> bool:
    if name_regex:
        name = str(meta.get("name", ""))
        if not re.search(name_regex, name):
            return False
    if iteration is not None and meta.get("iter") != iteration:
        return False
    if micro_batch is not None and meta.get("mb") != micro_batch:
        return False
    if layer is not None and meta.get("layer") != layer:
        return False
    return True


def _build_key(meta: Dict[str, Any], ignore_fields: Iterable[str]) -> Tuple[Tuple[str, Any], ...]:
    items = []
    ignore = set(ignore_fields)
    for key in sorted(meta.keys()):
        if key in ignore:
            continue
        items.append((key, meta[key]))
    return tuple(items)


def _collect_entries(
    root: str,
    ignore_fields: Iterable[str],
    name_regex: str | None,
    iteration: int | None,
    micro_batch: int | None,
    layer: int | None,
) -> Tuple[Dict[Tuple[Tuple[str, Any], ...], List[Dict[str, Any]]], List[Tuple[Tuple[str, Any], ...]]]:
    entries: Dict[Tuple[Tuple[str, Any], ...], List[Dict[str, Any]]] = {}
    duplicates: List[Tuple[Tuple[str, Any], ...]] = []
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            if not filename.endswith(".pt"):
                continue
            path = os.path.join(dirpath, filename)
            try:
                meta = parse_dump_filename(filename)
            except Exception as exc:
                raise RuntimeError(f"Failed to parse dump filename: {path}") from exc
            if not _meta_matches(meta, name_regex, iteration, micro_batch, layer):
                continue
            key = _build_key(meta, ignore_fields)
            entry = {"path": path, "meta": meta}
            if key in entries:
                entries[key].append(entry)
                duplicates.append(key)
            else:
                entries[key] = [entry]
    return entries, duplicates


def _extract_tensor_map(data: Any) -> Dict[str, torch.Tensor]:
    result: Dict[str, torch.Tensor] = {}

    def rec(obj: Any, prefix: str) -> None:
        if torch.is_tensor(obj):
            name = prefix or "data"
            result[name] = obj
            return
        if isinstance(obj, dict):
            for key, val in obj.items():
                next_prefix = f"{prefix}.{key}" if prefix else str(key)
                rec(val, next_prefix)
            return
        if isinstance(obj, (list, tuple)):
            for idx, val in enumerate(obj):
                next_prefix = f"{prefix}[{idx}]" if prefix else f"[{idx}]"
                rec(val, next_prefix)
            return

    rec(data, "")
    if result:
        return result

    try:
        tensor = torch.as_tensor(data)
    except Exception:
        return {}

    return {"data": tensor}


def _load_tensor_map(path: str) -> Dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "data" in payload:
        data = payload["data"]
    else:
        data = payload
    return _extract_tensor_map(data)


def _compare_tensors(
    a: torch.Tensor,
    b: torch.Tensor,
    trim_to_min_shape: bool,
) -> Dict[str, Any]:
    trimmed = False
    if a.shape != b.shape:
        if not trim_to_min_shape:
            return {"shape_mismatch": True}
        if a.dim() != b.dim():
            return {"shape_mismatch": True}
        min_shape = tuple(min(sa, sb) for sa, sb in zip(a.shape, b.shape))
        if any(dim == 0 for dim in min_shape):
            return {
                "shape_mismatch": False,
                "nonfinite": 0,
                "empty": True,
                "trimmed": True,
                "trim_shape": min_shape,
            }
        slices = tuple(slice(0, dim) for dim in min_shape)
        a = a[slices]
        b = b[slices]
        trimmed = True

    a = a.detach().cpu()
    b = b.detach().cpu()

    # Promote to float for numeric comparisons.
    if not torch.is_floating_point(a):
        a = a.float()
    else:
        a = a.float()
    if not torch.is_floating_point(b):
        b = b.float()
    else:
        b = b.float()

    diff = (a - b).abs()
    finite_mask = torch.isfinite(diff)
    nonfinite = diff.numel() - int(finite_mask.sum().item())
    if nonfinite:
        diff = diff[finite_mask]

    if diff.numel() == 0:
        return {
            "shape_mismatch": False,
            "nonfinite": nonfinite,
            "empty": True,
            "trimmed": trimmed,
        }

    max_abs = float(diff.max().item())
    sum_abs = float(diff.sum().item())
    sum_sq = float((diff * diff).sum().item())

    denom = max(float(a.abs().max().item()), float(b.abs().max().item()))
    rel_max = max_abs / (denom + 1e-12)

    return {
        "shape_mismatch": False,
        "nonfinite": nonfinite,
        "empty": False,
        "trimmed": trimmed,
        "max_abs": max_abs,
        "sum_abs": sum_abs,
        "sum_sq": sum_sq,
        "numel": int(diff.numel()),
        "rel_max": rel_max,
    }


def _init_layer_stats() -> Dict[str, Any]:
    return {
        "pairs": 0,
        "total_elems": 0,
        "sum_abs": 0.0,
        "sum_sq": 0.0,
        "max_abs": 0.0,
        "max_rel": 0.0,
        "shape_mismatch": 0,
        "missing_subtensor": 0,
        "skipped": 0,
        "nonfinite": 0,
        "trimmed": 0,
        "top": [],
    }


def _update_top(top_list: List[Dict[str, Any]], item: Dict[str, Any], topk: int) -> None:
    top_list.append(item)
    top_list.sort(key=lambda x: x["max_abs"], reverse=True)
    if len(top_list) > topk:
        del top_list[topk:]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare Megatron debug dumps layer-by-layer between two dump roots.",
    )
    parser.add_argument("--bshd", required=True, help="bshd dump root")
    parser.add_argument("--thd", required=True, help="thd dump root")
    parser.add_argument("--ignore-fields", default="", help="comma-separated metadata fields to ignore")
    parser.add_argument("--name-regex", default=None, help="only compare dumps whose name matches regex")
    parser.add_argument("--iteration", type=int, default=None, help="only compare a specific iteration")
    parser.add_argument("--micro-batch", type=int, default=None, help="only compare a specific micro-batch")
    parser.add_argument("--layer", type=int, default=None, help="only compare a specific layer")
    parser.add_argument("--topk", type=int, default=3, help="top-k largest diffs to report per layer")
    parser.add_argument("--output-json", default=None, help="write JSON summary to this path")
    parser.add_argument(
        "--trim-to-min-shape",
        action="store_true",
        help="if shapes differ, compare the common prefix (min shape per dimension)",
    )

    args = parser.parse_args()

    ignore_fields = _parse_ignore_fields(args.ignore_fields)

    b_entries, b_dups = _collect_entries(
        args.bshd,
        ignore_fields,
        args.name_regex,
        args.iteration,
        args.micro_batch,
        args.layer,
    )
    t_entries, t_dups = _collect_entries(
        args.thd,
        ignore_fields,
        args.name_regex,
        args.iteration,
        args.micro_batch,
        args.layer,
    )

    all_keys = set(b_entries.keys()) | set(t_entries.keys())
    unmatched_b: List[Dict[str, Any]] = []
    unmatched_t: List[Dict[str, Any]] = []

    layer_stats: Dict[Any, Dict[str, Any]] = {}

    def get_layer_key(meta: Dict[str, Any]) -> Any:
        layer_id = meta.get("layer", None)
        return layer_id if layer_id is not None else "no_layer"

    for key in sorted(all_keys):
        b_list = b_entries.get(key, [])
        t_list = t_entries.get(key, [])
        if not b_list:
            unmatched_t.extend(t_list)
            continue
        if not t_list:
            unmatched_b.extend(b_list)
            continue

        b_list_sorted = sorted(b_list, key=lambda x: x["path"])
        t_list_sorted = sorted(t_list, key=lambda x: x["path"])
        pairs = min(len(b_list_sorted), len(t_list_sorted))

        for idx in range(pairs):
            b_item = b_list_sorted[idx]
            t_item = t_list_sorted[idx]
            meta = b_item["meta"]
            layer_key = get_layer_key(meta)
            stats = layer_stats.setdefault(layer_key, _init_layer_stats())

            b_map = _load_tensor_map(b_item["path"])
            t_map = _load_tensor_map(t_item["path"])

            if not b_map or not t_map:
                stats["skipped"] += 1
                continue

            names = set(b_map.keys()) | set(t_map.keys())
            for name in sorted(names):
                if name not in b_map or name not in t_map:
                    stats["missing_subtensor"] += 1
                    continue

                result = _compare_tensors(b_map[name], t_map[name], args.trim_to_min_shape)
                if result.get("shape_mismatch"):
                    stats["shape_mismatch"] += 1
                    continue
                if result.get("empty"):
                    stats["skipped"] += 1
                    stats["nonfinite"] += int(result.get("nonfinite", 0))
                    stats["trimmed"] += int(bool(result.get("trimmed")))
                    continue

                stats["pairs"] += 1
                stats["nonfinite"] += int(result.get("nonfinite", 0))
                stats["trimmed"] += int(bool(result.get("trimmed")))
                stats["total_elems"] += int(result["numel"])
                stats["sum_abs"] += float(result["sum_abs"])
                stats["sum_sq"] += float(result["sum_sq"])
                stats["max_abs"] = max(stats["max_abs"], float(result["max_abs"]))
                stats["max_rel"] = max(stats["max_rel"], float(result["rel_max"]))

                _update_top(
                    stats["top"],
                    {
                        "name": meta.get("name", ""),
                        "subname": name,
                        "iter": meta.get("iter"),
                        "mb": meta.get("mb"),
                        "idx": meta.get("idx"),
                        "max_abs": float(result["max_abs"]),
                        "b_path": b_item["path"],
                        "t_path": t_item["path"],
                    },
                    args.topk,
                )

        if len(b_list_sorted) > pairs:
            unmatched_b.extend(b_list_sorted[pairs:])
        if len(t_list_sorted) > pairs:
            unmatched_t.extend(t_list_sorted[pairs:])

    def layer_sort_key(layer_key: Any) -> Tuple[int, int]:
        if layer_key == "no_layer":
            return (1, 1_000_000_000)
        return (0, int(layer_key))

    print("bshd:", args.bshd)
    print("thd:", args.thd)
    print("total keys - bshd:", len(b_entries), "thd:", len(t_entries))
    print("unmatched - bshd only:", len(unmatched_b), "thd only:", len(unmatched_t))
    if b_dups or t_dups:
        print("warning: duplicate metadata keys detected (may indicate ignored fields too broad)")
    print()

    header = (
        f"{'layer':>8} {'pairs':>8} {'elems':>12} {'max_abs':>12} "
        f"{'mean_abs':>12} {'rmse':>12} {'max_rel':>10} "
        f"{'shape':>8} {'trim':>6} {'missing':>8} {'skipped':>8} {'nonfinite':>10}"
    )
    print(header)
    print("-" * len(header))

    summary: Dict[str, Any] = {
        "bshd": args.bshd,
        "thd": args.thd,
        "unmatched_bshd": len(unmatched_b),
        "unmatched_thd": len(unmatched_t),
        "layers": {},
    }

    for layer_key in sorted(layer_stats.keys(), key=layer_sort_key):
        stats = layer_stats[layer_key]
        total_elems = stats["total_elems"]
        if total_elems > 0:
            mean_abs = stats["sum_abs"] / total_elems
            rmse = math.sqrt(stats["sum_sq"] / total_elems)
        else:
            mean_abs = float("nan")
            rmse = float("nan")

        layer_name = str(layer_key)
        print(
            f"{layer_name:>8} {stats['pairs']:>8} {total_elems:>12} "
            f"{stats['max_abs']:>12.6g} {mean_abs:>12.6g} {rmse:>12.6g} "
            f"{stats['max_rel']:>10.6g} {stats['shape_mismatch']:>8} "
            f"{stats['trimmed']:>6} {stats['missing_subtensor']:>8} "
            f"{stats['skipped']:>8} {stats['nonfinite']:>10}"
        )

        summary["layers"][layer_name] = {
            "pairs": stats["pairs"],
            "total_elems": total_elems,
            "max_abs": stats["max_abs"],
            "mean_abs": mean_abs,
            "rmse": rmse,
            "max_rel": stats["max_rel"],
            "shape_mismatch": stats["shape_mismatch"],
            "trimmed": stats["trimmed"],
            "missing_subtensor": stats["missing_subtensor"],
            "skipped": stats["skipped"],
            "nonfinite": stats["nonfinite"],
            "top": stats["top"],
        }

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print()
        print("wrote json:", args.output_json)


if __name__ == "__main__":
    main()
