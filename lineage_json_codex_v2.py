#!/usr/bin/env python3
"""
Data Lineage Extraction Utility
===============================

Extracts recursive column-level lineage from CSV/TSV data and writes one JSON
file per unique target node (target_table, target_column).

Key behavior:
- Graph-based lineage extraction (source -> target edges), not strict level hierarchy
- Uses CSV `level` values to bucket nested lineage keys as `LEVEL_<actual_level>`
- Case-insensitive matching with trimmed normalization
- Handles same-level and non-linear lineage dependencies
- Cycle-safe recursive traversal
- Optional filtering by target table/column
- Optional start-level restriction for selecting starting targets only

Input requirements:
- UTF-8 text file (BOM supported automatically)
- CSV/TSV (delimiter configurable)
- Required columns (case-insensitive):
  level, source_table, source_column, target_table, target_column
- Optional columns:
  transformation_logic, filter_logic

Usage:
  python lineage_utility.py INPUT_FILE --output OUTPUT_DIR [options]

Examples:
  python lineage_utility.py input.csv -d ',' --output out
  python lineage_utility.py input.tsv -d '\\t' --output out --target_table "SGCAFXBI.FX_TURNOVER"
  python lineage_utility.py input.csv --output out --target_column "NET_AMT" --missing-required-values error
  python lineage_utility.py input.csv --output out --metadata-columns pipeline_name run_id

Troubleshooting:
- "Missing required columns":
  Ensure the input header includes all required columns (case-insensitive).
- "No valid rows were found":
  Input rows may have missing required values and were skipped; adjust
  --missing-required-values.
- "No output generated":
  Filter criteria may not match any target nodes, or valid lineage rows were not loaded.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

REQUIRED_COLUMNS = (
    "level",
    "source_table",
    "source_column",
    "target_table",
    "target_column",
)

OPTIONAL_COLUMNS = (
    "transformation_logic",
    "filter_logic",
)


class LineageError(Exception):
    """Domain-specific error for lineage utility failures."""


def normalize_text(value: Any) -> str:
    """Normalize values for case-insensitive, whitespace-insensitive matching."""
    if value is None:
        return ""
    return str(value).strip().lower()


def clean_text(value: Any) -> str:
    """Return trimmed text, preserving case."""
    if value is None:
        return ""
    return str(value).strip()


def parse_delimiter(raw: str) -> str:
    """Support escaped tab notation from CLI."""
    if raw == r"\t":
        return "\t"
    return raw


def parse_int(value: str) -> Optional[int]:
    value = value.strip()
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def level_key(row_level: Optional[int]) -> str:
    """Return key name for a lineage level bucket."""
    if row_level is None:
        return "LEVEL_UNKNOWN"
    return f"LEVEL_{row_level}"


def safe_filename_component(value: str) -> str:
    cleaned = clean_text(value)
    cleaned = re.sub(r"[^\w.-]+", "_", cleaned, flags=re.ASCII)
    cleaned = cleaned.strip("._")
    return cleaned or "unknown"


def build_header_lookup(fieldnames: Sequence[str]) -> Dict[str, str]:
    """
    Map normalized header -> original header label.
    First occurrence wins if duplicate normalized names exist.
    """
    lookup: Dict[str, str] = {}
    for name in fieldnames:
        key = normalize_text(name)
        if key and key not in lookup:
            lookup[key] = name
    return lookup


def get_column_value(
    row: Dict[str, Any],
    header_lookup: Dict[str, str],
    column_name: str,
    default: str = "",
) -> str:
    original = header_lookup.get(column_name)
    if original is None:
        return default
    return clean_text(row.get(original, default))


@dataclass(frozen=True)
class Edge:
    line_number: int
    level: Optional[int]
    source_table: str
    source_column: str
    target_table: str
    target_column: str
    source_node: Tuple[str, str]
    target_node: Tuple[str, str]
    transformation_logic: str
    filter_logic: str
    metadata: Dict[str, str]

    def to_payload(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "level": self.level,
            "source_column": self.source_column,
            "target_column": self.target_column,
            "source_table": self.source_table,
            "target_table": self.target_table,
            "transformation_logic": self.transformation_logic,
            "filter_logic": self.filter_logic,
        }
        payload.update(self.metadata)
        return payload


def edge_sort_key(edge: Edge) -> Tuple[int, int, str, str, str, str]:
    """
    Deterministic edge ordering:
    - known levels first, ascending
    - then line number and stable text fields
    """
    level_rank = edge.level if edge.level is not None else 10**9
    return (
        level_rank,
        edge.line_number,
        edge.source_table.lower(),
        edge.source_column.lower(),
        edge.target_table.lower(),
        edge.target_column.lower(),
    )


def add_grouped_children(
    payload: Dict[str, Any],
    child_edges: List[Dict[str, Any]],
) -> None:
    """
    Group recursive child edges under LEVEL_<actual_level> keys.
    """
    by_level: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for child in child_edges:
        key = level_key(child.get("level"))
        by_level[key].append(child)

    # Sort keys by numeric level where possible, unknown last.
    def level_bucket_sort_key(name: str) -> Tuple[int, str]:
        if name == "LEVEL_UNKNOWN":
            return (10**9, name)
        raw = name.replace("LEVEL_", "", 1)
        try:
            return (int(raw), name)
        except ValueError:
            return (10**9 - 1, name)

    for key_name in sorted(by_level.keys(), key=level_bucket_sort_key):
        payload[key_name] = by_level[key_name]


def detect_cycle(nodes: Iterable[Tuple[str, str]], inbound: Dict[Tuple[str, str], List[Edge]]) -> bool:
    """
    Detect if target->source node graph contains any cycle.
    """
    adjacency: Dict[Tuple[str, str], Set[Tuple[str, str]]] = {}
    for node in nodes:
        adjacency[node] = set()

    for target_node, edges in inbound.items():
        refs = adjacency.setdefault(target_node, set())
        for edge in edges:
            refs.add(edge.source_node)
            adjacency.setdefault(edge.source_node, set())

    WHITE = 0
    GRAY = 1
    BLACK = 2
    color: Dict[Tuple[str, str], int] = {node: WHITE for node in adjacency}

    def dfs(node: Tuple[str, str]) -> bool:
        color[node] = GRAY
        for neighbor in adjacency[node]:
            n_color = color[neighbor]
            if n_color == GRAY:
                return True
            if n_color == WHITE and dfs(neighbor):
                return True
        color[node] = BLACK
        return False

    for node in adjacency:
        if color[node] == WHITE and dfs(node):
            return True
    return False


def build_upstream_acyclic(
    node: Tuple[str, str],
    inbound: Dict[Tuple[str, str], List[Edge]],
    cache: Dict[Tuple[str, str], List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    if node in cache:
        return copy.deepcopy(cache[node])

    branches: List[Dict[str, Any]] = []
    for edge in sorted(inbound.get(node, []), key=edge_sort_key):
        payload = edge.to_payload()
        upstream = build_upstream_acyclic(edge.source_node, inbound, cache)
        if upstream:
            add_grouped_children(payload, upstream)
        branches.append(payload)

    cache[node] = copy.deepcopy(branches)
    return branches


def build_upstream_cyclic(
    node: Tuple[str, str],
    inbound: Dict[Tuple[str, str], List[Edge]],
    path: Set[Tuple[str, str]],
) -> List[Dict[str, Any]]:
    if node in path:
        return []

    path.add(node)
    branches: List[Dict[str, Any]] = []
    for edge in sorted(inbound.get(node, []), key=edge_sort_key):
        payload = edge.to_payload()
        upstream = build_upstream_cyclic(edge.source_node, inbound, path)
        if upstream:
            add_grouped_children(payload, upstream)
        branches.append(payload)
    path.remove(node)
    return branches


def build_lineage_root(
    target_node: Tuple[str, str],
    target_display: Dict[Tuple[str, str], Tuple[str, str]],
    inbound: Dict[Tuple[str, str], List[Edge]],
    optional_default: str,
    is_acyclic: bool,
    cache: Dict[Tuple[str, str], List[Dict[str, Any]]],
) -> Dict[str, Any]:
    target_table, target_column = target_display[target_node]

    if is_acyclic:
        direct_edges = build_upstream_acyclic(target_node, inbound, cache)
    else:
        direct_edges = build_upstream_cyclic(target_node, inbound, set())

    root: Dict[str, Any] = {
        "level": None,
        "source_column": "",
        "target_column": target_column,
        "source_table": "",
        "target_table": target_table,
        "transformation_logic": optional_default,
        "filter_logic": optional_default,
    }
    if direct_edges:
        add_grouped_children(root, direct_edges)
    return root


def load_edges(
    input_file: Path,
    delimiter: str,
    missing_required_values: str,
    optional_default: str,
    metadata_columns: Sequence[str],
) -> Tuple[List[Edge], Set[Tuple[str, str]], Dict[Tuple[str, str], Tuple[str, str]], List[str]]:
    if missing_required_values not in {"skip", "error", "empty"}:
        raise LineageError("Invalid missing-required-values mode.")

    edges: List[Edge] = []
    all_nodes: Set[Tuple[str, str]] = set()
    target_display: Dict[Tuple[str, str], Tuple[str, str]] = {}
    warnings: List[str] = []

    try:
        with input_file.open("r", encoding="utf-8-sig", newline="") as fh:
            reader = csv.DictReader(fh, delimiter=delimiter)
            if not reader.fieldnames:
                raise LineageError("Input file has no header row.")

            header_lookup = build_header_lookup(reader.fieldnames)
            missing_required_cols = [col for col in REQUIRED_COLUMNS if col not in header_lookup]
            if missing_required_cols:
                raise LineageError(
                    "Missing required columns: " + ", ".join(missing_required_cols)
                )

            normalized_metadata = [normalize_text(name) for name in metadata_columns if name.strip()]
            seen_meta: Set[str] = set()
            deduped_metadata: List[str] = []
            for name in normalized_metadata:
                if name not in seen_meta:
                    deduped_metadata.append(name)
                    seen_meta.add(name)

            for line_number, row in enumerate(reader, start=2):
                required_values: Dict[str, str] = {
                    col: get_column_value(row, header_lookup, col, default="")
                    for col in REQUIRED_COLUMNS
                }

                missing_value_cols = [k for k, v in required_values.items() if not v]
                if missing_value_cols:
                    if missing_required_values == "error":
                        raise LineageError(
                            f"Missing required values at line {line_number}: {', '.join(missing_value_cols)}"
                        )
                    if missing_required_values == "skip":
                        warnings.append(
                            f"Skipped line {line_number} due to missing required values: {', '.join(missing_value_cols)}"
                        )
                        continue

                source_table = required_values["source_table"]
                source_column = required_values["source_column"]
                target_table = required_values["target_table"]
                target_column = required_values["target_column"]
                level_raw = required_values["level"]

                if missing_required_values == "empty":
                    source_table = source_table or ""
                    source_column = source_column or ""
                    target_table = target_table or ""
                    target_column = target_column or ""
                    level_raw = level_raw or ""

                level_value = parse_int(level_raw)
                if level_raw and level_value is None:
                    warnings.append(
                        f"Line {line_number} has non-integer level '{level_raw}'; stored as null."
                    )

                transformation_logic = get_column_value(
                    row, header_lookup, "transformation_logic", default=optional_default
                )
                filter_logic = get_column_value(
                    row, header_lookup, "filter_logic", default=optional_default
                )
                if not transformation_logic:
                    transformation_logic = optional_default
                if not filter_logic:
                    filter_logic = optional_default

                metadata_values: Dict[str, str] = {}
                for meta_col in deduped_metadata:
                    value = get_column_value(row, header_lookup, meta_col, default=optional_default)
                    metadata_values[meta_col] = value if value else optional_default

                source_node = (normalize_text(source_table), normalize_text(source_column))
                target_node = (normalize_text(target_table), normalize_text(target_column))

                if not source_node[0] or not source_node[1] or not target_node[0] or not target_node[1]:
                    if missing_required_values == "empty":
                        warnings.append(
                            f"Skipped line {line_number} because normalized node identity is empty."
                        )
                        continue
                    # For skip/error modes this branch is generally unreachable after checks above.
                    continue

                edge = Edge(
                    line_number=line_number,
                    level=level_value,
                    source_table=source_table,
                    source_column=source_column,
                    target_table=target_table,
                    target_column=target_column,
                    source_node=source_node,
                    target_node=target_node,
                    transformation_logic=transformation_logic,
                    filter_logic=filter_logic,
                    metadata=metadata_values,
                )
                edges.append(edge)
                all_nodes.add(source_node)
                all_nodes.add(target_node)
                target_display.setdefault(target_node, (target_table, target_column))
    except FileNotFoundError as exc:
        raise LineageError(f"Input file not found: {input_file}") from exc
    except OSError as exc:
        raise LineageError(f"Failed to read input file '{input_file}': {exc}") from exc
    except csv.Error as exc:
        raise LineageError(f"Failed to parse delimited input: {exc}") from exc

    return edges, all_nodes, target_display, warnings


def select_start_targets(
    inbound: Dict[Tuple[str, str], List[Edge]],
    target_display: Dict[Tuple[str, str], Tuple[str, str]],
    target_table_filter: Optional[str],
    target_column_filter: Optional[str],
    start_level: Optional[int],
) -> List[Tuple[str, str]]:
    table_filter = normalize_text(target_table_filter) if target_table_filter else ""
    column_filter = normalize_text(target_column_filter) if target_column_filter else ""

    selected: List[Tuple[str, str]] = []
    for node, edges in inbound.items():
        table_norm, column_norm = node
        if table_filter and table_norm != table_filter:
            continue
        if column_filter and column_norm != column_filter:
            continue
        if start_level is not None and not any(edge.level == start_level for edge in edges):
            continue
        if node not in target_display:
            continue
        selected.append(node)

    selected.sort(key=lambda n: (target_display[n][0].lower(), target_display[n][1].lower()))
    return selected


def write_lineage_files(
    output_dir: Path,
    targets: Sequence[Tuple[str, str]],
    target_display: Dict[Tuple[str, str], Tuple[str, str]],
    inbound: Dict[Tuple[str, str], List[Edge]],
    optional_default: str,
    is_acyclic: bool,
    json_indent: int,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    filename_counts: Dict[str, int] = defaultdict(int)
    cache: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}

    for target_node in targets:
        target_table, target_column = target_display[target_node]
        lineage_root = build_lineage_root(
            target_node=target_node,
            target_display=target_display,
            inbound=inbound,
            optional_default=optional_default,
            is_acyclic=is_acyclic,
            cache=cache,
        )

        payload = {
            "generated_column": target_column,
            "lineage": lineage_root,
        }

        base_name = (
            f"{safe_filename_component(target_table)}__{safe_filename_component(target_column)}"
        )
        filename_counts[base_name] += 1
        suffix = filename_counts[base_name]
        if suffix > 1:
            file_name = f"{base_name}_{suffix}.json"
        else:
            file_name = f"{base_name}.json"

        file_path = output_dir / file_name
        with file_path.open("w", encoding="utf-8") as out_fh:
            json.dump(payload, out_fh, ensure_ascii=False, indent=json_indent)
            out_fh.write("\n")
        written += 1

    return written


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract recursive column-level lineage from CSV/TSV and write JSON outputs."
    )
    parser.add_argument("input_file", help="Path to lineage input CSV/TSV file.")
    parser.add_argument(
        "-d",
        "--delimiter",
        default=",",
        help=r"Input delimiter (default: ','). Use '\t' for TSV.",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output directory for generated lineage JSON files.",
    )
    parser.add_argument(
        "--target_table",
        help="Optional target_table filter (case-insensitive exact match after trim).",
    )
    parser.add_argument(
        "--target_column",
        help="Optional target_column filter (case-insensitive exact match after trim).",
    )
    parser.add_argument(
        "--start-level",
        type=int,
        default=None,
        help="Optional starting level filter for selecting output targets only.",
    )
    parser.add_argument(
        "--missing-required-values",
        choices=("skip", "error", "empty"),
        default="skip",
        help="Behavior for rows missing required values: skip, error, or empty.",
    )
    parser.add_argument(
        "--optional-default",
        default="",
        help="Default fill value for missing optional fields and selected metadata.",
    )
    parser.add_argument(
        "--metadata-columns",
        nargs="*",
        default=(),
        help="Additional column names to include in output payload (if missing, fill with optional-default).",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indentation spaces (default: 2).",
    )
    return parser


def run(args: argparse.Namespace) -> int:
    input_file = Path(args.input_file)
    output_dir = Path(args.output)
    delimiter = parse_delimiter(args.delimiter)

    if len(delimiter) != 1:
        raise LineageError("Delimiter must be a single character (or '\\t').")

    edges, all_nodes, target_display, warnings = load_edges(
        input_file=input_file,
        delimiter=delimiter,
        missing_required_values=args.missing_required_values,
        optional_default=args.optional_default,
        metadata_columns=args.metadata_columns,
    )

    for warning in warnings:
        print(f"WARNING: {warning}", file=sys.stderr)

    if not edges:
        print("No valid rows were found. No output generated.")
        return 0

    inbound: Dict[Tuple[str, str], List[Edge]] = defaultdict(list)
    for edge in edges:
        inbound[edge.target_node].append(edge)

    targets = select_start_targets(
        inbound=inbound,
        target_display=target_display,
        target_table_filter=args.target_table,
        target_column_filter=args.target_column,
        start_level=args.start_level,
    )

    if not targets:
        print("No output generated: no targets matched the provided filters.")
        return 0

    is_acyclic = not detect_cycle(all_nodes, inbound)
    if not is_acyclic:
        print("INFO: Cycle detected in lineage graph; using cycle-safe traversal without global subtree cache.")

    written = write_lineage_files(
        output_dir=output_dir,
        targets=targets,
        target_display=target_display,
        inbound=inbound,
        optional_default=args.optional_default,
        is_acyclic=is_acyclic,
        json_indent=args.indent,
    )

    if written == 0:
        print("No output generated.")
        return 0

    print(f"Generated {written} lineage file(s) in: {output_dir.resolve()}")
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return run(args)
    except LineageError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
