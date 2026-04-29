"""
Data Lineage Extraction Utility
================================
Extracts column-level data lineage from a CSV/TSV file and outputs
complete recursive lineage as nested JSON files.

USAGE
-----
    python lineage_utility.py <input_file> [options]

ARGUMENTS
---------
    input_file              Path to the input CSV or TSV file (UTF-8 or UTF-8-BOM)

OPTIONS
-------
    -d, --delimiter         Field delimiter (default: ',')
    -o, --output            Output directory for JSON files (default: './lineage_output')
    --target_table          Filter: only process targets matching this table name
    --target_column         Filter: only process targets matching this column name
    --metadata_columns      Comma-separated list of extra metadata columns to include in output
    --skip_missing_values   Skip rows with missing required field values (default: False → include as empty)
    --all_levels            Generate lineage for ALL unique targets, not just minimum-level ones

INPUT FORMAT
------------
Required columns (case-insensitive):
    level                   Integer lineage depth (informational only)
    source_table            Name of the source table
    source_column           Name of the source column
    target_table            Name of the target table
    target_column           Name of the target column

Optional columns:
    transformation_logic    Transformation applied from source → target
    filter_logic            Filter logic applied during transformation

OUTPUT FORMAT
-------------
One JSON file per unique (target_column, target_table), named:
    <target_table>__<target_column>.json

Each file contains a fully recursive nested lineage tree. Example:
    {
        "generated_column": "<target_column>",
        "lineage": {
            "level": 1,
            "source_column": "...",
            "target_column": "...",
            "source_table": "...",
            "target_table": "...",
            "transformation_logic": "...",
            "filter_logic": "...",
            "LEVEL_2": [
                { ... "LEVEL_3": [ { ... } ] }
            ]
        }
    }

TROUBLESHOOTING
---------------
- "Missing required columns": Ensure your file has level, source_table, source_column,
  target_table, target_column headers (case-insensitive).
- "No output files generated": Check that --target_table / --target_column match actual
  values in your file; filters are case-insensitive but must match a real target.
- Encoding errors: Ensure the file is saved as UTF-8 or UTF-8-BOM.
- Infinite recursion suspected: The utility has built-in cycle detection; check your data
  for circular references and review the logged warnings.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)
log = logging.getLogger("lineage_utility")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
REQUIRED_COLUMNS: List[str] = [
    "level",
    "source_table",
    "source_column",
    "target_table",
    "target_column",
]
OPTIONAL_COLUMNS: List[str] = ["transformation_logic", "filter_logic"]

# A node key is (table_normalised, column_normalised)
NodeKey = Tuple[str, str]


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass
class LineageRow:
    level: int
    source_table: str
    source_column: str
    target_table: str
    target_column: str
    transformation_logic: str = ""
    filter_logic: str = ""
    metadata: Dict[str, str] = field(default_factory=dict)

    # Normalised keys (used for graph lookups)
    @property
    def source_key(self) -> NodeKey:
        return (_norm(self.source_table), _norm(self.source_column))

    @property
    def target_key(self) -> NodeKey:
        return (_norm(self.target_table), _norm(self.target_column))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _norm(value: str) -> str:
    """Normalise a string for case-insensitive, whitespace-trimmed comparison."""
    return value.strip().lower()


def _safe_filename(name: str) -> str:
    """Convert a string to a safe filesystem name."""
    return re.sub(r'[^\w\-.]', '_', name)


def _optional_value(row: dict, col: str) -> str:
    """Return the value of an optional column, defaulting to empty string."""
    val = row.get(col, "")
    return val.strip() if val else ""


# ---------------------------------------------------------------------------
# CSV / TSV reader
# ---------------------------------------------------------------------------
def load_rows(
    filepath: str,
    delimiter: str,
    skip_missing_values: bool,
    metadata_columns: List[str],
) -> List[LineageRow]:
    """
    Read and validate the input file. Returns a list of LineageRow objects.
    Raises SystemExit on unrecoverable errors.
    """
    path = Path(filepath)
    if not path.exists():
        log.error("Input file not found: %s", filepath)
        sys.exit(1)

    rows: List[LineageRow] = []
    skipped = 0

    # Open with UTF-8-sig to automatically strip BOM
    with path.open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh, delimiter=delimiter)

        # Validate required columns (case-insensitive header matching)
        if reader.fieldnames is None:
            log.error("Input file appears to be empty or unreadable.")
            sys.exit(1)

        normalised_headers = {_norm(h): h for h in reader.fieldnames if h}
        missing = [c for c in REQUIRED_COLUMNS if c not in normalised_headers]
        if missing:
            log.error(
                "Input file is missing required column(s): %s\n"
                "Found columns: %s",
                ", ".join(missing),
                ", ".join(reader.fieldnames),
            )
            sys.exit(1)

        # Build a mapping from required/optional col name → actual header name
        col_map = {c: normalised_headers[c] for c in REQUIRED_COLUMNS}
        opt_map = {
            c: normalised_headers[c]
            for c in OPTIONAL_COLUMNS
            if c in normalised_headers
        }
        meta_map = {
            c: normalised_headers[_norm(c)]
            for c in metadata_columns
            if _norm(c) in normalised_headers
        }

        for line_num, raw in enumerate(reader, start=2):
            # Extract required fields
            level_raw = raw.get(col_map["level"], "").strip()
            source_table = raw.get(col_map["source_table"], "").strip()
            source_column = raw.get(col_map["source_column"], "").strip()
            target_table = raw.get(col_map["target_table"], "").strip()
            target_column = raw.get(col_map["target_column"], "").strip()

            # Validate required values
            required_vals = {
                "level": level_raw,
                "source_table": source_table,
                "source_column": source_column,
                "target_table": target_table,
                "target_column": target_column,
            }
            missing_vals = [k for k, v in required_vals.items() if not v]
            if missing_vals:
                if skip_missing_values:
                    log.debug("Row %d skipped – missing values: %s", line_num, missing_vals)
                    skipped += 1
                    continue
                else:
                    # Use empty string for missing non-critical values; level defaults to 0
                    pass

            # Parse level
            try:
                level = int(level_raw) if level_raw else 0
            except ValueError:
                log.warning("Row %d: non-integer level '%s', defaulting to 0.", line_num, level_raw)
                level = 0

            # Optional fields
            transformation_logic = _optional_value(raw, opt_map.get("transformation_logic", ""))
            filter_logic = _optional_value(raw, opt_map.get("filter_logic", ""))

            # Metadata
            metadata = {c: _optional_value(raw, h) for c, h in meta_map.items()}

            rows.append(
                LineageRow(
                    level=level,
                    source_table=source_table,
                    source_column=source_column,
                    target_table=target_table,
                    target_column=target_column,
                    transformation_logic=transformation_logic,
                    filter_logic=filter_logic,
                    metadata=metadata,
                )
            )

    if skipped:
        log.info("Skipped %d row(s) with missing required values.", skipped)

    if not rows:
        log.error("No valid rows found in input file.")
        sys.exit(1)

    log.info("Loaded %d lineage row(s).", len(rows))
    return rows


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------
class LineageGraph:
    """
    Directed graph of lineage edges.
    edges_to_target[target_key]  → list of LineageRow whose target == target_key
    edges_from_source[source_key] → list of LineageRow whose source == source_key
    """

    def __init__(self, rows: List[LineageRow]) -> None:
        # target_key → all rows that produce this target
        self.edges_to_target: Dict[NodeKey, List[LineageRow]] = defaultdict(list)
        # source_key → all rows emitted from this source
        self.edges_from_source: Dict[NodeKey, List[LineageRow]] = defaultdict(list)

        for row in rows:
            self.edges_to_target[row.target_key].append(row)
            self.edges_from_source[row.source_key].append(row)

        # All distinct target nodes
        self.all_targets: Set[NodeKey] = set(self.edges_to_target.keys())

        # "Entry points": targets that are not themselves the source of any edge
        # (i.e. they are leaves in the forward direction – final outputs).
        # Used when --all_levels is NOT set.
        self._sources_as_targets: Set[NodeKey] = set()
        for row in rows:
            self._sources_as_targets.add(row.source_key)

    def terminal_targets(self) -> Set[NodeKey]:
        """Return target nodes that do NOT appear as source in any other row."""
        return self.all_targets - self._sources_as_targets

    def minimum_level_targets(self) -> Set[NodeKey]:
        """
        Return one representative (target_key, row) for each unique target,
        using the row with the minimum level value. These are typically the
        'final' output columns closest to the report layer.
        """
        best: Dict[NodeKey, LineageRow] = {}
        for key, row_list in self.edges_to_target.items():
            min_row = min(row_list, key=lambda r: r.level)
            best[key] = min_row
        return set(best.keys())


# ---------------------------------------------------------------------------
# Recursive lineage tree builder
# ---------------------------------------------------------------------------
def build_lineage_node(
    row: LineageRow,
    graph: LineageGraph,
    visited: Set[NodeKey],
    depth: int,
) -> dict:
    """
    Recursively build a lineage node dict for a given row.
    ``visited`` tracks the current DFS path to detect cycles.
    ``depth`` is the current nesting counter (used to name LEVEL_<n> keys).
    """
    node: dict = {
        "level": row.level,
        "source_column": row.source_column,
        "target_column": row.target_column,
        "source_table": row.source_table,
        "target_table": row.target_table,
        "transformation_logic": row.transformation_logic or "",
        "filter_logic": row.filter_logic or "",
    }
    if row.metadata:
        node.update(row.metadata)

    # Recurse into parents of this row's source
    parent_key = row.source_key
    if parent_key in visited:
        log.warning(
            "Cycle detected at (%s, %s); stopping recursion.",
            row.source_table,
            row.source_column,
        )
        return node

    parent_rows = graph.edges_to_target.get(parent_key, [])
    if not parent_rows:
        return node  # No further ancestry – omit LEVEL_<n+1>

    visited = visited | {parent_key}  # immutable update to avoid side-effects
    next_level_key = f"LEVEL_{depth + 1}"
    child_nodes = []
    for parent_row in parent_rows:
        child_node = build_lineage_node(parent_row, graph, visited, depth + 1)
        child_nodes.append(child_node)

    if child_nodes:
        node[next_level_key] = child_nodes

    return node


# ---------------------------------------------------------------------------
# File writer
# ---------------------------------------------------------------------------
def write_lineage_json(
    target_table: str,
    target_column: str,
    rows: List[LineageRow],
    graph: LineageGraph,
    output_dir: Path,
    starting_depth: int = 1,
) -> None:
    """
    For each row that produces (target_table, target_column), generate a JSON
    file containing the full recursive lineage tree.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    lineage_entries = []
    visited_root: Set[NodeKey] = {(_norm(target_table), _norm(target_column))}

    for row in rows:
        node = build_lineage_node(row, graph, visited_root, starting_depth)
        lineage_entries.append(node)

    # If there is exactly one root entry, use it directly; otherwise wrap in list
    if len(lineage_entries) == 1:
        lineage_root = lineage_entries[0]
    else:
        lineage_root = lineage_entries  # multiple rows map to same target

    output = {
        "generated_column": target_column,
        "lineage": lineage_root,
    }

    safe_table = _safe_filename(target_table)
    safe_col = _safe_filename(target_column)
    filename = f"{safe_table}__{safe_col}.json"
    out_path = output_dir / filename

    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2, ensure_ascii=False)

    log.info("Written: %s", out_path)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def run(
    input_file: str,
    delimiter: str,
    output_dir: str,
    target_table_filter: Optional[str],
    target_column_filter: Optional[str],
    skip_missing_values: bool,
    metadata_columns: List[str],
    all_levels: bool,
) -> None:
    rows = load_rows(input_file, delimiter, skip_missing_values, metadata_columns)
    graph = LineageGraph(rows)

    # Determine which targets to generate lineage for
    if target_table_filter or target_column_filter:
        # Filter mode
        tt_norm = _norm(target_table_filter) if target_table_filter else None
        tc_norm = _norm(target_column_filter) if target_column_filter else None

        candidate_keys: Set[NodeKey] = set()
        for key in graph.all_targets:
            t_table, t_col = key
            table_match = (tt_norm is None) or (t_table == tt_norm)
            col_match = (tc_norm is None) or (t_col == tc_norm)
            if table_match and col_match:
                candidate_keys.add(key)

        if not candidate_keys:
            log.warning(
                "No targets found matching --target_table='%s' --target_column='%s'.",
                target_table_filter or "*",
                target_column_filter or "*",
            )
            log.warning("No output files generated.")
            return
    elif all_levels:
        candidate_keys = graph.all_targets
    else:
        # Default: minimum-level targets (final output columns)
        candidate_keys = graph.minimum_level_targets()

    log.info("Generating lineage for %d unique target(s).", len(candidate_keys))
    out_path = Path(output_dir)
    generated = 0

    for target_key in sorted(candidate_keys):
        t_table_norm, t_col_norm = target_key
        target_rows = graph.edges_to_target[target_key]

        # Use original (non-normalised) values from first row
        first_row = target_rows[0]
        real_table = first_row.target_table
        real_col = first_row.target_column

        write_lineage_json(
            target_table=real_table,
            target_column=real_col,
            rows=target_rows,
            graph=graph,
            output_dir=out_path,
            starting_depth=1,
        )
        generated += 1

    if generated == 0:
        log.warning("No output files generated.")
    else:
        log.info("Done. %d file(s) written to: %s", generated, out_path.resolve())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="lineage_utility.py",
        description="Extract column-level data lineage from a CSV/TSV and output nested JSON files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "input_file",
        help="Path to the input CSV or TSV file.",
    )
    parser.add_argument(
        "-d", "--delimiter",
        default=",",
        help="Field delimiter character (default: ','). Use '\\t' for TSV.",
    )
    parser.add_argument(
        "-o", "--output",
        default="./lineage_output",
        help="Output directory for JSON files (default: './lineage_output').",
    )
    parser.add_argument(
        "--target_table",
        default=None,
        help="Filter: only generate lineage for targets in this table.",
    )
    parser.add_argument(
        "--target_column",
        default=None,
        help="Filter: only generate lineage for this target column.",
    )
    parser.add_argument(
        "--skip_missing_values",
        action="store_true",
        help="Skip rows that are missing required field values (default: include with empty values).",
    )
    parser.add_argument(
        "--metadata_columns",
        default="",
        help="Comma-separated list of extra metadata column names to include in output.",
    )
    parser.add_argument(
        "--all_levels",
        action="store_true",
        help="Generate lineage for ALL unique (target_table, target_column) pairs, not just minimum-level ones.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug-level logging.",
    )

    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    if args.verbose:
        log.setLevel(logging.DEBUG)

    # Handle tab delimiter shorthand
    delimiter = "\t" if args.delimiter in ("\\t", "\t", "tab") else args.delimiter

    metadata_cols = (
        [c.strip() for c in args.metadata_columns.split(",") if c.strip()]
        if args.metadata_columns
        else []
    )

    run(
        input_file=args.input_file,
        delimiter=delimiter,
        output_dir=args.output,
        target_table_filter=args.target_table,
        target_column_filter=args.target_column,
        skip_missing_values=args.skip_missing_values,
        metadata_columns=metadata_cols,
        all_levels=args.all_levels,
    )


if __name__ == "__main__":
    main()
