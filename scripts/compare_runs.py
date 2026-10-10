"""CLI script to compare two evaluation .metrics.json files and print metric deltas and outcome changes."""

import argparse
import json
import sys
from pathlib import Path

from annual_report_rag.evaluation.reporting import compare_metrics_dicts


def parse_args() -> argparse.Namespace:
    """Parses the two .metrics.json file paths to compare."""
    parser = argparse.ArgumentParser(
        description="Compare two evaluation .metrics.json files."
    )
    parser.add_argument("run_a", type=str, help="Path to baseline A.metrics.json")
    parser.add_argument("run_b", type=str, help="Path to candidate B.metrics.json")
    return parser.parse_args()


def main() -> None:
    """Loads both metrics JSON files and prints the formatted comparison report."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args()
    try:
        path_a = Path(args.run_a)
        path_b = Path(args.run_b)
        metrics_a = json.loads(path_a.read_text(encoding="utf-8"))
        metrics_b = json.loads(path_b.read_text(encoding="utf-8"))
        print(compare_metrics_dicts(metrics_a, metrics_b))
    except Exception as exc:  # noqa: BLE001
        msg = str(exc)[:300]
        print(f"Error ({type(exc).__name__}): {msg}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
