"""CLI script to run the 6 synthetic ExampleCo judge calibration cases and report agreement."""

import argparse
import sys
from pathlib import Path

from annual_report_rag.config import load_settings
from annual_report_rag.evaluation.judge import (
    CalibrationCase,
    evaluate_calibration_case,
    judge_answer,
    load_calibration_cases,
)
from annual_report_rag.providers.base import DailyQuotaExceeded, GenerationProvider
from annual_report_rag.providers.factory import get_generator


def run_calibration(
    cases: list[CalibrationCase],
    provider: GenerationProvider,
) -> tuple[int, int, str]:
    """Runs all calibration cases against provider and returns (agreed_count, total_count, formatted_report)."""
    lines: list[str] = []
    lines.append("# Judge Calibration Report (Synthetic `ExampleCo` Cases)")
    lines.append("")
    lines.append(
        "| Case | Description | Expected (`faithful` / `citations`) | Actual (`faithful` / `citations`) | Match |"
    )
    lines.append("| :--- | :--- | :--- | :--- | :--- |")

    agreed = 0
    for case in cases:
        verdict = judge_answer(
            question=case.question,
            answer=case.answer,
            cited_passages=case.cited_passages,
            provider=provider,
        )
        matched = evaluate_calibration_case(case, verdict)
        if matched:
            agreed += 1

        exp_f = "any" if case.expected_faithful is None else str(case.expected_faithful)
        exp_c = ", ".join(f"[{k}]={v}" for k, v in case.expected_citation_supports.items())
        act_c = ", ".join(f"[{c.n}]={c.supports}" for c in verdict.citations)
        status_str = "PASS" if matched else "FAIL"

        lines.append(
            f"| `{case.id}` | {case.description} | `faithful={exp_f}`, `{exp_c}` | "
            f"`faithful={verdict.faithful}`, `{act_c}` (`parse_ok={verdict.parse_ok}`) | **{status_str}** |"
        )

    total = len(cases)
    rate = (agreed / total) if total > 0 else 0.0
    lines.append("")
    lines.append(f"**Overall Agreement**: `{agreed}/{total}` (`{rate:.2%}`)")
    return agreed, total, "\n".join(lines)


def main() -> None:
    """CLI entry point for running judge calibration."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="Run synthetic ExampleCo judge calibration."
    )
    parser.add_argument(
        "--path",
        type=str,
        default="data/eval/judge_calibration.json",
        help="Path to judge_calibration.json",
    )
    args = parser.parse_args()

    try:
        settings = load_settings()
        cases = load_calibration_cases(Path(args.path))
        generator = get_generator(settings)
        _agreed, _total, report = run_calibration(cases, generator)
        print(report)
    except DailyQuotaExceeded as exc:
        msg = str(exc)[:300]
        print(f"DailyQuotaExceeded: {msg}", file=sys.stderr)
        sys.exit(2)
    except Exception as exc:  # noqa: BLE001
        msg = str(exc)[:300]
        print(f"Error ({type(exc).__name__}): {msg}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
