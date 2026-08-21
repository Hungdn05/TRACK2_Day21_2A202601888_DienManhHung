import argparse
import json
from pathlib import Path


MINIMUM_F1 = 0.65


class QualityGateError(RuntimeError):
    """Raised when a candidate model is not eligible for production."""


def check_quality_gate(candidate_f1: float, production_f1: float | None = None) -> list[str]:
    """Validate the minimum-quality and non-regression release policies."""
    messages = []
    if candidate_f1 < MINIMUM_F1:
        raise QualityGateError(
            f"FAILED minimum gate: f1_score {candidate_f1:.4f} < "
            f"{MINIMUM_F1:.2f}. Release is blocked."
        )
    messages.append(
        f"PASSED minimum gate: {candidate_f1:.4f} >= {MINIMUM_F1:.2f}"
    )

    if production_f1 is None:
        messages.append(
            "No production report exists yet; regression comparison skipped."
        )
    else:
        messages.append(
            f"Regression comparison: candidate={candidate_f1:.4f}, "
            f"production={production_f1:.4f}"
        )
        if candidate_f1 < production_f1:
            raise QualityGateError(
                f"FAILED regression gate: {candidate_f1:.4f} < "
                f"{production_f1:.4f}. Production model is retained."
            )

    messages.append("PASSED all quality gates. Candidate is eligible for release.")
    return messages


def read_f1(report_path: Path) -> float:
    return float(json.loads(report_path.read_text(encoding="utf-8"))["f1_score"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--production", type=Path)
    args = parser.parse_args()

    candidate_f1 = read_f1(args.candidate)
    production_f1 = None
    if args.production and args.production.exists():
        production_f1 = read_f1(args.production)

    try:
        messages = check_quality_gate(candidate_f1, production_f1)
    except QualityGateError as error:
        raise SystemExit(str(error)) from error

    print("\n".join(messages))


if __name__ == "__main__":
    main()
