import pytest

from src.quality_gate import QualityGateError, check_quality_gate


def test_quality_gate_accepts_equal_or_better_candidate():
    messages = check_quality_gate(candidate_f1=0.75, production_f1=0.74)
    assert messages[-1].startswith("PASSED all quality gates")


def test_quality_gate_blocks_below_minimum():
    with pytest.raises(QualityGateError, match="minimum gate"):
        check_quality_gate(candidate_f1=0.64, production_f1=0.60)


def test_quality_gate_blocks_regression():
    with pytest.raises(QualityGateError, match="regression gate"):
        check_quality_gate(candidate_f1=0.72, production_f1=0.74)
