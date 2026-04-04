"""Tests for CostDiff."""

from __future__ import annotations

import pytest
from pyrapide import Computation, Event

from trllm.cost_forensics.annotator import CostAnnotator
from trllm.cost_forensics.diff import CostDelta, CostDiff, diff_reports
from trllm.cost_forensics.pricing import ModelPrice, PricingRegistry
from trllm.cost_forensics.rollup import CausalCostRollup


def _make_report(
    pricing: PricingRegistry, input_tokens: int
) -> "ForensicReport":
    from trllm.cost_forensics.reports import ForensicReport

    comp = Computation()
    root = Event(name="root", payload={})
    llm = Event(
        name="llm_call",
        payload={
            "model": "gpt-4o",
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": 100,
            },
        },
    )
    comp.record(root)
    comp.record(llm, caused_by=[root])

    annotations = CostAnnotator(pricing).annotate(comp)
    return CausalCostRollup().rollup(comp, annotations)


class TestDiffReports:
    def test_identical_reports(self, pricing: PricingRegistry) -> None:
        report = _make_report(pricing, 1000)
        result = diff_reports(report, report)

        assert result.total_delta == pytest.approx(0.0)
        assert result.pct_change == pytest.approx(0.0)
        assert len(result.regressions) == 0
        assert len(result.improvements) == 0

    def test_regression_detected(self, pricing: PricingRegistry) -> None:
        before = _make_report(pricing, 1000)
        after = _make_report(pricing, 5000)

        result = diff_reports(before, after)
        assert result.total_delta > 0
        assert result.pct_change > 0
        assert len(result.regressions) > 0

    def test_improvement_detected(self, pricing: PricingRegistry) -> None:
        before = _make_report(pricing, 5000)
        after = _make_report(pricing, 1000)

        result = diff_reports(before, after)
        assert result.total_delta < 0
        assert len(result.improvements) > 0

    def test_custom_threshold(self, pricing: PricingRegistry) -> None:
        before = _make_report(pricing, 1000)
        after = _make_report(pricing, 1050)

        # With default 10% threshold
        result_default = diff_reports(before, after)
        # With 1% threshold
        result_strict = diff_reports(before, after, regression_threshold_pct=1.0)

        # Strict threshold should flag more regressions
        assert len(result_strict.regressions) >= len(result_default.regressions)

    def test_summary_output(self, pricing: PricingRegistry) -> None:
        before = _make_report(pricing, 1000)
        after = _make_report(pricing, 5000)

        result = diff_reports(before, after)
        summary = result.summary()
        assert "Cost diff" in summary
        assert "$" in summary

    def test_to_dict(self, pricing: PricingRegistry) -> None:
        before = _make_report(pricing, 1000)
        after = _make_report(pricing, 5000)

        result = diff_reports(before, after)
        d = result.to_dict()
        assert "before_total" in d
        assert "after_total" in d
        assert "deltas" in d

    def test_cost_delta_repr(self) -> None:
        cd = CostDelta("test", 1.0, 2.0, 1.0, 100.0)
        assert "test" in repr(cd)
