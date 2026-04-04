"""Tests for ForensicReport and WasteReport."""

from __future__ import annotations

import json

import pytest
from pyrapide import Computation, Event

from trllm.cost_forensics.annotator import CostAnnotator
from trllm.cost_forensics.pricing import PricingRegistry
from trllm.cost_forensics.reports import ForensicReport, WasteReport
from trllm.cost_forensics.rollup import CausalCostRollup
from trllm.cost_forensics.waste import WasteDetector


class TestForensicReport:
    def test_ascii_tree_deterministic(
        self, pricing: PricingRegistry, simple_comp: Computation
    ) -> None:
        annotations = CostAnnotator(pricing).annotate(simple_comp)
        report = CausalCostRollup().rollup(simple_comp, annotations)

        tree1 = report.ascii_tree()
        tree2 = report.ascii_tree()
        assert tree1 == tree2
        assert "user_request" in tree1
        assert "$" in tree1

    def test_ascii_tree_max_depth(
        self, pricing: PricingRegistry, simple_comp: Computation
    ) -> None:
        annotations = CostAnnotator(pricing).annotate(simple_comp)
        report = CausalCostRollup().rollup(simple_comp, annotations)

        shallow = report.ascii_tree(max_depth=1)
        deep = report.ascii_tree(max_depth=6)
        # Shallow tree should have fewer lines
        assert shallow.count("\n") <= deep.count("\n")

    def test_ascii_tree_with_waste_markers(
        self, pricing: PricingRegistry, retry_comp: Computation
    ) -> None:
        annotations = CostAnnotator(pricing).annotate(retry_comp)
        report = CausalCostRollup().rollup(retry_comp, annotations)
        waste = WasteDetector().detect(retry_comp, annotations)
        report.attach_waste(waste)

        tree = report.ascii_tree()
        assert "\u26a0" in tree  # Warning marker

    def test_to_dict_is_json_serializable(
        self, pricing: PricingRegistry, simple_comp: Computation
    ) -> None:
        annotations = CostAnnotator(pricing).annotate(simple_comp)
        report = CausalCostRollup().rollup(simple_comp, annotations)

        d = report.to_dict()
        # Should not raise
        json_str = json.dumps(d)
        parsed = json.loads(json_str)
        assert parsed["total_cost"] == report.total_cost

    def test_to_dict_with_waste(
        self, pricing: PricingRegistry, retry_comp: Computation
    ) -> None:
        annotations = CostAnnotator(pricing).annotate(retry_comp)
        report = CausalCostRollup().rollup(retry_comp, annotations)
        waste = WasteDetector().detect(retry_comp, annotations)
        report.attach_waste(waste)

        d = report.to_dict()
        assert "waste" in d

    def test_top_costs(
        self, pricing: PricingRegistry, large_comp: Computation
    ) -> None:
        annotations = CostAnnotator(pricing).annotate(large_comp)
        report = CausalCostRollup().rollup(large_comp, annotations)

        top5 = report.top_costs(5)
        assert len(top5) == 5
        # Should be sorted descending
        for i in range(len(top5) - 1):
            assert top5[i].causal_cost >= top5[i + 1].causal_cost

    def test_attach_waste(
        self, pricing: PricingRegistry, simple_comp: Computation
    ) -> None:
        annotations = CostAnnotator(pricing).annotate(simple_comp)
        report = CausalCostRollup().rollup(simple_comp, annotations)

        assert report.waste is None
        waste = WasteReport(instances=[], total_waste=0.0)
        report.attach_waste(waste)
        assert report.waste is waste


class TestWasteReport:
    def test_summary_no_waste(self) -> None:
        report = WasteReport(instances=[], total_waste=0.0)
        assert report.summary() == "No waste detected."

    def test_to_dict(
        self, pricing: PricingRegistry, retry_comp: Computation
    ) -> None:
        annotations = CostAnnotator(pricing).annotate(retry_comp)
        report = WasteDetector().detect(retry_comp, annotations)

        d = report.to_dict()
        json_str = json.dumps(d)
        parsed = json.loads(json_str)
        assert parsed["total_waste"] == report.total_waste

    def test_by_severity(
        self, pricing: PricingRegistry, retry_comp: Computation
    ) -> None:
        annotations = CostAnnotator(pricing).annotate(retry_comp)
        report = WasteDetector().detect(retry_comp, annotations)

        by_sev = report.by_severity()
        assert isinstance(by_sev, dict)
        assert all(k in by_sev for k in ("high", "medium", "low"))
