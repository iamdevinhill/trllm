"""Tests for WasteDetector and built-in patterns."""

from __future__ import annotations

import pytest
from pyrapide import Computation, Event

from trllm.cost_forensics.annotator import CostAnnotator
from trllm.cost_forensics.pricing import PricingRegistry
from trllm.cost_forensics.waste import (
    AbandonedBranch,
    DeadEndToolCall,
    RedundantContext,
    RetryStorm,
    WasteDetector,
    WasteInstance,
    WastePattern,
)


class TestRetryStorm:
    def test_detects_retry_storm(
        self, pricing: PricingRegistry, retry_comp: Computation
    ) -> None:
        annotations = CostAnnotator(pricing).annotate(retry_comp)
        instances = RetryStorm().detect(retry_comp, annotations)

        assert len(instances) == 1
        inst = instances[0]
        assert inst.pattern_name == "RetryStorm"
        assert inst.severity == "high"
        assert len(inst.events) == 4
        # Waste = cost of 3 retries (all but first)
        per_call = 0.001  # query_db price
        assert inst.estimated_waste == pytest.approx(per_call * 3)

    def test_no_storm_with_fewer_than_3(
        self, pricing: PricingRegistry
    ) -> None:
        comp = Computation()
        root = Event(name="root", payload={})
        comp.record(root)
        for _ in range(2):
            tool = Event(name="query_db", payload={})
            comp.record(tool, caused_by=[root])

        annotations = CostAnnotator(pricing).annotate(comp)
        instances = RetryStorm().detect(comp, annotations)
        assert len(instances) == 0


class TestDeadEndToolCall:
    def test_detects_dead_end(
        self, pricing: PricingRegistry, dead_end_comp: Computation
    ) -> None:
        annotations = CostAnnotator(pricing).annotate(dead_end_comp)
        instances = DeadEndToolCall().detect(dead_end_comp, annotations)

        assert len(instances) == 1
        inst = instances[0]
        assert inst.pattern_name == "DeadEndToolCall"
        assert inst.severity == "medium"
        assert "web_search" in inst.description
        assert inst.estimated_waste == pytest.approx(0.005)

    def test_no_dead_end_when_consumed(
        self, pricing: PricingRegistry
    ) -> None:
        comp = Computation()
        root = Event(name="root", payload={})
        tool = Event(name="web_search", payload={})
        llm = Event(
            name="llm_call",
            payload={
                "model": "gpt-4o",
                "usage": {"input_tokens": 100, "output_tokens": 50},
            },
        )
        comp.record(root)
        comp.record(tool, caused_by=[root])
        comp.record(llm, caused_by=[tool])

        annotations = CostAnnotator(pricing).annotate(comp)
        instances = DeadEndToolCall().detect(comp, annotations)
        assert len(instances) == 0


class TestRedundantContext:
    def test_detects_redundant_context(
        self, pricing: PricingRegistry, redundant_context_comp: Computation
    ) -> None:
        annotations = CostAnnotator(pricing).annotate(redundant_context_comp)
        instances = RedundantContext().detect(redundant_context_comp, annotations)

        assert len(instances) == 1
        inst = instances[0]
        assert inst.pattern_name == "RedundantContext"
        assert inst.severity == "low"
        assert len(inst.events) == 2

    def test_no_redundancy_with_different_sizes(
        self, pricing: PricingRegistry
    ) -> None:
        comp = Computation()
        root = Event(name="root", payload={})
        llm1 = Event(
            name="llm_call",
            payload={
                "model": "gpt-4o",
                "usage": {"input_tokens": 1000, "output_tokens": 100},
            },
        )
        llm2 = Event(
            name="llm_call",
            payload={
                "model": "gpt-4o",
                "usage": {"input_tokens": 5000, "output_tokens": 100},
            },
        )
        comp.record(root)
        comp.record(llm1, caused_by=[root])
        comp.record(llm2, caused_by=[root])

        annotations = CostAnnotator(pricing).annotate(comp)
        instances = RedundantContext().detect(comp, annotations)
        assert len(instances) == 0


class TestAbandonedBranch:
    def test_detects_abandoned_branch(
        self, pricing: PricingRegistry, multi_agent_comp: Computation
    ) -> None:
        annotations = CostAnnotator(pricing).annotate(multi_agent_comp)
        instances = AbandonedBranch().detect(multi_agent_comp, annotations)

        # Both branches are abandoned (neither has external effects)
        assert len(instances) >= 1
        for inst in instances:
            assert inst.pattern_name == "AbandonedBranch"
            assert inst.severity == "high"
            assert inst.estimated_waste > 0

    def test_no_abandoned_if_shallow(
        self, pricing: PricingRegistry
    ) -> None:
        comp = Computation()
        root = Event(name="root", payload={})
        child_a = Event(name="branch_a", payload={})
        child_b = Event(name="branch_b", payload={})
        comp.record(root)
        comp.record(child_a, caused_by=[root])
        comp.record(child_b, caused_by=[root])

        annotations = CostAnnotator(pricing).annotate(comp)
        instances = AbandonedBranch().detect(comp, annotations)
        # Depth < 2, should not detect
        assert len(instances) == 0


class TestWasteDetector:
    def test_runs_all_builtin_patterns(
        self, pricing: PricingRegistry, retry_comp: Computation
    ) -> None:
        annotations = CostAnnotator(pricing).annotate(retry_comp)
        report = WasteDetector().detect(retry_comp, annotations)

        assert report.total_waste > 0
        assert len(report.instances) > 0

    def test_custom_pattern(self, pricing: PricingRegistry) -> None:
        class AlwaysWaste(WastePattern):
            name = "AlwaysWaste"
            description = "Always finds waste"

            def detect(
                self,
                comp: Computation,
                annotations: dict,
            ) -> list[WasteInstance]:
                return [
                    WasteInstance(
                        pattern_name=self.name,
                        description="test",
                        events=[],
                        estimated_waste=1.0,
                        severity="low",
                    )
                ]

        comp = Computation()
        comp.record(Event(name="root", payload={}))
        annotations = CostAnnotator(pricing).annotate(comp)

        detector = WasteDetector(patterns=[AlwaysWaste()])
        report = detector.detect(comp, annotations)
        assert report.total_waste == pytest.approx(1.0)
        assert report.instances[0].pattern_name == "AlwaysWaste"

    def test_waste_report_summary(
        self, pricing: PricingRegistry, retry_comp: Computation
    ) -> None:
        annotations = CostAnnotator(pricing).annotate(retry_comp)
        report = WasteDetector().detect(retry_comp, annotations)

        summary = report.summary()
        assert "Waste detected" in summary
        assert "$" in summary

    def test_waste_report_to_dict(
        self, pricing: PricingRegistry, retry_comp: Computation
    ) -> None:
        annotations = CostAnnotator(pricing).annotate(retry_comp)
        report = WasteDetector().detect(retry_comp, annotations)

        d = report.to_dict()
        assert "total_waste" in d
        assert "instances" in d
        assert isinstance(d["instances"], list)

    def test_waste_report_by_severity(
        self, pricing: PricingRegistry, retry_comp: Computation
    ) -> None:
        annotations = CostAnnotator(pricing).annotate(retry_comp)
        report = WasteDetector().detect(retry_comp, annotations)

        by_sev = report.by_severity()
        assert "high" in by_sev
        assert "medium" in by_sev
        assert "low" in by_sev

    def test_empty_waste_report_summary(self) -> None:
        from trllm.cost_forensics.reports import WasteReport

        report = WasteReport(instances=[], total_waste=0.0)
        assert report.summary() == "No waste detected."

    def test_repr(self) -> None:
        detector = WasteDetector()
        assert "WasteDetector" in repr(detector)
