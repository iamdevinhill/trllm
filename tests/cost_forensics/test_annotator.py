"""Tests for CostAnnotator."""

from __future__ import annotations

from pyrapide import Computation, Event

from trllm.cost_forensics.annotator import CostAnnotation, CostAnnotator
from trllm.cost_forensics.pricing import PricingRegistry


class TestCostAnnotator:
    def test_annotate_simple_comp(
        self, pricing: PricingRegistry, simple_comp: Computation
    ) -> None:
        annotator = CostAnnotator(pricing)
        annotations = annotator.annotate(simple_comp)

        # Should have an annotation for every event
        assert len(annotations) == len(list(simple_comp))

        # All annotations should have USD currency
        for ann in annotations.values():
            assert ann.currency == "USD"

    def test_llm_event_cost(self, pricing: PricingRegistry) -> None:
        comp = Computation()
        ev = Event(
            name="llm_call",
            payload={
                "model": "gpt-4o",
                "usage": {"input_tokens": 1_000_000, "output_tokens": 0},
            },
        )
        comp.record(ev)

        annotations = CostAnnotator(pricing).annotate(comp)
        # 1M input tokens at $2.50/M = $2.50
        assert annotations[ev.id].direct_cost == pytest.approx(2.50)

    def test_tool_event_cost(self, pricing: PricingRegistry) -> None:
        comp = Computation()
        ev = Event(name="query_db", payload={})
        comp.record(ev)

        annotations = CostAnnotator(pricing).annotate(comp)
        assert annotations[ev.id].direct_cost == pytest.approx(0.001)

    def test_unknown_event_cost_is_zero(
        self, pricing: PricingRegistry
    ) -> None:
        comp = Computation()
        ev = Event(name="unknown_event", payload={})
        comp.record(ev)

        annotations = CostAnnotator(pricing).annotate(comp)
        assert annotations[ev.id].direct_cost == 0.0

    def test_custom_currency(self, pricing: PricingRegistry) -> None:
        comp = Computation()
        ev = Event(name="query_db", payload={})
        comp.record(ev)

        annotations = CostAnnotator(pricing, currency="EUR").annotate(comp)
        assert annotations[ev.id].currency == "EUR"

    def test_does_not_mutate_computation(
        self, pricing: PricingRegistry, simple_comp: Computation
    ) -> None:
        events_before = list(simple_comp)
        CostAnnotator(pricing).annotate(simple_comp)
        events_after = list(simple_comp)
        assert events_before == events_after

    def test_repr(self, pricing: PricingRegistry) -> None:
        annotator = CostAnnotator(pricing)
        assert "USD" in repr(annotator)


import pytest  # noqa: E402
