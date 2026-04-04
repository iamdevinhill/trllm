"""Tests for CausalCostRollup."""

from __future__ import annotations

import pytest
from pyrapide import Computation, Event

from trllm.cost_forensics.annotator import CostAnnotator
from trllm.cost_forensics.pricing import PricingRegistry
from trllm.cost_forensics.reports import RollupNode
from trllm.cost_forensics.rollup import CausalCostRollup


class TestCausalCostRollup:
    def test_simple_chain_rollup(
        self, pricing: PricingRegistry, simple_comp: Computation
    ) -> None:
        annotations = CostAnnotator(pricing).annotate(simple_comp)
        report = CausalCostRollup().rollup(simple_comp, annotations)

        assert len(report.roots) == 1
        root = report.roots[0]
        assert root.event.name == "user_request"
        assert root.depth == 0

        # Root causal_cost should equal total_cost
        assert root.causal_cost == pytest.approx(report.total_cost)

        # Root direct cost is 0 (user_request has no pricing)
        assert root.direct_cost == pytest.approx(0.0)

        # Causal cost should be sum of all event costs
        total_direct = sum(a.direct_cost for a in annotations.values())
        assert report.total_cost == pytest.approx(total_direct)

    def test_causal_cost_includes_descendants(
        self, pricing: PricingRegistry
    ) -> None:
        comp = Computation()
        root = Event(name="root", payload={})
        child = Event(
            name="llm_call",
            payload={
                "model": "gpt-4o",
                "usage": {"input_tokens": 1000, "output_tokens": 0},
            },
        )
        comp.record(root)
        comp.record(child, caused_by=[root])

        annotations = CostAnnotator(pricing).annotate(comp)
        report = CausalCostRollup().rollup(comp, annotations)

        root_node = report.roots[0]
        # root has 0 direct cost, but causal cost includes child
        assert root_node.direct_cost == 0.0
        assert root_node.causal_cost > 0.0
        assert len(root_node.children) == 1
        assert root_node.children[0].direct_cost == root_node.causal_cost

    def test_multiple_roots(self, pricing: PricingRegistry) -> None:
        comp = Computation()
        r1 = Event(name="root_a", payload={})
        r2 = Event(name="root_b", payload={})
        comp.record(r1)
        comp.record(r2)

        annotations = CostAnnotator(pricing).annotate(comp)
        report = CausalCostRollup().rollup(comp, annotations)

        assert len(report.roots) == 2
        assert report.total_cost == pytest.approx(0.0)

    def test_branching_rollup(self, pricing: PricingRegistry) -> None:
        comp = Computation()
        root = Event(name="root", payload={})
        llm1 = Event(
            name="llm_a",
            payload={
                "model": "gpt-4o",
                "usage": {"input_tokens": 1000, "output_tokens": 100},
            },
        )
        llm2 = Event(
            name="llm_b",
            payload={
                "model": "gpt-4o",
                "usage": {"input_tokens": 2000, "output_tokens": 200},
            },
        )
        comp.record(root)
        comp.record(llm1, caused_by=[root])
        comp.record(llm2, caused_by=[root])

        annotations = CostAnnotator(pricing).annotate(comp)
        report = CausalCostRollup().rollup(comp, annotations)

        root_node = report.roots[0]
        assert len(root_node.children) == 2
        child_costs = sum(c.causal_cost for c in root_node.children)
        assert root_node.causal_cost == pytest.approx(child_costs)

    def test_depth_tracking(
        self, pricing: PricingRegistry, simple_comp: Computation
    ) -> None:
        annotations = CostAnnotator(pricing).annotate(simple_comp)
        report = CausalCostRollup().rollup(simple_comp, annotations)

        # Walk the chain and check depths
        node = report.roots[0]
        depth = 0
        while node.children:
            assert node.depth == depth
            node = node.children[0]
            depth += 1
        assert node.depth == depth

    def test_large_comp_performance(
        self, pricing: PricingRegistry, large_comp: Computation
    ) -> None:
        annotations = CostAnnotator(pricing).annotate(large_comp)
        report = CausalCostRollup().rollup(large_comp, annotations)

        assert report.total_cost > 0
        assert len(report.roots) == 1
        top = report.top_costs(5)
        assert len(top) == 5

    def test_repr(self) -> None:
        assert "CausalCostRollup" in repr(CausalCostRollup())
