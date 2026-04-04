"""Tests for budget constraint integration."""

from __future__ import annotations

import pytest
from pyrapide import Computation, Event

from trllm.cost_forensics.constraints import BudgetExceeded, CostPerRootExceeded
from trllm.cost_forensics.pricing import PricingRegistry


def _make_comp(pricing: PricingRegistry) -> Computation:
    """Build a computation with known cost."""
    comp = Computation()
    root = Event(name="user_request", payload={})
    llm = Event(
        name="llm_call",
        payload={
            "model": "gpt-4o",
            "usage": {"input_tokens": 1_000_000, "output_tokens": 0},
        },
    )
    comp.record(root)
    comp.record(llm, caused_by=[root])
    return comp  # total cost = $2.50


class TestBudgetExceeded:
    def test_no_violation_under_budget(self, pricing: PricingRegistry) -> None:
        comp = _make_comp(pricing)
        constraint = BudgetExceeded(budget=10.0, pricing=pricing)
        violations = constraint.check(comp)
        assert len(violations) == 0

    def test_violation_over_budget(self, pricing: PricingRegistry) -> None:
        comp = _make_comp(pricing)
        constraint = BudgetExceeded(budget=1.0, pricing=pricing)
        violations = constraint.check(comp)
        assert len(violations) == 1
        assert violations[0].constraint_name == "BudgetExceeded"
        assert violations[0].violation_type == "budget_exceeded"
        assert "$" in violations[0].description

    def test_custom_message(self, pricing: PricingRegistry) -> None:
        comp = _make_comp(pricing)
        constraint = BudgetExceeded(
            budget=0.01, pricing=pricing, message="Too expensive!"
        )
        violations = constraint.check(comp)
        assert violations[0].description == "Too expensive!"

    def test_custom_currency(self, pricing: PricingRegistry) -> None:
        comp = _make_comp(pricing)
        constraint = BudgetExceeded(
            budget=1.0, pricing=pricing, currency="EUR"
        )
        violations = constraint.check(comp)
        assert len(violations) == 1

    def test_repr(self, pricing: PricingRegistry) -> None:
        c = BudgetExceeded(budget=5.0, pricing=pricing)
        assert "BudgetExceeded" in repr(c)
        assert "$5.00" in repr(c)


class TestCostPerRootExceeded:
    def test_no_violation_under_budget(self, pricing: PricingRegistry) -> None:
        comp = _make_comp(pricing)
        constraint = CostPerRootExceeded(
            per_root_budget=10.0, pricing=pricing
        )
        violations = constraint.check(comp)
        assert len(violations) == 0

    def test_violation_over_budget(self, pricing: PricingRegistry) -> None:
        comp = _make_comp(pricing)
        constraint = CostPerRootExceeded(
            per_root_budget=0.01, pricing=pricing
        )
        violations = constraint.check(comp)
        assert len(violations) == 1
        assert violations[0].constraint_name == "CostPerRootExceeded"
        assert "user_request" in violations[0].description

    def test_multiple_roots_one_over(self, pricing: PricingRegistry) -> None:
        comp = Computation()
        cheap_root = Event(name="cheap_root", payload={})
        expensive_root = Event(name="expensive_root", payload={})
        llm = Event(
            name="llm_call",
            payload={
                "model": "gpt-4o",
                "usage": {"input_tokens": 1_000_000, "output_tokens": 0},
            },
        )
        comp.record(cheap_root)
        comp.record(expensive_root)
        comp.record(llm, caused_by=[expensive_root])

        constraint = CostPerRootExceeded(
            per_root_budget=1.0, pricing=pricing
        )
        violations = constraint.check(comp)
        assert len(violations) == 1
        assert "expensive_root" in violations[0].description

    def test_repr(self, pricing: PricingRegistry) -> None:
        c = CostPerRootExceeded(per_root_budget=0.50, pricing=pricing)
        assert "CostPerRootExceeded" in repr(c)
        assert "$0.50" in repr(c)
