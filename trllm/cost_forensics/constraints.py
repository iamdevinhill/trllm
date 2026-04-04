"""PyRapide constraint integration for budget alerts."""

from __future__ import annotations

from pyrapide import Computation  # type: ignore[import-untyped]
from pyrapide.constraints import Constraint, ConstraintViolation  # type: ignore[import-untyped]

from .annotator import CostAnnotator
from .pricing import PricingRegistry
from .rollup import CausalCostRollup


class BudgetExceeded(Constraint):  # type: ignore[misc]
    """Constraint that fails if total causal cost exceeds a budget."""

    name = "BudgetExceeded"
    description = "Total computation cost must not exceed budget"

    def __init__(
        self,
        budget: float,
        pricing: PricingRegistry,
        currency: str = "USD",
        message: str | None = None,
    ) -> None:
        self._budget = budget
        self._pricing = pricing
        self._currency = currency
        self._message = message

    def __repr__(self) -> str:
        return (
            f"BudgetExceeded(budget=${self._budget:.2f}, "
            f"currency={self._currency!r})"
        )

    def check(self, computation: Computation) -> list[ConstraintViolation]:
        annotator = CostAnnotator(self._pricing, self._currency)
        annotations = annotator.annotate(computation)
        total = sum(a.direct_cost for a in annotations.values())

        if total > self._budget:
            msg = self._message or (
                f"Total cost ${total:.4f} exceeds budget "
                f"${self._budget:.4f}"
            )
            return [
                ConstraintViolation(
                    constraint_name=self.name,
                    violation_type="budget_exceeded",
                    description=msg,
                    matched_events=list(computation),
                )
            ]
        return []


class CostPerRootExceeded(Constraint):  # type: ignore[misc]
    """Constraint that fails if any root's causal cost exceeds per_root_budget."""

    name = "CostPerRootExceeded"
    description = "Per-root causal cost must not exceed budget"

    def __init__(
        self,
        per_root_budget: float,
        pricing: PricingRegistry,
        currency: str = "USD",
    ) -> None:
        self._per_root_budget = per_root_budget
        self._pricing = pricing
        self._currency = currency

    def __repr__(self) -> str:
        return (
            f"CostPerRootExceeded(per_root_budget="
            f"${self._per_root_budget:.2f})"
        )

    def check(self, computation: Computation) -> list[ConstraintViolation]:
        annotator = CostAnnotator(self._pricing, self._currency)
        annotations = annotator.annotate(computation)
        rollup = CausalCostRollup()
        report = rollup.rollup(computation, annotations)

        violations: list[ConstraintViolation] = []
        for root_node in report.roots:
            if root_node.causal_cost > self._per_root_budget:
                violations.append(
                    ConstraintViolation(
                        constraint_name=self.name,
                        violation_type="per_root_budget_exceeded",
                        description=(
                            f"Root '{root_node.event.name}' cost "
                            f"${root_node.causal_cost:.4f} exceeds "
                            f"per-root budget "
                            f"${self._per_root_budget:.4f}"
                        ),
                        matched_events=[root_node.event],
                    )
                )
        return violations
