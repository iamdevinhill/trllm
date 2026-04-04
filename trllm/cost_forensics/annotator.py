"""Cost annotation for computation events."""

from __future__ import annotations

from dataclasses import dataclass

from pyrapide import Computation  # type: ignore[import-untyped]

from .pricing import PricingRegistry


@dataclass
class CostAnnotation:
    """Cost annotation for a single event."""

    event_id: str
    direct_cost: float  # cost of this event alone
    currency: str = "USD"


class CostAnnotator:
    """Annotates every event in a Computation with its direct cost."""

    def __init__(
        self, pricing: PricingRegistry, currency: str = "USD"
    ) -> None:
        self._pricing = pricing
        self._currency = currency

    def __repr__(self) -> str:
        return f"CostAnnotator(currency={self._currency!r})"

    def annotate(self, comp: Computation) -> dict[str, CostAnnotation]:
        """Walk every event in comp and compute direct_cost.

        Returns a mapping of event_id -> CostAnnotation.
        Does not mutate the Computation.
        """
        annotations: dict[str, CostAnnotation] = {}
        for event in comp:
            cost = self._pricing.cost_for_event(event)
            annotations[event.id] = CostAnnotation(
                event_id=event.id,
                direct_cost=cost,
                currency=self._currency,
            )
        return annotations
