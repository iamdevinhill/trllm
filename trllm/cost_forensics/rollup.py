"""Causal cost rollup — walks the DAG and sums costs per subtree."""

from __future__ import annotations

from typing import Any

from pyrapide import Computation, Event  # type: ignore[import-untyped]

from .annotator import CostAnnotation
from .reports import ForensicReport, RollupNode


class CausalCostRollup:
    """Builds a causal cost tree from a Computation and its annotations."""

    def __repr__(self) -> str:
        return "CausalCostRollup()"

    def rollup(
        self,
        comp: Computation,
        annotations: dict[str, CostAnnotation],
    ) -> ForensicReport:
        """Build a tree of RollupNodes rooted at events with no causes.

        causal_cost for each node = direct_cost + sum(child.causal_cost).
        """
        poset: Any = comp._poset  # noqa: SLF001
        roots: frozenset[Event] = comp.root_events()

        root_nodes: list[RollupNode] = []
        for root_event in roots:
            node = self._build_node(root_event, poset, annotations, depth=0)
            root_nodes.append(node)

        total_cost = sum(n.causal_cost for n in root_nodes)

        return ForensicReport(
            roots=root_nodes,
            total_cost=total_cost,
            annotations=annotations,
        )

    def _build_node(
        self,
        event: Event,
        poset: Any,
        annotations: dict[str, CostAnnotation],
        depth: int,
    ) -> RollupNode:
        annotation = annotations.get(event.id)
        direct_cost = annotation.direct_cost if annotation else 0.0

        # Get direct effects (children in the causal DAG)
        children_events: frozenset[Event] = poset.effects(event)
        child_nodes: list[RollupNode] = []
        for child_event in children_events:
            child_node = self._build_node(
                child_event, poset, annotations, depth + 1
            )
            child_nodes.append(child_node)

        causal_cost = direct_cost + sum(c.causal_cost for c in child_nodes)

        return RollupNode(
            event=event,
            direct_cost=direct_cost,
            causal_cost=causal_cost,
            children=child_nodes,
            depth=depth,
        )
