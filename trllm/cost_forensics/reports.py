"""Forensic and waste reports."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pyrapide import Event  # type: ignore[import-untyped]

from .annotator import CostAnnotation


@dataclass
class RollupNode:
    """A node in the causal cost tree."""

    event: Event
    direct_cost: float
    causal_cost: float  # direct + all descendants
    children: list[RollupNode]
    depth: int

    def __repr__(self) -> str:
        return (
            f"RollupNode(event={self.event.name!r}, "
            f"direct=${self.direct_cost:.4f}, "
            f"causal=${self.causal_cost:.4f}, "
            f"children={len(self.children)}, depth={self.depth})"
        )


@dataclass
class WasteInstance:
    """A single detected waste instance."""

    pattern_name: str
    description: str
    events: list[Event]
    estimated_waste: float  # USD
    severity: str  # "low", "medium", "high"

    def __repr__(self) -> str:
        return (
            f"WasteInstance(pattern={self.pattern_name!r}, "
            f"waste=${self.estimated_waste:.4f}, "
            f"severity={self.severity!r})"
        )


@dataclass
class WasteReport:
    """Report of detected waste patterns."""

    instances: list[WasteInstance]
    total_waste: float
    currency: str = "USD"

    def summary(self) -> str:
        """Human-readable waste summary."""
        if not self.instances:
            return "No waste detected."

        lines: list[str] = [
            f"Waste detected: ${self.total_waste:.3f} "
            f"across {len(self.instances)} pattern(s)"
        ]
        for inst in self.instances:
            event_names = ", ".join(e.name for e in inst.events[:3])
            if len(inst.events) > 3:
                event_names += f" (+{len(inst.events) - 3} more)"
            lines.append(
                f"  \u26a0 {inst.pattern_name} ({inst.severity})"
                f"  ${inst.estimated_waste:.3f}"
                f"  {event_names}"
            )
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable dict."""
        return {
            "total_waste": self.total_waste,
            "currency": self.currency,
            "instances": [
                {
                    "pattern_name": inst.pattern_name,
                    "description": inst.description,
                    "events": [e.id for e in inst.events],
                    "estimated_waste": inst.estimated_waste,
                    "severity": inst.severity,
                }
                for inst in self.instances
            ],
        }

    def by_severity(self) -> dict[str, list[WasteInstance]]:
        """Group instances by severity."""
        result: dict[str, list[WasteInstance]] = {
            "high": [],
            "medium": [],
            "low": [],
        }
        for inst in self.instances:
            result.setdefault(inst.severity, []).append(inst)
        return result


@dataclass
class ForensicReport:
    """Full causal cost forensic report."""

    roots: list[RollupNode]
    total_cost: float
    annotations: dict[str, CostAnnotation]
    currency: str = "USD"
    waste: WasteReport | None = field(default=None, repr=False)

    def ascii_tree(self, max_depth: int = 6) -> str:
        """Render the causal cost tree as indented ASCII.

        Siblings are sorted by causal_cost descending for deterministic output.
        """
        waste_event_ids: set[str] = set()
        if self.waste is not None:
            for inst in self.waste.instances:
                for ev in inst.events:
                    waste_event_ids.add(ev.id)

        lines: list[str] = []
        sorted_roots = sorted(
            self.roots, key=lambda n: n.causal_cost, reverse=True
        )
        for root in sorted_roots:
            self._render_node(root, lines, "", True, max_depth, waste_event_ids)
        return "\n".join(lines)

    def _render_node(
        self,
        node: RollupNode,
        lines: list[str],
        prefix: str,
        is_last: bool,
        max_depth: int,
        waste_ids: set[str],
    ) -> None:
        if node.depth > max_depth:
            return

        connector = "\u2514\u2500 " if is_last else "\u251c\u2500 "
        if node.depth == 0:
            connector = ""
            child_prefix = ""
        else:
            child_prefix = prefix + ("\u2502  " if not is_last else "   ")

        warn = " \u26a0" if node.event.id in waste_ids else ""
        line = (
            f"{prefix}{connector}{node.event.name}"
            f"  ${node.causal_cost:.4f}"
            f"  [direct: ${node.direct_cost:.4f}]{warn}"
        )
        lines.append(line)

        sorted_children = sorted(
            node.children, key=lambda n: n.causal_cost, reverse=True
        )
        for i, child in enumerate(sorted_children):
            is_child_last = i == len(sorted_children) - 1
            self._render_node(
                child, lines, child_prefix, is_child_last, max_depth, waste_ids
            )

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable dict."""

        def _node_to_dict(node: RollupNode) -> dict[str, Any]:
            return {
                "event_id": node.event.id,
                "event_name": node.event.name,
                "direct_cost": node.direct_cost,
                "causal_cost": node.causal_cost,
                "depth": node.depth,
                "children": [_node_to_dict(c) for c in node.children],
            }

        result: dict[str, Any] = {
            "total_cost": self.total_cost,
            "currency": self.currency,
            "roots": [_node_to_dict(r) for r in self.roots],
        }
        if self.waste is not None:
            result["waste"] = self.waste.to_dict()
        return result

    def top_costs(self, n: int = 10) -> list[RollupNode]:
        """Return the top n nodes by causal_cost descending."""
        all_nodes: list[RollupNode] = []
        self._collect_nodes(self.roots, all_nodes)
        all_nodes.sort(key=lambda node: node.causal_cost, reverse=True)
        return all_nodes[:n]

    def _collect_nodes(
        self, nodes: list[RollupNode], acc: list[RollupNode]
    ) -> None:
        for node in nodes:
            acc.append(node)
            self._collect_nodes(node.children, acc)

    def attach_waste(self, waste: WasteReport) -> None:
        """Attach a WasteReport for annotated ascii_tree output."""
        self.waste = waste
