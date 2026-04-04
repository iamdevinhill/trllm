"""Waste detection patterns for LLM/agent pipelines."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from pyrapide import Computation, Event  # type: ignore[import-untyped]

from .annotator import CostAnnotation
from .reports import WasteInstance, WasteReport


class WastePattern(ABC):
    """Base class for waste detection patterns."""

    name: str
    description: str

    @abstractmethod
    def detect(
        self,
        comp: Computation,
        annotations: dict[str, CostAnnotation],
    ) -> list[WasteInstance]: ...


@dataclass
class RetryStorm(WastePattern):
    """Detects same tool called 3+ times under the same root."""

    name: str = "RetryStorm"
    description: str = (
        "Same tool name called 3+ times with causes tracing to the same root"
    )

    def detect(
        self,
        comp: Computation,
        annotations: dict[str, CostAnnotation],
    ) -> list[WasteInstance]:
        roots = comp.root_events()
        instances: list[WasteInstance] = []

        for root in roots:
            descendants = comp.descendants(root)
            # Group descendants by event name
            by_name: dict[str, list[Event]] = defaultdict(list)
            for event in descendants:
                by_name[event.name].append(event)

            for tool_name, events in by_name.items():
                if len(events) < 3:
                    continue
                # Check they share a common causal parent (same cause)
                # Sort by timestamp for deterministic ordering
                sorted_events = sorted(events, key=lambda e: e.timestamp)
                total_cost = sum(
                    annotations.get(e.id, CostAnnotation(e.id, 0.0)).direct_cost
                    for e in sorted_events
                )
                # Waste = all but the first call
                first_cost = annotations.get(
                    sorted_events[0].id,
                    CostAnnotation(sorted_events[0].id, 0.0),
                ).direct_cost
                waste = total_cost - first_cost

                instances.append(
                    WasteInstance(
                        pattern_name=self.name,
                        description=(
                            f"{tool_name} called {len(events)}x "
                            f"under root {root.name}"
                        ),
                        events=sorted_events,
                        estimated_waste=waste,
                        severity="high",
                    )
                )
        return instances


@dataclass
class DeadEndToolCall(WastePattern):
    """Detects tool calls whose results caused zero LLM events downstream."""

    name: str = "DeadEndToolCall"
    description: str = (
        "Tool call event whose results caused zero LLM events downstream"
    )

    def detect(
        self,
        comp: Computation,
        annotations: dict[str, CostAnnotation],
    ) -> list[WasteInstance]:
        poset: Any = comp._poset  # noqa: SLF001
        instances: list[WasteInstance] = []

        for event in comp:
            # Heuristic: tool call events have "tool" in the name
            # or are not LLM events (no "usage" in payload)
            if "usage" in event.payload:
                continue  # This is an LLM event, skip

            effects: frozenset[Event] = poset.effects(event)
            if effects:
                continue  # Has downstream effects

            # Check that this event is not a root (roots aren't tool calls)
            causes = poset.causes(event)
            if not causes:
                continue  # Root event, not a dead-end tool call

            cost = annotations.get(
                event.id, CostAnnotation(event.id, 0.0)
            ).direct_cost

            instances.append(
                WasteInstance(
                    pattern_name=self.name,
                    description=f"{event.name} result never consumed",
                    events=[event],
                    estimated_waste=cost,
                    severity="medium",
                )
            )
        return instances


@dataclass
class RedundantContext(WastePattern):
    """Detects two LLM calls with nearly identical prompt token counts."""

    name: str = "RedundantContext"
    description: str = (
        "Two LLM call events with prompt_tokens differing by less than 5%"
    )

    def detect(
        self,
        comp: Computation,
        annotations: dict[str, CostAnnotation],
    ) -> list[WasteInstance]:
        instances: list[WasteInstance] = []
        roots = comp.root_events()

        # Collect LLM events per root subtree
        for root in roots:
            descendants = comp.descendants(root)
            llm_events: list[Event] = []
            for event in descendants:
                if "usage" in event.payload:
                    llm_events.append(event)
            # Also check root itself
            if "usage" in root.payload:
                llm_events.append(root)

            # Compare pairs
            seen_pairs: set[tuple[str, str]] = set()
            for i, ev_a in enumerate(llm_events):
                tokens_a = ev_a.payload.get("usage", {}).get(
                    "input_tokens",
                    ev_a.payload.get("usage", {}).get("prompt_tokens", 0),
                )
                if tokens_a == 0:
                    continue
                for ev_b in llm_events[i + 1 :]:
                    pair_key = tuple(sorted([ev_a.id, ev_b.id]))
                    if pair_key in seen_pairs:
                        continue
                    seen_pairs.add(pair_key)

                    tokens_b = ev_b.payload.get("usage", {}).get(
                        "input_tokens",
                        ev_b.payload.get("usage", {}).get("prompt_tokens", 0),
                    )
                    if tokens_b == 0:
                        continue

                    max_tokens = max(tokens_a, tokens_b)
                    diff_pct = abs(tokens_a - tokens_b) / max_tokens * 100
                    if diff_pct < 5.0:
                        cost_a = annotations.get(
                            ev_a.id, CostAnnotation(ev_a.id, 0.0)
                        ).direct_cost
                        cost_b = annotations.get(
                            ev_b.id, CostAnnotation(ev_b.id, 0.0)
                        ).direct_cost
                        waste = min(cost_a, cost_b)
                        instances.append(
                            WasteInstance(
                                pattern_name=self.name,
                                description=(
                                    f"same ~{max_tokens}-token context "
                                    f"sent twice"
                                ),
                                events=[ev_a, ev_b],
                                estimated_waste=waste,
                                severity="low",
                            )
                        )
        return instances


@dataclass
class AbandonedBranch(WastePattern):
    """Detects subtrees (depth >= 2) whose leaves produced no output consumed by siblings."""

    name: str = "AbandonedBranch"
    description: str = (
        "Subtree of depth >= 2 where leaf events produced no output "
        "consumed by any sibling branch"
    )

    def detect(
        self,
        comp: Computation,
        annotations: dict[str, CostAnnotation],
    ) -> list[WasteInstance]:
        poset: Any = comp._poset  # noqa: SLF001
        instances: list[WasteInstance] = []
        roots = comp.root_events()

        for root in roots:
            children = poset.effects(root)
            if len(children) < 2:
                continue

            for branch_root in children:
                branch_descendants = comp.descendants(branch_root)
                branch_all = {branch_root} | branch_descendants

                # Check depth of this branch
                max_depth = self._max_depth(branch_root, poset, 0)
                if max_depth < 2:
                    continue

                # Get leaf events in this branch
                branch_leaves = {
                    e for e in branch_all if not poset.effects(e) & branch_all
                }

                # Check if any leaf has effects outside this branch
                has_external_effects = False
                for leaf in branch_leaves:
                    external = poset.effects(leaf) - branch_all
                    if external:
                        has_external_effects = True
                        break

                if not has_external_effects:
                    total_cost = sum(
                        annotations.get(
                            e.id, CostAnnotation(e.id, 0.0)
                        ).direct_cost
                        for e in branch_all
                    )
                    instances.append(
                        WasteInstance(
                            pattern_name=self.name,
                            description=(
                                f"branch rooted at {branch_root.name} "
                                f"({len(branch_all)} events) abandoned"
                            ),
                            events=sorted(
                                list(branch_all), key=lambda e: e.timestamp
                            ),
                            estimated_waste=total_cost,
                            severity="high",
                        )
                    )
        return instances

    def _max_depth(
        self, event: Event, poset: Any, current: int
    ) -> int:
        children: frozenset[Event] = poset.effects(event)
        if not children:
            return current
        return max(
            self._max_depth(child, poset, current + 1) for child in children
        )


_BUILTIN_PATTERNS: list[WastePattern] = [
    RetryStorm(),
    DeadEndToolCall(),
    RedundantContext(),
    AbandonedBranch(),
]


class WasteDetector:
    """Runs waste detection patterns against a Computation."""

    def __init__(self, patterns: list[WastePattern] | None = None) -> None:
        self._patterns = patterns if patterns is not None else list(_BUILTIN_PATTERNS)

    def __repr__(self) -> str:
        pattern_names = [p.name for p in self._patterns]
        return f"WasteDetector(patterns={pattern_names!r})"

    def detect(
        self,
        comp: Computation,
        annotations: dict[str, CostAnnotation],
    ) -> WasteReport:
        """Run all patterns and return a WasteReport."""
        all_instances: list[WasteInstance] = []
        for pattern in self._patterns:
            all_instances.extend(pattern.detect(comp, annotations))

        total_waste = sum(inst.estimated_waste for inst in all_instances)
        return WasteReport(
            instances=all_instances,
            total_waste=total_waste,
        )
