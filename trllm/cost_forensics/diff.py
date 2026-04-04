"""Cost diffing between two forensic reports."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from .reports import ForensicReport, RollupNode


@dataclass
class CostDelta:
    """Cost change for a single event name."""

    event_name: str
    before: float
    after: float
    delta: float  # after - before
    pct_change: float

    def __repr__(self) -> str:
        return (
            f"CostDelta({self.event_name!r}, "
            f"${self.before:.4f} -> ${self.after:.4f}, "
            f"{self.pct_change:+.1f}%)"
        )


@dataclass
class CostDiff:
    """Diff between two ForensicReports."""

    before_total: float
    after_total: float
    total_delta: float
    pct_change: float
    deltas: list[CostDelta]
    regressions: list[CostDelta]
    improvements: list[CostDelta]

    def summary(self) -> str:
        """Human-readable diff summary."""
        lines: list[str] = [
            f"Cost diff: ${self.before_total:.4f} -> "
            f"${self.after_total:.4f} "
            f"({self.pct_change:+.1f}%)"
        ]
        if self.regressions:
            lines.append(f"\nRegressions ({len(self.regressions)}):")
            for d in sorted(
                self.regressions, key=lambda x: x.delta, reverse=True
            ):
                lines.append(
                    f"  \u2191 {d.event_name}: "
                    f"${d.before:.4f} -> ${d.after:.4f} "
                    f"({d.pct_change:+.1f}%)"
                )
        if self.improvements:
            lines.append(f"\nImprovements ({len(self.improvements)}):")
            for d in sorted(self.improvements, key=lambda x: x.delta):
                lines.append(
                    f"  \u2193 {d.event_name}: "
                    f"${d.before:.4f} -> ${d.after:.4f} "
                    f"({d.pct_change:+.1f}%)"
                )
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable dict."""
        return {
            "before_total": self.before_total,
            "after_total": self.after_total,
            "total_delta": self.total_delta,
            "pct_change": self.pct_change,
            "deltas": [
                {
                    "event_name": d.event_name,
                    "before": d.before,
                    "after": d.after,
                    "delta": d.delta,
                    "pct_change": d.pct_change,
                }
                for d in self.deltas
            ],
        }


def _aggregate_costs_by_name(report: ForensicReport) -> dict[str, float]:
    """Sum direct costs by event name across the entire report."""
    costs: dict[str, float] = defaultdict(float)

    def _walk(nodes: list[RollupNode]) -> None:
        for node in nodes:
            costs[node.event.name] += node.direct_cost
            _walk(node.children)

    _walk(report.roots)
    return dict(costs)


def diff_reports(
    before: ForensicReport,
    after: ForensicReport,
    regression_threshold_pct: float = 10.0,
) -> CostDiff:
    """Compare two ForensicReports by aggregating costs per event name."""
    before_costs = _aggregate_costs_by_name(before)
    after_costs = _aggregate_costs_by_name(after)

    all_names = sorted(set(before_costs) | set(after_costs))
    deltas: list[CostDelta] = []
    regressions: list[CostDelta] = []
    improvements: list[CostDelta] = []

    for name in all_names:
        b = before_costs.get(name, 0.0)
        a = after_costs.get(name, 0.0)
        delta = a - b
        pct = (delta / b * 100) if b > 0 else (100.0 if a > 0 else 0.0)
        cd = CostDelta(
            event_name=name,
            before=b,
            after=a,
            delta=delta,
            pct_change=pct,
        )
        deltas.append(cd)
        if delta > 0 and pct > regression_threshold_pct:
            regressions.append(cd)
        elif delta < 0:
            improvements.append(cd)

    total_delta = after.total_cost - before.total_cost
    pct_change = (
        (total_delta / before.total_cost * 100)
        if before.total_cost > 0
        else (100.0 if after.total_cost > 0 else 0.0)
    )

    return CostDiff(
        before_total=before.total_cost,
        after_total=after.total_cost,
        total_delta=total_delta,
        pct_change=pct_change,
        deltas=deltas,
        regressions=regressions,
        improvements=improvements,
    )
