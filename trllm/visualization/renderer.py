"""Visualization wrappers around PyRapide's analysis.visualization module."""

from __future__ import annotations

from pyrapide import Computation, visualization


def render_summary(computation: Computation) -> str:
    return visualization.summary(computation)


def render_mermaid(computation: Computation) -> str:
    return visualization.to_mermaid(computation)


def render_dot(computation: Computation) -> str:
    return visualization.to_dot(computation)


def render_ascii(computation: Computation) -> str:
    return visualization.to_ascii(computation)
