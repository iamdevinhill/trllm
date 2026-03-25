"""Pipeline constraints using PyRapide's pattern algebra.

Constraints are context-aware — they only fire when the relevant event
types are present in the computation. A tool_completion constraint won't
trigger in a pipeline that never makes tool calls.
"""

from __future__ import annotations

from pyrapide import Computation, Pattern, must_match, never


def grounding_constraint():
    """Every final output must be causally grounded in at least one retrieved chunk."""
    return must_match(
        Pattern.match("chunk.injected") >> Pattern.match("final.response"),
        name="grounding",
        description="Final output must be grounded in at least one retrieved chunk",
    )


def tool_completion_constraint():
    """Every tool call must produce a result."""
    return must_match(
        Pattern.match("tool.call") >> Pattern.match("tool.result"),
        name="tool_completion",
        description="Every tool call must produce a result",
    )


def llm_request_response_constraint():
    """Every LLM response must have a preceding request."""
    return must_match(
        Pattern.match("llm.request") >> Pattern.match("llm.response"),
        name="llm_request_response",
        description="Every LLM response must have a preceding request",
    )


# Map constraint to the event types that must be present for it to apply
CONSTRAINT_REGISTRY = [
    {
        "constraint": grounding_constraint,
        "requires": {"chunk.injected", "final.response"},
    },
    {
        "constraint": tool_completion_constraint,
        "requires": {"tool.call"},
    },
    {
        "constraint": llm_request_response_constraint,
        "requires": {"llm.request", "llm.response"},
    },
]


def check_constraints(computation: Computation) -> list:
    event_names = {e.name for e in computation.events}

    violations = []
    for entry in CONSTRAINT_REGISTRY:
        # Only check constraints whose required event types are present
        if not entry["requires"].issubset(event_names):
            continue
        constraint = entry["constraint"]()
        result = constraint.check(computation)
        violations.extend(result)
    return violations
