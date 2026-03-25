"""Pipeline constraints using PyRapide's pattern algebra."""

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


PIPELINE_CONSTRAINTS = [
    grounding_constraint(),
    tool_completion_constraint(),
    llm_request_response_constraint(),
]


def check_constraints(computation: Computation) -> list[dict]:
    violations = []
    for constraint in PIPELINE_CONSTRAINTS:
        result = constraint.check(computation)
        violations.extend(result)
    return violations
