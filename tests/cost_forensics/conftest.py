"""Shared fixtures with realistic Computation objects."""

from __future__ import annotations

import pytest
from pyrapide import Computation, Event

from trllm.cost_forensics.pricing import (
    ModelPrice,
    PricingRegistry,
    ToolPrice,
)


def _llm_event(
    name: str = "llm_call",
    model: str = "gpt-4o",
    input_tokens: int = 1000,
    output_tokens: int = 200,
) -> Event:
    return Event(
        name=name,
        payload={
            "model": model,
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            },
        },
        source="llm",
    )


def _tool_event(name: str = "tool_call", payload: dict | None = None) -> Event:
    return Event(
        name=name,
        payload=payload or {},
        source="tool",
    )


def _root_event(name: str = "user_request") -> Event:
    return Event(name=name, payload={}, source="user")


@pytest.fixture()
def pricing() -> PricingRegistry:
    """Simple pricing registry for tests."""
    registry = PricingRegistry()
    registry.register_model(
        "gpt-4o", ModelPrice(2.50, 10.00, "GPT-4o")
    )
    registry.register_model(
        "gpt-4o-mini", ModelPrice(0.15, 0.60, "GPT-4o Mini")
    )
    registry.register_tool("query_db", ToolPrice(0.001, "query_db"))
    registry.register_tool("web_search", ToolPrice(0.005, "web_search"))
    return registry


@pytest.fixture()
def simple_comp() -> Computation:
    """Linear chain: user_request -> llm_call -> tool_call -> llm_call."""
    comp = Computation()
    e1 = _root_event("user_request")
    e2 = _llm_event("llm_call", input_tokens=500, output_tokens=100)
    e3 = _tool_event("query_db")
    e4 = _llm_event("llm_call", input_tokens=800, output_tokens=150)

    comp.record(e1)
    comp.record(e2, caused_by=[e1])
    comp.record(e3, caused_by=[e2])
    comp.record(e4, caused_by=[e3])
    return comp


@pytest.fixture()
def retry_comp() -> Computation:
    """Retry storm: root -> llm_call -> tool_call x4 (same tool, same parent)."""
    comp = Computation()
    root = _root_event("user_request")
    llm = _llm_event("llm_call", input_tokens=1000, output_tokens=200)

    comp.record(root)
    comp.record(llm, caused_by=[root])

    # 4 retries of the same tool
    for _ in range(4):
        tool = _tool_event("query_db")
        comp.record(tool, caused_by=[llm])

    return comp


@pytest.fixture()
def dead_end_comp() -> Computation:
    """Dead end: root -> llm_call -> tool_call (no effects from tool result)."""
    comp = Computation()
    root = _root_event("user_request")
    llm = _llm_event("llm_call", input_tokens=500, output_tokens=100)
    tool = _tool_event("web_search")

    comp.record(root)
    comp.record(llm, caused_by=[root])
    comp.record(tool, caused_by=[llm])
    # tool has no effects — dead end
    return comp


@pytest.fixture()
def multi_agent_comp() -> Computation:
    """Two agent branches from one root, one branch abandoned."""
    comp = Computation()
    root = _root_event("user_request")
    comp.record(root)

    # Branch 1: active branch (leads to output consumed externally)
    b1_llm1 = _llm_event("agent_a.llm", input_tokens=1000, output_tokens=300)
    b1_tool = _tool_event("query_db")
    b1_llm2 = _llm_event("agent_a.llm", input_tokens=1200, output_tokens=400)
    comp.record(b1_llm1, caused_by=[root])
    comp.record(b1_tool, caused_by=[b1_llm1])
    comp.record(b1_llm2, caused_by=[b1_tool])

    # Branch 2: abandoned branch (depth >= 2, no external consumption)
    b2_llm1 = _llm_event("agent_b.llm", input_tokens=800, output_tokens=200)
    b2_tool = _tool_event("web_search")
    b2_llm2 = _llm_event("agent_b.llm", input_tokens=900, output_tokens=250)
    comp.record(b2_llm1, caused_by=[root])
    comp.record(b2_tool, caused_by=[b2_llm1])
    comp.record(b2_llm2, caused_by=[b2_tool])

    return comp


@pytest.fixture()
def redundant_context_comp() -> Computation:
    """Two LLM calls with nearly identical input token counts."""
    comp = Computation()
    root = _root_event("user_request")
    comp.record(root)

    llm1 = _llm_event("llm_call", input_tokens=10000, output_tokens=200)
    llm2 = _llm_event("llm_call", input_tokens=10200, output_tokens=250)
    comp.record(llm1, caused_by=[root])
    comp.record(llm2, caused_by=[root])
    return comp


@pytest.fixture()
def large_comp() -> Computation:
    """50+ events with mixed patterns for performance tests."""
    comp = Computation()
    root = _root_event("orchestrator")
    comp.record(root)

    events_so_far: list[Event] = [root]

    # Create 5 agent branches
    for i in range(5):
        agent_llm = _llm_event(
            f"agent_{i}.llm",
            input_tokens=1000 + i * 200,
            output_tokens=100 + i * 50,
        )
        comp.record(agent_llm, caused_by=[root])
        events_so_far.append(agent_llm)

        prev = agent_llm
        # Each branch has ~10 events
        for j in range(10):
            if j % 2 == 0:
                ev = _tool_event(f"agent_{i}.tool_{j}")
            else:
                ev = _llm_event(
                    f"agent_{i}.llm",
                    input_tokens=500 + j * 100,
                    output_tokens=50 + j * 20,
                )
            comp.record(ev, caused_by=[prev])
            events_so_far.append(ev)
            prev = ev

    return comp
