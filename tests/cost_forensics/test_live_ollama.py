"""Live integration test using Ollama.

Run with: pytest tests/cost_forensics/test_live_ollama.py -v
Requires Ollama running locally with qwen3:0.6b available.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
import urllib.request

from pyrapide import Computation, Event

from trllm.cost_forensics import (
    CausalCostRollup,
    CostAnnotator,
    ForensicReport,
    ModelPrice,
    PricingRegistry,
    ToolPrice,
    WasteDetector,
    WasteReport,
    diff_reports,
)

OLLAMA_BASE = "http://localhost:11434"
MODEL = "qwen3:0.6b"


def ollama_available() -> bool:
    try:
        req = urllib.request.Request(f"{OLLAMA_BASE}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=2) as resp:
            return resp.status == 200
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not ollama_available(), reason="Ollama not running"
)


def _ollama_chat(prompt: str, model: str = MODEL) -> dict[str, Any]:
    """Call Ollama chat API and return the response with usage stats."""
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }).encode()
    req = urllib.request.Request(
        f"{OLLAMA_BASE}/api/chat",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def _build_live_computation() -> tuple[Computation, list[Event]]:
    """Run a real 2-step LLM pipeline through Ollama and record it."""
    comp = Computation()

    # Step 1: user request
    root = Event(name="user_request", payload={"prompt": "What is 2+2?"})
    comp.record(root)

    # Step 2: first LLM call
    resp1 = _ollama_chat("What is 2+2? Answer in one word.")
    llm1 = Event(
        name="llm_call",
        payload={
            "model": MODEL,
            "usage": {
                "input_tokens": resp1.get("prompt_eval_count", 0),
                "output_tokens": resp1.get("eval_count", 0),
            },
            "response": resp1.get("message", {}).get("content", ""),
        },
        source="ollama",
    )
    comp.record(llm1, caused_by=[root])

    # Step 3: simulated tool call (lookup based on LLM output)
    tool = Event(
        name="calculator",
        payload={"input": "2+2", "output": "4"},
        source="tool",
    )
    comp.record(tool, caused_by=[llm1])

    # Step 4: second LLM call — summarize
    resp2 = _ollama_chat("Confirm: does 2+2 equal 4? Answer yes or no.")
    llm2 = Event(
        name="llm_call",
        payload={
            "model": MODEL,
            "usage": {
                "input_tokens": resp2.get("prompt_eval_count", 0),
                "output_tokens": resp2.get("eval_count", 0),
            },
            "response": resp2.get("message", {}).get("content", ""),
        },
        source="ollama",
    )
    comp.record(llm2, caused_by=[tool])

    return comp, [root, llm1, tool, llm2]


@pytest.fixture(scope="module")
def live_pipeline() -> tuple[Computation, list[Event], PricingRegistry]:
    """Run the live pipeline once and share across tests."""
    comp, events = _build_live_computation()

    pricing = PricingRegistry()
    # Ollama is free, but assign fake prices to test the math
    pricing.register_model(MODEL, ModelPrice(0.50, 1.00, "qwen3-0.6b"))
    pricing.register_tool("calculator", ToolPrice(0.002, "calculator"))

    return comp, events, pricing


class TestLiveAnnotation:
    def test_events_have_real_token_counts(
        self, live_pipeline: tuple[Computation, list[Event], PricingRegistry]
    ) -> None:
        comp, events, pricing = live_pipeline
        annotations = CostAnnotator(pricing).annotate(comp)

        # LLM events should have non-zero costs (real tokens from Ollama)
        llm_events = [e for e in events if e.name == "llm_call"]
        for ev in llm_events:
            ann = annotations[ev.id]
            assert ann.direct_cost > 0, (
                f"LLM event should have non-zero cost, got {ann.direct_cost}. "
                f"Tokens: {ev.payload.get('usage')}"
            )

    def test_tool_event_has_cost(
        self, live_pipeline: tuple[Computation, list[Event], PricingRegistry]
    ) -> None:
        comp, events, pricing = live_pipeline
        annotations = CostAnnotator(pricing).annotate(comp)

        tool_events = [e for e in events if e.name == "calculator"]
        assert len(tool_events) == 1
        assert annotations[tool_events[0].id].direct_cost == pytest.approx(0.002)


class TestLiveRollup:
    def test_causal_cost_flows_to_root(
        self, live_pipeline: tuple[Computation, list[Event], PricingRegistry]
    ) -> None:
        comp, events, pricing = live_pipeline
        annotations = CostAnnotator(pricing).annotate(comp)
        report = CausalCostRollup().rollup(comp, annotations)

        assert report.total_cost > 0
        assert len(report.roots) == 1
        root = report.roots[0]
        assert root.event.name == "user_request"
        assert root.causal_cost == pytest.approx(report.total_cost)
        # Root has no direct cost
        assert root.direct_cost == 0.0

    def test_ascii_tree_shows_real_costs(
        self, live_pipeline: tuple[Computation, list[Event], PricingRegistry]
    ) -> None:
        comp, events, pricing = live_pipeline
        annotations = CostAnnotator(pricing).annotate(comp)
        report = CausalCostRollup().rollup(comp, annotations)

        tree = report.ascii_tree()
        assert "user_request" in tree
        assert "llm_call" in tree
        assert "calculator" in tree
        assert "$" in tree
        print("\n--- Live ASCII Tree ---")
        print(tree)


class TestLiveWaste:
    def test_no_false_positives_on_clean_pipeline(
        self, live_pipeline: tuple[Computation, list[Event], PricingRegistry]
    ) -> None:
        comp, events, pricing = live_pipeline
        annotations = CostAnnotator(pricing).annotate(comp)
        waste = WasteDetector().detect(comp, annotations)

        # Clean linear pipeline should have no retry storms
        retry_instances = [
            i for i in waste.instances if i.pattern_name == "RetryStorm"
        ]
        assert len(retry_instances) == 0

        print("\n--- Live Waste Summary ---")
        print(waste.summary())


class TestLiveDiff:
    def test_diff_two_live_runs(
        self, live_pipeline: tuple[Computation, list[Event], PricingRegistry]
    ) -> None:
        comp, events, pricing = live_pipeline
        annotations = CostAnnotator(pricing).annotate(comp)
        report = CausalCostRollup().rollup(comp, annotations)

        # Diff against itself — should be zero delta
        diff = diff_reports(report, report)
        assert diff.total_delta == pytest.approx(0.0)
        assert len(diff.regressions) == 0


class TestLiveEndToEnd:
    def test_full_pipeline(
        self, live_pipeline: tuple[Computation, list[Event], PricingRegistry]
    ) -> None:
        """The acceptance criteria flow from the spec, using live data."""
        comp, events, pricing = live_pipeline

        annotations = CostAnnotator(pricing).annotate(comp)
        report = CausalCostRollup().rollup(comp, annotations)
        waste = WasteDetector().detect(comp, annotations)
        report.attach_waste(waste)

        # Everything should work without errors
        tree = report.ascii_tree()
        summary = waste.summary()
        report_dict = report.to_dict()
        top = report.top_costs(3)

        assert report.total_cost > 0
        assert isinstance(tree, str)
        assert isinstance(summary, str)
        assert isinstance(report_dict, dict)
        assert len(top) > 0

        # Print the full output for visibility
        print("\n--- Full Live Analysis ---")
        print(tree)
        print()
        print(summary)
        print(f"\nTotal cost: ${report.total_cost:.6f}")
        print(f"Top cost node: {top[0].event.name} (${top[0].causal_cost:.6f})")

    def test_serialization_roundtrip(
        self, live_pipeline: tuple[Computation, list[Event], PricingRegistry]
    ) -> None:
        """Serialize computation to JSON, reload, re-analyze — same results."""
        comp, events, pricing = live_pipeline

        # Serialize
        data = comp.to_dict()
        json_str = json.dumps(data)

        # Reload
        comp2 = Computation.from_dict(json.loads(json_str))

        # Re-analyze
        ann1 = CostAnnotator(pricing).annotate(comp)
        ann2 = CostAnnotator(pricing).annotate(comp2)

        total1 = sum(a.direct_cost for a in ann1.values())
        total2 = sum(a.direct_cost for a in ann2.values())
        assert total1 == pytest.approx(total2)
