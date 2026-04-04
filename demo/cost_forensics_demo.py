"""
TRLLM Cost Forensics Demo — Ollama pipeline with waste detection.

Runs a multi-step agent pipeline through Ollama, then analyzes costs
and detects waste patterns.

Usage:
    python demo/cost_forensics_demo.py
    python demo/cost_forensics_demo.py --model qwen3:1.7b
    python demo/cost_forensics_demo.py --budget 0.01

Requires: Ollama running at localhost:11434 with the specified model pulled.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from typing import Any

from pyrapide import Computation, Event  # type: ignore[import-untyped]

from trllm.cost_forensics import (
    CausalCostRollup,
    CostAnnotator,
    ModelPrice,
    PricingRegistry,
    ToolPrice,
    WasteDetector,
)

OLLAMA_BASE = "http://localhost:11434"
DEFAULT_MODEL = "qwen3:0.6b"


def ollama_chat(prompt: str, model: str) -> dict[str, Any]:
    """Call Ollama and return the response."""
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
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())


def llm_event(resp: dict[str, Any], model: str) -> Event:
    """Create an LLM event from an Ollama response."""
    return Event(
        name="llm_call",
        payload={
            "model": model,
            "usage": {
                "input_tokens": resp.get("prompt_eval_count", 0),
                "output_tokens": resp.get("eval_count", 0),
            },
            "response": resp.get("message", {}).get("content", ""),
        },
        source="ollama",
    )


def run_pipeline(model: str) -> Computation:
    """Run a multi-step agent pipeline and return the Computation."""
    comp = Computation()

    # Root: user request
    root = Event(name="user_request", payload={"query": "What are the moons of Mars?"})
    comp.record(root)
    print("  [1/5] User request recorded")

    # Step 1: Planning LLM call
    resp1 = ollama_chat(
        "I need to answer: What are the moons of Mars? "
        "List the steps I should take. Be brief.",
        model,
    )
    plan = llm_event(resp1, model)
    comp.record(plan, caused_by=[root])
    print(f"  [2/5] Planning call: {resp1.get('prompt_eval_count', 0)} in / {resp1.get('eval_count', 0)} out tokens")

    # Step 2: Tool call (simulated knowledge base lookup)
    tool = Event(
        name="knowledge_lookup",
        payload={"query": "Mars moons", "result": "Phobos, Deimos"},
        source="tool",
    )
    comp.record(tool, caused_by=[plan])
    print("  [3/5] Tool call: knowledge_lookup")

    # Step 3: Synthesis LLM call
    resp2 = ollama_chat(
        "Based on this info: Mars has two moons called Phobos and Deimos. "
        "Answer the question: What are the moons of Mars?",
        model,
    )
    synthesis = llm_event(resp2, model)
    comp.record(synthesis, caused_by=[tool])
    print(f"  [4/5] Synthesis call: {resp2.get('prompt_eval_count', 0)} in / {resp2.get('eval_count', 0)} out tokens")

    # Step 4: Dead-end tool call (fetched but never used — waste!)
    dead_end = Event(
        name="web_search",
        payload={"query": "Mars moons discovery date"},
        source="tool",
    )
    comp.record(dead_end, caused_by=[plan])
    print("  [5/5] Dead-end tool call: web_search (result unused)")

    return comp


def main() -> None:
    parser = argparse.ArgumentParser(description="TRLLM Cost Forensics Demo")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model to use")
    parser.add_argument("--budget", type=float, default=None, help="Budget limit in USD")
    parser.add_argument("--json", action="store_true", help="Output JSON instead of text")
    args = parser.parse_args()

    print(f"\n=== TRLLM Cost Forensics Demo (model: {args.model}) ===\n")
    print("Running pipeline...")
    comp = run_pipeline(args.model)

    # Set up pricing (Ollama is free, but we assign realistic prices)
    pricing = PricingRegistry()
    pricing.register_model(args.model, ModelPrice(0.50, 1.00, args.model))
    pricing.register_tool("knowledge_lookup", ToolPrice(0.001, "knowledge_lookup"))
    pricing.register_tool("web_search", ToolPrice(0.005, "web_search"))

    # Analyze
    print("\nAnalyzing costs...\n")
    annotations = CostAnnotator(pricing).annotate(comp)
    report = CausalCostRollup().rollup(comp, annotations)
    waste = WasteDetector().detect(comp, annotations)
    report.attach_waste(waste)

    if args.json:
        output = report.to_dict()
        output["waste"] = waste.to_dict()
        print(json.dumps(output, indent=2))
    else:
        print("--- Causal Cost Tree ---")
        print(report.ascii_tree())
        print(f"\nTotal cost: ${report.total_cost:.6f}")
        print()
        print("--- Waste Detection ---")
        print(waste.summary())

        top = report.top_costs(3)
        print("\n--- Top 3 Most Expensive Subtrees ---")
        for node in top:
            print(f"  {node.event.name}: ${node.causal_cost:.6f}")

    if args.budget is not None and report.total_cost > args.budget:
        print(f"\nBUDGET EXCEEDED: ${report.total_cost:.6f} > ${args.budget:.6f}")
        sys.exit(1)

    # Save computation for CLI replay
    comp_path = "demo_computation.json"
    with open(comp_path, "w") as f:
        json.dump(comp.to_dict(), f, indent=2)
    print(f"\nComputation saved to {comp_path}")
    print(f"  Replay: trllm-cost-forensics analyze {comp_path}")


if __name__ == "__main__":
    main()
