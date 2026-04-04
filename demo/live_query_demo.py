"""
TRLLM Live Query Demo — Use the OpenAI adapter with Ollama to run real
LLM calls and get cost forensics in real time.

Usage:
    python demo/live_query_demo.py
    python demo/live_query_demo.py --model qwen3:1.7b

Requires: Ollama running at localhost:11434 with the specified model pulled.
"""

from __future__ import annotations

import argparse
import json

from openai import OpenAI

from trllm.cost_forensics import (
    CausalCostRollup,
    CostAnnotator,
    ModelPrice,
    PricingRegistry,
    ToolPrice,
    WasteDetector,
)
from trllm.cost_forensics.adapters.openai import InstrumentedOpenAI

OLLAMA_BASE = "http://localhost:11434/v1"
DEFAULT_MODEL = "qwen3:0.6b"


def main() -> None:
    parser = argparse.ArgumentParser(description="TRLLM Live Query Demo")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    print(f"\n=== TRLLM Live Query Demo (model: {args.model}) ===\n")

    # Wrap Ollama's OpenAI-compatible API
    client = InstrumentedOpenAI(
        OpenAI(base_url=OLLAMA_BASE, api_key="ollama")
    )

    # --- Step 1: Planning call ---
    print("[1/4] Planning...")
    plan = client.chat.completions.create(
        model=args.model,
        messages=[{
            "role": "user",
            "content": (
                "I need to answer: What are the three largest moons of Jupiter? "
                "List the steps I should take. Be brief, 2-3 steps max."
            ),
        }],
    )
    plan_text = plan.choices[0].message.content
    print(f"      Plan: {plan_text[:120]}...")

    # --- Step 2: Simulated tool call (knowledge lookup) ---
    print("[2/4] Tool call: knowledge_lookup")
    client.record_tool_call("knowledge_lookup", {
        "query": "Jupiter largest moons",
        "result": "Ganymede, Callisto, Io",
    })

    # --- Step 3: Synthesis call ---
    print("[3/4] Synthesizing...")
    synthesis = client.chat.completions.create(
        model=args.model,
        messages=[
            {"role": "user", "content": "What are the three largest moons of Jupiter?"},
            {"role": "assistant", "content": plan_text},
            {
                "role": "user",
                "content": (
                    "Tool result: The three largest moons of Jupiter are "
                    "Ganymede, Callisto, and Io. "
                    "Now give a final concise answer."
                ),
            },
        ],
    )
    answer = synthesis.choices[0].message.content
    print(f"      Answer: {answer[:200]}")

    # --- Step 4: Dead-end call (waste — fetched but unused) ---
    print("[4/4] Dead-end tool call: web_search (simulated waste)")
    client.record_tool_call("web_search", {"query": "Jupiter moon discovery dates"})

    # --- Analyze ---
    print("\n--- Cost Forensics ---\n")

    comp = client.computation()
    pricing = PricingRegistry()
    pricing.register_model(args.model, ModelPrice(0.50, 1.00, args.model))
    pricing.register_tool("knowledge_lookup", ToolPrice(0.001, "knowledge_lookup"))
    pricing.register_tool("web_search", ToolPrice(0.005, "web_search"))

    annotations = CostAnnotator(pricing).annotate(comp)
    report = CausalCostRollup().rollup(comp, annotations)
    waste = WasteDetector().detect(comp, annotations)
    report.attach_waste(waste)

    print(report.ascii_tree())
    print(f"\nTotal cost: ${report.total_cost:.6f}")
    print()
    print(waste.summary())

    # Save for CLI replay
    comp_path = "live_query_computation.json"
    with open(comp_path, "w") as f:
        json.dump(comp.to_dict(), f, indent=2)
    print(f"\nSaved to {comp_path}")
    print(f"  Replay: trllm-cost-forensics analyze {comp_path} "
          f"--model-price \"{args.model}=0.50,1.00\" "
          f"--tool-price knowledge_lookup=0.001 --tool-price web_search=0.005")


if __name__ == "__main__":
    main()
