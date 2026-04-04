"""CLI entry point for cost-forensics analysis."""

from __future__ import annotations

import argparse
import json
import sys

from pyrapide import Computation  # type: ignore[import-untyped]

from .annotator import CostAnnotator
from .diff import diff_reports
from .pricing import ModelPrice, PricingRegistry, ToolPrice
from .rollup import CausalCostRollup
from .waste import WasteDetector


def _get_pricing(
    args: argparse.Namespace, comp: Computation | None = None
) -> PricingRegistry:
    match args.provider:
        case "openai":
            registry = PricingRegistry.openai()
        case "anthropic":
            registry = PricingRegistry.anthropic()
        case "bedrock":
            registry = PricingRegistry.bedrock()
        case "ollama":
            registry = PricingRegistry.ollama(comp)
        case _:
            print(f"Unknown provider: {args.provider}", file=sys.stderr)
            sys.exit(1)

    # Register custom model prices: --model-price name=input,output
    for spec in getattr(args, "model_price", None) or []:
        if "=" not in spec:
            print(
                f"Invalid --model-price format: {spec!r} "
                f"(expected name=input,output)",
                file=sys.stderr,
            )
            sys.exit(1)
        name, prices = spec.rsplit("=", 1)
        parts = prices.split(",")
        if len(parts) != 2:
            print(
                f"Invalid --model-price format: {spec!r} "
                f"(expected name=input,output)",
                file=sys.stderr,
            )
            sys.exit(1)
        registry.register_model(
            name, ModelPrice(float(parts[0]), float(parts[1]), name)
        )

    # Register custom tool prices: --tool-price name=per_call
    for spec in getattr(args, "tool_price", None) or []:
        if "=" not in spec:
            print(
                f"Invalid --tool-price format: {spec!r} "
                f"(expected name=per_call)",
                file=sys.stderr,
            )
            sys.exit(1)
        name, cost = spec.rsplit("=", 1)
        registry.register_tool(name, ToolPrice(float(cost), name))

    return registry


def _load_computation(path: str) -> Computation:
    with open(path) as f:
        data = json.load(f)
    return Computation.from_dict(data)


def cmd_analyze(args: argparse.Namespace) -> None:
    """Run full analysis on a serialized Computation."""
    comp = _load_computation(args.computation)
    pricing = _get_pricing(args, comp)

    annotations = CostAnnotator(pricing).annotate(comp)
    report = CausalCostRollup().rollup(comp, annotations)
    waste = WasteDetector().detect(comp, annotations)
    report.attach_waste(waste)

    if args.json:
        output = report.to_dict()
        output["waste"] = waste.to_dict()
        print(json.dumps(output, indent=2))
    else:
        print(report.ascii_tree())
        print()
        print(waste.summary())

    if args.budget is not None and report.total_cost > args.budget:
        print(
            f"\nBudget exceeded: ${report.total_cost:.4f} > "
            f"${args.budget:.4f}",
            file=sys.stderr,
        )
        sys.exit(1)


def cmd_diff(args: argparse.Namespace) -> None:
    """Diff two serialized Computations."""
    before_comp = _load_computation(args.before)
    after_comp = _load_computation(args.after)
    pricing = _get_pricing(args)

    annotator = CostAnnotator(pricing)
    before_report = CausalCostRollup().rollup(
        before_comp, annotator.annotate(before_comp)
    )
    after_report = CausalCostRollup().rollup(
        after_comp, annotator.annotate(after_comp)
    )

    cost_diff = diff_reports(before_report, after_report)

    if args.json:
        print(json.dumps(cost_diff.to_dict(), indent=2))
    else:
        print(cost_diff.summary())


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="trllm-cost-forensics",
        description="Causal cost attribution and waste detection for LLM pipelines",
    )

    # Shared arguments added to each subparser for flexible ordering
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument(
        "--provider",
        choices=["openai", "anthropic", "bedrock", "ollama"],
        default="openai",
        help="Pricing provider (default: openai)",
    )
    shared.add_argument(
        "--json",
        action="store_true",
        help="Output machine-readable JSON",
    )
    shared.add_argument(
        "--model-price",
        action="append",
        metavar="MODEL=INPUT,OUTPUT",
        help="Custom model price (USD per 1M tokens), e.g. qwen3:0.6b=0.50,1.00",
    )
    shared.add_argument(
        "--tool-price",
        action="append",
        metavar="TOOL=COST",
        help="Custom tool price (USD per call), e.g. web_search=0.005",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # analyze
    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Analyze a serialized Computation",
        parents=[shared],
    )
    analyze_parser.add_argument(
        "computation", help="Path to computation JSON file"
    )
    analyze_parser.add_argument(
        "--budget",
        type=float,
        default=None,
        help="Fail with exit code 1 if total cost exceeds budget",
    )

    # diff
    diff_parser = subparsers.add_parser(
        "diff",
        help="Diff two serialized Computations",
        parents=[shared],
    )
    diff_parser.add_argument("before", help="Path to before computation JSON")
    diff_parser.add_argument("after", help="Path to after computation JSON")

    args = parser.parse_args()

    match args.command:
        case "analyze":
            cmd_analyze(args)
        case "diff":
            cmd_diff(args)


if __name__ == "__main__":
    main()
