"""Pricing registry for LLM models and tools."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pyrapide import Computation, Event  # type: ignore[import-untyped]


@dataclass
class ModelPrice:
    """Price definition for an LLM model."""

    input_per_million: float  # USD per 1M input tokens
    output_per_million: float  # USD per 1M output tokens
    name: str


@dataclass
class ToolPrice:
    """Price definition for a tool invocation."""

    per_call: float  # USD per invocation
    name: str


class PricingRegistry:
    """Registry mapping model IDs and tool names to their pricing."""

    def __init__(self) -> None:
        self._models: dict[str, ModelPrice] = {}
        self._tools: dict[str, ToolPrice] = {}

    def __repr__(self) -> str:
        return (
            f"PricingRegistry(models={len(self._models)}, "
            f"tools={len(self._tools)})"
        )

    def register_model(self, model_id: str, price: ModelPrice) -> None:
        """Register pricing for an LLM model."""
        self._models[model_id] = price

    def register_tool(self, tool_name: str, price: ToolPrice) -> None:
        """Register pricing for a tool."""
        self._tools[tool_name] = price

    def cost_for_event(self, event: Event) -> float:
        """Compute the USD cost of an event. Returns 0.0 if unknown."""
        payload: dict[str, Any] = event.payload

        # Try LLM event pricing
        usage = payload.get("usage")
        if isinstance(usage, dict):
            model_id = payload.get("model", event.name)
            price = self._models.get(model_id)
            if price is not None:
                input_tokens: int = int(usage.get("input_tokens", 0))
                output_tokens: int = int(usage.get("output_tokens", 0))
                return float(
                    input_tokens * price.input_per_million / 1_000_000
                    + output_tokens * price.output_per_million / 1_000_000
                )

        # Try tool pricing
        tool_price = self._tools.get(event.name)
        if tool_price is not None:
            return tool_price.per_call

        return 0.0

    @classmethod
    def openai(cls) -> PricingRegistry:
        """Pre-filled registry with current OpenAI prices (as of Apr 2026)."""
        registry = cls()
        models = {
            # GPT-4.1 family
            "gpt-4.1": ModelPrice(2.00, 8.00, "GPT-4.1"),
            "gpt-4.1-mini": ModelPrice(0.40, 1.60, "GPT-4.1 Mini"),
            "gpt-4.1-nano": ModelPrice(0.10, 0.40, "GPT-4.1 Nano"),
            # GPT-4o family
            "gpt-4o": ModelPrice(2.50, 10.00, "GPT-4o"),
            "gpt-4o-mini": ModelPrice(0.15, 0.60, "GPT-4o Mini"),
            # Reasoning models
            "o3": ModelPrice(10.00, 40.00, "o3"),
            "o4-mini": ModelPrice(1.10, 4.40, "o4-mini"),
            "o3-mini": ModelPrice(1.10, 4.40, "o3-mini"),
            "o1": ModelPrice(15.00, 60.00, "o1"),
            "o1-mini": ModelPrice(1.10, 4.40, "o1-mini"),
            # Older models
            "gpt-4-turbo": ModelPrice(10.00, 30.00, "GPT-4 Turbo"),
            "gpt-4": ModelPrice(30.00, 60.00, "GPT-4"),
            "gpt-3.5-turbo": ModelPrice(0.50, 1.50, "GPT-3.5 Turbo"),
        }
        for model_id, price in models.items():
            registry.register_model(model_id, price)
        return registry

    @classmethod
    def anthropic(cls) -> PricingRegistry:
        """Pre-filled registry with current Anthropic prices (as of Apr 2026)."""
        registry = cls()
        models = {
            # Claude 4.6
            "claude-opus-4-6-20260401": ModelPrice(5.00, 25.00, "Claude Opus 4.6"),
            "claude-sonnet-4-6-20260401": ModelPrice(3.00, 15.00, "Claude Sonnet 4.6"),
            # Claude 4.5
            "claude-opus-4-5-20260301": ModelPrice(5.00, 25.00, "Claude Opus 4.5"),
            "claude-sonnet-4-5-20260301": ModelPrice(3.00, 15.00, "Claude Sonnet 4.5"),
            "claude-haiku-4-5-20251001": ModelPrice(1.00, 5.00, "Claude Haiku 4.5"),
            # Claude 4.1
            "claude-opus-4-1-20250610": ModelPrice(15.00, 75.00, "Claude Opus 4.1"),
            # Claude 4
            "claude-opus-4-20250514": ModelPrice(15.00, 75.00, "Claude Opus 4"),
            "claude-sonnet-4-20250514": ModelPrice(3.00, 15.00, "Claude Sonnet 4"),
            # Claude 3.x
            "claude-3-5-haiku-20241022": ModelPrice(0.80, 4.00, "Claude 3.5 Haiku"),
            "claude-3-5-sonnet-20241022": ModelPrice(3.00, 15.00, "Claude 3.5 Sonnet"),
            "claude-3-opus-20240229": ModelPrice(15.00, 75.00, "Claude 3 Opus"),
            "claude-3-haiku-20240307": ModelPrice(0.25, 1.25, "Claude 3 Haiku"),
        }
        for model_id, price in models.items():
            registry.register_model(model_id, price)
        return registry

    @classmethod
    def bedrock(cls) -> PricingRegistry:
        """Pre-filled registry with current AWS Bedrock prices (as of Apr 2026).

        Prices are on-demand USD per 1M tokens for the US East (N. Virginia)
        region.  Cross-region inference and provisioned throughput have
        different pricing — register overrides with ``register_model()``.
        """
        registry = cls()
        models = {
            # Anthropic on Bedrock
            "anthropic.claude-3-5-haiku-20241022-v1:0": ModelPrice(
                0.80, 4.00, "Claude 3.5 Haiku"
            ),
            "anthropic.claude-3-5-sonnet-20241022-v2:0": ModelPrice(
                3.00, 15.00, "Claude 3.5 Sonnet v2"
            ),
            "anthropic.claude-3-opus-20240229-v1:0": ModelPrice(
                15.00, 75.00, "Claude 3 Opus"
            ),
            "anthropic.claude-3-haiku-20240307-v1:0": ModelPrice(
                0.25, 1.25, "Claude 3 Haiku"
            ),
            # Meta Llama on Bedrock
            "meta.llama3-70b-instruct-v1:0": ModelPrice(
                2.65, 3.50, "Llama 3 70B"
            ),
            "meta.llama3-8b-instruct-v1:0": ModelPrice(
                0.22, 0.22, "Llama 3 8B"
            ),
            "meta.llama3-1-405b-instruct-v1:0": ModelPrice(
                5.32, 16.00, "Llama 3.1 405B"
            ),
            # Amazon Nova
            "amazon.nova-pro-v1:0": ModelPrice(
                0.80, 3.20, "Amazon Nova Pro"
            ),
            "amazon.nova-lite-v1:0": ModelPrice(
                0.06, 0.24, "Amazon Nova Lite"
            ),
            "amazon.nova-micro-v1:0": ModelPrice(
                0.035, 0.14, "Amazon Nova Micro"
            ),
            # Mistral on Bedrock
            "mistral.mistral-large-2407-v1:0": ModelPrice(
                2.00, 6.00, "Mistral Large"
            ),
            "mistral.mistral-small-2402-v1:0": ModelPrice(
                0.10, 0.30, "Mistral Small"
            ),
        }
        for model_id, price in models.items():
            registry.register_model(model_id, price)
        return registry

    @classmethod
    def ollama(cls, comp: Computation | None = None) -> PricingRegistry:
        """Registry for Ollama models (free, tracks tokens only).

        If a Computation is provided, auto-discovers model names from
        event payloads and registers them at zero cost.
        """
        registry = cls()
        if comp is not None:
            for event in comp:
                model = event.payload.get("model")
                if model and model not in registry._models:
                    registry.register_model(
                        model, ModelPrice(0.0, 0.0, model)
                    )
        return registry
