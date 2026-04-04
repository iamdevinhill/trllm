"""Tests for PricingRegistry factory methods."""

from __future__ import annotations

import pytest
from pyrapide import Event

from trllm.cost_forensics.pricing import PricingRegistry


class TestPricingFactories:
    def test_openai_registry_has_models(self) -> None:
        registry = PricingRegistry.openai()
        ev = Event(
            name="llm_call",
            payload={
                "model": "gpt-4o",
                "usage": {"input_tokens": 1_000_000, "output_tokens": 0},
            },
        )
        assert registry.cost_for_event(ev) == pytest.approx(2.50)

    def test_openai_gpt4_turbo(self) -> None:
        registry = PricingRegistry.openai()
        ev = Event(
            name="llm_call",
            payload={
                "model": "gpt-4-turbo",
                "usage": {"input_tokens": 1_000_000, "output_tokens": 0},
            },
        )
        assert registry.cost_for_event(ev) == pytest.approx(10.00)

    def test_openai_output_tokens(self) -> None:
        registry = PricingRegistry.openai()
        ev = Event(
            name="llm_call",
            payload={
                "model": "gpt-4o",
                "usage": {"input_tokens": 0, "output_tokens": 1_000_000},
            },
        )
        assert registry.cost_for_event(ev) == pytest.approx(10.00)

    def test_anthropic_registry_has_models(self) -> None:
        registry = PricingRegistry.anthropic()
        ev = Event(
            name="llm_call",
            payload={
                "model": "claude-opus-4-20250514",
                "usage": {"input_tokens": 1_000_000, "output_tokens": 0},
            },
        )
        assert registry.cost_for_event(ev) == pytest.approx(15.00)

    def test_anthropic_sonnet(self) -> None:
        registry = PricingRegistry.anthropic()
        ev = Event(
            name="llm_call",
            payload={
                "model": "claude-sonnet-4-20250514",
                "usage": {"input_tokens": 1_000_000, "output_tokens": 0},
            },
        )
        assert registry.cost_for_event(ev) == pytest.approx(3.00)

    def test_unknown_model_returns_zero(self) -> None:
        registry = PricingRegistry.openai()
        ev = Event(
            name="llm_call",
            payload={
                "model": "nonexistent-model",
                "usage": {"input_tokens": 1000, "output_tokens": 100},
            },
        )
        assert registry.cost_for_event(ev) == 0.0

    def test_repr(self) -> None:
        registry = PricingRegistry.openai()
        r = repr(registry)
        assert "PricingRegistry" in r
        assert "models=13" in r
