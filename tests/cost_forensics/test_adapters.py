"""Tests for SDK adapters using mock clients."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from trllm.cost_forensics.adapters.openai import InstrumentedOpenAI
from trllm.cost_forensics.adapters.anthropic import InstrumentedAnthropic
from trllm.cost_forensics.annotator import CostAnnotator
from trllm.cost_forensics.pricing import ModelPrice, PricingRegistry
from trllm.cost_forensics.rollup import CausalCostRollup


# --- Mock OpenAI SDK objects ---


@dataclass
class _MockUsageOpenAI:
    prompt_tokens: int = 100
    completion_tokens: int = 50


@dataclass
class _MockChatResponse:
    model: str = "gpt-4o"
    usage: _MockUsageOpenAI | None = None
    choices: list[Any] | None = None

    def __post_init__(self) -> None:
        if self.usage is None:
            self.usage = _MockUsageOpenAI()
        if self.choices is None:
            self.choices = []


class _MockCompletions:
    def __init__(self, responses: list[_MockChatResponse] | None = None) -> None:
        self._responses = responses or [_MockChatResponse()]
        self._call_count = 0

    def create(self, **kwargs: Any) -> _MockChatResponse:
        resp = self._responses[min(self._call_count, len(self._responses) - 1)]
        self._call_count += 1
        return resp


class _MockChat:
    def __init__(self, completions: _MockCompletions | None = None) -> None:
        self.completions = completions or _MockCompletions()


class _MockOpenAIClient:
    def __init__(self, chat: _MockChat | None = None) -> None:
        self.chat = chat or _MockChat()


# --- Mock Anthropic SDK objects ---


@dataclass
class _MockUsageAnthropic:
    input_tokens: int = 200
    output_tokens: int = 100


@dataclass
class _MockMessageResponse:
    model: str = "claude-sonnet-4-20250514"
    usage: _MockUsageAnthropic | None = None
    content: list[Any] | None = None

    def __post_init__(self) -> None:
        if self.usage is None:
            self.usage = _MockUsageAnthropic()
        if self.content is None:
            self.content = []


class _MockMessages:
    def __init__(self, responses: list[_MockMessageResponse] | None = None) -> None:
        self._responses = responses or [_MockMessageResponse()]
        self._call_count = 0

    def create(self, **kwargs: Any) -> _MockMessageResponse:
        resp = self._responses[min(self._call_count, len(self._responses) - 1)]
        self._call_count += 1
        return resp


class _MockAnthropicClient:
    def __init__(self, messages: _MockMessages | None = None) -> None:
        self.messages = messages or _MockMessages()


# --- Tests ---


class TestInstrumentedOpenAI:
    def test_single_call_records_events(self) -> None:
        client = InstrumentedOpenAI(_MockOpenAIClient())
        client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "hi"}],
        )

        comp = client.computation()
        events = list(comp)
        # session root + 1 llm_call
        assert len(events) == 2

    def test_multiple_calls_chain_causally(self) -> None:
        client = InstrumentedOpenAI(_MockOpenAIClient())
        client.chat.completions.create(model="gpt-4o", messages=[])
        client.chat.completions.create(model="gpt-4o", messages=[])
        client.chat.completions.create(model="gpt-4o", messages=[])

        comp = client.computation()
        events = list(comp)
        # session root + 3 llm_calls
        assert len(events) == 4

        # Should be a linear chain
        roots = comp.root_events()
        assert len(roots) == 1

    def test_token_counts_captured(self) -> None:
        mock = _MockOpenAIClient(
            chat=_MockChat(
                completions=_MockCompletions([
                    _MockChatResponse(
                        usage=_MockUsageOpenAI(prompt_tokens=5000, completion_tokens=500)
                    )
                ])
            )
        )
        client = InstrumentedOpenAI(mock)
        client.chat.completions.create(model="gpt-4o", messages=[])

        comp = client.computation()
        pricing = PricingRegistry.openai()
        annotations = CostAnnotator(pricing).annotate(comp)
        report = CausalCostRollup().rollup(comp, annotations)

        assert report.total_cost > 0

    def test_record_tool_call(self) -> None:
        client = InstrumentedOpenAI(_MockOpenAIClient())
        client.chat.completions.create(model="gpt-4o", messages=[])
        client.record_tool_call("web_search", {"query": "test"})
        client.chat.completions.create(model="gpt-4o", messages=[])

        comp = client.computation()
        events = list(comp)
        # session + llm + tool + llm
        assert len(events) == 4
        tool_events = [e for e in events if e.source == "tool"]
        assert len(tool_events) == 1
        assert tool_events[0].name == "web_search"

    def test_reset(self) -> None:
        client = InstrumentedOpenAI(_MockOpenAIClient())
        client.chat.completions.create(model="gpt-4o", messages=[])
        assert len(list(client.computation())) == 2

        client.reset()
        assert len(list(client.computation())) == 1  # just the root

    def test_returns_original_response(self) -> None:
        expected = _MockChatResponse(model="gpt-4o")
        mock = _MockOpenAIClient(
            chat=_MockChat(completions=_MockCompletions([expected]))
        )
        client = InstrumentedOpenAI(mock)
        response = client.chat.completions.create(model="gpt-4o", messages=[])
        assert response is expected

    def test_repr(self) -> None:
        client = InstrumentedOpenAI(_MockOpenAIClient())
        assert "InstrumentedOpenAI" in repr(client)

    def test_full_analysis_pipeline(self) -> None:
        mock = _MockOpenAIClient(
            chat=_MockChat(
                completions=_MockCompletions([
                    _MockChatResponse(
                        usage=_MockUsageOpenAI(prompt_tokens=1000, completion_tokens=200)
                    ),
                    _MockChatResponse(
                        usage=_MockUsageOpenAI(prompt_tokens=2000, completion_tokens=400)
                    ),
                ])
            )
        )
        client = InstrumentedOpenAI(mock)
        client.chat.completions.create(model="gpt-4o", messages=[])
        client.record_tool_call("query_db")
        client.chat.completions.create(model="gpt-4o", messages=[])

        comp = client.computation()
        pricing = PricingRegistry.openai()
        annotations = CostAnnotator(pricing).annotate(comp)
        report = CausalCostRollup().rollup(comp, annotations)

        tree = report.ascii_tree()
        assert "session" in tree
        assert "llm_call" in tree
        assert "query_db" in tree


class TestInstrumentedAnthropic:
    def test_single_call_records_events(self) -> None:
        client = InstrumentedAnthropic(_MockAnthropicClient())
        client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{"role": "user", "content": "hi"}],
        )

        comp = client.computation()
        events = list(comp)
        assert len(events) == 2

    def test_token_counts_captured(self) -> None:
        mock = _MockAnthropicClient(
            messages=_MockMessages([
                _MockMessageResponse(
                    usage=_MockUsageAnthropic(input_tokens=5000, output_tokens=500)
                )
            ])
        )
        client = InstrumentedAnthropic(mock)
        client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[],
        )

        comp = client.computation()
        pricing = PricingRegistry.anthropic()
        annotations = CostAnnotator(pricing).annotate(comp)
        report = CausalCostRollup().rollup(comp, annotations)

        assert report.total_cost > 0

    def test_multiple_calls_chain(self) -> None:
        client = InstrumentedAnthropic(_MockAnthropicClient())
        for _ in range(3):
            client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1024,
                messages=[],
            )

        comp = client.computation()
        events = list(comp)
        assert len(events) == 4  # root + 3 calls
        assert len(comp.root_events()) == 1

    def test_record_tool_call(self) -> None:
        client = InstrumentedAnthropic(_MockAnthropicClient())
        client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[],
        )
        client.record_tool_call("calculator", {"input": "2+2"})

        comp = client.computation()
        tool_events = [e for e in comp if e.source == "tool"]
        assert len(tool_events) == 1

    def test_reset(self) -> None:
        client = InstrumentedAnthropic(_MockAnthropicClient())
        client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[],
        )
        client.reset()
        assert len(list(client.computation())) == 1

    def test_repr(self) -> None:
        client = InstrumentedAnthropic(_MockAnthropicClient())
        assert "InstrumentedAnthropic" in repr(client)
