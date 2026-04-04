"""Tests for async SDK adapters using mock clients."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from trllm.cost_forensics.adapters.openai_async import AsyncInstrumentedOpenAI
from trllm.cost_forensics.adapters.anthropic_async import (
    AsyncInstrumentedAnthropic,
)
from trllm.cost_forensics.annotator import CostAnnotator
from trllm.cost_forensics.pricing import ModelPrice, PricingRegistry
from trllm.cost_forensics.rollup import CausalCostRollup


# --- Mock async OpenAI SDK objects ---


@dataclass
class _MockUsageOpenAI:
    prompt_tokens: int = 100
    completion_tokens: int = 50


@dataclass
class _MockChatResponse:
    model: str = "gpt-4o"
    usage: _MockUsageOpenAI | None = None

    def __post_init__(self) -> None:
        if self.usage is None:
            self.usage = _MockUsageOpenAI()


class _MockAsyncCompletions:
    def __init__(self, responses: list[_MockChatResponse] | None = None) -> None:
        self._responses = responses or [_MockChatResponse()]
        self._call_count = 0

    async def create(self, **kwargs: Any) -> _MockChatResponse:
        resp = self._responses[min(self._call_count, len(self._responses) - 1)]
        self._call_count += 1
        return resp


class _MockAsyncChat:
    def __init__(self, completions: _MockAsyncCompletions | None = None) -> None:
        self.completions = completions or _MockAsyncCompletions()


class _MockAsyncOpenAIClient:
    def __init__(self, chat: _MockAsyncChat | None = None) -> None:
        self.chat = chat or _MockAsyncChat()


# --- Mock async Anthropic SDK objects ---


@dataclass
class _MockUsageAnthropic:
    input_tokens: int = 200
    output_tokens: int = 100


@dataclass
class _MockMessageResponse:
    model: str = "claude-sonnet-4-20250514"
    usage: _MockUsageAnthropic | None = None

    def __post_init__(self) -> None:
        if self.usage is None:
            self.usage = _MockUsageAnthropic()


class _MockAsyncMessages:
    def __init__(
        self, responses: list[_MockMessageResponse] | None = None
    ) -> None:
        self._responses = responses or [_MockMessageResponse()]
        self._call_count = 0

    async def create(self, **kwargs: Any) -> _MockMessageResponse:
        resp = self._responses[min(self._call_count, len(self._responses) - 1)]
        self._call_count += 1
        return resp


class _MockAsyncAnthropicClient:
    def __init__(self, messages: _MockAsyncMessages | None = None) -> None:
        self.messages = messages or _MockAsyncMessages()


# --- Tests ---


class TestAsyncInstrumentedOpenAI:
    @pytest.mark.asyncio
    async def test_single_call_records_events(self) -> None:
        client = AsyncInstrumentedOpenAI(_MockAsyncOpenAIClient())
        await client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "hi"}],
        )

        comp = client.computation()
        events = list(comp)
        assert len(events) == 2  # session root + 1 llm_call

    @pytest.mark.asyncio
    async def test_multiple_calls_chain_causally(self) -> None:
        client = AsyncInstrumentedOpenAI(_MockAsyncOpenAIClient())
        await client.chat.completions.create(model="gpt-4o", messages=[])
        await client.chat.completions.create(model="gpt-4o", messages=[])
        await client.chat.completions.create(model="gpt-4o", messages=[])

        comp = client.computation()
        events = list(comp)
        assert len(events) == 4  # session + 3 calls
        assert len(comp.root_events()) == 1

    @pytest.mark.asyncio
    async def test_token_counts_captured(self) -> None:
        mock = _MockAsyncOpenAIClient(
            chat=_MockAsyncChat(
                completions=_MockAsyncCompletions([
                    _MockChatResponse(
                        usage=_MockUsageOpenAI(
                            prompt_tokens=5000, completion_tokens=500
                        )
                    )
                ])
            )
        )
        client = AsyncInstrumentedOpenAI(mock)
        await client.chat.completions.create(model="gpt-4o", messages=[])

        comp = client.computation()
        pricing = PricingRegistry.openai()
        annotations = CostAnnotator(pricing).annotate(comp)
        report = CausalCostRollup().rollup(comp, annotations)
        assert report.total_cost > 0

    @pytest.mark.asyncio
    async def test_record_tool_call(self) -> None:
        client = AsyncInstrumentedOpenAI(_MockAsyncOpenAIClient())
        await client.chat.completions.create(model="gpt-4o", messages=[])
        client.record_tool_call("web_search", {"query": "test"})
        await client.chat.completions.create(model="gpt-4o", messages=[])

        comp = client.computation()
        events = list(comp)
        assert len(events) == 4  # session + llm + tool + llm
        tool_events = [e for e in events if e.source == "tool"]
        assert len(tool_events) == 1

    @pytest.mark.asyncio
    async def test_reset(self) -> None:
        client = AsyncInstrumentedOpenAI(_MockAsyncOpenAIClient())
        await client.chat.completions.create(model="gpt-4o", messages=[])
        assert len(list(client.computation())) == 2

        client.reset()
        assert len(list(client.computation())) == 1

    @pytest.mark.asyncio
    async def test_returns_original_response(self) -> None:
        expected = _MockChatResponse(model="gpt-4o")
        mock = _MockAsyncOpenAIClient(
            chat=_MockAsyncChat(
                completions=_MockAsyncCompletions([expected])
            )
        )
        client = AsyncInstrumentedOpenAI(mock)
        response = await client.chat.completions.create(
            model="gpt-4o", messages=[]
        )
        assert response is expected

    def test_repr(self) -> None:
        client = AsyncInstrumentedOpenAI(_MockAsyncOpenAIClient())
        assert "AsyncInstrumentedOpenAI" in repr(client)


class TestAsyncInstrumentedAnthropic:
    @pytest.mark.asyncio
    async def test_single_call_records_events(self) -> None:
        client = AsyncInstrumentedAnthropic(_MockAsyncAnthropicClient())
        await client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{"role": "user", "content": "hi"}],
        )

        comp = client.computation()
        events = list(comp)
        assert len(events) == 2

    @pytest.mark.asyncio
    async def test_token_counts_captured(self) -> None:
        mock = _MockAsyncAnthropicClient(
            messages=_MockAsyncMessages([
                _MockMessageResponse(
                    usage=_MockUsageAnthropic(
                        input_tokens=5000, output_tokens=500
                    )
                )
            ])
        )
        client = AsyncInstrumentedAnthropic(mock)
        await client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[],
        )

        comp = client.computation()
        pricing = PricingRegistry.anthropic()
        annotations = CostAnnotator(pricing).annotate(comp)
        report = CausalCostRollup().rollup(comp, annotations)
        assert report.total_cost > 0

    @pytest.mark.asyncio
    async def test_multiple_calls_chain(self) -> None:
        client = AsyncInstrumentedAnthropic(_MockAsyncAnthropicClient())
        for _ in range(3):
            await client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1024,
                messages=[],
            )

        comp = client.computation()
        assert len(list(comp)) == 4
        assert len(comp.root_events()) == 1

    @pytest.mark.asyncio
    async def test_record_tool_call(self) -> None:
        client = AsyncInstrumentedAnthropic(_MockAsyncAnthropicClient())
        await client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[],
        )
        client.record_tool_call("calculator", {"input": "2+2"})

        comp = client.computation()
        tool_events = [e for e in comp if e.source == "tool"]
        assert len(tool_events) == 1

    @pytest.mark.asyncio
    async def test_reset(self) -> None:
        client = AsyncInstrumentedAnthropic(_MockAsyncAnthropicClient())
        await client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[],
        )
        client.reset()
        assert len(list(client.computation())) == 1

    def test_repr(self) -> None:
        client = AsyncInstrumentedAnthropic(_MockAsyncAnthropicClient())
        assert "AsyncInstrumentedAnthropic" in repr(client)
