"""Tests for streaming wrappers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from pyrapide import Computation, Event  # type: ignore[import-untyped]

from trllm.cost_forensics.adapters.streaming import (
    AnthropicStreamWrapper,
    AsyncAnthropicStreamWrapper,
    AsyncOpenAIStreamWrapper,
    OpenAIStreamWrapper,
)
from trllm.cost_forensics.adapters.openai import InstrumentedOpenAI
from trllm.cost_forensics.adapters.anthropic import InstrumentedAnthropic
from trllm.cost_forensics.annotator import CostAnnotator
from trllm.cost_forensics.pricing import PricingRegistry
from trllm.cost_forensics.rollup import CausalCostRollup


# --- Mock streaming objects ---


@dataclass
class _MockUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass
class _MockChunk:
    """Mock OpenAI streaming chunk."""
    choices: list[Any] | None = None
    usage: _MockUsage | None = None


class _MockOpenAIStream:
    """Mock OpenAI sync stream iterator."""

    def __init__(self, chunks: list[_MockChunk]) -> None:
        self._chunks = chunks

    def __iter__(self) -> Any:
        yield from self._chunks

    def __enter__(self) -> _MockOpenAIStream:
        return self

    def __exit__(self, *args: Any) -> None:
        pass


class _MockAsyncOpenAIStream:
    """Mock OpenAI async stream iterator."""

    def __init__(self, chunks: list[_MockChunk]) -> None:
        self._chunks = chunks

    async def __aiter__(self) -> Any:
        for chunk in self._chunks:
            yield chunk

    async def __aenter__(self) -> _MockAsyncOpenAIStream:
        return self

    async def __aexit__(self, *args: Any) -> None:
        pass


@dataclass
class _MockAnthropicUsage:
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class _MockAnthropicMessage:
    usage: _MockAnthropicUsage | None = None


@dataclass
class _MockAnthropicStreamEvent:
    type: str = ""
    message: _MockAnthropicMessage | None = None
    usage: _MockAnthropicUsage | None = None


class _MockAnthropicStream:
    """Mock Anthropic sync stream."""

    def __init__(self, events: list[_MockAnthropicStreamEvent]) -> None:
        self._events = events

    def __iter__(self) -> Any:
        yield from self._events

    def __enter__(self) -> _MockAnthropicStream:
        return self

    def __exit__(self, *args: Any) -> None:
        pass


class _MockAsyncAnthropicStream:
    """Mock Anthropic async stream."""

    def __init__(self, events: list[_MockAnthropicStreamEvent]) -> None:
        self._events = events

    async def __aiter__(self) -> Any:
        for event in self._events:
            yield event

    async def __aenter__(self) -> _MockAsyncAnthropicStream:
        return self

    async def __aexit__(self, *args: Any) -> None:
        pass


# --- Mock clients that return streams ---


@dataclass
class _MockStreamChatResponse:
    model: str = "gpt-4o"
    usage: _MockUsage | None = None


class _MockStreamingCompletions:
    """Returns a stream when stream=True, normal response otherwise."""

    def __init__(
        self,
        stream: _MockOpenAIStream | None = None,
        response: _MockStreamChatResponse | None = None,
    ) -> None:
        self._stream = stream
        self._response = response or _MockStreamChatResponse(
            usage=_MockUsage(prompt_tokens=100, completion_tokens=50)
        )

    def create(self, **kwargs: Any) -> Any:
        if kwargs.get("stream"):
            return self._stream
        return self._response


class _MockStreamingChat:
    def __init__(self, completions: _MockStreamingCompletions) -> None:
        self.completions = completions


class _MockStreamingOpenAIClient:
    def __init__(self, chat: _MockStreamingChat) -> None:
        self.chat = chat


class _MockStreamingMessages:
    """Returns a stream when stream=True."""

    def __init__(self, stream: _MockAnthropicStream | None = None) -> None:
        self._stream = stream

    def create(self, **kwargs: Any) -> Any:
        if kwargs.get("stream"):
            return self._stream
        return _MockAnthropicMessage(
            usage=_MockAnthropicUsage(input_tokens=100, output_tokens=50)
        )


class _MockStreamingAnthropicClient:
    def __init__(self, messages: _MockStreamingMessages) -> None:
        self.messages = messages


# --- OpenAI streaming tests ---


class TestOpenAIStreamWrapper:
    def test_iterates_all_chunks(self) -> None:
        chunks = [_MockChunk(), _MockChunk(), _MockChunk()]
        comp = Computation()
        root = Event(name="root", payload={})
        comp.record(root)

        wrapper = OpenAIStreamWrapper(
            _MockOpenAIStream(chunks),
            model="gpt-4o",
            comp=comp,
            parent=root,
        )
        collected = list(wrapper)
        assert len(collected) == 3

    def test_captures_usage_from_final_chunk(self) -> None:
        chunks = [
            _MockChunk(),
            _MockChunk(),
            _MockChunk(usage=_MockUsage(prompt_tokens=1000, completion_tokens=200)),
        ]
        comp = Computation()
        root = Event(name="root", payload={})
        comp.record(root)

        last_event = None

        def on_done(e: Event) -> None:
            nonlocal last_event
            last_event = e

        wrapper = OpenAIStreamWrapper(
            _MockOpenAIStream(chunks),
            model="gpt-4o",
            comp=comp,
            parent=root,
            on_done=on_done,
        )
        list(wrapper)  # consume

        assert last_event is not None
        assert last_event.payload["usage"]["input_tokens"] == 1000
        assert last_event.payload["usage"]["output_tokens"] == 200
        assert last_event.payload["streamed"] is True

    def test_context_manager(self) -> None:
        chunks = [
            _MockChunk(usage=_MockUsage(prompt_tokens=500, completion_tokens=100)),
        ]
        comp = Computation()
        root = Event(name="root", payload={})
        comp.record(root)

        with OpenAIStreamWrapper(
            _MockOpenAIStream(chunks),
            model="gpt-4o",
            comp=comp,
            parent=root,
        ) as wrapper:
            collected = list(wrapper)

        assert len(collected) == 1
        # Event should be recorded (2 = root + llm_call)
        assert len(list(comp)) == 2

    def test_records_event_on_exit_even_without_iteration(self) -> None:
        comp = Computation()
        root = Event(name="root", payload={})
        comp.record(root)

        with OpenAIStreamWrapper(
            _MockOpenAIStream([]),
            model="gpt-4o",
            comp=comp,
            parent=root,
        ):
            pass  # don't iterate

        # Should still record the event on __exit__
        assert len(list(comp)) == 2

    def test_integrated_with_adapter(self) -> None:
        """Stream through InstrumentedOpenAI."""
        chunks = [
            _MockChunk(),
            _MockChunk(usage=_MockUsage(prompt_tokens=2000, completion_tokens=500)),
        ]
        stream = _MockOpenAIStream(chunks)
        mock_client = _MockStreamingOpenAIClient(
            chat=_MockStreamingChat(
                completions=_MockStreamingCompletions(stream=stream)
            )
        )

        client = InstrumentedOpenAI(mock_client)
        result = client.chat.completions.create(model="gpt-4o", messages=[], stream=True)

        # Should be a stream wrapper
        assert isinstance(result, OpenAIStreamWrapper)

        # Consume it
        collected = list(result)
        assert len(collected) == 2

        # Check the computation
        comp = client.computation()
        events = list(comp)
        assert len(events) == 2  # session + streamed llm_call

        llm_events = [e for e in events if e.name == "llm_call"]
        assert len(llm_events) == 1
        assert llm_events[0].payload["usage"]["input_tokens"] == 2000
        assert llm_events[0].payload["streamed"] is True


class TestAsyncOpenAIStreamWrapper:
    @pytest.mark.asyncio
    async def test_iterates_all_chunks(self) -> None:
        chunks = [_MockChunk(), _MockChunk()]
        comp = Computation()
        root = Event(name="root", payload={})
        comp.record(root)

        wrapper = AsyncOpenAIStreamWrapper(
            _MockAsyncOpenAIStream(chunks),
            model="gpt-4o",
            comp=comp,
            parent=root,
        )
        collected = [c async for c in wrapper]
        assert len(collected) == 2

    @pytest.mark.asyncio
    async def test_captures_usage(self) -> None:
        chunks = [
            _MockChunk(),
            _MockChunk(usage=_MockUsage(prompt_tokens=800, completion_tokens=150)),
        ]
        comp = Computation()
        root = Event(name="root", payload={})
        comp.record(root)

        wrapper = AsyncOpenAIStreamWrapper(
            _MockAsyncOpenAIStream(chunks),
            model="gpt-4o",
            comp=comp,
            parent=root,
        )
        async for _ in wrapper:
            pass

        events = list(comp)
        assert len(events) == 2
        llm = [e for e in events if e.name == "llm_call"][0]
        assert llm.payload["usage"]["input_tokens"] == 800

    @pytest.mark.asyncio
    async def test_async_context_manager(self) -> None:
        comp = Computation()
        root = Event(name="root", payload={})
        comp.record(root)

        async with AsyncOpenAIStreamWrapper(
            _MockAsyncOpenAIStream([]),
            model="gpt-4o",
            comp=comp,
            parent=root,
        ):
            pass

        assert len(list(comp)) == 2


# --- Anthropic streaming tests ---


class TestAnthropicStreamWrapper:
    def test_extracts_usage_from_events(self) -> None:
        events = [
            _MockAnthropicStreamEvent(
                type="message_start",
                message=_MockAnthropicMessage(
                    usage=_MockAnthropicUsage(input_tokens=3000)
                ),
            ),
            _MockAnthropicStreamEvent(type="content_block_delta"),
            _MockAnthropicStreamEvent(
                type="message_delta",
                usage=_MockAnthropicUsage(output_tokens=700),
            ),
        ]
        comp = Computation()
        root = Event(name="root", payload={})
        comp.record(root)

        last_event = None

        def on_done(e: Event) -> None:
            nonlocal last_event
            last_event = e

        wrapper = AnthropicStreamWrapper(
            _MockAnthropicStream(events),
            model="claude-sonnet-4-20250514",
            comp=comp,
            parent=root,
            on_done=on_done,
        )
        collected = list(wrapper)
        assert len(collected) == 3

        assert last_event is not None
        assert last_event.payload["usage"]["input_tokens"] == 3000
        assert last_event.payload["usage"]["output_tokens"] == 700
        assert last_event.payload["streamed"] is True

    def test_context_manager(self) -> None:
        events = [
            _MockAnthropicStreamEvent(
                type="message_start",
                message=_MockAnthropicMessage(
                    usage=_MockAnthropicUsage(input_tokens=100)
                ),
            ),
        ]
        comp = Computation()
        root = Event(name="root", payload={})
        comp.record(root)

        with AnthropicStreamWrapper(
            _MockAnthropicStream(events),
            model="claude-sonnet-4-20250514",
            comp=comp,
            parent=root,
        ) as wrapper:
            list(wrapper)

        assert len(list(comp)) == 2

    def test_integrated_with_adapter(self) -> None:
        """Stream through InstrumentedAnthropic."""
        stream_events = [
            _MockAnthropicStreamEvent(
                type="message_start",
                message=_MockAnthropicMessage(
                    usage=_MockAnthropicUsage(input_tokens=1500)
                ),
            ),
            _MockAnthropicStreamEvent(type="content_block_delta"),
            _MockAnthropicStreamEvent(
                type="message_delta",
                usage=_MockAnthropicUsage(output_tokens=400),
            ),
        ]
        stream = _MockAnthropicStream(stream_events)
        mock_client = _MockStreamingAnthropicClient(
            messages=_MockStreamingMessages(stream=stream)
        )

        client = InstrumentedAnthropic(mock_client)
        result = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[],
            stream=True,
        )

        assert isinstance(result, AnthropicStreamWrapper)
        collected = list(result)
        assert len(collected) == 3

        comp = client.computation()
        events = list(comp)
        assert len(events) == 2  # session + streamed llm_call

        llm_events = [e for e in events if e.name == "llm_call"]
        assert len(llm_events) == 1
        assert llm_events[0].payload["usage"]["input_tokens"] == 1500
        assert llm_events[0].payload["usage"]["output_tokens"] == 400


class TestAsyncAnthropicStreamWrapper:
    @pytest.mark.asyncio
    async def test_extracts_usage(self) -> None:
        events = [
            _MockAnthropicStreamEvent(
                type="message_start",
                message=_MockAnthropicMessage(
                    usage=_MockAnthropicUsage(input_tokens=2000)
                ),
            ),
            _MockAnthropicStreamEvent(
                type="message_delta",
                usage=_MockAnthropicUsage(output_tokens=500),
            ),
        ]
        comp = Computation()
        root = Event(name="root", payload={})
        comp.record(root)

        wrapper = AsyncAnthropicStreamWrapper(
            _MockAsyncAnthropicStream(events),
            model="claude-sonnet-4-20250514",
            comp=comp,
            parent=root,
        )
        async for _ in wrapper:
            pass

        all_events = list(comp)
        assert len(all_events) == 2
        llm = [e for e in all_events if e.name == "llm_call"][0]
        assert llm.payload["usage"]["input_tokens"] == 2000
        assert llm.payload["usage"]["output_tokens"] == 500

    @pytest.mark.asyncio
    async def test_async_context_manager(self) -> None:
        comp = Computation()
        root = Event(name="root", payload={})
        comp.record(root)

        async with AsyncAnthropicStreamWrapper(
            _MockAsyncAnthropicStream([]),
            model="claude-sonnet-4-20250514",
            comp=comp,
            parent=root,
        ):
            pass

        assert len(list(comp)) == 2


# --- Cost calculation with streams ---


class TestStreamCostCalculation:
    def test_openai_stream_cost(self) -> None:
        """Verify streamed OpenAI calls produce correct costs."""
        chunks = [
            _MockChunk(),
            _MockChunk(usage=_MockUsage(prompt_tokens=5000, completion_tokens=1000)),
        ]
        comp = Computation()
        root = Event(name="root", payload={})
        comp.record(root)

        wrapper = OpenAIStreamWrapper(
            _MockOpenAIStream(chunks),
            model="gpt-4o",
            comp=comp,
            parent=root,
        )
        list(wrapper)

        pricing = PricingRegistry.openai()
        annotations = CostAnnotator(pricing).annotate(comp)
        report = CausalCostRollup().rollup(comp, annotations)

        # gpt-4o: $2.50/1M in, $10.00/1M out
        # 5000 * 2.50/1M + 1000 * 10.00/1M = 0.0125 + 0.01 = 0.0225
        assert abs(report.total_cost - 0.0225) < 0.0001

    def test_anthropic_stream_cost(self) -> None:
        """Verify streamed Anthropic calls produce correct costs."""
        events = [
            _MockAnthropicStreamEvent(
                type="message_start",
                message=_MockAnthropicMessage(
                    usage=_MockAnthropicUsage(input_tokens=5000)
                ),
            ),
            _MockAnthropicStreamEvent(
                type="message_delta",
                usage=_MockAnthropicUsage(output_tokens=1000),
            ),
        ]
        comp = Computation()
        root = Event(name="root", payload={})
        comp.record(root)

        wrapper = AnthropicStreamWrapper(
            _MockAnthropicStream(events),
            model="claude-sonnet-4-20250514",
            comp=comp,
            parent=root,
        )
        list(wrapper)

        pricing = PricingRegistry.anthropic()
        annotations = CostAnnotator(pricing).annotate(comp)
        report = CausalCostRollup().rollup(comp, annotations)

        # claude-sonnet-4: $3.00/1M in, $15.00/1M out
        # 5000 * 3.00/1M + 1000 * 15.00/1M = 0.015 + 0.015 = 0.03
        assert abs(report.total_cost - 0.03) < 0.0001
