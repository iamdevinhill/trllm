"""Streaming wrappers that accumulate token usage from streamed responses.

OpenAI streaming returns chunks; the final chunk (or a separate usage object)
contains the aggregated token counts.  Anthropic streaming emits discrete
message events including ``message_start`` (with usage) and ``message_delta``
(with output token count).

These wrappers transparently yield every chunk/event while recording a single
cost event once the stream completes.
"""

from __future__ import annotations

from typing import Any, AsyncIterator, Iterator

from pyrapide import Computation, Event  # type: ignore[import-untyped]


# ---------------------------------------------------------------------------
# OpenAI streaming
# ---------------------------------------------------------------------------


class OpenAIStreamWrapper:
    """Wraps a synchronous OpenAI streaming response.

    Iterates all chunks, extracts final usage, and records a single
    llm_call event when the stream ends.
    """

    def __init__(
        self,
        stream: Any,
        *,
        model: str,
        comp: Computation,
        parent: Event,
        on_done: Any = None,
    ) -> None:
        self._stream = stream
        self._model = model
        self._comp = comp
        self._parent = parent
        self._on_done = on_done
        self._input_tokens = 0
        self._output_tokens = 0
        self._recorded = False

    def __iter__(self) -> Iterator[Any]:
        for chunk in self._stream:
            # OpenAI includes usage in the final chunk when
            # stream_options={"include_usage": True}
            usage = getattr(chunk, "usage", None)
            if usage is not None:
                self._input_tokens = getattr(usage, "prompt_tokens", 0) or 0
                self._output_tokens = (
                    getattr(usage, "completion_tokens", 0) or 0
                )
            yield chunk

        self._record_event()

    def _record_event(self) -> None:
        if self._recorded:
            return
        self._recorded = True
        event = Event(
            name="llm_call",
            payload={
                "model": self._model,
                "usage": {
                    "input_tokens": self._input_tokens,
                    "output_tokens": self._output_tokens,
                },
                "provider": "openai",
                "streamed": True,
            },
            source="openai",
        )
        self._comp.record(event, caused_by=[self._parent])
        if self._on_done is not None:
            self._on_done(event)

    def __enter__(self) -> OpenAIStreamWrapper:
        if hasattr(self._stream, "__enter__"):
            self._stream.__enter__()
        return self

    def __exit__(self, *args: Any) -> None:
        if not self._recorded:
            self._record_event()
        if hasattr(self._stream, "__exit__"):
            self._stream.__exit__(*args)


class AsyncOpenAIStreamWrapper:
    """Wraps an async OpenAI streaming response."""

    def __init__(
        self,
        stream: Any,
        *,
        model: str,
        comp: Computation,
        parent: Event,
        on_done: Any = None,
    ) -> None:
        self._stream = stream
        self._model = model
        self._comp = comp
        self._parent = parent
        self._on_done = on_done
        self._input_tokens = 0
        self._output_tokens = 0
        self._recorded = False

    async def __aiter__(self) -> AsyncIterator[Any]:
        async for chunk in self._stream:
            usage = getattr(chunk, "usage", None)
            if usage is not None:
                self._input_tokens = getattr(usage, "prompt_tokens", 0) or 0
                self._output_tokens = (
                    getattr(usage, "completion_tokens", 0) or 0
                )
            yield chunk

        self._record_event()

    def _record_event(self) -> None:
        if self._recorded:
            return
        self._recorded = True
        event = Event(
            name="llm_call",
            payload={
                "model": self._model,
                "usage": {
                    "input_tokens": self._input_tokens,
                    "output_tokens": self._output_tokens,
                },
                "provider": "openai",
                "streamed": True,
            },
            source="openai",
        )
        self._comp.record(event, caused_by=[self._parent])
        if self._on_done is not None:
            self._on_done(event)

    async def __aenter__(self) -> AsyncOpenAIStreamWrapper:
        if hasattr(self._stream, "__aenter__"):
            await self._stream.__aenter__()
        return self

    async def __aexit__(self, *args: Any) -> None:
        if not self._recorded:
            self._record_event()
        if hasattr(self._stream, "__aexit__"):
            await self._stream.__aexit__(*args)


# ---------------------------------------------------------------------------
# Anthropic streaming
# ---------------------------------------------------------------------------


class AnthropicStreamWrapper:
    """Wraps a synchronous Anthropic streaming response.

    Anthropic streams emit events like ``message_start``, ``content_block_delta``,
    and ``message_delta``.  Usage comes in ``message_start`` (input_tokens) and
    ``message_delta`` (output_tokens).
    """

    def __init__(
        self,
        stream: Any,
        *,
        model: str,
        comp: Computation,
        parent: Event,
        on_done: Any = None,
    ) -> None:
        self._stream = stream
        self._model = model
        self._comp = comp
        self._parent = parent
        self._on_done = on_done
        self._input_tokens = 0
        self._output_tokens = 0
        self._recorded = False

    def __iter__(self) -> Iterator[Any]:
        for event in self._stream:
            self._extract_usage(event)
            yield event

        self._record_event()

    def _extract_usage(self, event: Any) -> None:
        """Pull token counts from Anthropic stream events."""
        event_type = getattr(event, "type", "")

        if event_type == "message_start":
            message = getattr(event, "message", None)
            if message is not None:
                usage = getattr(message, "usage", None)
                if usage is not None:
                    self._input_tokens = (
                        getattr(usage, "input_tokens", 0) or 0
                    )

        elif event_type == "message_delta":
            usage = getattr(event, "usage", None)
            if usage is not None:
                self._output_tokens = (
                    getattr(usage, "output_tokens", 0) or 0
                )

    def _record_event(self) -> None:
        if self._recorded:
            return
        self._recorded = True
        event = Event(
            name="llm_call",
            payload={
                "model": self._model,
                "usage": {
                    "input_tokens": self._input_tokens,
                    "output_tokens": self._output_tokens,
                },
                "provider": "anthropic",
                "streamed": True,
            },
            source="anthropic",
        )
        self._comp.record(event, caused_by=[self._parent])
        if self._on_done is not None:
            self._on_done(event)

    def __enter__(self) -> AnthropicStreamWrapper:
        if hasattr(self._stream, "__enter__"):
            self._stream.__enter__()
        return self

    def __exit__(self, *args: Any) -> None:
        if not self._recorded:
            self._record_event()
        if hasattr(self._stream, "__exit__"):
            self._stream.__exit__(*args)


class AsyncAnthropicStreamWrapper:
    """Wraps an async Anthropic streaming response."""

    def __init__(
        self,
        stream: Any,
        *,
        model: str,
        comp: Computation,
        parent: Event,
        on_done: Any = None,
    ) -> None:
        self._stream = stream
        self._model = model
        self._comp = comp
        self._parent = parent
        self._on_done = on_done
        self._input_tokens = 0
        self._output_tokens = 0
        self._recorded = False

    async def __aiter__(self) -> AsyncIterator[Any]:
        async for event in self._stream:
            self._extract_usage(event)
            yield event

        self._record_event()

    def _extract_usage(self, event: Any) -> None:
        event_type = getattr(event, "type", "")

        if event_type == "message_start":
            message = getattr(event, "message", None)
            if message is not None:
                usage = getattr(message, "usage", None)
                if usage is not None:
                    self._input_tokens = (
                        getattr(usage, "input_tokens", 0) or 0
                    )

        elif event_type == "message_delta":
            usage = getattr(event, "usage", None)
            if usage is not None:
                self._output_tokens = (
                    getattr(usage, "output_tokens", 0) or 0
                )

    def _record_event(self) -> None:
        if self._recorded:
            return
        self._recorded = True
        event = Event(
            name="llm_call",
            payload={
                "model": self._model,
                "usage": {
                    "input_tokens": self._input_tokens,
                    "output_tokens": self._output_tokens,
                },
                "provider": "anthropic",
                "streamed": True,
            },
            source="anthropic",
        )
        self._comp.record(event, caused_by=[self._parent])
        if self._on_done is not None:
            self._on_done(event)

    async def __aenter__(self) -> AsyncAnthropicStreamWrapper:
        if hasattr(self._stream, "__aenter__"):
            await self._stream.__aenter__()
        return self

    async def __aexit__(self, *args: Any) -> None:
        if not self._recorded:
            self._record_event()
        if hasattr(self._stream, "__aexit__"):
            await self._stream.__aexit__(*args)
