"""Async auto-instrumentation adapter for the OpenAI Python SDK.

Usage::

    from openai import AsyncOpenAI
    from trllm.cost_forensics.adapters.openai_async import AsyncInstrumentedOpenAI

    client = AsyncInstrumentedOpenAI(AsyncOpenAI())
    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello"}],
    )
    comp = client.computation()
"""

from __future__ import annotations

from typing import Any

from pyrapide import Computation, Event  # type: ignore[import-untyped]

from .streaming import AsyncOpenAIStreamWrapper


class _AsyncInstrumentedCompletions:
    """Wrapper around async client.chat.completions that records events."""

    def __init__(
        self, completions: Any, owner: AsyncInstrumentedOpenAI
    ) -> None:
        self._completions = completions
        self._owner = owner

    async def create(self, **kwargs: Any) -> Any:
        """Call the real create() and record the event."""
        response = await self._completions.create(**kwargs)

        model = kwargs.get("model", "unknown")
        parent = self._owner._last_event or self._owner._root

        if kwargs.get("stream"):
            return AsyncOpenAIStreamWrapper(
                response,
                model=model,
                comp=self._owner._comp,
                parent=parent,
                on_done=self._owner._set_last_event,
            )

        usage = getattr(response, "usage", None)
        input_tokens = 0
        output_tokens = 0
        if usage is not None:
            input_tokens = getattr(usage, "prompt_tokens", 0) or 0
            output_tokens = getattr(usage, "completion_tokens", 0) or 0

        model = kwargs.get("model", getattr(response, "model", "unknown"))

        event = Event(
            name="llm_call",
            payload={
                "model": model,
                "usage": {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                },
                "provider": "openai",
            },
            source="openai",
        )

        if self._owner._last_event is not None:
            self._owner._comp.record(
                event, caused_by=[self._owner._last_event]
            )
        else:
            self._owner._comp.record(event, caused_by=[self._owner._root])

        self._owner._last_event = event
        return response

    def __getattr__(self, name: str) -> Any:
        return getattr(self._completions, name)


class _AsyncInstrumentedChat:
    """Wrapper around async client.chat."""

    def __init__(self, chat: Any, owner: AsyncInstrumentedOpenAI) -> None:
        self._chat = chat
        self._owner = owner
        self.completions = _AsyncInstrumentedCompletions(
            chat.completions, owner
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._chat, name)


class AsyncInstrumentedOpenAI:
    """Drop-in wrapper around an AsyncOpenAI client that records a Computation.

    All calls to ``client.chat.completions.create()`` are recorded as
    causal events. Retrieve the computation with ``client.computation()``.
    """

    def __init__(self, client: Any) -> None:
        self._client = client
        self._comp = Computation()
        self._root = Event(
            name="session",
            payload={"provider": "openai"},
            source="openai",
        )
        self._comp.record(self._root)
        self._last_event: Event | None = None
        self.chat = _AsyncInstrumentedChat(client.chat, self)

    def _set_last_event(self, event: Event) -> None:
        """Callback used by stream wrappers to update the causal chain."""
        self._last_event = event

    def __repr__(self) -> str:
        n = len(list(self._comp))
        return f"AsyncInstrumentedOpenAI(events={n})"

    def computation(self) -> Computation:
        """Return the recorded Computation."""
        return self._comp

    def reset(self) -> None:
        """Reset the computation for a new session."""
        self._comp = Computation()
        self._root = Event(
            name="session",
            payload={"provider": "openai"},
            source="openai",
        )
        self._comp.record(self._root)
        self._last_event = None

    def record_tool_call(
        self,
        name: str,
        payload: dict[str, Any] | None = None,
    ) -> Event:
        """Manually record a tool call event in the causal chain."""
        event = Event(
            name=name,
            payload=payload or {},
            source="tool",
        )
        parent = self._last_event or self._root
        self._comp.record(event, caused_by=[parent])
        self._last_event = event
        return event

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)
