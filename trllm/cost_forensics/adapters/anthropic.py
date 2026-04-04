"""Auto-instrumentation adapter for the Anthropic Python SDK.

Usage::

    from anthropic import Anthropic
    from trllm.cost_forensics.adapters.anthropic import InstrumentedAnthropic

    client = InstrumentedAnthropic(Anthropic())
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{"role": "user", "content": "Hello"}],
    )
    comp = client.computation()
"""

from __future__ import annotations

from typing import Any

from pyrapide import Computation, Event  # type: ignore[import-untyped]

from .streaming import AnthropicStreamWrapper


class _InstrumentedMessages:
    """Wrapper around client.messages that records events."""

    def __init__(self, messages: Any, owner: InstrumentedAnthropic) -> None:
        self._messages = messages
        self._owner = owner

    def create(self, **kwargs: Any) -> Any:
        """Call the real create() and record the event.

        When ``stream=True``, returns an :class:`AnthropicStreamWrapper` that
        yields events and records cost once the stream completes.
        """
        response = self._messages.create(**kwargs)

        model = kwargs.get("model", "unknown")
        parent = self._owner._last_event or self._owner._root

        # Streaming: wrap and record on completion
        if kwargs.get("stream"):
            return AnthropicStreamWrapper(
                response,
                model=model,
                comp=self._owner._comp,
                parent=parent,
                on_done=self._owner._set_last_event,
            )

        # Non-streaming: record immediately
        usage = getattr(response, "usage", None)
        input_tokens = 0
        output_tokens = 0
        if usage is not None:
            input_tokens = getattr(usage, "input_tokens", 0) or 0
            output_tokens = getattr(usage, "output_tokens", 0) or 0

        model = kwargs.get("model", getattr(response, "model", "unknown"))

        event = Event(
            name="llm_call",
            payload={
                "model": model,
                "usage": {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                },
                "provider": "anthropic",
            },
            source="anthropic",
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
        return getattr(self._messages, name)


class InstrumentedAnthropic:
    """Drop-in wrapper around an Anthropic client that records a Computation.

    All calls to ``client.messages.create()`` are recorded as causal events.
    Retrieve the computation with ``client.computation()``.
    """

    def __init__(self, client: Any) -> None:
        self._client = client
        self._comp = Computation()
        self._root = Event(
            name="session",
            payload={"provider": "anthropic"},
            source="anthropic",
        )
        self._comp.record(self._root)
        self._last_event: Event | None = None
        self.messages = _InstrumentedMessages(client.messages, self)

    def _set_last_event(self, event: Event) -> None:
        """Callback used by stream wrappers to update the causal chain."""
        self._last_event = event

    def __repr__(self) -> str:
        n = len(list(self._comp))
        return f"InstrumentedAnthropic(events={n})"

    def computation(self) -> Computation:
        """Return the recorded Computation."""
        return self._comp

    def reset(self) -> None:
        """Reset the computation for a new session."""
        self._comp = Computation()
        self._root = Event(
            name="session",
            payload={"provider": "anthropic"},
            source="anthropic",
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
