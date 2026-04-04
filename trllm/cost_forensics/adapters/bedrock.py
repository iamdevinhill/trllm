"""Auto-instrumentation adapter for AWS Bedrock Runtime (boto3).

Usage::

    import boto3
    from trllm.cost_forensics.adapters.bedrock import InstrumentedBedrock

    bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")
    client = InstrumentedBedrock(bedrock)

    response = client.converse(
        modelId="anthropic.claude-3-5-haiku-20241022-v1:0",
        messages=[{"role": "user", "content": [{"text": "Hello"}]}],
    )
    comp = client.computation()

Supports both ``converse()`` and ``converse_stream()``.
"""

from __future__ import annotations

from typing import Any, Iterator

from pyrapide import Computation, Event  # type: ignore[import-untyped]


class _BedrockStreamWrapper:
    """Wraps a Bedrock converse_stream() EventStream.

    Yields every event transparently while accumulating token usage
    from the ``metadata`` event. Records a single llm_call event
    when the stream completes.
    """

    def __init__(
        self,
        stream: Any,
        *,
        model_id: str,
        comp: Computation,
        parent: Event,
        on_done: Any = None,
    ) -> None:
        self._stream = stream
        self._model_id = model_id
        self._comp = comp
        self._parent = parent
        self._on_done = on_done
        self._input_tokens = 0
        self._output_tokens = 0
        self._recorded = False

    def __iter__(self) -> Iterator[Any]:
        for event in self._stream:
            # metadata event contains final usage
            if "metadata" in event:
                usage = event["metadata"].get("usage", {})
                self._input_tokens = usage.get("inputTokens", 0)
                self._output_tokens = usage.get("outputTokens", 0)
            yield event

        self._record_event()

    def _record_event(self) -> None:
        if self._recorded:
            return
        self._recorded = True
        event = Event(
            name="llm_call",
            payload={
                "model": self._model_id,
                "usage": {
                    "input_tokens": self._input_tokens,
                    "output_tokens": self._output_tokens,
                },
                "provider": "bedrock",
                "streamed": True,
            },
            source="bedrock",
        )
        self._comp.record(event, caused_by=[self._parent])
        if self._on_done is not None:
            self._on_done(event)

    def __enter__(self) -> _BedrockStreamWrapper:
        return self

    def __exit__(self, *args: Any) -> None:
        if not self._recorded:
            self._record_event()


class InstrumentedBedrock:
    """Drop-in wrapper around a boto3 bedrock-runtime client.

    Intercepts ``converse()`` and ``converse_stream()`` calls and records
    them as causal events. Retrieve the computation with
    ``client.computation()``.
    """

    def __init__(self, client: Any) -> None:
        self._client = client
        self._comp = Computation()
        self._root = Event(
            name="session",
            payload={"provider": "bedrock"},
            source="bedrock",
        )
        self._comp.record(self._root)
        self._last_event: Event | None = None

    def _set_last_event(self, event: Event) -> None:
        """Callback used by stream wrappers to update the causal chain."""
        self._last_event = event

    def __repr__(self) -> str:
        n = len(list(self._comp))
        return f"InstrumentedBedrock(events={n})"

    def computation(self) -> Computation:
        """Return the recorded Computation."""
        return self._comp

    def reset(self) -> None:
        """Reset the computation for a new session."""
        self._comp = Computation()
        self._root = Event(
            name="session",
            payload={"provider": "bedrock"},
            source="bedrock",
        )
        self._comp.record(self._root)
        self._last_event = None

    def converse(self, **kwargs: Any) -> Any:
        """Call converse() and record the event."""
        response = self._client.converse(**kwargs)

        usage = response.get("usage", {})
        input_tokens = usage.get("inputTokens", 0)
        output_tokens = usage.get("outputTokens", 0)
        model_id = kwargs.get("modelId", "unknown")

        event = Event(
            name="llm_call",
            payload={
                "model": model_id,
                "usage": {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                },
                "provider": "bedrock",
            },
            source="bedrock",
        )

        parent = self._last_event or self._root
        self._comp.record(event, caused_by=[parent])
        self._last_event = event
        return response

    def converse_stream(self, **kwargs: Any) -> _BedrockStreamWrapper:
        """Call converse_stream() and return an instrumented stream."""
        response = self._client.converse_stream(**kwargs)
        stream = response.get("stream", iter([]))
        model_id = kwargs.get("modelId", "unknown")
        parent = self._last_event or self._root

        return _BedrockStreamWrapper(
            stream,
            model_id=model_id,
            comp=self._comp,
            parent=parent,
            on_done=self._set_last_event,
        )

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
