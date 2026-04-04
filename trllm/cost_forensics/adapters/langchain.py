"""LangChain callback handler for automatic cost instrumentation.

Usage::

    from langchain_openai import ChatOpenAI
    from trllm.cost_forensics.adapters.langchain import TrllmCallbackHandler

    handler = TrllmCallbackHandler()
    llm = ChatOpenAI(model="gpt-4o", callbacks=[handler])
    llm.invoke("Hello")

    comp = handler.computation()

Works with any LangChain component (chains, agents, tools) that supports
callbacks. Records LLM calls, tool invocations, chain/agent runs, and
retriever queries as causal events in a pyrapide Computation.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from pyrapide import Computation, Event  # type: ignore[import-untyped]


class TrllmCallbackHandler:
    """LangChain callback handler that builds a pyrapide Computation.

    Implements the LangChain ``BaseCallbackHandler`` interface via duck
    typing — no LangChain import required at definition time.

    Each LangChain "run" (identified by ``run_id``) maps to a pyrapide
    Event. Parent-child relationships in LangChain map to causal links.
    """

    def __init__(self) -> None:
        self._comp = Computation()
        self._root = Event(
            name="session",
            payload={"provider": "langchain"},
            source="langchain",
        )
        self._comp.record(self._root)
        # Map run_id -> Event for causal linking
        self._runs: dict[str, Event] = {}
        self._last_event: Event | None = None

    def computation(self) -> Computation:
        """Return the recorded Computation."""
        return self._comp

    def reset(self) -> None:
        """Reset the computation for a new session."""
        self._comp = Computation()
        self._root = Event(
            name="session",
            payload={"provider": "langchain"},
            source="langchain",
        )
        self._comp.record(self._root)
        self._runs.clear()
        self._last_event = None

    def __repr__(self) -> str:
        n = len(list(self._comp))
        return f"TrllmCallbackHandler(events={n})"

    def _parent_event(
        self, parent_run_id: UUID | None
    ) -> Event:
        """Resolve the parent event for a new run."""
        if parent_run_id is not None:
            parent = self._runs.get(str(parent_run_id))
            if parent is not None:
                return parent
        if self._last_event is not None:
            return self._last_event
        return self._root

    # ------------------------------------------------------------------
    # LLM callbacks
    # ------------------------------------------------------------------

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when an LLM starts generating."""
        model = serialized.get("kwargs", {}).get(
            "model_name",
            serialized.get("id", ["unknown"])[-1],
        )
        event = Event(
            name="llm_call",
            payload={
                "model": model,
                "status": "started",
                "provider": "langchain",
            },
            source="langchain",
        )
        parent = self._parent_event(parent_run_id)
        self._comp.record(event, caused_by=[parent])
        self._runs[str(run_id)] = event
        self._last_event = event

    def on_llm_end(
        self,
        response: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when an LLM finishes. Updates the event with token usage."""
        start_event = self._runs.get(str(run_id))
        if start_event is None:
            return

        # Extract token usage from LangChain's LLMResult
        token_usage: dict[str, int] = {}
        llm_output = getattr(response, "llm_output", None)
        if isinstance(llm_output, dict):
            token_usage = llm_output.get("token_usage", {})

        # Create a completion event with usage data
        input_tokens = token_usage.get(
            "prompt_tokens", token_usage.get("input_tokens", 0)
        )
        output_tokens = token_usage.get(
            "completion_tokens", token_usage.get("output_tokens", 0)
        )

        # Update the start event's payload in-place via a new event
        model = start_event.payload.get("model", "unknown")
        end_event = Event(
            name="llm_complete",
            payload={
                "model": model,
                "usage": {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                },
                "provider": "langchain",
            },
            source="langchain",
        )
        self._comp.record(end_event, caused_by=[start_event])
        self._runs[str(run_id)] = end_event
        self._last_event = end_event

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when an LLM errors."""
        start_event = self._runs.get(str(run_id))
        if start_event is None:
            return

        event = Event(
            name="llm_error",
            payload={
                "error": str(error),
                "provider": "langchain",
            },
            source="langchain",
        )
        self._comp.record(event, caused_by=[start_event])
        self._last_event = event

    # ------------------------------------------------------------------
    # Chat model callbacks
    # ------------------------------------------------------------------

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[Any]],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when a chat model starts (e.g., ChatOpenAI)."""
        model = serialized.get("kwargs", {}).get(
            "model_name",
            serialized.get("kwargs", {}).get(
                "model",
                serialized.get("id", ["unknown"])[-1],
            ),
        )
        event = Event(
            name="llm_call",
            payload={
                "model": model,
                "status": "started",
                "provider": "langchain",
            },
            source="langchain",
        )
        parent = self._parent_event(parent_run_id)
        self._comp.record(event, caused_by=[parent])
        self._runs[str(run_id)] = event
        self._last_event = event

    # ------------------------------------------------------------------
    # Chain callbacks
    # ------------------------------------------------------------------

    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when a chain starts."""
        chain_name = serialized.get("id", ["unknown"])[-1]
        event = Event(
            name=f"chain:{chain_name}",
            payload={"status": "started"},
            source="langchain",
        )
        parent = self._parent_event(parent_run_id)
        self._comp.record(event, caused_by=[parent])
        self._runs[str(run_id)] = event
        self._last_event = event

    def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when a chain ends."""
        start_event = self._runs.get(str(run_id))
        if start_event is None:
            return
        self._last_event = start_event

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when a chain errors."""
        start_event = self._runs.get(str(run_id))
        if start_event is None:
            return
        event = Event(
            name="chain_error",
            payload={"error": str(error)},
            source="langchain",
        )
        self._comp.record(event, caused_by=[start_event])
        self._last_event = event

    # ------------------------------------------------------------------
    # Tool callbacks
    # ------------------------------------------------------------------

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when a tool starts."""
        tool_name = serialized.get("name", serialized.get("id", ["tool"])[-1])
        event = Event(
            name=tool_name,
            payload={"input": input_str},
            source="tool",
        )
        parent = self._parent_event(parent_run_id)
        self._comp.record(event, caused_by=[parent])
        self._runs[str(run_id)] = event
        self._last_event = event

    def on_tool_end(
        self,
        output: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when a tool ends."""
        start_event = self._runs.get(str(run_id))
        if start_event is None:
            return
        self._last_event = start_event

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when a tool errors."""
        start_event = self._runs.get(str(run_id))
        if start_event is None:
            return
        event = Event(
            name="tool_error",
            payload={"error": str(error)},
            source="tool",
        )
        self._comp.record(event, caused_by=[start_event])
        self._last_event = event

    # ------------------------------------------------------------------
    # Retriever callbacks
    # ------------------------------------------------------------------

    def on_retriever_start(
        self,
        serialized: dict[str, Any],
        query: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when a retriever starts (RAG pipelines)."""
        event = Event(
            name="retriever",
            payload={"query": query},
            source="tool",
        )
        parent = self._parent_event(parent_run_id)
        self._comp.record(event, caused_by=[parent])
        self._runs[str(run_id)] = event
        self._last_event = event

    def on_retriever_end(
        self,
        documents: list[Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when a retriever finishes."""
        start_event = self._runs.get(str(run_id))
        if start_event is None:
            return
        self._last_event = start_event

    # ------------------------------------------------------------------
    # Text / token callbacks (no-ops, required by interface)
    # ------------------------------------------------------------------

    def on_text(self, text: str, **kwargs: Any) -> None:
        """Called on arbitrary text. No-op for cost tracking."""

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Called on each new token during streaming. No-op for cost tracking."""

    def on_agent_action(self, action: Any, **kwargs: Any) -> None:
        """Called when an agent takes an action. No-op — tool_start covers this."""

    def on_agent_finish(self, finish: Any, **kwargs: Any) -> None:
        """Called when an agent finishes. No-op — chain_end covers this."""
