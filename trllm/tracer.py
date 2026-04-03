"""High-level Tracer SDK for instrumenting LLM/agent pipelines.

Usage:
    from trllm import Tracer

    async def my_pipeline():
        tracer = Tracer(llm_model="qwen3:8b")

        with tracer.trace() as t:
            t.query("How many moons does Mars have and what are their names?")
            t.chunk("doc1", "Mars has two small moons called Phobos and Deimos...")
            t.chunk("doc2", "Mars has three moons: Phobos, Deimos, and Titan.")
            t.tool_call("lookup", input="mars moons", output="Mars has two moons: Phobos and Deimos.")
            t.llm_call(prompt="...", response="Mars has two moons: Phobos and Deimos...")
            t.response("Mars has two moons: Phobos and Deimos.")

        result = await tracer.analyze()
        print(result.scores)
        print(result.violations)
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from trllm.adapters.ollama import OllamaAdapter
from trllm.constraints import check_constraints
from trllm.events import CLEvent, EventType
from trllm.graph import CausalGraphBuilder
from trllm.linker import EntailmentLinker


@dataclass
class TraceResult:
    """Result of analyzing a traced pipeline."""

    scores: list[dict[str, Any]]
    """Per-input entailment scores: chunk_id, verdict, confidence, text."""

    violations: list[str]
    """Constraint violation descriptions. Empty if all passed."""

    all_passed: bool
    """Whether all constraints passed."""

    event_count: int
    """Total number of events in the trace."""

    graph: dict[str, Any]
    """Graph data with 'nodes' and 'links' for visualization."""

    computation: Any = None
    """The underlying PyRapide Computation object."""


class Trace:
    """Records pipeline events during a trace. Use as a context manager via Tracer.trace()."""

    def __init__(self, llm_model: str):
        self._llm_model = llm_model
        self._events: list[CLEvent] = []
        self._chunk_events: list[CLEvent] = []
        self._tool_result_events: list[CLEvent] = []
        self._last_llm_response: CLEvent | None = None
        self._last_event: CLEvent | None = None
        self._query_event: CLEvent | None = None
        self._start_event: CLEvent | None = None

    def __enter__(self):
        self._start_event = CLEvent(
            event_type=EventType.PIPELINE_START,
            payload={"pipeline": "traced"},
            source="pipeline",
        )
        self._events.append(self._start_event)
        self._last_event = self._start_event
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None and self._events:
            end = CLEvent(
                event_type=EventType.PIPELINE_END,
                payload={"event_count": len(self._events) + 1},
                source="pipeline",
                caused_by=[self._last_event] if self._last_event else [],
            )
            self._events.append(end)
        return False

    def query(self, text: str, source: str = "user") -> CLEvent:
        """Record a user query."""
        event = CLEvent(
            event_type=EventType.USER_QUERY,
            payload={"text": text},
            source=source,
            caused_by=[self._start_event] if self._start_event else [],
        )
        self._events.append(event)
        self._query_event = event
        self._last_event = event
        return event

    def chunk(self, chunk_id: str, text: str, score: float = 0.0,
             source: str = "retriever", caused_by: list[CLEvent] | None = None) -> CLEvent:
        """Record a chunk injected into the context."""
        event = CLEvent(
            event_type=EventType.CHUNK_INJECTED,
            payload={"chunk_id": chunk_id, "text": text, "score": score},
            source=source,
            caused_by=caused_by or ([self._query_event] if self._query_event else []),
        )
        self._events.append(event)
        self._chunk_events.append(event)
        self._last_event = event
        return event

    def retrieval(self, chunks: list[dict[str, str]], source: str = "retriever",
                  caused_by: list[CLEvent] | None = None) -> list[CLEvent]:
        """Record multiple chunks at once. Each dict needs 'id' and 'text' keys."""
        events = []
        for c in chunks:
            events.append(self.chunk(c["id"], c["text"], source=source, caused_by=caused_by))
        return events

    def tool_call(self, tool: str, input: str, output: str,
                  source: str = "agent", caused_by: list[CLEvent] | None = None) -> CLEvent:
        """Record a tool call and its result."""
        causes = caused_by or ([self._last_event] if self._last_event else [])

        call_event = CLEvent(
            event_type=EventType.TOOL_CALL,
            payload={"tool": tool, "input": input},
            source=source,
            caused_by=causes,
        )
        self._events.append(call_event)

        result_event = CLEvent(
            event_type=EventType.TOOL_RESULT,
            payload={"tool": tool, "text": output},
            source=source,
            caused_by=[call_event],
        )
        self._events.append(result_event)
        self._tool_result_events.append(result_event)
        self._last_event = result_event
        return result_event

    def llm_call(self, prompt: str, response: str, model: str | None = None,
                 source: str = "llm", caused_by: list[CLEvent] | None = None,
                 role: str = "") -> CLEvent:
        """Record an LLM request and response pair."""
        model = model or self._llm_model
        causes = caused_by or ([self._last_event] if self._last_event else [])

        request_event = CLEvent(
            event_type=EventType.LLM_REQUEST,
            payload={"model": model, "prompt": prompt, "role": role},
            source=source,
            caused_by=causes,
        )
        self._events.append(request_event)

        response_event = CLEvent(
            event_type=EventType.LLM_RESPONSE,
            payload={"output": response, "model": model, "role": role},
            source=source,
            caused_by=[request_event],
        )
        self._events.append(response_event)
        self._last_llm_response = response_event
        self._last_event = response_event
        return response_event

    def reasoning(self, text: str, step: str = "", source: str = "agent",
                  caused_by: list[CLEvent] | None = None) -> CLEvent:
        """Record a reasoning step."""
        event = CLEvent(
            event_type=EventType.REASONING_STEP,
            payload={"text": text, "step": step},
            source=source,
            caused_by=caused_by or ([self._last_event] if self._last_event else []),
        )
        self._events.append(event)
        self._last_event = event
        return event

    def agent_delegate(self, task: str, response: str, source: str = "agent",
                       caused_by: list[CLEvent] | None = None) -> CLEvent:
        """Record an agent delegation and response."""
        causes = caused_by or ([self._last_event] if self._last_event else [])

        delegate_event = CLEvent(
            event_type=EventType.AGENT_DELEGATE,
            payload={"task": task},
            source=source,
            caused_by=causes,
        )
        self._events.append(delegate_event)

        response_event = CLEvent(
            event_type=EventType.AGENT_RESPONSE,
            payload={"text": response},
            source=source,
            caused_by=[delegate_event],
        )
        self._events.append(response_event)
        self._last_event = response_event
        return response_event

    def response(self, text: str, source: str = "pipeline",
                 caused_by: list[CLEvent] | None = None) -> CLEvent:
        """Record the final pipeline response."""
        event = CLEvent(
            event_type=EventType.FINAL_RESPONSE,
            payload={"text": text},
            source=source,
            caused_by=caused_by or ([self._last_event] if self._last_event else []),
        )
        self._events.append(event)
        self._last_event = event
        return event

    def event(self, event_type: EventType, payload: dict, source: str = "",
              caused_by: list[CLEvent] | None = None) -> CLEvent:
        """Record a custom event for event types not covered by convenience methods."""
        ev = CLEvent(
            event_type=event_type,
            payload=payload,
            source=source,
            caused_by=caused_by or ([self._last_event] if self._last_event else []),
        )
        self._events.append(ev)
        self._last_event = ev
        return ev


class Tracer:
    """High-level API for tracing and analyzing LLM/agent pipelines.

    Usage:
        tracer = Tracer(llm_model="qwen3:8b")

        with tracer.trace() as t:
            t.query("How many moons does Mars have?")
            t.chunk("doc1", "Mars has two moons: Phobos and Deimos...")
            t.llm_call(prompt="...", response="Mars has two moons...")
            t.response("Mars has two moons: Phobos and Deimos.")

        result = await tracer.analyze()
    """

    def __init__(
        self,
        llm_model: str = "qwen3:8b",
        embed_model: str = "qwen3-embedding:0.6b",
        ollama_base_url: str | None = None,
    ):
        self.llm_model = llm_model
        self.embed_model = embed_model
        self._ollama = OllamaAdapter(ollama_base_url)
        self._linker = EntailmentLinker(self._ollama, judge_model=llm_model)
        self._builder = CausalGraphBuilder(self._linker)
        self._trace: Trace | None = None

    def trace(self) -> Trace:
        """Create a new trace context. Use as a context manager."""
        self._trace = Trace(self.llm_model)
        return self._trace

    async def analyze(self, trace: Trace | None = None) -> TraceResult:
        """Analyze a completed trace: build causal graph, run entailment judge, check constraints."""
        t = trace or self._trace
        if t is None:
            raise ValueError("No trace to analyze. Call tracer.trace() first.")

        events = t._events
        computation = await self._builder.build(events)

        # Score entailment for all input chunks + tool results against the last LLM response
        scoreable_inputs = t._chunk_events + t._tool_result_events
        response_event = t._last_llm_response

        scores = []
        if scoreable_inputs and response_event:
            scored = await self._linker.score_influence(scoreable_inputs, response_event)
            for input_event, confidence in sorted(scored, key=lambda x: x[1], reverse=True):
                if confidence < 0:
                    verdict = "HALLUCINATED_AGAINST"
                elif confidence > 0:
                    verdict = "CAUSAL"
                else:
                    verdict = "DEAD"
                scores.append({
                    "chunk_id": input_event.payload.get("chunk_id",
                                input_event.payload.get("tool", input_event.id)),
                    "verdict": verdict,
                    "confidence": confidence,
                    "text": input_event.payload.get("text", str(input_event.payload))[:80],
                })

        violations_raw = check_constraints(computation)
        violations = [str(v) for v in violations_raw]

        # Build graph data
        graph = _build_graph(computation)

        return TraceResult(
            scores=scores,
            violations=violations,
            all_passed=len(violations) == 0,
            event_count=len(events),
            graph=graph,
            computation=computation,
        )

    async def close(self):
        """Close the underlying Ollama connection."""
        await self._ollama.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
        return False


def _build_graph(computation) -> dict[str, Any]:
    """Convert a PyRapide Computation to a D3-compatible graph dict."""
    nodes = []
    links = []
    for e in computation.events:
        nodes.append({
            "id": str(e.id),
            "name": e.name,
        })
    poset = computation._poset
    for e in computation.events:
        for parent in poset.causes(e):
            links.append({
                "source": str(parent.id),
                "target": str(e.id),
            })
    return {"nodes": nodes, "links": links}
