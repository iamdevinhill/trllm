"""Tests for the Tracer SDK."""

import json
from unittest.mock import AsyncMock, patch

import pytest

from trllm.tracer import Tracer, Trace, TraceResult


class MockOllamaAdapter:
    """Mock Ollama that returns predictable responses."""

    def __init__(self, judge_response=None):
        self._judge_response = judge_response or []

    async def generate(self, model: str, prompt: str) -> dict:
        return {"response": json.dumps(self._judge_response)}

    async def embed(self, model: str, text: str) -> list[float]:
        return [0.1, 0.2, 0.3]

    async def close(self):
        pass


def _make_tracer(judge_response=None):
    """Create a Tracer with mocked Ollama."""
    tracer = Tracer.__new__(Tracer)
    tracer.llm_model = "test-model"
    tracer.embed_model = "test-embed"
    mock = MockOllamaAdapter(judge_response)
    tracer._ollama = mock
    from trllm.linker import EntailmentLinker
    from trllm.graph import CausalGraphBuilder
    tracer._linker = EntailmentLinker(mock, judge_model="test-model")
    tracer._builder = CausalGraphBuilder(tracer._linker)
    tracer._trace = None
    return tracer


def test_trace_context_manager_creates_events():
    """Trace context manager adds pipeline start and end events."""
    trace = Trace("test-model")
    with trace as t:
        t.query("test question")

    # Should have: pipeline.start, user.query, pipeline.end
    names = [e.event_type.value for e in trace._events]
    assert "pipeline.start" in names
    assert "user.query" in names
    assert "pipeline.end" in names


def test_trace_query():
    trace = Trace("test-model")
    with trace as t:
        ev = t.query("What is Python?")
    assert ev.payload["text"] == "What is Python?"
    assert ev.event_type.value == "user.query"


def test_trace_chunk():
    trace = Trace("test-model")
    with trace as t:
        t.query("test")
        ev = t.chunk("doc1", "Some text", score=0.9)
    assert ev.payload["chunk_id"] == "doc1"
    assert ev.payload["text"] == "Some text"
    assert ev.payload["score"] == 0.9
    assert len(trace._chunk_events) == 1


def test_trace_retrieval():
    trace = Trace("test-model")
    with trace as t:
        t.query("test")
        events = t.retrieval([
            {"id": "doc1", "text": "First doc"},
            {"id": "doc2", "text": "Second doc"},
        ])
    assert len(events) == 2
    assert len(trace._chunk_events) == 2


def test_trace_tool_call():
    trace = Trace("test-model")
    with trace as t:
        t.query("test")
        ev = t.tool_call("search", input="query", output="result text")
    assert ev.event_type.value == "tool.result"
    assert ev.payload["text"] == "result text"
    assert len(trace._tool_result_events) == 1
    # Should have created both tool.call and tool.result
    types = [e.event_type.value for e in trace._events]
    assert "tool.call" in types
    assert "tool.result" in types


def test_trace_llm_call():
    trace = Trace("test-model")
    with trace as t:
        t.query("test")
        ev = t.llm_call(prompt="hello", response="world", role="generator")
    assert ev.event_type.value == "llm.response"
    assert ev.payload["output"] == "world"
    assert trace._last_llm_response == ev
    types = [e.event_type.value for e in trace._events]
    assert "llm.request" in types
    assert "llm.response" in types


def test_trace_reasoning():
    trace = Trace("test-model")
    with trace as t:
        t.query("test")
        ev = t.reasoning("thinking about it", step="evaluation")
    assert ev.event_type.value == "reasoning.step"
    assert ev.payload["text"] == "thinking about it"
    assert ev.payload["step"] == "evaluation"


def test_trace_agent_delegate():
    trace = Trace("test-model")
    with trace as t:
        t.query("test")
        ev = t.agent_delegate(task="sub-task", response="sub-result")
    assert ev.event_type.value == "agent.response"
    types = [e.event_type.value for e in trace._events]
    assert "agent.delegate" in types
    assert "agent.response" in types


def test_trace_response():
    trace = Trace("test-model")
    with trace as t:
        t.query("test")
        ev = t.response("Final answer")
    assert ev.event_type.value == "final.response"
    assert ev.payload["text"] == "Final answer"


def test_trace_custom_event():
    trace = Trace("test-model")
    with trace as t:
        from trllm.events import EventType
        ev = t.event(EventType.SYNTHESIS, {"text": "synthesized"}, source="agent")
    assert ev.event_type.value == "synthesis"


def test_trace_causal_linking():
    """Events are automatically linked to the previous event."""
    trace = Trace("test-model")
    with trace as t:
        q = t.query("test")
        c = t.chunk("doc1", "text")
    # chunk should be caused by query (auto-linked)
    assert q in c.caused_by


def test_trace_explicit_caused_by():
    """Explicit caused_by overrides auto-linking."""
    trace = Trace("test-model")
    with trace as t:
        q = t.query("test")
        c1 = t.chunk("doc1", "text1")
        c2 = t.chunk("doc2", "text2", caused_by=[q])
    # c2 should be caused by query, not c1
    assert q in c2.caused_by
    assert c1 not in c2.caused_by


@pytest.mark.asyncio
async def test_analyze_returns_result():
    """Analyze produces a TraceResult with scores and graph."""
    judge_response = [
        {"chunk_id": "doc1", "verdict": "CAUSAL", "confidence": 0.95, "evidence": "used"},
        {"chunk_id": "doc2", "verdict": "DEAD", "confidence": 0.0, "evidence": "not used"},
    ]
    tracer = _make_tracer(judge_response)

    with tracer.trace() as t:
        t.query("test question")
        t.chunk("doc1", "Relevant info")
        t.chunk("doc2", "Irrelevant info")
        t.llm_call(prompt="...", response="Answer using relevant info")
        t.response("Answer using relevant info")

    result = await tracer.analyze()

    assert isinstance(result, TraceResult)
    assert result.event_count > 0
    assert result.all_passed is True
    assert len(result.violations) == 0
    assert "nodes" in result.graph
    assert "links" in result.graph
    assert result.computation is not None


@pytest.mark.asyncio
async def test_analyze_scores_causal_and_dead():
    """Verify that CAUSAL and DEAD verdicts come through correctly."""
    judge_response = [
        {"chunk_id": "doc1", "verdict": "CAUSAL", "confidence": 1.0, "evidence": "direct match"},
        {"chunk_id": "doc2", "verdict": "DEAD", "confidence": 0.0, "evidence": "unused"},
    ]
    tracer = _make_tracer(judge_response)

    with tracer.trace() as t:
        t.query("test")
        t.chunk("doc1", "Useful text")
        t.chunk("doc2", "Useless text")
        t.llm_call(prompt="...", response="Used useful text")
        t.response("Used useful text")

    result = await tracer.analyze()

    causal = [s for s in result.scores if s["verdict"] == "CAUSAL"]
    dead = [s for s in result.scores if s["verdict"] == "DEAD"]
    assert len(causal) == 1
    assert causal[0]["chunk_id"] == "doc1"
    assert causal[0]["confidence"] == 1.0
    assert len(dead) == 1
    assert dead[0]["chunk_id"] == "doc2"


@pytest.mark.asyncio
async def test_analyze_scores_hallucination():
    """Verify HALLUCINATED_AGAINST verdict."""
    judge_response = [
        {"chunk_id": "doc1", "verdict": "HALLUCINATED_AGAINST", "confidence": 0.9, "evidence": "contradicted"},
    ]
    tracer = _make_tracer(judge_response)

    with tracer.trace() as t:
        t.query("test")
        t.chunk("doc1", "Mars has two moons")
        t.llm_call(prompt="...", response="Mars has three moons")
        t.response("Mars has three moons")

    result = await tracer.analyze()

    assert len(result.scores) == 1
    assert result.scores[0]["verdict"] == "HALLUCINATED_AGAINST"
    assert result.scores[0]["confidence"] == -0.9


@pytest.mark.asyncio
async def test_analyze_with_tool_calls():
    """Tool results are included in entailment scoring."""
    judge_response = [
        {"chunk_id": "doc1", "verdict": "CAUSAL", "confidence": 1.0, "evidence": "used"},
        {"chunk_id": "search", "verdict": "CAUSAL", "confidence": 0.8, "evidence": "tool used"},
    ]
    tracer = _make_tracer(judge_response)

    with tracer.trace() as t:
        t.query("test")
        t.chunk("doc1", "Some fact")
        t.tool_call("search", input="query", output="Tool output fact")
        t.llm_call(prompt="...", response="Combined answer")
        t.response("Combined answer")

    result = await tracer.analyze()

    assert len(result.scores) == 2
    assert any(s["verdict"] == "CAUSAL" for s in result.scores)


@pytest.mark.asyncio
async def test_analyze_no_trace_raises():
    """Calling analyze without a trace raises ValueError."""
    tracer = _make_tracer()
    with pytest.raises(ValueError, match="No trace"):
        await tracer.analyze()


@pytest.mark.asyncio
async def test_tracer_async_context_manager():
    """Tracer works as an async context manager."""
    tracer = _make_tracer()
    async with tracer:
        with tracer.trace() as t:
            t.query("test")
            t.llm_call(prompt="...", response="answer")
            t.response("answer")
        result = await tracer.analyze()
    assert result.event_count > 0


@pytest.mark.asyncio
async def test_full_agent_pipeline_trace():
    """End-to-end test matching the agent pipeline pattern."""
    # The linker assigns chunk_ids as chunk_0, chunk_1, chunk_2 for inputs without chunk_id
    judge_response = [
        {"chunk_id": "doc1", "verdict": "CAUSAL", "confidence": 1.0, "evidence": "used"},
        {"chunk_id": "doc2", "verdict": "DEAD", "confidence": 0.0, "evidence": "unused"},
        {"chunk_id": "chunk_2", "verdict": "CAUSAL", "confidence": 0.9, "evidence": "tool used"},
    ]
    tracer = _make_tracer(judge_response)

    with tracer.trace() as t:
        q = t.query("How many moons does Mars have?")

        # Planning
        plan = t.llm_call(prompt="Plan for: ...", response="Sub-questions: ...", role="planner")
        reasoning = t.reasoning("Query decomposed", step="planning")

        # Parallel branches
        tool = t.tool_call("lookup", input="mars moons", output="Mars has 2 moons", caused_by=[reasoning])
        c1 = t.chunk("doc1", "Mars has Phobos and Deimos", caused_by=[reasoning])
        c2 = t.chunk("doc2", "Jupiter has 95 moons", caused_by=[reasoning])

        # Merge
        eval_step = t.reasoning("Evaluating evidence", step="evaluation", caused_by=[tool, c1, c2])

        # Synthesis
        t.llm_call(prompt="Synthesize...", response="Mars has 2 moons: Phobos and Deimos", role="synthesizer")
        t.response("Mars has 2 moons: Phobos and Deimos")

    result = await tracer.analyze()

    assert result.event_count >= 12
    assert result.all_passed is True
    assert len(result.graph["nodes"]) >= 12
    assert len(result.graph["links"]) >= 10

    causal = [s for s in result.scores if s["verdict"] == "CAUSAL"]
    dead = [s for s in result.scores if s["verdict"] == "DEAD"]
    assert len(causal) == 2
    assert len(dead) == 1
