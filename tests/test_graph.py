"""Tests for CausalGraphBuilder."""

import pytest

from trllm.events import CLEvent, EventType
from trllm.graph import CausalGraphBuilder
from trllm.linker import SemanticLinker


class MockOllamaAdapter:
    async def embed(self, model: str, text: str) -> list[float]:
        h = hash(text) % 1000
        return [h / 1000, (h * 7 % 1000) / 1000, (h * 13 % 1000) / 1000, (h * 31 % 1000) / 1000]

    async def close(self):
        pass


@pytest.fixture
def builder():
    mock = MockOllamaAdapter()
    linker = SemanticLinker(mock, model="test", similarity_threshold=0.45)
    return CausalGraphBuilder(linker)


@pytest.mark.asyncio
async def test_build_creates_computation(builder):
    query = CLEvent(event_type=EventType.USER_QUERY, payload={"text": "hello"}, source="user")
    request = CLEvent(
        event_type=EventType.LLM_REQUEST,
        payload={"prompt": "hello"},
        source="llm",
        caused_by=[query],
    )
    response = CLEvent(
        event_type=EventType.LLM_RESPONSE,
        payload={"output": "world"},
        source="llm",
        caused_by=[request],
    )

    comp = await builder.build([query, request, response])
    events = list(comp.events)
    assert len(events) == 3


@pytest.mark.asyncio
async def test_explicit_causal_links_preserved(builder):
    cause = CLEvent(event_type=EventType.USER_QUERY, payload={"text": "q"}, source="user")
    effect = CLEvent(
        event_type=EventType.RETRIEVAL_REQUEST,
        payload={"query": "q"},
        source="retriever",
        caused_by=[cause],
    )

    comp = await builder.build([cause, effect])
    events = list(comp.events)
    cause_ev = next(e for e in events if e.name == "user.query")
    effect_ev = next(e for e in events if e.name == "retrieval.request")
    assert comp.is_ancestor(cause_ev, effect_ev)


@pytest.mark.asyncio
async def test_full_pipeline_graph(builder):
    """Build a minimal pipeline and verify the causal structure."""
    query = CLEvent(event_type=EventType.USER_QUERY, payload={"text": "What is Python?"}, source="user")
    retrieval = CLEvent(
        event_type=EventType.RETRIEVAL_REQUEST,
        payload={"query": "What is Python?"},
        source="retriever",
        caused_by=[query],
    )
    chunk = CLEvent(
        event_type=EventType.CHUNK_INJECTED,
        payload={"text": "Python is a programming language"},
        source="retriever",
        caused_by=[retrieval],
    )
    request = CLEvent(
        event_type=EventType.LLM_REQUEST,
        payload={"prompt": "Context: Python is a programming language\nQ: What is Python?"},
        source="llm",
        caused_by=[chunk],
    )
    response = CLEvent(
        event_type=EventType.LLM_RESPONSE,
        payload={"output": "Python is a programming language created by Guido"},
        source="llm",
        caused_by=[request],
    )
    final = CLEvent(
        event_type=EventType.FINAL_RESPONSE,
        payload={"text": "Python is a programming language created by Guido"},
        source="pipeline",
        caused_by=[response],
    )

    comp = await builder.build([query, retrieval, chunk, request, response, final])
    events = list(comp.events)
    assert len(events) == 6

    # Verify query is ancestor of final
    query_ev = next(e for e in events if e.name == "user.query")
    final_ev = next(e for e in events if e.name == "final.response")
    assert comp.is_ancestor(query_ev, final_ev)

    # Verify root and leaf
    roots = list(comp.root_events())
    leaves = list(comp.leaf_events())
    assert any(e.name == "user.query" for e in roots)
    assert any(e.name == "final.response" for e in leaves)
