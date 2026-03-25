"""Tests for SemanticLinker with mocked Ollama embeddings."""

import pytest

from trllm.events import CLEvent, EventType
from trllm.linker import SemanticLinker


class MockOllamaAdapter:
    """Returns fixed embeddings for testing."""

    def __init__(self, embeddings: dict[str, list[float]] | None = None):
        self._embeddings = embeddings or {}
        self._default_dim = 4

    async def embed(self, model: str, text: str) -> list[float]:
        if text in self._embeddings:
            return self._embeddings[text]
        # Return a deterministic embedding based on text hash
        h = hash(text) % 1000
        return [h / 1000, (h * 7 % 1000) / 1000, (h * 13 % 1000) / 1000, (h * 31 % 1000) / 1000]

    async def close(self):
        pass


@pytest.fixture
def similar_embeddings():
    """Embeddings where doc1 is very similar to output, doc2 is not."""
    return MockOllamaAdapter(embeddings={
        "Python was created by Guido van Rossum": [0.9, 0.1, 0.1, 0.0],
        "The capital of France is Paris": [0.0, 0.1, 0.9, 0.1],
        "Python was made by Guido in 1991": [0.85, 0.15, 0.1, 0.0],
    })


@pytest.fixture
def linker(similar_embeddings):
    return SemanticLinker(similar_embeddings, model="test-model", similarity_threshold=0.45)


@pytest.mark.asyncio
async def test_score_influence_returns_scores(linker):
    chunk1 = CLEvent(
        event_type=EventType.CHUNK_INJECTED,
        payload={"text": "Python was created by Guido van Rossum"},
        source="retriever",
    )
    chunk2 = CLEvent(
        event_type=EventType.CHUNK_INJECTED,
        payload={"text": "The capital of France is Paris"},
        source="retriever",
    )
    output = CLEvent(
        event_type=EventType.LLM_RESPONSE,
        payload={"output": "Python was made by Guido in 1991"},
        source="llm",
    )

    scored = await linker.score_influence([chunk1, chunk2], output)
    assert len(scored) == 2
    # chunk1 should score higher than chunk2
    scores = {ev.payload["text"][:10]: conf for ev, conf in scored}
    assert scores["Python was"] > scores["The capita"]


@pytest.mark.asyncio
async def test_score_influence_empty_inputs(linker):
    output = CLEvent(
        event_type=EventType.LLM_RESPONSE,
        payload={"output": "test"},
        source="llm",
    )
    scored = await linker.score_influence([], output)
    assert scored == []


@pytest.mark.asyncio
async def test_type_weighting():
    # Use embeddings with ~0.7 cosine similarity to output so weight multipliers differentiate
    mock = MockOllamaAdapter(embeddings={
        "tool output": [0.5, 0.5, 0.0, 0.0],
        "chunk text": [0.5, 0.5, 0.0, 0.0],
        "result text": [0.5, 0.0, 0.5, 0.0],
    })
    linker = SemanticLinker(mock, model="test", similarity_threshold=0.0)

    tool_event = CLEvent(
        event_type=EventType.TOOL_RESULT,
        payload={"output": "tool output"},
        source="tool",
    )
    chunk_event = CLEvent(
        event_type=EventType.CHUNK_INJECTED,
        payload={"text": "chunk text"},
        source="retriever",
    )
    output = CLEvent(
        event_type=EventType.LLM_RESPONSE,
        payload={"output": "result text"},
        source="llm",
    )

    scored = await linker.score_influence([tool_event, chunk_event], output)
    tool_score = next(s for e, s in scored if e.event_type == EventType.TOOL_RESULT)
    chunk_score = next(s for e, s in scored if e.event_type == EventType.CHUNK_INJECTED)
    # Tool result has 1.3x weight, chunk has 1.1x — same base similarity, different weighted score
    assert tool_score > chunk_score


def test_extract_text_priority():
    linker = SemanticLinker(MockOllamaAdapter(), model="test")

    # "output" key takes priority
    ev = CLEvent(event_type=EventType.LLM_RESPONSE, payload={"output": "out", "text": "txt"}, source="llm")
    assert linker._extract_text(ev) == "out"

    # Falls through to "text"
    ev2 = CLEvent(event_type=EventType.USER_QUERY, payload={"text": "query"}, source="user")
    assert linker._extract_text(ev2) == "query"

    # Falls back to str(payload)
    ev3 = CLEvent(event_type=EventType.PIPELINE_START, payload={"foo": 123}, source="pipeline")
    assert "foo" in linker._extract_text(ev3)


def test_cosine_similarity():
    linker = SemanticLinker(MockOllamaAdapter(), model="test")
    assert linker._cosine_similarity([1, 0, 0], [1, 0, 0]) == pytest.approx(1.0)
    assert linker._cosine_similarity([1, 0, 0], [0, 1, 0]) == pytest.approx(0.0)
    assert linker._cosine_similarity([1, 0, 0], [-1, 0, 0]) == pytest.approx(-1.0)
    assert linker._cosine_similarity([0, 0, 0], [1, 0, 0]) == 0.0
