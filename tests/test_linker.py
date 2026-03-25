"""Tests for EntailmentLinker with mocked Ollama generate calls."""

import json

import pytest

from trllm.events import CLEvent, EventType
from trllm.linker import EntailmentLinker


class MockOllamaAdapter:
    """Returns predefined judge responses for testing."""

    def __init__(self, judge_response: str = "[]"):
        self._judge_response = judge_response

    async def generate(self, model: str, prompt: str) -> dict:
        return {"response": self._judge_response}

    async def embed(self, model: str, text: str) -> list[float]:
        return [0.0] * 4

    async def close(self):
        pass


def _make_judge_response(verdicts: list[dict]) -> str:
    return json.dumps(verdicts)


@pytest.mark.asyncio
async def test_causal_chunk_scores_high():
    response = _make_judge_response([
        {"chunk_id": "doc1", "verdict": "CAUSAL", "confidence": 0.95, "evidence": "response cites 1991 and Guido from doc1"},
        {"chunk_id": "doc2", "verdict": "DEAD", "confidence": 0.1, "evidence": "no information from doc2 used"},
    ])
    mock = MockOllamaAdapter(judge_response=response)
    linker = EntailmentLinker(mock, judge_model="test")

    chunk1 = CLEvent(
        event_type=EventType.CHUNK_INJECTED,
        payload={"chunk_id": "doc1", "text": "Python was created by Guido van Rossum in 1991."},
        source="retriever",
    )
    chunk2 = CLEvent(
        event_type=EventType.CHUNK_INJECTED,
        payload={"chunk_id": "doc2", "text": "The capital of France is Paris."},
        source="retriever",
    )
    output = CLEvent(
        event_type=EventType.LLM_RESPONSE,
        payload={"output": "Python was created in 1991 by Guido van Rossum."},
        source="llm",
    )

    scored = await linker.score_influence([chunk1, chunk2], output)
    assert len(scored) == 2
    doc1_score = next(s for e, s in scored if e.payload["chunk_id"] == "doc1")
    doc2_score = next(s for e, s in scored if e.payload["chunk_id"] == "doc2")
    assert doc1_score == 0.95
    assert doc2_score == 0.0  # DEAD verdicts get score 0.0


@pytest.mark.asyncio
async def test_hallucination_scores_negative():
    response = _make_judge_response([
        {"chunk_id": "doc1", "verdict": "HALLUCINATED_AGAINST", "confidence": 0.9, "evidence": "response says Linus Torvalds but chunk says Guido"},
    ])
    mock = MockOllamaAdapter(judge_response=response)
    linker = EntailmentLinker(mock, judge_model="test")

    chunk1 = CLEvent(
        event_type=EventType.CHUNK_INJECTED,
        payload={"chunk_id": "doc1", "text": "Python was created by Guido van Rossum."},
        source="retriever",
    )
    output = CLEvent(
        event_type=EventType.LLM_RESPONSE,
        payload={"output": "Python was created by Linus Torvalds."},
        source="llm",
    )

    scored = await linker.score_influence([chunk1], output)
    assert len(scored) == 1
    assert scored[0][1] == pytest.approx(-0.9)


@pytest.mark.asyncio
async def test_empty_inputs():
    mock = MockOllamaAdapter()
    linker = EntailmentLinker(mock, judge_model="test")

    output = CLEvent(
        event_type=EventType.LLM_RESPONSE,
        payload={"output": "test"},
        source="llm",
    )
    scored = await linker.score_influence([], output)
    assert scored == []


@pytest.mark.asyncio
async def test_malformed_judge_response_returns_zeros():
    mock = MockOllamaAdapter(judge_response="this is not json at all")
    linker = EntailmentLinker(mock, judge_model="test")

    chunk = CLEvent(
        event_type=EventType.CHUNK_INJECTED,
        payload={"chunk_id": "doc1", "text": "some text"},
        source="retriever",
    )
    output = CLEvent(
        event_type=EventType.LLM_RESPONSE,
        payload={"output": "some output"},
        source="llm",
    )

    scored = await linker.score_influence([chunk], output)
    assert len(scored) == 1
    assert scored[0][1] == 0.0


@pytest.mark.asyncio
async def test_judge_response_with_thinking_tags():
    """Judge models sometimes wrap output in thinking tags or markdown."""
    raw = '<think>Let me analyze...</think>\n```json\n' + _make_judge_response([
        {"chunk_id": "doc1", "verdict": "CAUSAL", "confidence": 0.8, "evidence": "used fact X"},
    ]) + '\n```'
    mock = MockOllamaAdapter(judge_response=raw)
    linker = EntailmentLinker(mock, judge_model="test")

    chunk = CLEvent(
        event_type=EventType.CHUNK_INJECTED,
        payload={"chunk_id": "doc1", "text": "some text"},
        source="retriever",
    )
    output = CLEvent(
        event_type=EventType.LLM_RESPONSE,
        payload={"output": "some output"},
        source="llm",
    )

    scored = await linker.score_influence([chunk], output)
    assert len(scored) == 1
    assert scored[0][1] == 0.8


def test_extract_text_priority():
    mock = MockOllamaAdapter()
    linker = EntailmentLinker(mock, judge_model="test")

    # "output" key takes priority
    ev = CLEvent(event_type=EventType.LLM_RESPONSE, payload={"output": "out", "text": "txt"}, source="llm")
    assert linker._extract_text(ev) == "out"

    # Falls through to "text"
    ev2 = CLEvent(event_type=EventType.USER_QUERY, payload={"text": "query"}, source="user")
    assert linker._extract_text(ev2) == "query"

    # Falls back to str(payload)
    ev3 = CLEvent(event_type=EventType.PIPELINE_START, payload={"foo": 123}, source="pipeline")
    assert "foo" in linker._extract_text(ev3)


@pytest.mark.asyncio
async def test_all_dead_chunks():
    response = _make_judge_response([
        {"chunk_id": "doc1", "verdict": "DEAD", "confidence": 0.1, "evidence": "not used"},
        {"chunk_id": "doc2", "verdict": "DEAD", "confidence": 0.05, "evidence": "not used"},
    ])
    mock = MockOllamaAdapter(judge_response=response)
    linker = EntailmentLinker(mock, judge_model="test")

    chunks = [
        CLEvent(event_type=EventType.CHUNK_INJECTED, payload={"chunk_id": "doc1", "text": "irrelevant"}, source="r"),
        CLEvent(event_type=EventType.CHUNK_INJECTED, payload={"chunk_id": "doc2", "text": "also irrelevant"}, source="r"),
    ]
    output = CLEvent(event_type=EventType.LLM_RESPONSE, payload={"output": "I don't know"}, source="llm")

    scored = await linker.score_influence(chunks, output)
    assert all(s == 0.0 for _, s in scored)
