"""Tests for FastAPI endpoints using mocked Ollama."""

import pytest
from fastapi.testclient import TestClient

from trllm.api import server
from trllm.api.server import app, computations, run_events
from trllm.events import CLEvent, EventType
from trllm.graph import CausalGraphBuilder
from trllm.linker import EntailmentLinker

import json


class MockOllamaAdapter:
    async def generate(self, model: str, prompt: str) -> dict:
        return {"response": json.dumps([])}

    async def close(self):
        pass


@pytest.fixture(autouse=True)
def mock_builder(monkeypatch):
    mock_ollama = MockOllamaAdapter()
    mock_linker = EntailmentLinker(mock_ollama, judge_model="test")
    mock_builder = CausalGraphBuilder(mock_linker)
    monkeypatch.setattr(server, "builder", mock_builder)
    computations.clear()
    run_events.clear()
    yield
    computations.clear()
    run_events.clear()


@pytest.fixture
def client():
    return TestClient(app)


def _make_pipeline_events() -> list[dict]:
    query = CLEvent(event_type=EventType.USER_QUERY, payload={"text": "What is Python?"}, source="user")
    chunk = CLEvent(
        event_type=EventType.CHUNK_INJECTED,
        payload={"text": "Python is a language"},
        source="retriever",
        caused_by=[query],
    )
    request = CLEvent(
        event_type=EventType.LLM_REQUEST,
        payload={"prompt": "Python is a language\nQ: What is Python?"},
        source="llm",
        caused_by=[chunk],
    )
    response = CLEvent(
        event_type=EventType.LLM_RESPONSE,
        payload={"output": "Python is a programming language"},
        source="llm",
        caused_by=[request],
    )
    final = CLEvent(
        event_type=EventType.FINAL_RESPONSE,
        payload={"text": "Python is a programming language"},
        source="pipeline",
        caused_by=[response],
    )
    return [e.to_dict() for e in [query, chunk, request, response, final]]


def test_ingest(client):
    events = _make_pipeline_events()
    resp = client.post("/ingest", json={"run_id": "test-1", "events": events})
    assert resp.status_code == 200
    data = resp.json()
    assert data["run_id"] == "test-1"
    assert data["event_count"] == 5


def test_ingest_and_query_what_caused(client):
    events = _make_pipeline_events()
    client.post("/ingest", json={"run_id": "test-1", "events": events})

    final_id = events[-1]["id"]
    resp = client.post("/query", json={
        "run_id": "test-1",
        "question_type": "what_caused",
        "target_event_id": final_id,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["causes"]) > 0


def test_ingest_and_query_dead_nodes(client):
    events = _make_pipeline_events()
    client.post("/ingest", json={"run_id": "test-1", "events": events})

    resp = client.post("/query", json={
        "run_id": "test-1",
        "question_type": "dead_nodes",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data["dead_events"], list)


def test_query_nonexistent_run(client):
    resp = client.post("/query", json={
        "run_id": "nonexistent",
        "question_type": "dead_nodes",
    })
    assert resp.status_code == 404


def test_visualization_endpoint(client):
    events = _make_pipeline_events()
    client.post("/ingest", json={"run_id": "test-1", "events": events})

    resp = client.get("/runs/test-1/visualization?format=mermaid")
    assert resp.status_code == 200
    data = resp.json()
    assert "mermaid" in data


def test_visualization_summary(client):
    events = _make_pipeline_events()
    client.post("/ingest", json={"run_id": "test-1", "events": events})

    resp = client.get("/runs/test-1/visualization?format=summary")
    assert resp.status_code == 200
    data = resp.json()
    assert "summary" in data


def test_constraints_endpoint(client):
    events = _make_pipeline_events()
    client.post("/ingest", json={"run_id": "test-1", "events": events})

    resp = client.get("/runs/test-1/constraints")
    assert resp.status_code == 200
    data = resp.json()
    assert "violations" in data
    assert "all_passed" in data


def test_visualization_not_found(client):
    resp = client.get("/runs/nonexistent/visualization")
    assert resp.status_code == 404


def test_query_counterfactual(client):
    events = _make_pipeline_events()
    client.post("/ingest", json={"run_id": "test-1", "events": events})

    # Remove the chunk event (index 1)
    chunk_id = events[1]["id"]
    resp = client.post("/query", json={
        "run_id": "test-1",
        "question_type": "counterfactual",
        "remove_event_id": chunk_id,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "affected_events" in data
    assert "output_affected" in data
