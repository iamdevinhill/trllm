"""Tests for event serialization/deserialization."""

from trllm.events import CLEvent, EventType


def test_event_creation():
    ev = CLEvent(
        event_type=EventType.USER_QUERY,
        payload={"text": "hello"},
        source="user",
    )
    assert ev.event_type == EventType.USER_QUERY
    assert ev.payload == {"text": "hello"}
    assert ev.source == "user"
    assert ev.id  # auto-generated
    assert ev.timestamp > 0
    assert ev.caused_by == []
    assert ev.causal_confidence == 1.0


def test_event_to_dict():
    ev = CLEvent(
        event_type=EventType.LLM_REQUEST,
        payload={"model": "test", "prompt": "hi"},
        source="llm",
    )
    d = ev.to_dict()
    assert d["event_type"] == "llm.request"
    assert d["payload"]["model"] == "test"
    assert d["source"] == "llm"
    assert d["caused_by"] == []
    assert d["id"] == ev.id


def test_event_to_dict_with_causes():
    cause = CLEvent(event_type=EventType.USER_QUERY, payload={"text": "q"}, source="user")
    effect = CLEvent(
        event_type=EventType.RETRIEVAL_REQUEST,
        payload={"query": "q"},
        source="retriever",
        caused_by=[cause],
    )
    d = effect.to_dict()
    assert d["caused_by"] == [cause.id]


def test_event_from_dict():
    original = CLEvent(
        event_type=EventType.CHUNK_INJECTED,
        payload={"chunk_id": "doc1", "text": "some text"},
        source="retriever",
    )
    d = original.to_dict()
    restored = CLEvent.from_dict(d)
    assert restored.id == original.id
    assert restored.event_type == EventType.CHUNK_INJECTED
    assert restored.payload == original.payload
    assert restored.source == "retriever"


def test_event_from_dict_with_causal_lookup():
    cause = CLEvent(event_type=EventType.USER_QUERY, payload={"text": "q"}, source="user")
    effect = CLEvent(
        event_type=EventType.RETRIEVAL_REQUEST,
        payload={"query": "q"},
        source="retriever",
        caused_by=[cause],
    )

    lookup = {cause.id: cause}
    d = effect.to_dict()
    restored = CLEvent.from_dict(d, event_lookup=lookup)
    assert len(restored.caused_by) == 1
    assert restored.caused_by[0].id == cause.id


def test_all_event_types_have_values():
    for et in EventType:
        assert isinstance(et.value, str)
        assert "." in et.value or et.value == "synthesis"


def test_event_roundtrip_preserves_metadata():
    ev = CLEvent(
        event_type=EventType.TOOL_RESULT,
        payload={"result": "42"},
        source="tool",
        metadata={"custom_key": "custom_value"},
    )
    d = ev.to_dict()
    restored = CLEvent.from_dict(d)
    assert restored.metadata == {"custom_key": "custom_value"}
