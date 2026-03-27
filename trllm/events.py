"""TRLLM event schema and types."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
import uuid


class EventType(Enum):
    # Pipeline lifecycle
    PIPELINE_START = "pipeline.start"
    PIPELINE_END = "pipeline.end"

    # User interaction
    USER_QUERY = "user.query"
    FINAL_RESPONSE = "final.response"

    # Retrieval
    RETRIEVAL_REQUEST = "retrieval.request"
    RETRIEVAL_RESULT = "retrieval.result"
    CHUNK_SELECTED = "chunk.selected"
    CHUNK_INJECTED = "chunk.injected"

    # LLM calls
    LLM_REQUEST = "llm.request"
    LLM_RESPONSE = "llm.response"
    PROMPT_ASSEMBLED = "prompt.assembled"

    # Tool usage
    TOOL_CALL = "tool.call"
    TOOL_RESULT = "tool.result"

    # Agent delegation
    AGENT_DELEGATE = "agent.delegate"
    AGENT_RESPONSE = "agent.response"

    # Reasoning
    REASONING_STEP = "reasoning.step"
    SYNTHESIS = "synthesis"


@dataclass
class CLEvent:
    event_type: EventType
    payload: dict[str, Any]
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=lambda: datetime.now(timezone.utc).timestamp())
    source: str = ""
    caused_by: list[CLEvent] = field(default_factory=list)
    causal_confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "event_type": self.event_type.value,
            "payload": self.payload,
            "timestamp": self.timestamp,
            "source": self.source,
            "caused_by": [e.id for e in self.caused_by],
            "causal_confidence": self.causal_confidence,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], event_lookup: dict[str, CLEvent] | None = None) -> CLEvent:
        event_lookup = event_lookup or {}
        caused_by = [
            event_lookup[eid]
            for eid in data.get("caused_by", [])
            if eid in event_lookup
        ]
        return cls(
            id=data["id"],
            event_type=EventType(data["event_type"]),
            payload=data.get("payload", {}),
            timestamp=data.get("timestamp", datetime.now(timezone.utc).timestamp()),
            source=data.get("source", ""),
            caused_by=caused_by,
            causal_confidence=data.get("causal_confidence", 1.0),
            metadata=data.get("metadata", {}),
        )
