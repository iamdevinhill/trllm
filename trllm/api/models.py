"""Pydantic request/response models for the TRLLM API."""

from __future__ import annotations

from pydantic import BaseModel


class PipelineRun(BaseModel):
    run_id: str
    events: list[dict]


class CausalQuery(BaseModel):
    run_id: str
    question_type: str  # "what_caused", "dead_nodes", "min_path", "counterfactual"
    target_event_id: str | None = None
    destination_event_id: str | None = None
    remove_event_id: str | None = None
