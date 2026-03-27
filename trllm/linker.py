"""Entailment-based causal influence scorer.

Determines whether each input (chunk, tool result, etc.) actually caused
specific claims in the LLM output by asking a judge model to trace each
claim back to its source — not just checking topical similarity.
"""

from __future__ import annotations

import json
import re

from trllm.adapters.ollama import OllamaAdapter
from trllm.events import CLEvent, EventType

ENTAILMENT_PROMPT = """You are a grounding judge. Your job is to determine which source chunks actually contributed specific information to the response.

## Response to analyze
{response}

## Source chunks
{chunks}

## Instructions
For each source chunk, determine:
1. Does the response contain ANY specific claim, fact, or detail that comes from this chunk and is NOT common/general knowledge?
2. If yes, what specific claim(s) in the response came from this chunk?

A chunk is CAUSAL if the response contains specific information that could only come from that chunk (names, dates, numbers, specific technical details, etc.).
A chunk is DEAD if the response doesn't use any specific information from it, OR only overlaps on general/common knowledge.
A chunk is HALLUCINATED_AGAINST if the response directly contradicts a specific fact in the chunk.

Respond with ONLY a JSON array, one object per chunk, in order:
[
  {{"chunk_id": "...", "verdict": "CAUSAL|DEAD|HALLUCINATED_AGAINST", "confidence": 0.0-1.0, "evidence": "brief explanation"}},
  ...
]

No other text. Just the JSON array."""


class EntailmentLinker:
    def __init__(
        self,
        ollama: OllamaAdapter,
        judge_model: str = "qwen3:30b",
    ):
        self.ollama = ollama
        self.judge_model = judge_model
        self.confidence_threshold = 0.45

    async def score_influence(
        self, inputs: list[CLEvent], output: CLEvent
    ) -> list[tuple[CLEvent, float]]:
        if not inputs:
            return []

        response_text = self._extract_text(output)

        chunks_text = ""
        chunk_ids = []
        for i, inp in enumerate(inputs):
            chunk_id = inp.payload.get("chunk_id", f"chunk_{i}")
            chunk_ids.append(chunk_id)
            text = self._extract_text(inp)
            chunks_text += f"[{chunk_id}]: {text}\n"

        prompt = ENTAILMENT_PROMPT.format(
            response=response_text,
            chunks=chunks_text.strip(),
        )

        result = await self.ollama.generate(self.judge_model, prompt)
        raw = result.get("response", "")
        verdicts = self._parse_verdicts(raw, chunk_ids)

        scored = []
        for i, inp in enumerate(inputs):
            chunk_id = chunk_ids[i]
            if chunk_id in verdicts:
                v = verdicts[chunk_id]
                confidence = v["confidence"]
                if v["verdict"] == "DEAD":
                    confidence = 0.0
                elif v["verdict"] == "HALLUCINATED_AGAINST":
                    confidence = -1.0 * v["confidence"]
                scored.append((inp, confidence))
            else:
                scored.append((inp, 0.0))

        return scored

    def _parse_verdicts(self, raw: str, expected_ids: list[str]) -> dict:
        # Extract JSON from response (may have thinking tags or markdown)
        json_match = re.search(r'\[.*\]', raw, re.DOTALL)
        if not json_match:
            return {}

        try:
            verdicts_list = json.loads(json_match.group())
        except json.JSONDecodeError:
            return {}

        verdicts = {}
        for v in verdicts_list:
            if isinstance(v, dict) and "chunk_id" in v:
                verdicts[v["chunk_id"]] = {
                    "verdict": v.get("verdict", "DEAD"),
                    "confidence": float(v.get("confidence", 0.0)),
                    "evidence": v.get("evidence", ""),
                }
        return verdicts

    def _extract_text(self, event: CLEvent) -> str:
        for key in ("output", "text", "content", "msg", "prompt", "result", "response"):
            if key in event.payload:
                val = event.payload[key]
                if isinstance(val, str):
                    return val
        return str(event.payload)
