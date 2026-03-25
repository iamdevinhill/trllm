"""Semantic influence scorer — determines causal influence via embedding similarity."""

from __future__ import annotations

import numpy as np

from trllm.adapters.ollama import OllamaAdapter
from trllm.events import CLEvent, EventType


class SemanticLinker:
    def __init__(
        self,
        ollama: OllamaAdapter,
        model: str = "qwen3-embedding:0.6b",
        similarity_threshold: float = 0.45,
    ):
        self.ollama = ollama
        self.model = model
        self.similarity_threshold = similarity_threshold

    async def score_influence(
        self, inputs: list[CLEvent], output: CLEvent
    ) -> list[tuple[CLEvent, float]]:
        if not inputs:
            return []

        output_text = self._extract_text(output)

        # Batch all texts for embedding
        all_texts = [self._extract_text(inp) for inp in inputs] + [output_text]
        embeddings = await self._batch_embed(all_texts)

        output_emb = np.array(embeddings[-1])
        output_norm = np.linalg.norm(output_emb)
        if output_norm == 0:
            return [(inp, 0.0) for inp in inputs]

        scored = []
        for i, inp in enumerate(inputs):
            inp_emb = np.array(embeddings[i])
            inp_norm = np.linalg.norm(inp_emb)
            if inp_norm == 0:
                scored.append((inp, 0.0))
                continue

            similarity = float(np.dot(inp_emb, output_emb) / (inp_norm * output_norm))
            weight = self._type_weight(inp.event_type)
            confidence = min(similarity * weight, 1.0)
            scored.append((inp, confidence))

        return scored

    async def _batch_embed(self, texts: list[str]) -> list[list[float]]:
        results = []
        for text in texts:
            emb = await self.ollama.embed(self.model, text)
            results.append(emb)
        return results

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        a_arr, b_arr = np.array(a), np.array(b)
        norm_a, norm_b = np.linalg.norm(a_arr), np.linalg.norm(b_arr)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a_arr, b_arr) / (norm_a * norm_b))

    def _type_weight(self, event_type: EventType) -> float:
        weights = {
            EventType.TOOL_RESULT: 1.3,
            EventType.CHUNK_INJECTED: 1.1,
            EventType.RETRIEVAL_RESULT: 1.0,
            EventType.REASONING_STEP: 1.2,
            EventType.AGENT_RESPONSE: 1.1,
            EventType.USER_QUERY: 0.9,
        }
        return weights.get(event_type, 1.0)

    def _extract_text(self, event: CLEvent) -> str:
        for key in ("output", "text", "content", "msg", "prompt", "result", "response"):
            if key in event.payload:
                val = event.payload[key]
                if isinstance(val, str):
                    return val
        return str(event.payload)
