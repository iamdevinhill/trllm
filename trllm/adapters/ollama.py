"""Async Ollama HTTP adapter for generate and embed endpoints."""

from __future__ import annotations

import os

import httpx


class OllamaAdapter:
    def __init__(self, ollama_base_url: str | None = None):
        ollama_base_url = ollama_base_url or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        self.base_url = ollama_base_url
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0))

    async def generate(self, model: str, prompt: str) -> dict:
        response = await self.client.post(
            f"{self.base_url}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
        )
        response.raise_for_status()
        return response.json()

    async def embed(self, model: str, text: str) -> list[float]:
        response = await self.client.post(
            f"{self.base_url}/api/embed",
            json={"model": model, "input": text},
            timeout=30.0,
        )
        response.raise_for_status()
        return response.json()["embeddings"][0]

    async def close(self):
        await self.client.aclose()
