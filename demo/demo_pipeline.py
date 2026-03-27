"""
TRLLM Demo — A minimal RAG pipeline with entailment-based causal tracing.
Uses Ollama for the LLM (qwen3:30b), embeddings for retrieval (qwen3-embedding:0.6b),
and the same LLM as the entailment judge.

Run: python demo/demo_pipeline.py
Requires: Ollama running at localhost:11434 with qwen3:30b and qwen3-embedding:0.6b pulled.
"""

import asyncio

import httpx
import numpy as np

from trllm.adapters.ollama import OllamaAdapter
from trllm.constraints import check_constraints
from trllm.events import CLEvent, EventType
from trllm.graph import CausalGraphBuilder
from trllm.linker import EntailmentLinker
from trllm.visualization.renderer import render_summary

# Simulated document store
DOCUMENTS = [
    {"id": "doc1", "text": "Python was created by Guido van Rossum and first released in 1991."},
    {"id": "doc2", "text": "The capital of France is Paris, known for the Eiffel Tower."},
    {"id": "doc3", "text": "Machine learning is a subset of artificial intelligence."},
    {"id": "doc4", "text": "Tennessee is known as the Volunteer State."},
    {"id": "doc5", "text": "FastAPI is a modern Python web framework based on Starlette."},
]

LLM_MODEL = "qwen3:30b"
EMBED_MODEL = "qwen3-embedding:0.6b"


def cosine_similarity(a: list[float], b: list[float]) -> float:
    a_arr, b_arr = np.array(a), np.array(b)
    norm_a, norm_b = np.linalg.norm(a_arr), np.linalg.norm(b_arr)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a_arr, b_arr) / (norm_a * norm_b))


async def run_demo():
    ollama = OllamaAdapter()
    linker = EntailmentLinker(ollama, judge_model=LLM_MODEL)
    builder = CausalGraphBuilder(linker)

    events: list[CLEvent] = []

    # 1. Pipeline start
    start_event = CLEvent(
        event_type=EventType.PIPELINE_START,
        payload={"pipeline": "rag_demo"},
        source="pipeline",
    )
    events.append(start_event)

    # 2. User query
    user_query = "What year was Python created and who made it?"
    query_event = CLEvent(
        event_type=EventType.USER_QUERY,
        payload={"text": user_query},
        source="user",
        caused_by=[start_event],
    )
    events.append(query_event)

    # 3. Retrieval — embed query and find similar docs
    retrieval_event = CLEvent(
        event_type=EventType.RETRIEVAL_REQUEST,
        payload={"query": user_query},
        source="retriever",
        caused_by=[query_event],
    )
    events.append(retrieval_event)

    print("Embedding query and documents...")
    query_embedding = await ollama.embed(EMBED_MODEL, user_query)
    scored_docs = []
    for doc in DOCUMENTS:
        doc_embedding = await ollama.embed(EMBED_MODEL, doc["text"])
        sim = cosine_similarity(query_embedding, doc_embedding)
        scored_docs.append((doc, sim))

    scored_docs.sort(key=lambda x: x[1], reverse=True)
    top_chunks = scored_docs[:3]

    print(f"Top 3 chunks selected (out of {len(DOCUMENTS)}):")
    for doc, score in top_chunks:
        print(f"  {doc['id']}: similarity={score:.3f} — {doc['text'][:60]}...")

    # 4. Create chunk events
    chunk_events = []
    for doc, score in top_chunks:
        chunk_event = CLEvent(
            event_type=EventType.CHUNK_INJECTED,
            payload={"chunk_id": doc["id"], "text": doc["text"], "score": score},
            source="retriever",
            caused_by=[retrieval_event],
        )
        events.append(chunk_event)
        chunk_events.append(chunk_event)

    # 5. Assemble prompt
    context_text = "\n".join(ce.payload["text"] for ce in chunk_events)
    prompt = f"Context:\n{context_text}\n\nQuestion: {user_query}\nAnswer:"

    prompt_event = CLEvent(
        event_type=EventType.PROMPT_ASSEMBLED,
        payload={"prompt": prompt, "chunk_count": len(chunk_events)},
        source="pipeline",
        caused_by=chunk_events,
    )
    events.append(prompt_event)

    # 6. LLM call
    request_event = CLEvent(
        event_type=EventType.LLM_REQUEST,
        payload={"model": LLM_MODEL, "prompt": prompt},
        source="llm",
        caused_by=[prompt_event],
    )
    events.append(request_event)

    print(f"\nCalling {LLM_MODEL}...")
    result = await ollama.generate(LLM_MODEL, prompt)
    llm_output = result["response"]

    response_event = CLEvent(
        event_type=EventType.LLM_RESPONSE,
        payload={"output": llm_output, "model": LLM_MODEL},
        source="llm",
        caused_by=[request_event],
    )
    events.append(response_event)

    # 7. Final output
    final_event = CLEvent(
        event_type=EventType.FINAL_RESPONSE,
        payload={"text": llm_output},
        source="pipeline",
        caused_by=[response_event],
    )
    events.append(final_event)

    # 8. Pipeline end
    end_event = CLEvent(
        event_type=EventType.PIPELINE_END,
        payload={"event_count": len(events)},
        source="pipeline",
        caused_by=[final_event],
    )
    events.append(end_event)

    # 9. Push to API server (if running) so the dashboard can display it
    run_id = "demo-rag-1"
    api_url = "http://localhost:8000"
    try:
        async with httpx.AsyncClient() as http:
            resp = await http.post(
                f"{api_url}/ingest",
                json={"run_id": run_id, "events": [e.to_dict() for e in events]},
                timeout=30.0,
            )
            if resp.status_code == 200:
                print(f"\nIngested run '{run_id}' to API — view at {api_url}/dashboard")
            else:
                print(f"\nAPI ingest returned {resp.status_code} — dashboard won't have this run")
    except httpx.ConnectError:
        print(f"\nAPI server not running at {api_url} — skipping dashboard ingest")

    # 10. Build causal graph (includes entailment-based influence scoring)
    print("Building causal graph (running entailment judge)...")
    computation = await builder.build(events)

    # 11. Print results
    print("\n" + "=" * 60)
    print("CAUSAL GRAPH SUMMARY")
    print("=" * 60)
    print(render_summary(computation))

    print("\n" + "=" * 60)
    print("LLM OUTPUT")
    print("=" * 60)
    print(llm_output)

    print("\n" + "=" * 60)
    print("ENTAILMENT-BASED CAUSAL SCORES")
    print("=" * 60)
    scored = await linker.score_influence(chunk_events, response_event)
    for chunk_event, confidence in sorted(scored, key=lambda x: x[1], reverse=True):
        if confidence < 0:
            status = "HALLUCINATED_AGAINST"
        elif confidence > 0:
            status = "CAUSAL"
        else:
            status = "DEAD"
        print(
            f"  [{status}] {chunk_event.payload['chunk_id']}: "
            f"confidence={confidence:+.2f} | "
            f"text={chunk_event.payload['text'][:60]}..."
        )

    print("\n" + "=" * 60)
    print("CONSTRAINT VIOLATIONS")
    print("=" * 60)
    violations = check_constraints(computation)
    if violations:
        for v in violations:
            print(f"  VIOLATION: {v}")
    else:
        print("  All constraints passed.")

    await ollama.close()


if __name__ == "__main__":
    asyncio.run(run_demo())
