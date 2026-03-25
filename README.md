# TRLLM

Causal tracing for LLM and agent pipelines.

TRLLM reconstructs **causal relationships** in LLM pipeline executions. Unlike tracing tools that capture chronological spans, TRLLM answers: *which inputs actually caused which outputs?*

Built on [PyRapide](https://pypi.org/project/pyrapide/) — a causal event-driven architecture library based on Stanford's RAPIDE 1.0 specification.

## The Problem

When an LLM agent pipeline runs (retrieval → reasoning → tool calls → synthesis), existing observability tools show you a timeline. They cannot tell you:

- Which retrieval chunk actually influenced the final output
- Whether a tool call was causally relevant or dead weight
- What the minimum causal path from query to answer looks like
- If you removed a data source, which outputs would lose their grounding

TRLLM fills this gap by modeling pipeline executions as **causal posets** — directed acyclic graphs where edges represent causal influence, not just temporal sequence.

## How It Works

1. **Instrument** your LLM pipeline to emit `CLEvent` objects at each step (query, retrieval, chunk injection, LLM call, tool use, final response)
2. **Build** a causal graph — explicit causal links from your pipeline + inferred links from the **Entailment Linker**, which uses an LLM judge to verify whether specific claims in the output actually came from each input
3. **Query** the graph — trace causes, find dead nodes, compute shortest causal paths, run counterfactual analysis
4. **Enforce constraints** — declarative rules like "every output must be grounded in at least one retrieved chunk"

## Quick Start

### Requirements

- Python 3.11+
- [Ollama](https://ollama.ai) running locally (for the demo and semantic linking)

### Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### Pull Ollama Models

```bash
ollama pull qwen3:30b          # LLM for demo pipeline
ollama pull qwen3-embedding:0.6b   # Embeddings for semantic linker
```

### Run the Demo

```bash
python demo/demo_pipeline.py
```

This runs a RAG pipeline against a small document store, builds the causal graph, and prints which chunks actually influenced the output:

```
Embedding query and documents...
Top 3 chunks selected (out of 5):
  doc1: similarity=0.811 — Python was created by Guido van Rossum and first released in...
  doc5: similarity=0.464 — FastAPI is a modern Python web framework based on Starlette....
  doc3: similarity=0.305 — Machine learning is a subset of artificial intelligence....

Calling qwen3:30b...

Building causal graph (running entailment judge)...

============================================================
CAUSAL GRAPH SUMMARY
============================================================
Computation: 11 events, 12 causal edges, depth 8, 1 root events, 1 leaf events

============================================================
ENTAILMENT-BASED CAUSAL SCORES
============================================================
  [CAUSAL] doc1: confidence=+1.00 | text=Python was created by Guido van Rossum and first released in...
  [DEAD] doc5: confidence=+0.00 | text=FastAPI is a modern Python web framework based on Starlette....
  [DEAD] doc3: confidence=+0.00 | text=Machine learning is a subset of artificial intelligence....

============================================================
CONSTRAINT VIOLATIONS
============================================================
  All constraints passed.
```

**Reading the output:**
- **CAUSAL** — the LLM judge determined that specific claims in the output came from this input (e.g. the date "1991" and name "Guido" trace back to doc1).
- **DEAD** — this input was present in the prompt but no specific information from it appears in the output. Dead weight.
- **HALLUCINATED_AGAINST** — the output directly contradicts a fact in this input (negative confidence score).
- **Confidence scores** reflect the judge's assessment of how strongly the input contributed to the output.

### Run Tests

```bash
pytest tests/ -v
```

Tests use mocked Ollama calls — no running Ollama instance required.

### Start the API Server

```bash
uvicorn trllm.api.server:app --reload
```

Then open `http://localhost:8000/docs` for the interactive API docs, or `http://localhost:8000/dashboard` for the DAG viewer.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/ingest` | Submit a pipeline run (events) for causal analysis |
| `POST` | `/query` | Query the causal graph (`what_caused`, `dead_nodes`, `min_path`, `counterfactual`) |
| `GET` | `/runs/{id}/visualization` | Get Mermaid, ASCII, or DOT visualization |
| `GET` | `/runs/{id}/constraints` | Check constraint violations |
| `GET` | `/dashboard` | DAG viewer UI |

## Project Structure

```
trllm/
├── trllm/
│   ├── events.py          # CLEvent schema, 16 EventType values
│   ├── graph.py           # CausalGraphBuilder → PyRapide Computation
│   ├── linker.py          # EntailmentLinker (LLM judge-based causal verification)
│   ├── constraints.py     # Pipeline constraints via PyRapide patterns
│   ├── adapters/
│   │   └── ollama.py      # Async Ollama HTTP adapter
│   ├── api/
│   │   ├── models.py      # Pydantic request/response models
│   │   └── server.py      # FastAPI endpoints
│   └── visualization/
│       └── renderer.py    # PyRapide visualization wrappers
├── demo/
│   └── demo_pipeline.py   # End-to-end RAG demo
├── tests/                 # 26 tests (all mocked, no Ollama needed)
└── dashboard/
    └── index.html         # Mermaid.js DAG viewer
```

## Instrumenting Your Own Pipeline

Create `CLEvent` objects at each step, linking them with `caused_by`:

```python
import asyncio
from trllm.adapters.ollama import OllamaAdapter
from trllm.events import CLEvent, EventType
from trllm.graph import CausalGraphBuilder
from trllm.linker import EntailmentLinker
from trllm.constraints import check_constraints
from trllm.visualization.renderer import render_summary

async def my_pipeline():
    ollama = OllamaAdapter()
    linker = EntailmentLinker(ollama)
    builder = CausalGraphBuilder(linker)

    events = []

    # 1. Emit events at each pipeline step, linking causes
    query = CLEvent(
        event_type=EventType.USER_QUERY,
        payload={"text": "What is Python?"},
        source="user",
    )
    events.append(query)

    chunk = CLEvent(
        event_type=EventType.CHUNK_INJECTED,
        payload={"text": "Python is a programming language created in 1991."},
        source="retriever",
        caused_by=[query],  # explicit causal link
    )
    events.append(chunk)

    # ... add LLM_REQUEST, LLM_RESPONSE, FINAL_RESPONSE, etc.

    # 2. Build the causal graph (adds semantic-inferred links automatically)
    computation = await builder.build(events)

    # 3. Inspect results
    print(render_summary(computation))

    # 4. Check constraints
    violations = check_constraints(computation)
    for v in violations:
        print(f"VIOLATION: {v}")

    await ollama.close()

asyncio.run(my_pipeline())
```

Each event's `caused_by` list defines **explicit** causal links (what you know from your pipeline logic). The Entailment Linker then adds **inferred** causal links by asking an LLM judge to trace specific claims in the output back to their source inputs.

## Event Types

TRLLM supports 16 event types covering the full lifecycle of RAG and agent pipelines:

| Category | Event Type | Description |
|----------|-----------|-------------|
| Pipeline | `PIPELINE_START`, `PIPELINE_END` | Pipeline lifecycle boundaries |
| User | `USER_QUERY`, `FINAL_RESPONSE` | Input from user, final output to user |
| Retrieval | `RETRIEVAL_REQUEST`, `RETRIEVAL_RESULT` | Vector search request and raw results |
| Retrieval | `CHUNK_SELECTED`, `CHUNK_INJECTED` | Chunk picked from results, chunk added to prompt |
| LLM | `LLM_REQUEST`, `LLM_RESPONSE`, `PROMPT_ASSEMBLED` | Full LLM call lifecycle |
| Tools | `TOOL_CALL`, `TOOL_RESULT` | Tool invocation and its output |
| Agents | `AGENT_DELEGATE`, `AGENT_RESPONSE` | Sub-agent delegation and response |
| Reasoning | `REASONING_STEP`, `SYNTHESIS` | Chain-of-thought steps, final synthesis |

## Key Concepts

**Entailment Linker** — The core differentiator. Instead of cosine similarity (which only measures topical overlap — a hallucinated response about Python scores just as high as a grounded one), the linker uses an LLM judge to trace each specific claim in the response back to its source chunk. This catches:
- **True grounding** — "response says 1991, chunk says 1991" → CAUSAL
- **Dead weight** — chunk about FastAPI, response doesn't use it → DEAD
- **Hallucination** — "response says Linus Torvalds, chunk says Guido" → HALLUCINATED_AGAINST (negative score)

One judge call evaluates all chunks at once. The same model used for generation can serve as the judge.

**Causal Graph Builder** — Combines explicit causal links (from your pipeline instrumentation) with inferred links (from the entailment linker) into a PyRapide `Computation`. Supports both Engine-driven (live) and post-hoc (recorded) graph construction.

**Constraints** — Declarative rules using PyRapide's pattern algebra (`>>` for causal sequence):
- Every final output must be grounded in a retrieved chunk
- Every tool call must produce a result
- Every LLM response must have a preceding request

Constraints are **context-aware** — they only fire when the relevant event types are present. A `tool_completion` constraint won't trigger in a pipeline that never makes tool calls.

## Tech Stack

- **PyRapide** — causal event modeling (posets, patterns, constraints, visualization)
- **Ollama** — local LLM inference + embeddings
- **FastAPI** — async API layer
- **httpx** — async HTTP client
- **numpy** — cosine similarity (retrieval in demo)
- **networkx** / **pydantic** — via PyRapide

## License

MIT
