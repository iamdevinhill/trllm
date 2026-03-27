"""FastAPI server for TRLLM."""

from __future__ import annotations

import asyncio
import json

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
from sse_starlette.sse import EventSourceResponse
from pyrapide import Computation, visualization

from trllm.adapters.ollama import OllamaAdapter
from trllm.api.models import CausalQuery, PipelineRun
from trllm.constraints import check_constraints
from trllm.events import CLEvent, EventType
from trllm.graph import CausalGraphBuilder
from trllm.linker import EntailmentLinker

app = FastAPI(title="TRLLM", version="0.1.0")

# In-memory storage
computations: dict[str, Computation] = {}
run_events: dict[str, list[CLEvent]] = {}

# Shared instances
ollama = OllamaAdapter()
linker = EntailmentLinker(ollama)
builder = CausalGraphBuilder(linker)


@app.post("/ingest")
async def ingest_pipeline_run(run: PipelineRun):
    # Deserialize events with causal link resolution
    event_lookup: dict[str, CLEvent] = {}
    events: list[CLEvent] = []
    for e_dict in run.events:
        ev = CLEvent.from_dict(e_dict, event_lookup)
        event_lookup[ev.id] = ev
        events.append(ev)

    computation = await builder.build(events)
    computations[run.run_id] = computation
    run_events[run.run_id] = events

    return {
        "run_id": run.run_id,
        "event_count": len(events),
        "events_in_poset": len(list(computation.events)),
    }


@app.post("/query")
async def query_causal_graph(query: CausalQuery):
    comp = computations.get(query.run_id)
    if not comp:
        raise HTTPException(404, "Run not found")

    if query.question_type == "what_caused":
        if not query.target_event_id:
            raise HTTPException(400, "target_event_id required")
        target = _find_event(comp, query.target_event_id)
        if not target:
            raise HTTPException(404, "Target event not found")
        ancestors = comp.ancestors(target)
        return {
            "target": query.target_event_id,
            "causes": [_event_summary(e) for e in ancestors],
        }

    elif query.question_type == "dead_nodes":
        leaves = list(comp.leaf_events())
        final_outputs = [e for e in leaves if e.name == "final.response"]
        if not final_outputs:
            return {"dead_events": [_event_summary(e) for e in comp.events]}
        final = final_outputs[0]
        all_ancestors = comp.ancestors(final) | {final}
        dead = [e for e in comp.events if e not in all_ancestors]
        return {"dead_events": [_event_summary(e) for e in dead]}

    elif query.question_type == "min_path":
        if not query.target_event_id or not query.destination_event_id:
            raise HTTPException(400, "target_event_id and destination_event_id required")
        source = _find_event(comp, query.target_event_id)
        dest = _find_event(comp, query.destination_event_id)
        if not source or not dest:
            raise HTTPException(404, "Event not found")
        chain = comp.causal_chain(source, dest)
        return {"path": [_event_summary(e) for e in chain]}

    elif query.question_type == "counterfactual":
        if not query.remove_event_id:
            raise HTTPException(400, "remove_event_id required")
        target = _find_event(comp, query.remove_event_id)
        if not target:
            raise HTTPException(404, "Event not found")
        dependent = comp.descendants(target)
        return {
            "removed": query.remove_event_id,
            "affected_events": [_event_summary(e) for e in dependent],
            "output_affected": any(e.name == "final.response" for e in dependent),
        }

    raise HTTPException(400, f"Unknown question_type: {query.question_type}")


@app.get("/runs/{run_id}/visualization")
async def get_visualization(run_id: str, format: str = "mermaid"):
    comp = computations.get(run_id)
    if not comp:
        raise HTTPException(404, "Run not found")

    if format == "mermaid":
        return {"mermaid": visualization.to_mermaid(comp)}
    elif format == "ascii":
        return {"ascii": visualization.to_ascii(comp)}
    elif format == "dot":
        return {"dot": visualization.to_dot(comp)}
    elif format == "summary":
        return {"summary": visualization.summary(comp)}
    raise HTTPException(400, f"Unknown format: {format}")


@app.get("/runs/{run_id}/constraints")
async def get_constraint_violations(run_id: str):
    comp = computations.get(run_id)
    if not comp:
        raise HTTPException(404, "Run not found")

    violations = check_constraints(comp)
    return {
        "violations": [str(v) for v in violations],
        "all_passed": len(violations) == 0,
    }


DEMO_DOCUMENTS = [
    {"id": "doc1", "text": "Python was created by Guido van Rossum and first released in 1991."},
    {"id": "doc2", "text": "The capital of France is Paris, known for the Eiffel Tower."},
    {"id": "doc3", "text": "Machine learning is a subset of artificial intelligence."},
    {"id": "doc4", "text": "Tennessee is known as the Volunteer State."},
    {"id": "doc5", "text": "FastAPI is a modern Python web framework based on Starlette."},
]

DEMO_LLM_MODEL = "qwen3:30b"
DEMO_EMBED_MODEL = "qwen3-embedding:0.6b"


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    a_arr, b_arr = np.array(a), np.array(b)
    norm_a, norm_b = np.linalg.norm(a_arr), np.linalg.norm(b_arr)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a_arr, b_arr) / (norm_a * norm_b))


class DemoRequest(BaseModel):
    query: str = "What year was Python created and who made it?"
    documents: list[dict] | None = None
    top_k: int = 3
    llm_model: str = DEMO_LLM_MODEL
    embed_model: str = DEMO_EMBED_MODEL
    pipeline: str = "rag"  # "rag" or "agent"


def _emit(step, data):
    return json.dumps({"step": step, **data})


async def _finalize(events, chunk_events, response_event, req, run_id_prefix):
    """Shared finalization: build graph, score entailment, check constraints."""
    yield _emit("status", {"message": "Building causal graph (running entailment judge)..."})

    computation = await builder.build(events)

    run_id = f"{run_id_prefix}-{len(computations) + 1}"
    computations[run_id] = computation
    run_events[run_id] = events

    yield _emit("status", {"message": "Scoring entailment..."})

    scored = await linker.score_influence(chunk_events, response_event)
    scores = []
    for chunk_event, confidence in sorted(scored, key=lambda x: x[1], reverse=True):
        if confidence < 0:
            verdict = "HALLUCINATED_AGAINST"
        elif confidence > 0:
            verdict = "CAUSAL"
        else:
            verdict = "DEAD"
        scores.append({
            "chunk_id": chunk_event.payload.get("chunk_id", chunk_event.id),
            "verdict": verdict,
            "confidence": confidence,
            "text": (chunk_event.payload.get("text", str(chunk_event.payload)))[:80],
        })

    yield _emit("scores", {"message": "Entailment scores ready", "scores": scores})

    violations = check_constraints(computation)
    graph_data = _build_graph_data(computation)

    yield _emit("complete", {
        "message": "Pipeline complete",
        "run_id": run_id,
        "event_count": len(events),
        "scores": scores,
        "violations": [str(v) for v in violations],
        "all_passed": len(violations) == 0,
        "graph": graph_data,
    })


async def _rag_pipeline(req, documents):
    """Simple RAG: query → retrieve → assemble → LLM → response."""
    events: list[CLEvent] = []

    yield _emit("status", {"message": "Starting RAG pipeline..."})

    start_event = CLEvent(
        event_type=EventType.PIPELINE_START,
        payload={"pipeline": "rag"},
        source="pipeline",
    )
    events.append(start_event)

    user_query = req.query
    query_event = CLEvent(
        event_type=EventType.USER_QUERY,
        payload={"text": user_query},
        source="user",
        caused_by=[start_event],
    )
    events.append(query_event)

    retrieval_event = CLEvent(
        event_type=EventType.RETRIEVAL_REQUEST,
        payload={"query": user_query},
        source="retriever",
        caused_by=[query_event],
    )
    events.append(retrieval_event)

    yield _emit("status", {"message": "Embedding query and documents..."})

    query_embedding = await ollama.embed(req.embed_model, user_query)
    scored_docs = []
    for doc in documents:
        doc_embedding = await ollama.embed(req.embed_model, doc["text"])
        sim = _cosine_similarity(query_embedding, doc_embedding)
        scored_docs.append((doc, sim))

    scored_docs.sort(key=lambda x: x[1], reverse=True)
    top_chunks = scored_docs[:req.top_k]

    yield _emit("retrieval", {
        "message": f"Retrieved top {len(top_chunks)} chunks",
        "chunks": [
            {"id": doc["id"], "similarity": round(sim, 3), "text": doc["text"][:80]}
            for doc, sim in top_chunks
        ],
    })

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

    context_text = "\n".join(ce.payload["text"] for ce in chunk_events)
    prompt = f"Context:\n{context_text}\n\nQuestion: {user_query}\nAnswer:"

    prompt_event = CLEvent(
        event_type=EventType.PROMPT_ASSEMBLED,
        payload={"prompt": prompt, "chunk_count": len(chunk_events)},
        source="pipeline",
        caused_by=chunk_events,
    )
    events.append(prompt_event)

    request_event = CLEvent(
        event_type=EventType.LLM_REQUEST,
        payload={"model": req.llm_model, "prompt": prompt},
        source="llm",
        caused_by=[prompt_event],
    )
    events.append(request_event)

    yield _emit("status", {"message": f"Calling {req.llm_model}..."})

    result = await ollama.generate(req.llm_model, prompt)
    llm_output = result["response"]

    yield _emit("llm_response", {"message": "LLM responded", "output": llm_output})

    response_event = CLEvent(
        event_type=EventType.LLM_RESPONSE,
        payload={"output": llm_output, "model": req.llm_model},
        source="llm",
        caused_by=[request_event],
    )
    events.append(response_event)

    final_event = CLEvent(
        event_type=EventType.FINAL_RESPONSE,
        payload={"text": llm_output},
        source="pipeline",
        caused_by=[response_event],
    )
    events.append(final_event)

    end_event = CLEvent(
        event_type=EventType.PIPELINE_END,
        payload={"event_count": len(events)},
        source="pipeline",
        caused_by=[final_event],
    )
    events.append(end_event)

    async for msg in _finalize(events, chunk_events, response_event, req, "rag"):
        yield msg


async def _agent_pipeline(req, documents):
    """Multi-step agent: query → planning LLM → tool call → retrieval → reasoning → synthesis LLM → response.

    Creates a branching graph with parallel tool/retrieval paths and multiple LLM calls.
    """
    events: list[CLEvent] = []

    yield _emit("status", {"message": "Starting agent pipeline..."})

    start_event = CLEvent(
        event_type=EventType.PIPELINE_START,
        payload={"pipeline": "agent"},
        source="pipeline",
    )
    events.append(start_event)

    user_query = req.query
    query_event = CLEvent(
        event_type=EventType.USER_QUERY,
        payload={"text": user_query},
        source="user",
        caused_by=[start_event],
    )
    events.append(query_event)

    # Step 1: Planning LLM — analyze query and produce a plan
    yield _emit("status", {"message": f"Planning: analyzing query with {req.llm_model}..."})

    planning_prompt = (
        f"You are a research planner. Given this question, identify 2-3 specific sub-questions "
        f"that need to be answered. Be brief.\n\nQuestion: {user_query}\n\nSub-questions:"
    )
    planning_request = CLEvent(
        event_type=EventType.LLM_REQUEST,
        payload={"model": req.llm_model, "prompt": planning_prompt, "role": "planner"},
        source="planner",
        caused_by=[query_event],
    )
    events.append(planning_request)

    plan_result = await ollama.generate(req.llm_model, planning_prompt)
    plan_text = plan_result["response"]

    planning_response = CLEvent(
        event_type=EventType.LLM_RESPONSE,
        payload={"output": plan_text, "model": req.llm_model, "role": "planner"},
        source="planner",
        caused_by=[planning_request],
    )
    events.append(planning_response)

    reasoning_plan = CLEvent(
        event_type=EventType.REASONING_STEP,
        payload={"text": plan_text, "step": "query_decomposition"},
        source="planner",
        caused_by=[planning_response],
    )
    events.append(reasoning_plan)

    yield _emit("status", {"message": "Plan ready. Running tool call and retrieval in parallel..."})

    # Step 2a: Tool call branch — simulate a knowledge lookup tool
    tool_call_event = CLEvent(
        event_type=EventType.TOOL_CALL,
        payload={"tool": "knowledge_lookup", "input": user_query},
        source="agent",
        caused_by=[reasoning_plan],
    )
    events.append(tool_call_event)

    tool_prompt = (
        f"You are a knowledge lookup tool. Provide 2-3 brief factual statements relevant to "
        f"this query. Only state facts, no opinions.\n\nQuery: {user_query}\n\nFacts:"
    )
    tool_llm_result = await ollama.generate(req.llm_model, tool_prompt)
    tool_output = tool_llm_result["response"]

    tool_result_event = CLEvent(
        event_type=EventType.TOOL_RESULT,
        payload={"tool": "knowledge_lookup", "text": tool_output},
        source="agent",
        caused_by=[tool_call_event],
    )
    events.append(tool_result_event)

    # Step 2b: Retrieval branch (parallel with tool call in the graph)
    retrieval_event = CLEvent(
        event_type=EventType.RETRIEVAL_REQUEST,
        payload={"query": user_query},
        source="retriever",
        caused_by=[reasoning_plan],
    )
    events.append(retrieval_event)

    yield _emit("status", {"message": "Embedding query and documents..."})

    query_embedding = await ollama.embed(req.embed_model, user_query)
    scored_docs = []
    for doc in documents:
        doc_embedding = await ollama.embed(req.embed_model, doc["text"])
        sim = _cosine_similarity(query_embedding, doc_embedding)
        scored_docs.append((doc, sim))

    scored_docs.sort(key=lambda x: x[1], reverse=True)
    top_chunks = scored_docs[:req.top_k]

    yield _emit("retrieval", {
        "message": f"Retrieved top {len(top_chunks)} chunks",
        "chunks": [
            {"id": doc["id"], "similarity": round(sim, 3), "text": doc["text"][:80]}
            for doc, sim in top_chunks
        ],
    })

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

    # Step 3: Reasoning — merge tool results and retrieval
    yield _emit("status", {"message": "Reasoning: evaluating evidence..."})

    evidence_prompt = (
        f"You are evaluating evidence for a question. Rate the relevance of each piece of "
        f"evidence on a scale of HIGH/MEDIUM/LOW. Be brief.\n\n"
        f"Question: {user_query}\n\n"
        f"Tool output:\n{tool_output}\n\n"
        f"Retrieved chunks:\n" +
        "\n".join(f"- {ce.payload['text']}" for ce in chunk_events) +
        "\n\nEvaluation:"
    )
    reasoning_result = await ollama.generate(req.llm_model, evidence_prompt)
    reasoning_text = reasoning_result["response"]

    reasoning_eval = CLEvent(
        event_type=EventType.REASONING_STEP,
        payload={"text": reasoning_text, "step": "evidence_evaluation"},
        source="agent",
        caused_by=[tool_result_event] + chunk_events,
    )
    events.append(reasoning_eval)

    # Step 4: Synthesis — final LLM call with all context
    context_parts = [
        f"Plan:\n{plan_text}",
        f"Tool output:\n{tool_output}",
        "Retrieved documents:\n" + "\n".join(ce.payload["text"] for ce in chunk_events),
        f"Evidence evaluation:\n{reasoning_text}",
    ]
    synthesis_prompt = (
        f"Using the following research, answer the question thoroughly.\n\n"
        + "\n\n".join(context_parts) +
        f"\n\nQuestion: {user_query}\nAnswer:"
    )

    prompt_event = CLEvent(
        event_type=EventType.PROMPT_ASSEMBLED,
        payload={"prompt": synthesis_prompt, "chunk_count": len(chunk_events)},
        source="pipeline",
        caused_by=[reasoning_eval, reasoning_plan],
    )
    events.append(prompt_event)

    synthesis_request = CLEvent(
        event_type=EventType.LLM_REQUEST,
        payload={"model": req.llm_model, "prompt": synthesis_prompt, "role": "synthesizer"},
        source="llm",
        caused_by=[prompt_event],
    )
    events.append(synthesis_request)

    yield _emit("status", {"message": f"Synthesizing answer with {req.llm_model}..."})

    final_result = await ollama.generate(req.llm_model, synthesis_prompt)
    final_output = final_result["response"]

    yield _emit("llm_response", {"message": "LLM responded", "output": final_output})

    synthesis_response = CLEvent(
        event_type=EventType.LLM_RESPONSE,
        payload={"output": final_output, "model": req.llm_model, "role": "synthesizer"},
        source="llm",
        caused_by=[synthesis_request],
    )
    events.append(synthesis_response)

    # Step 5: Synthesis event merging both reasoning paths
    synthesis_event = CLEvent(
        event_type=EventType.SYNTHESIS,
        payload={"text": final_output},
        source="agent",
        caused_by=[synthesis_response, reasoning_eval],
    )
    events.append(synthesis_event)

    final_event = CLEvent(
        event_type=EventType.FINAL_RESPONSE,
        payload={"text": final_output},
        source="pipeline",
        caused_by=[synthesis_event],
    )
    events.append(final_event)

    end_event = CLEvent(
        event_type=EventType.PIPELINE_END,
        payload={"event_count": len(events)},
        source="pipeline",
        caused_by=[final_event],
    )
    events.append(end_event)

    # Include tool result in the scored inputs alongside chunks
    all_inputs = chunk_events + [tool_result_event]

    async for msg in _finalize(events, all_inputs, synthesis_response, req, "agent"):
        yield msg


@app.post("/demo/run")
async def run_demo(req: DemoRequest):
    """Run a pipeline with custom query and documents, streaming progress via SSE."""
    documents = req.documents or DEMO_DOCUMENTS
    for i, doc in enumerate(documents):
        if "id" not in doc:
            doc["id"] = f"doc{i+1}"

    if req.pipeline == "agent":
        return EventSourceResponse(_agent_pipeline(req, documents))
    return EventSourceResponse(_rag_pipeline(req, documents))


@app.get("/runs/{run_id}/graph")
async def get_graph_data(run_id: str):
    """Return graph nodes and edges for D3 force layout."""
    comp = computations.get(run_id)
    if not comp:
        raise HTTPException(404, "Run not found")
    return _build_graph_data(comp)


def _build_graph_data(comp: Computation) -> dict:
    """Convert a Computation to D3-compatible nodes and links."""
    events_list = list(comp.events)
    poset = comp._poset

    nodes = []
    for e in events_list:
        nodes.append({
            "id": e.metadata.get("cl_event_id", str(id(e))),
            "name": e.name,
            "source": e.source,
            "payload_keys": list(e.payload.keys()),
        })

    links = []
    for e in events_list:
        for cause in poset.causes(e):
            links.append({
                "source": cause.metadata.get("cl_event_id", str(id(cause))),
                "target": e.metadata.get("cl_event_id", str(id(e))),
            })

    return {"nodes": nodes, "links": links}


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    import pathlib
    html_path = pathlib.Path(__file__).resolve().parent.parent.parent / "dashboard" / "index.html"
    with open(html_path) as f:
        return f.read()


def _find_event(comp: Computation, cl_event_id: str):
    for e in comp.events:
        if e.metadata.get("cl_event_id") == cl_event_id:
            return e
    return None


def _event_summary(event) -> dict:
    return {
        "id": event.metadata.get("cl_event_id", event.id),
        "name": event.name,
        "source": event.source,
        "payload_keys": list(event.payload.keys()),
    }
