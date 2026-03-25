"""FastAPI server for CausalLens."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pyrapide import Computation, visualization

from trllm.adapters.ollama import OllamaAdapter
from trllm.api.models import CausalQuery, PipelineRun
from trllm.constraints import check_constraints
from trllm.events import CLEvent
from trllm.graph import CausalGraphBuilder
from trllm.linker import SemanticLinker

app = FastAPI(title="CausalLens", version="0.1.0")

# In-memory storage
computations: dict[str, Computation] = {}
run_events: dict[str, list[CLEvent]] = {}

# Shared instances
ollama = OllamaAdapter()
linker = SemanticLinker(ollama)
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


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    with open("dashboard/index.html") as f:
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
