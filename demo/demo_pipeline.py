"""
TRLLM Demo — Agent pipeline with entailment-based causal tracing.
Runs a multi-step agent pipeline: planning, parallel tool call + retrieval,
evidence evaluation, and synthesis, then traces causal relationships.

Usage:
  python demo/demo_pipeline.py
  python demo/demo_pipeline.py --query "How many moons does Mars have?"
  python demo/demo_pipeline.py --query "..." --docs docs.txt
  python demo/demo_pipeline.py --query "..." --docs docs.txt --llm qwen3:8b --top-k 3

Requires: Ollama running at localhost:11434 with the specified models pulled.
"""

import argparse
import asyncio

import numpy as np

from trllm.adapters.ollama import OllamaAdapter
from trllm.constraints import check_constraints
from trllm.events import CLEvent, EventType
from trllm.graph import CausalGraphBuilder
from trllm.linker import EntailmentLinker
from trllm.visualization.renderer import render_summary

DEFAULT_QUERY = "How many moons does Mars have and what are their names?"
DEFAULT_DOCUMENTS = [
    {"id": "doc1", "text": "Mars has two small moons called Phobos and Deimos, discovered by Asaph Hall in 1877."},
    {"id": "doc2", "text": "Jupiter is the largest planet in our solar system with at least 95 known moons."},
    {"id": "doc3", "text": "Phobos orbits Mars at a distance of only 6,000 km and is slowly spiraling inward."},
    {"id": "doc4", "text": "Mars has three moons: Phobos, Deimos, and Titan."},
    {"id": "doc5", "text": "The Martian surface features Olympus Mons, the tallest volcano in the solar system."},
]
DEFAULT_LLM = "qwen3:8b"
DEFAULT_EMBED = "qwen3-embedding:0.6b"
DEFAULT_TOP_K = 3


def cosine_similarity(a: list[float], b: list[float]) -> float:
    a_arr, b_arr = np.array(a), np.array(b)
    norm_a, norm_b = np.linalg.norm(a_arr), np.linalg.norm(b_arr)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a_arr, b_arr) / (norm_a * norm_b))


async def run_demo(user_query, documents, llm_model, embed_model, top_k):
    ollama = OllamaAdapter()
    linker = EntailmentLinker(ollama, judge_model=llm_model)
    builder = CausalGraphBuilder(linker)
    events: list[CLEvent] = []

    # 1. Pipeline start
    start_event = CLEvent(
        event_type=EventType.PIPELINE_START,
        payload={"pipeline": "agent_demo"},
        source="pipeline",
    )
    events.append(start_event)

    # 2. User query
    query_event = CLEvent(
        event_type=EventType.USER_QUERY,
        payload={"text": user_query},
        source="user",
        caused_by=[start_event],
    )
    events.append(query_event)

    # 3. Planning LLM — decompose the query
    print("Step 1: Planning query decomposition...")
    planning_request = CLEvent(
        event_type=EventType.LLM_REQUEST,
        payload={"model": llm_model, "prompt": user_query, "role": "planner"},
        source="llm",
        caused_by=[query_event],
    )
    events.append(planning_request)

    plan_prompt = (
        f"You are a research planner. Break this question into 2-3 sub-questions "
        f"that would help answer it completely. Be brief.\n\nQuestion: {user_query}\n\nSub-questions:"
    )
    plan_result = await ollama.generate(llm_model, plan_prompt)
    plan_text = plan_result["response"]

    planning_response = CLEvent(
        event_type=EventType.LLM_RESPONSE,
        payload={"output": plan_text, "model": llm_model, "role": "planner"},
        source="llm",
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

    print(f"  Plan: {plan_text[:100]}...")

    # 4. Parallel: Tool call + Retrieval
    print("Step 2: Running tool call and retrieval in parallel...")

    # 4a. Tool call — knowledge lookup
    tool_call_event = CLEvent(
        event_type=EventType.TOOL_CALL,
        payload={"tool": "knowledge_lookup", "input": user_query},
        source="agent",
        caused_by=[reasoning_plan],
    )
    events.append(tool_call_event)

    # 4b. Retrieval request
    retrieval_event = CLEvent(
        event_type=EventType.RETRIEVAL_REQUEST,
        payload={"query": user_query},
        source="retriever",
        caused_by=[reasoning_plan],
    )
    events.append(retrieval_event)

    # Run both in parallel
    async def _tool_call():
        tool_prompt = (
            f"You are a knowledge lookup tool. Provide 2-3 brief factual statements relevant to "
            f"this query. Only state facts, no opinions.\n\nQuery: {user_query}\n\nFacts:"
        )
        result = await ollama.generate(llm_model, tool_prompt)
        return result["response"]

    async def _retrieve():
        query_embedding = await ollama.embed(embed_model, user_query)
        scored = []
        for doc in documents:
            doc_embedding = await ollama.embed(embed_model, doc["text"])
            sim = cosine_similarity(query_embedding, doc_embedding)
            scored.append((doc, sim))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    tool_output, top_chunks = await asyncio.gather(_tool_call(), _retrieve())

    tool_result_event = CLEvent(
        event_type=EventType.TOOL_RESULT,
        payload={"tool": "knowledge_lookup", "text": tool_output},
        source="agent",
        caused_by=[tool_call_event],
    )
    events.append(tool_result_event)

    print(f"  Tool output: {tool_output[:80]}...")
    print(f"  Top {len(top_chunks)} chunks retrieved (out of {len(documents)}):")
    for doc, score in top_chunks:
        print(f"    {doc['id']}: similarity={score:.3f} — {doc['text'][:60]}...")

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

    # 5. Reasoning — evaluate evidence
    print("Step 3: Evaluating evidence...")
    evidence_prompt = (
        f"You are evaluating evidence for a question. Rate the relevance of each piece of "
        f"evidence on a scale of HIGH/MEDIUM/LOW. Be brief.\n\n"
        f"Question: {user_query}\n\n"
        f"Tool output:\n{tool_output}\n\n"
        f"Retrieved chunks:\n"
        + "\n".join(f"- {ce.payload['text']}" for ce in chunk_events)
        + "\n\nEvaluation:"
    )
    reasoning_result = await ollama.generate(llm_model, evidence_prompt)
    reasoning_text = reasoning_result["response"]

    reasoning_eval = CLEvent(
        event_type=EventType.REASONING_STEP,
        payload={"text": reasoning_text, "step": "evidence_evaluation"},
        source="agent",
        caused_by=[tool_result_event] + chunk_events,
    )
    events.append(reasoning_eval)

    print(f"  Evaluation: {reasoning_text[:100]}...")

    # 6. Synthesis — final LLM call
    print("Step 4: Synthesizing answer...")
    context_parts = [
        f"Plan:\n{plan_text}",
        f"Tool output:\n{tool_output}",
        "Retrieved documents:\n" + "\n".join(ce.payload["text"] for ce in chunk_events),
        f"Evidence evaluation:\n{reasoning_text}",
    ]
    synthesis_prompt = (
        f"Using the following research, answer the question thoroughly.\n\n"
        + "\n\n".join(context_parts)
        + f"\n\nQuestion: {user_query}\nAnswer:"
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
        payload={"model": llm_model, "prompt": synthesis_prompt, "role": "synthesizer"},
        source="llm",
        caused_by=[prompt_event],
    )
    events.append(synthesis_request)

    final_result = await ollama.generate(llm_model, synthesis_prompt)
    final_output = final_result["response"]

    synthesis_response = CLEvent(
        event_type=EventType.LLM_RESPONSE,
        payload={"output": final_output, "model": llm_model, "role": "synthesizer"},
        source="llm",
        caused_by=[synthesis_request],
    )
    events.append(synthesis_response)

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

    # 7. Build causal graph
    print("\nBuilding causal graph (running entailment judge)...")
    computation = await builder.build(events)

    # 9. Print results
    print("\n" + "=" * 60)
    print("CAUSAL GRAPH SUMMARY")
    print("=" * 60)
    print(render_summary(computation))

    print("\n" + "=" * 60)
    print("LLM OUTPUT")
    print("=" * 60)
    print(final_output)

    print("\n" + "=" * 60)
    print("ENTAILMENT-BASED CAUSAL SCORES")
    print("=" * 60)
    all_inputs = chunk_events + [tool_result_event]
    scored = await linker.score_influence(all_inputs, synthesis_response)
    for chunk_event, confidence in sorted(scored, key=lambda x: x[1], reverse=True):
        if confidence < 0:
            status = "HALLUCINATED_AGAINST"
        elif confidence > 0:
            status = "CAUSAL"
        else:
            status = "DEAD"
        chunk_id = chunk_event.payload.get("chunk_id", chunk_event.payload.get("tool", chunk_event.id))
        print(
            f"  [{status}] {chunk_id}: "
            f"confidence={confidence:+.2f} | "
            f"text={chunk_event.payload.get('text', '')[:60]}..."
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


def parse_args():
    parser = argparse.ArgumentParser(description="TRLLM Demo — Agent pipeline with causal tracing")
    parser.add_argument("--query", "-q", default=DEFAULT_QUERY, help="Question to ask")
    parser.add_argument("--docs", "-d", default=None, help="Path to documents file (one per line)")
    parser.add_argument("--llm", default=DEFAULT_LLM, help=f"LLM model (default: {DEFAULT_LLM})")
    parser.add_argument("--embed", default=DEFAULT_EMBED, help=f"Embed model (default: {DEFAULT_EMBED})")
    parser.add_argument("--top-k", "-k", type=int, default=DEFAULT_TOP_K, help=f"Top K chunks (default: {DEFAULT_TOP_K})")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.docs:
        with open(args.docs) as f:
            docs = [
                {"id": f"doc{i+1}", "text": line.strip()}
                for i, line in enumerate(f) if line.strip()
            ]
    else:
        docs = DEFAULT_DOCUMENTS

    asyncio.run(run_demo(args.query, docs, args.llm, args.embed, args.top_k))
