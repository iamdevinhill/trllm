"""Causal graph builder — converts CLEvents into a PyRapide Computation."""

from __future__ import annotations

from pyrapide import Computation, Event, Poset

from trllm.events import CLEvent, EventType
from trllm.linker import EntailmentLinker


class CausalGraphBuilder:
    def __init__(self, linker: EntailmentLinker):
        self.linker = linker

    async def build(self, events: list[CLEvent]) -> Computation:
        poset = Poset()
        rapide_events: dict[str, Event] = {}

        # Phase 1: Add all events as nodes
        for ev in events:
            rapide_event = Event(
                name=ev.event_type.value,
                payload=ev.payload,
                source=ev.source,
                timestamp=ev.timestamp,
                metadata={"cl_event_id": ev.id, **ev.metadata},
            )
            poset.add(rapide_event)
            rapide_events[ev.id] = rapide_event

        # Phase 2: Add explicit causal links from caused_by
        for ev in events:
            for cause in ev.caused_by:
                if cause.id in rapide_events:
                    poset.add_causal_link(
                        rapide_events[cause.id],
                        rapide_events[ev.id],
                    )

        # Phase 3: Infer implicit causal links via entailment checking
        llm_responses = [e for e in events if e.event_type == EventType.LLM_RESPONSE]
        potential_causes = [
            e for e in events
            if e.event_type in (
                EventType.CHUNK_INJECTED,
                EventType.TOOL_RESULT,
                EventType.REASONING_STEP,
                EventType.AGENT_RESPONSE,
            )
        ]

        for response_event in llm_responses:
            scored = await self.linker.score_influence(potential_causes, response_event)
            for cause_event, confidence in scored:
                if confidence >= self.linker.similarity_threshold:
                    cause_re = rapide_events[cause_event.id]
                    effect_re = rapide_events[response_event.id]
                    # Avoid duplicate edges (may already exist from explicit caused_by)
                    if not poset.is_ancestor(cause_re, effect_re):
                        poset.add_causal_link(cause_re, effect_re)

        # Phase 4: Build Computation
        computation = Computation(poset=poset)
        return computation

    def build_from_rapide_events(
        self, rapide_events: list[Event], causal_pairs: list[tuple[Event, Event]]
    ) -> Computation:
        """Build a Computation directly from PyRapide Events (no semantic linking)."""
        poset = Poset()
        for ev in rapide_events:
            poset.add(ev)
        for cause, effect in causal_pairs:
            poset.add_causal_link(cause, effect)
        return Computation(poset=poset)
