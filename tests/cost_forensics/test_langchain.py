"""Tests for LangChain callback handler."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from trllm.cost_forensics.adapters.langchain import TrllmCallbackHandler
from trllm.cost_forensics.annotator import CostAnnotator
from trllm.cost_forensics.pricing import ModelPrice, PricingRegistry
from trllm.cost_forensics.rollup import CausalCostRollup
from trllm.cost_forensics.waste import WasteDetector


# --- Mock LangChain response objects ---


class _MockLLMResult:
    def __init__(self, token_usage: dict[str, int] | None = None) -> None:
        self.llm_output: dict[str, Any] = {}
        if token_usage:
            self.llm_output["token_usage"] = token_usage


# --- Tests ---


class TestTrllmCallbackHandler:
    def test_init(self) -> None:
        handler = TrllmCallbackHandler()
        comp = handler.computation()
        events = list(comp)
        assert len(events) == 1  # session root only
        assert events[0].name == "session"

    def test_repr(self) -> None:
        handler = TrllmCallbackHandler()
        assert "TrllmCallbackHandler" in repr(handler)

    def test_llm_start_records_event(self) -> None:
        handler = TrllmCallbackHandler()
        run_id = uuid4()

        handler.on_llm_start(
            serialized={"kwargs": {"model_name": "gpt-4o"}, "id": ["openai"]},
            prompts=["Hello"],
            run_id=run_id,
        )

        events = list(handler.computation())
        assert len(events) == 2
        llm_events = [e for e in events if e.name == "llm_call"]
        assert len(llm_events) == 1
        assert llm_events[0].payload["model"] == "gpt-4o"

    def test_llm_end_records_usage(self) -> None:
        handler = TrllmCallbackHandler()
        run_id = uuid4()

        handler.on_llm_start(
            serialized={"kwargs": {"model_name": "gpt-4o"}, "id": ["openai"]},
            prompts=["Hello"],
            run_id=run_id,
        )

        result = _MockLLMResult(
            token_usage={"prompt_tokens": 100, "completion_tokens": 50}
        )
        handler.on_llm_end(result, run_id=run_id)

        events = list(handler.computation())
        assert len(events) == 3  # session + llm_call + llm_complete

        complete = [e for e in events if e.name == "llm_complete"]
        assert len(complete) == 1
        assert complete[0].payload["usage"]["input_tokens"] == 100
        assert complete[0].payload["usage"]["output_tokens"] == 50

    def test_llm_error_records_event(self) -> None:
        handler = TrllmCallbackHandler()
        run_id = uuid4()

        handler.on_llm_start(
            serialized={"id": ["openai"]},
            prompts=["Hello"],
            run_id=run_id,
        )
        handler.on_llm_error(
            ValueError("rate limited"), run_id=run_id
        )

        events = list(handler.computation())
        error_events = [e for e in events if e.name == "llm_error"]
        assert len(error_events) == 1
        assert "rate limited" in error_events[0].payload["error"]

    def test_chat_model_start(self) -> None:
        handler = TrllmCallbackHandler()
        run_id = uuid4()

        handler.on_chat_model_start(
            serialized={"kwargs": {"model": "gpt-4o"}, "id": ["ChatOpenAI"]},
            messages=[[{"role": "user", "content": "Hi"}]],
            run_id=run_id,
        )

        events = list(handler.computation())
        llm_events = [e for e in events if e.name == "llm_call"]
        assert len(llm_events) == 1
        assert llm_events[0].payload["model"] == "gpt-4o"

    def test_chain_start_and_end(self) -> None:
        handler = TrllmCallbackHandler()
        chain_id = uuid4()

        handler.on_chain_start(
            serialized={"id": ["langchain", "chains", "LLMChain"]},
            inputs={"input": "test"},
            run_id=chain_id,
        )
        handler.on_chain_end(
            outputs={"output": "result"}, run_id=chain_id
        )

        events = list(handler.computation())
        chain_events = [
            e for e in events if e.name.startswith("chain:")
        ]
        assert len(chain_events) == 1
        assert chain_events[0].name == "chain:LLMChain"

    def test_chain_error(self) -> None:
        handler = TrllmCallbackHandler()
        chain_id = uuid4()

        handler.on_chain_start(
            serialized={"id": ["MyChain"]},
            inputs={},
            run_id=chain_id,
        )
        handler.on_chain_error(
            RuntimeError("chain failed"), run_id=chain_id
        )

        events = list(handler.computation())
        errors = [e for e in events if e.name == "chain_error"]
        assert len(errors) == 1

    def test_tool_start_and_end(self) -> None:
        handler = TrllmCallbackHandler()
        tool_id = uuid4()

        handler.on_tool_start(
            serialized={"name": "web_search", "id": ["tool"]},
            input_str="search query",
            run_id=tool_id,
        )
        handler.on_tool_end("search results", run_id=tool_id)

        events = list(handler.computation())
        tool_events = [e for e in events if e.source == "tool"]
        assert len(tool_events) == 1
        assert tool_events[0].name == "web_search"
        assert tool_events[0].payload["input"] == "search query"

    def test_tool_error(self) -> None:
        handler = TrllmCallbackHandler()
        tool_id = uuid4()

        handler.on_tool_start(
            serialized={"name": "calculator"},
            input_str="1/0",
            run_id=tool_id,
        )
        handler.on_tool_error(
            ZeroDivisionError("division by zero"), run_id=tool_id
        )

        events = list(handler.computation())
        errors = [e for e in events if e.name == "tool_error"]
        assert len(errors) == 1

    def test_retriever_start_and_end(self) -> None:
        handler = TrllmCallbackHandler()
        ret_id = uuid4()

        handler.on_retriever_start(
            serialized={"id": ["Retriever"]},
            query="What is the capital of France?",
            run_id=ret_id,
        )
        handler.on_retriever_end(
            documents=["Paris is the capital of France."],
            run_id=ret_id,
        )

        events = list(handler.computation())
        ret_events = [e for e in events if e.name == "retriever"]
        assert len(ret_events) == 1
        assert ret_events[0].payload["query"] == "What is the capital of France?"

    def test_parent_child_linking(self) -> None:
        """Chain → LLM → Tool should create a proper causal chain."""
        handler = TrllmCallbackHandler()
        chain_id = uuid4()
        llm_id = uuid4()
        tool_id = uuid4()

        handler.on_chain_start(
            serialized={"id": ["AgentExecutor"]},
            inputs={"input": "test"},
            run_id=chain_id,
        )
        handler.on_llm_start(
            serialized={"kwargs": {"model_name": "gpt-4o"}, "id": ["openai"]},
            prompts=["test"],
            run_id=llm_id,
            parent_run_id=chain_id,
        )
        handler.on_tool_start(
            serialized={"name": "web_search"},
            input_str="query",
            run_id=tool_id,
            parent_run_id=chain_id,
        )

        comp = handler.computation()
        assert len(comp.root_events()) == 1  # just the session root

    def test_reset(self) -> None:
        handler = TrllmCallbackHandler()

        handler.on_llm_start(
            serialized={"id": ["openai"]},
            prompts=["test"],
            run_id=uuid4(),
        )
        assert len(list(handler.computation())) == 2

        handler.reset()
        assert len(list(handler.computation())) == 1

    def test_noop_callbacks(self) -> None:
        """on_text, on_llm_new_token, on_agent_action, on_agent_finish are no-ops."""
        handler = TrllmCallbackHandler()
        handler.on_text("some text")
        handler.on_llm_new_token("token")
        handler.on_agent_action(None)
        handler.on_agent_finish(None)
        # Should not crash, no events recorded beyond session
        assert len(list(handler.computation())) == 1

    def test_full_agent_pipeline(self) -> None:
        """Simulate a full agent pipeline: chain → llm → tool → llm → end."""
        handler = TrllmCallbackHandler()
        chain_id = uuid4()
        llm1_id = uuid4()
        tool_id = uuid4()
        llm2_id = uuid4()

        # Agent chain starts
        handler.on_chain_start(
            serialized={"id": ["AgentExecutor"]},
            inputs={"input": "What's the weather?"},
            run_id=chain_id,
        )

        # First LLM call (decides to use tool)
        handler.on_chat_model_start(
            serialized={"kwargs": {"model": "gpt-4o"}, "id": ["ChatOpenAI"]},
            messages=[[]],
            run_id=llm1_id,
            parent_run_id=chain_id,
        )
        handler.on_llm_end(
            _MockLLMResult(
                token_usage={"prompt_tokens": 500, "completion_tokens": 100}
            ),
            run_id=llm1_id,
        )

        # Tool call
        handler.on_tool_start(
            serialized={"name": "weather_api"},
            input_str="San Francisco",
            run_id=tool_id,
            parent_run_id=chain_id,
        )
        handler.on_tool_end("72°F, Sunny", run_id=tool_id)

        # Second LLM call (final answer)
        handler.on_chat_model_start(
            serialized={"kwargs": {"model": "gpt-4o"}, "id": ["ChatOpenAI"]},
            messages=[[]],
            run_id=llm2_id,
            parent_run_id=chain_id,
        )
        handler.on_llm_end(
            _MockLLMResult(
                token_usage={"prompt_tokens": 800, "completion_tokens": 200}
            ),
            run_id=llm2_id,
        )

        # Chain ends
        handler.on_chain_end(
            outputs={"output": "It's 72°F and sunny in SF"},
            run_id=chain_id,
        )

        comp = handler.computation()
        events = list(comp)

        # session + chain + llm_call + llm_complete + tool + llm_call + llm_complete = 7
        assert len(events) == 7

        # Cost analysis
        pricing = PricingRegistry.openai()
        pricing.register_tool("weather_api", _tool_price(0.002))
        annotations = CostAnnotator(pricing).annotate(comp)
        report = CausalCostRollup().rollup(comp, annotations)

        assert report.total_cost > 0
        tree = report.ascii_tree()
        assert "session" in tree

    def test_cost_with_langchain_usage(self) -> None:
        """Verify costs are calculated from llm_complete events."""
        handler = TrllmCallbackHandler()
        run_id = uuid4()

        handler.on_chat_model_start(
            serialized={"kwargs": {"model": "gpt-4o"}, "id": ["ChatOpenAI"]},
            messages=[[]],
            run_id=run_id,
        )
        handler.on_llm_end(
            _MockLLMResult(
                token_usage={"prompt_tokens": 1000, "completion_tokens": 500}
            ),
            run_id=run_id,
        )

        comp = handler.computation()
        pricing = PricingRegistry.openai()
        annotations = CostAnnotator(pricing).annotate(comp)
        report = CausalCostRollup().rollup(comp, annotations)

        # gpt-4o: $2.50/1M in, $10.00/1M out
        # 1000 * 2.50/1M + 500 * 10.00/1M = 0.0025 + 0.005 = 0.0075
        assert abs(report.total_cost - 0.0075) < 0.0001

    def test_waste_detection_with_langchain(self) -> None:
        """Dead-end tool calls should be detected."""
        handler = TrllmCallbackHandler()
        chain_id = uuid4()
        llm_id = uuid4()
        tool1_id = uuid4()
        tool2_id = uuid4()

        handler.on_chain_start(
            serialized={"id": ["AgentExecutor"]},
            inputs={},
            run_id=chain_id,
        )

        # LLM call
        handler.on_chat_model_start(
            serialized={"kwargs": {"model": "gpt-4o"}, "id": ["ChatOpenAI"]},
            messages=[[]],
            run_id=llm_id,
            parent_run_id=chain_id,
        )
        handler.on_llm_end(
            _MockLLMResult(token_usage={"prompt_tokens": 100, "completion_tokens": 50}),
            run_id=llm_id,
        )

        # Tool used (has downstream)
        handler.on_tool_start(
            serialized={"name": "used_tool"},
            input_str="input",
            run_id=tool1_id,
            parent_run_id=chain_id,
        )
        handler.on_tool_end("result", run_id=tool1_id)

        # Dead-end tool (no downstream)
        handler.on_tool_start(
            serialized={"name": "dead_tool"},
            input_str="input",
            run_id=tool2_id,
            parent_run_id=chain_id,
        )
        handler.on_tool_end("unused result", run_id=tool2_id)

        handler.on_chain_end(outputs={}, run_id=chain_id)

        comp = handler.computation()
        pricing = PricingRegistry.openai()
        pricing.register_tool("used_tool", _tool_price(0.01))
        pricing.register_tool("dead_tool", _tool_price(0.01))
        annotations = CostAnnotator(pricing).annotate(comp)
        waste = WasteDetector().detect(comp, annotations)

        # dead_tool should be flagged
        dead_ends = [
            w for w in waste.instances if w.pattern_name == "DeadEndToolCall"
        ]
        assert len(dead_ends) >= 1


def _tool_price(cost: float) -> Any:
    from trllm.cost_forensics.pricing import ToolPrice

    return ToolPrice(cost, "tool")
