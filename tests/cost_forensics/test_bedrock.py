"""Tests for Bedrock adapter using mock boto3 client."""

from __future__ import annotations

from typing import Any

import pytest

from trllm.cost_forensics.adapters.bedrock import InstrumentedBedrock
from trllm.cost_forensics.annotator import CostAnnotator
from trllm.cost_forensics.pricing import PricingRegistry
from trllm.cost_forensics.rollup import CausalCostRollup


# --- Mock boto3 bedrock-runtime objects ---

MODEL_ID = "anthropic.claude-3-5-haiku-20241022-v1:0"


def _mock_converse_response(
    input_tokens: int = 100, output_tokens: int = 50
) -> dict[str, Any]:
    return {
        "output": {
            "message": {
                "role": "assistant",
                "content": [{"text": "Hello!"}],
            }
        },
        "stopReason": "end_turn",
        "usage": {
            "inputTokens": input_tokens,
            "outputTokens": output_tokens,
            "totalTokens": input_tokens + output_tokens,
        },
    }


class _MockBedrockStream:
    """Mock EventStream from converse_stream()."""

    def __init__(self, events: list[dict[str, Any]]) -> None:
        self._events = events

    def __iter__(self) -> Any:
        yield from self._events


class _MockBedrockClient:
    """Mock boto3 bedrock-runtime client."""

    def __init__(
        self,
        responses: list[dict[str, Any]] | None = None,
        stream_events: list[dict[str, Any]] | None = None,
    ) -> None:
        self._responses = responses or [_mock_converse_response()]
        self._stream_events = stream_events or []
        self._call_count = 0

    def converse(self, **kwargs: Any) -> dict[str, Any]:
        resp = self._responses[
            min(self._call_count, len(self._responses) - 1)
        ]
        self._call_count += 1
        return resp

    def converse_stream(self, **kwargs: Any) -> dict[str, Any]:
        return {"stream": _MockBedrockStream(self._stream_events)}


# --- Tests ---


class TestInstrumentedBedrock:
    def test_single_call_records_events(self) -> None:
        client = InstrumentedBedrock(_MockBedrockClient())
        client.converse(
            modelId=MODEL_ID,
            messages=[{"role": "user", "content": [{"text": "hi"}]}],
        )

        comp = client.computation()
        events = list(comp)
        assert len(events) == 2  # session + llm_call

    def test_multiple_calls_chain_causally(self) -> None:
        client = InstrumentedBedrock(_MockBedrockClient())
        client.converse(modelId=MODEL_ID, messages=[])
        client.converse(modelId=MODEL_ID, messages=[])
        client.converse(modelId=MODEL_ID, messages=[])

        comp = client.computation()
        events = list(comp)
        assert len(events) == 4  # session + 3 calls
        assert len(comp.root_events()) == 1

    def test_token_counts_captured(self) -> None:
        mock = _MockBedrockClient(
            responses=[_mock_converse_response(5000, 500)]
        )
        client = InstrumentedBedrock(mock)
        client.converse(modelId=MODEL_ID, messages=[])

        comp = client.computation()
        pricing = PricingRegistry.bedrock()
        annotations = CostAnnotator(pricing).annotate(comp)
        report = CausalCostRollup().rollup(comp, annotations)
        assert report.total_cost > 0

    def test_correct_cost_calculation(self) -> None:
        mock = _MockBedrockClient(
            responses=[_mock_converse_response(1000, 200)]
        )
        client = InstrumentedBedrock(mock)
        client.converse(modelId=MODEL_ID, messages=[])

        comp = client.computation()
        pricing = PricingRegistry.bedrock()
        annotations = CostAnnotator(pricing).annotate(comp)
        report = CausalCostRollup().rollup(comp, annotations)

        # Claude 3.5 Haiku: $0.80/1M in, $4.00/1M out
        # 1000 * 0.80/1M + 200 * 4.00/1M = 0.0008 + 0.0008 = 0.0016
        assert abs(report.total_cost - 0.0016) < 0.0001

    def test_record_tool_call(self) -> None:
        client = InstrumentedBedrock(_MockBedrockClient())
        client.converse(modelId=MODEL_ID, messages=[])
        client.record_tool_call("web_search", {"query": "test"})
        client.converse(modelId=MODEL_ID, messages=[])

        comp = client.computation()
        events = list(comp)
        assert len(events) == 4  # session + llm + tool + llm
        tool_events = [e for e in events if e.source == "tool"]
        assert len(tool_events) == 1
        assert tool_events[0].name == "web_search"

    def test_reset(self) -> None:
        client = InstrumentedBedrock(_MockBedrockClient())
        client.converse(modelId=MODEL_ID, messages=[])
        assert len(list(client.computation())) == 2

        client.reset()
        assert len(list(client.computation())) == 1

    def test_returns_original_response(self) -> None:
        expected = _mock_converse_response()
        mock = _MockBedrockClient(responses=[expected])
        client = InstrumentedBedrock(mock)
        response = client.converse(modelId=MODEL_ID, messages=[])
        assert response is expected

    def test_repr(self) -> None:
        client = InstrumentedBedrock(_MockBedrockClient())
        assert "InstrumentedBedrock" in repr(client)

    def test_getattr_passthrough(self) -> None:
        """Non-converse methods pass through to the underlying client."""
        mock = _MockBedrockClient()
        mock.some_other_method = lambda: "passthrough"  # type: ignore[attr-defined]
        client = InstrumentedBedrock(mock)
        assert client.some_other_method() == "passthrough"  # type: ignore[attr-defined]


class TestBedrockStreaming:
    def test_stream_captures_usage(self) -> None:
        stream_events = [
            {"contentBlockDelta": {"delta": {"text": "Hello"}}},
            {"contentBlockDelta": {"delta": {"text": " world"}}},
            {
                "metadata": {
                    "usage": {
                        "inputTokens": 2000,
                        "outputTokens": 400,
                    }
                }
            },
        ]
        mock = _MockBedrockClient(stream_events=stream_events)
        client = InstrumentedBedrock(mock)

        stream = client.converse_stream(modelId=MODEL_ID, messages=[])
        collected = list(stream)
        assert len(collected) == 3

        comp = client.computation()
        events = list(comp)
        assert len(events) == 2  # session + streamed llm_call

        llm_events = [e for e in events if e.name == "llm_call"]
        assert len(llm_events) == 1
        assert llm_events[0].payload["usage"]["input_tokens"] == 2000
        assert llm_events[0].payload["usage"]["output_tokens"] == 400
        assert llm_events[0].payload["streamed"] is True

    def test_stream_context_manager(self) -> None:
        stream_events = [
            {
                "metadata": {
                    "usage": {"inputTokens": 100, "outputTokens": 50}
                }
            },
        ]
        mock = _MockBedrockClient(stream_events=stream_events)
        client = InstrumentedBedrock(mock)

        with client.converse_stream(modelId=MODEL_ID, messages=[]) as stream:
            list(stream)

        comp = client.computation()
        assert len(list(comp)) == 2

    def test_stream_records_on_exit_without_iteration(self) -> None:
        mock = _MockBedrockClient(stream_events=[])
        client = InstrumentedBedrock(mock)

        with client.converse_stream(modelId=MODEL_ID, messages=[]):
            pass  # don't iterate

        comp = client.computation()
        assert len(list(comp)) == 2  # session + llm_call (zero tokens)

    def test_stream_cost_calculation(self) -> None:
        stream_events = [
            {
                "metadata": {
                    "usage": {"inputTokens": 5000, "outputTokens": 1000}
                }
            },
        ]
        mock = _MockBedrockClient(stream_events=stream_events)
        client = InstrumentedBedrock(mock)

        stream = client.converse_stream(modelId=MODEL_ID, messages=[])
        list(stream)

        comp = client.computation()
        pricing = PricingRegistry.bedrock()
        annotations = CostAnnotator(pricing).annotate(comp)
        report = CausalCostRollup().rollup(comp, annotations)

        # Claude 3.5 Haiku: $0.80/1M in, $4.00/1M out
        # 5000 * 0.80/1M + 1000 * 4.00/1M = 0.004 + 0.004 = 0.008
        assert abs(report.total_cost - 0.008) < 0.0001

    def test_stream_then_converse_chains(self) -> None:
        stream_events = [
            {
                "metadata": {
                    "usage": {"inputTokens": 100, "outputTokens": 50}
                }
            },
        ]
        mock = _MockBedrockClient(stream_events=stream_events)
        client = InstrumentedBedrock(mock)

        # Stream first
        stream = client.converse_stream(modelId=MODEL_ID, messages=[])
        list(stream)

        # Then regular call
        client.converse(modelId=MODEL_ID, messages=[])

        comp = client.computation()
        events = list(comp)
        assert len(events) == 3  # session + streamed + regular
        assert len(comp.root_events()) == 1


class TestFullBedrockPipeline:
    def test_end_to_end_analysis(self) -> None:
        mock = _MockBedrockClient(
            responses=[
                _mock_converse_response(1000, 200),
                _mock_converse_response(2000, 400),
            ]
        )
        client = InstrumentedBedrock(mock)
        client.converse(modelId=MODEL_ID, messages=[])
        client.record_tool_call("query_db")
        client.converse(modelId=MODEL_ID, messages=[])

        comp = client.computation()
        pricing = PricingRegistry.bedrock()
        annotations = CostAnnotator(pricing).annotate(comp)
        report = CausalCostRollup().rollup(comp, annotations)

        tree = report.ascii_tree()
        assert "session" in tree
        assert "llm_call" in tree
        assert "query_db" in tree
        assert report.total_cost > 0
