# TRLLM

Causal cost attribution and waste detection for LLM/agent pipelines.

TRLLM answers two questions about your LLM pipeline runs: **which root cause is responsible for what fraction of the total bill**, and **where is money being wasted?**

Built on [PyRapide](https://pypi.org/project/pyrapide/) — a causal event-driven architecture library that models distributed systems as causal DAGs.

## The Problem

You're running multi-step LLM pipelines — retrieval, reasoning, tool calls, agent handoffs — and the bill is growing. Existing observability tools show you total token counts. They can't tell you:

- Which agent branch caused 80% of the cost
- Whether that retry storm on the DB tool wasted $0.50
- That two LLM calls sent nearly identical 12k-token contexts
- That an entire agent branch was abandoned and its work was never consumed

TRLLM fills this gap by walking the **causal DAG** of your pipeline execution and attributing every dollar to the root cause that triggered it.

## Quick Start

```bash
pip install -e ".[dev]"
```

### Auto-instrument OpenAI

```python
from openai import OpenAI
from trllm.cost_forensics import PricingRegistry, CostAnnotator, CausalCostRollup, WasteDetector
from trllm.cost_forensics.adapters.openai import InstrumentedOpenAI

client = InstrumentedOpenAI(OpenAI())

# Use client.chat.completions.create() as normal — calls are recorded automatically
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "What is 2+2?"}],
)

# Get the causal computation
comp = client.computation()

# Analyze
pricing = PricingRegistry.openai()
annotations = CostAnnotator(pricing).annotate(comp)
report = CausalCostRollup().rollup(comp, annotations)
waste = WasteDetector().detect(comp, annotations)
report.attach_waste(waste)

print(report.ascii_tree())
print(waste.summary())
```

### Auto-instrument Anthropic

```python
from anthropic import Anthropic
from trllm.cost_forensics.adapters.anthropic import InstrumentedAnthropic

client = InstrumentedAnthropic(Anthropic())
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    messages=[{"role": "user", "content": "What is 2+2?"}],
)

comp = client.computation()
# ... same analysis as above with PricingRegistry.anthropic()
```

### Auto-instrument Bedrock

```python
import boto3
from trllm.cost_forensics import PricingRegistry, CostAnnotator, CausalCostRollup
from trllm.cost_forensics.adapters.bedrock import InstrumentedBedrock

bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")
client = InstrumentedBedrock(bedrock)

response = client.converse(
    modelId="anthropic.claude-3-5-haiku-20241022-v1:0",
    messages=[{"role": "user", "content": [{"text": "What is 2+2?"}]}],
)

# Streaming works too
stream = client.converse_stream(modelId="meta.llama3-70b-instruct-v1:0", messages=[...])
for event in stream:
    ...  # events pass through, usage captured from metadata event

comp = client.computation()
pricing = PricingRegistry.bedrock()
# ... analyze as usual
```

### Manual instrumentation

If you're using a provider without an adapter (Ollama, etc.), record events manually:

```python
from pyrapide import Computation, Event

comp = Computation()
root = Event(name="user_request", payload={"prompt": "..."})
comp.record(root)

llm = Event(name="llm_call", payload={
    "model": "gpt-4o",
    "usage": {"input_tokens": 5000, "output_tokens": 500},
})
comp.record(llm, caused_by=[root])
# ... continue recording your pipeline
```

### Async & Streaming

All adapters have async variants and support `stream=True`:

```python
from openai import AsyncOpenAI
from trllm.cost_forensics.adapters.openai_async import AsyncInstrumentedOpenAI

client = AsyncInstrumentedOpenAI(AsyncOpenAI())
response = await client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello"}],
)

# Streaming — usage is captured when the stream completes
stream = client.chat.completions.create(model="gpt-4o", messages=[...], stream=True)
for chunk in stream:
    ...  # chunks pass through transparently
comp = client.computation()  # cost event recorded automatically
```

### LangChain

Drop in `TrllmCallbackHandler` to instrument any LangChain component — no code changes to your chains or agents:

```python
from trllm.cost_forensics.adapters.langchain import TrllmCallbackHandler

handler = TrllmCallbackHandler()
llm = ChatOpenAI(model="gpt-4o", callbacks=[handler])
llm.invoke("Hello")

comp = handler.computation()
# ... analyze as usual
```

## CLI

```bash
# Analyze a serialized computation
trllm-cost-forensics analyze comp.json --provider openai

# Fail CI if cost exceeds budget
trllm-cost-forensics analyze comp.json --budget 0.10

# JSON output for dashboards
trllm-cost-forensics analyze comp.json --json

# Compare before/after a deploy
trllm-cost-forensics diff before.json after.json
```

## Waste Detection

Four built-in patterns that catch real money being burned:

| Pattern | Severity | What it catches |
|---------|----------|-----------------|
| **RetryStorm** | high | Same tool called 3+ times under one root |
| **DeadEndToolCall** | medium | Tool result fetched but never consumed downstream |
| **RedundantContext** | low | Two LLM calls with <5% difference in input tokens |
| **AbandonedBranch** | high | Entire agent branch (depth >= 2) whose output was never used |

Custom patterns:

```python
from trllm.cost_forensics import WastePattern, WasteInstance

class MyPattern(WastePattern):
    name = "MyPattern"
    description = "Detects my anti-pattern"

    def detect(self, comp, annotations):
        return [WasteInstance(...)]

detector = WasteDetector(patterns=[MyPattern()])
```

## Budget Constraints

Integrate with PyRapide's constraint system for real-time alerts:

```python
from trllm.cost_forensics import BudgetExceeded, CostPerRootExceeded

# Fail if total > $1.00
constraint = BudgetExceeded(budget=1.00, pricing=pricing)
violations = constraint.check(comp)

# Fail if any single root subtree > $0.50
constraint = CostPerRootExceeded(per_root_budget=0.50, pricing=pricing)
```

## Cost Diffing

Compare costs across deploys:

```python
from trllm.cost_forensics import diff_reports

diff = diff_reports(before_report, after_report)
print(diff.summary())
# Cost diff: $0.0485 -> $0.0912 (+88.0%)
# Regressions (1):
#   ↑ llm_call: $0.0300 -> $0.0712 (+137.3%)
```

## Architecture

```
trllm/
    cost_forensics/
        pricing.py         # PricingRegistry with OpenAI + Anthropic tables
        annotator.py       # CostAnnotator — per-event cost calculation
        rollup.py          # CausalCostRollup — DAG cost attribution
        waste.py           # WasteDetector + 4 built-in patterns
        reports.py         # ForensicReport, WasteReport, ASCII tree
        diff.py            # CostDiff for before/after comparison
        constraints.py     # Budget constraints (PyRapide integration)
        cli.py             # CLI entry point
        adapters/
            openai.py          # Sync OpenAI adapter
            openai_async.py    # Async OpenAI adapter
            anthropic.py       # Sync Anthropic adapter
            anthropic_async.py # Async Anthropic adapter
            bedrock.py         # AWS Bedrock adapter (converse + converse_stream)
            streaming.py       # Stream wrappers (sync + async)
            langchain.py       # LangChain callback handler
```

## Testing

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Unit tests (143 tests, no external dependencies)
pytest tests/cost_forensics/ -v --ignore=tests/cost_forensics/test_live_ollama.py

# Coverage
pytest tests/cost_forensics/ --cov=trllm.cost_forensics --cov-report=term-missing --ignore=tests/cost_forensics/test_live_ollama.py

# Type checking
mypy trllm/cost_forensics/ --strict
```

### Live integration tests (Ollama)

The live test suite uses [Ollama](https://ollama.com/) because it is **free and runs locally** — no API keys, no token costs, no rate limits. This makes it practical to run real LLM calls in tests without incurring charges. The tests assign simulated prices to verify cost math works correctly with real, nondeterministic token counts.

Ollama is a testing convenience, not a target use case. In production, you'd use the OpenAI, Anthropic, or LangChain adapters with real pricing.

```bash
# Requires: Ollama running locally with qwen3:0.6b pulled
ollama pull qwen3:0.6b
pytest tests/cost_forensics/test_live_ollama.py -v

# Live demo (also requires Ollama + pip install openai)
pip install openai
python demo/live_query_demo.py
```

If Ollama is not running, the live tests skip automatically — CI only runs the 143 unit tests.

## License

MIT
