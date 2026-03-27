# Contributing to TRLLM

Thanks for your interest in contributing to TRLLM!

## Getting Started

```bash
git clone https://github.com/iamdevinhill/trllm.git
cd trllm
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest tests/ -v
```

All tests use mocked Ollama calls — no running Ollama instance required.

## Running the Demo

You'll need [Ollama](https://ollama.ai) running locally:

```bash
ollama pull qwen3:30b
ollama pull qwen3-embedding:0.6b
python demo/demo_pipeline.py
```

## Code Style

This project uses [Ruff](https://docs.astral.sh/ruff/) for linting:

```bash
ruff check trllm/ tests/
```

## Submitting Changes

1. Fork the repo and create a branch from `main`
2. Make your changes
3. Add or update tests as needed
4. Ensure all tests pass (`pytest tests/ -v`)
5. Run the linter (`ruff check`)
6. Open a pull request

## Project Structure

- `trllm/` — core library (events, graph builder, entailment linker, constraints)
- `trllm/api/` — FastAPI server
- `trllm/adapters/` — LLM provider adapters (currently Ollama)
- `trllm/visualization/` — rendering wrappers
- `demo/` — end-to-end demo pipeline
- `dashboard/` — Mermaid.js DAG viewer
- `tests/` — test suite

## Reporting Issues

Open an issue at https://github.com/iamdevinhill/trllm/issues with:
- What you expected to happen
- What actually happened
- Steps to reproduce

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
