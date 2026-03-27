FROM python:3.12-slim

WORKDIR /app

# Install dependencies first for layer caching
COPY pyproject.toml README.md ./
COPY trllm/__init__.py trllm/__init__.py
RUN pip install --no-cache-dir .

# Copy full source
COPY . .
RUN pip install --no-cache-dir -e .

EXPOSE 8000

CMD ["uvicorn", "trllm.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
