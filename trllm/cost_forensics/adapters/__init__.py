"""SDK adapters for auto-instrumentation.

Sync adapters:
    - :class:`~.openai.InstrumentedOpenAI`
    - :class:`~.anthropic.InstrumentedAnthropic`
    - :class:`~.bedrock.InstrumentedBedrock`

Async adapters:
    - :class:`~.openai_async.AsyncInstrumentedOpenAI`
    - :class:`~.anthropic_async.AsyncInstrumentedAnthropic`

Framework adapters:
    - :class:`~.langchain.TrllmCallbackHandler`

All adapters support streaming (``stream=True``) — token usage is
recorded once the stream completes.
"""
