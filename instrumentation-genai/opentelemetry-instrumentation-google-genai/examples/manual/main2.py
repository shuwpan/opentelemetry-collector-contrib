# pylint: skip-file
# ruff: noqa: E402
"""
Per-feature verification app for google-genai instrumentation.

Each function exercises one instrumented code path. Comment out calls
in main() to test individual scenarios.

Run:
    python main2.py
"""

import os

from dotenv import load_dotenv

load_dotenv()

import google.genai
from google.genai.types import Content, GenerateContentConfig, Part

from opentelemetry import _logs as otel_logs
from opentelemetry import metrics as otel_metrics
from opentelemetry import trace as otel_trace
from opentelemetry.exporter.otlp.proto.grpc._log_exporter import (
    OTLPLogExporter,
)
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
    OTLPMetricExporter,
)
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
    OTLPSpanExporter,
)
from opentelemetry.instrumentation.google_genai import (
    GoogleGenAiSdkInstrumentor,
)
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor


def setup_otel_tracing():
    otel_trace.set_tracer_provider(TracerProvider())
    otel_trace.get_tracer_provider().add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter())
    )


def setup_otel_logs():
    otel_logs.set_logger_provider(LoggerProvider())
    otel_logs.get_logger_provider().add_log_record_processor(
        BatchLogRecordProcessor(OTLPLogExporter())
    )


def setup_otel_metrics():
    meter_provider = MeterProvider(
        metric_readers=[
            PeriodicExportingMetricReader(
                OTLPMetricExporter(),
            ),
        ]
    )
    otel_metrics.set_meter_provider(meter_provider)


def setup_opentelemetry():
    setup_otel_tracing()
    setup_otel_logs()
    setup_otel_metrics()


def instrument_google_genai():
    GoogleGenAiSdkInstrumentor().instrument()


MODEL = os.getenv("MODEL", "gemini-2.5-flash")


def _client() -> google.genai.Client:
    return google.genai.Client()


# ---------------------------------------------------------------------------
# 1. Simple sync generate_content
# ---------------------------------------------------------------------------
def simple():
    print("\n=== 1. Simple generate_content ===")
    client = _client()
    response = client.models.generate_content(
        model=MODEL,
        contents="Explain the concept of observability in one paragraph.",
    )
    print(response.text)


# ---------------------------------------------------------------------------
# 2. System instructions + GenerateContentConfig attributes
# ---------------------------------------------------------------------------
def system_config():
    print("\n=== 2. System instructions + config ===")
    client = _client()
    response = client.models.generate_content(
        model=MODEL,
        contents="What are the three pillars of observability?",
        config=GenerateContentConfig(
            system_instruction="You are a concise SRE expert. Answer in bullet points.",
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            max_output_tokens=256,
            stop_sequences=["\n\n\n"],
        ),
    )
    print(response.text)


# ---------------------------------------------------------------------------
# 3. Sync streaming
# ---------------------------------------------------------------------------
def streaming():
    print("\n=== 3. Streaming generate_content ===")
    client = _client()
    stream = client.models.generate_content_stream(
        model=MODEL,
        contents="Write a short poem about distributed tracing.",
    )
    for chunk in stream:
        print(chunk.text, end="", flush=True)
    print()


# ---------------------------------------------------------------------------
# 4. Async generate_content
# ---------------------------------------------------------------------------
async def async_basic():
    print("\n=== 4. Async generate_content ===")
    client = _client()
    response = await client.aio.models.generate_content(
        model=MODEL,
        contents="What is OpenTelemetry in one sentence?",
    )
    print(response.text)


# ---------------------------------------------------------------------------
# 5. Async streaming
# ---------------------------------------------------------------------------
async def async_streaming():
    print("\n=== 5. Async streaming generate_content ===")
    client = _client()
    stream = await client.aio.models.generate_content_stream(
        model=MODEL,
        contents="List three benefits of structured logging.",
    )
    async for chunk in stream:
        print(chunk.text, end="", flush=True)
    print()


# ---------------------------------------------------------------------------
# 6. Automatic function calling (execute_tool spans)
# ---------------------------------------------------------------------------
def get_current_weather(location: str) -> dict:
    """Get the current weather for a location (stub)."""
    weather_data = {
        "san francisco": {"temperature": "18°C", "condition": "Foggy"},
        "new york": {"temperature": "25°C", "condition": "Sunny"},
        "london": {"temperature": "14°C", "condition": "Rainy"},
    }
    return weather_data.get(
        location.lower(),
        {"temperature": "Unknown", "condition": "Unknown"},
    )


def function_call():
    print("\n=== 6. Function calling (automatic) ===")
    client = _client()
    response = client.models.generate_content(
        model=MODEL,
        contents="What is the weather in San Francisco and London?",
        config=GenerateContentConfig(
            tools=[get_current_weather],
        ),
    )
    print(response.text)


# ---------------------------------------------------------------------------
# 7. Multi-turn conversation
# ---------------------------------------------------------------------------
def multi_turn():
    print("\n=== 7. Multi-turn conversation ===")
    client = _client()
    history = [
        Content(
            role="user",
            parts=[Part.from_text(text="Hi! My name is Alex.")],
        ),
        Content(
            role="model",
            parts=[
                Part.from_text(text="Hello Alex! How can I help you today?")
            ],
        ),
        Content(
            role="user",
            parts=[Part.from_text(text="What was my name again?")],
        ),
    ]
    response = client.models.generate_content(
        model=MODEL,
        contents=history,
    )
    print(response.text)


# ---------------------------------------------------------------------------
# 8. Embeddings
# ---------------------------------------------------------------------------
def embeddings():
    print("\n=== 8. Embeddings ===")
    client = _client()
    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents="Observability depends on traces, metrics, and logs.",
    )
    for i, emb in enumerate(response.embeddings):
        print(f"  Embedding {i}: {len(emb.values)} dimensions")


# ---------------------------------------------------------------------------
# main — comment out scenarios you don't need
# ---------------------------------------------------------------------------
def main():
    setup_opentelemetry()
    instrument_google_genai()

    # simple()  # PR1
    # system_config()  # PR1
    # streaming()  # PR2
    # asyncio.run(async_basic())  # PR1
    # asyncio.run(async_streaming())  # PR2
    # function_call()  # PR3
    # multi_turn()  # PR1
    embeddings()  # PR4


if __name__ == "__main__":
    main()
