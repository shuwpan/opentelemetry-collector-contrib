# Google GenAI Travel Planner — Demo App

A multi-step travel planner using the Google GenAI SDK (`google-genai`) with
automatic OpenTelemetry instrumentation via `GoogleGenAiSdkInstrumentor`.

## Overview

The app runs three steps, each exercising a different instrumented code path:

1. **Embeddings** — embeds a user query and five destination descriptions with
   `embed_content`, then ranks destinations by cosine similarity.
2. **Tool calling** — calls `generate_content` with a `get_weather` tool
   (automatic function calling). The SDK executes the tool and the
   instrumentation creates an `execute_tool` span.
3. **Streaming** — calls `generate_content_stream` to produce a polished
   travel summary, demonstrating streaming-specific telemetry
   (time-to-first-chunk).

## Prerequisites

- Python 3.10+
- GCP project with Vertex AI API enabled, **or** a Gemini API key
- Application Default Credentials (ADC) or a service-account JSON for Vertex AI

## Setup

```bash
cd instrumentation-genai/opentelemetry-instrumentation-google-genai/examples/demo-app
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create environment configuration:

```bash
cp .env.example .env
# Edit .env with your credentials and OTLP endpoint
```

### Required Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GOOGLE_API_KEY` | Gemini Developer API key | — |
| `GOOGLE_CLOUD_PROJECT` | GCP project (Vertex AI) | — |
| `GOOGLE_CLOUD_LOCATION` | GCP region | `us-central1` |

### OpenTelemetry Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `OTEL_EXPORTER_OTLP_ENDPOINT` | OTLP collector endpoint | `http://localhost:4317` |
| `OTEL_EXPORTER_OTLP_PROTOCOL` | OTLP protocol | `grpc` |
| `OTEL_SERVICE_NAME` | Service name in traces | `google-genai-travel-planner` |
| `OTEL_INSTRUMENTATION_GENAI_EMITTERS` | Emitter selection | `span_metric_event` |
| `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT` | Capture prompt/response content | `true` |

## Running

```bash
source .venv/bin/activate
python main.py                         # default query
python main.py --query "adventure trip with hiking and northern lights"
```

## Sample Output

```
============================================================
  Google GenAI Travel Planner
  Backend : Vertex AI
  Model   : gemini-2.5-flash
  OTLP    : http://localhost:4317
============================================================

--- Step 1: Finding best destination (embeddings) ---
  Query: "relaxing beach vacation with temples and wellness"
  Query embedding: 3072 dimensions
  Rankings:
    1. Bali, Indonesia (0.805) <-- best match
    2. Tokyo, Japan (0.606)
    3. Marrakech, Morocco (0.582)
    4. Reykjavik, Iceland (0.551)
    5. Paris, France (0.526)

--- Step 2: Planning trip (tool calling) ---
  Destination: Bali, Indonesia
  Origin: Los Angeles
  Calling model with tools...
    [TOOL CALLED] get_weather(Bali, Indonesia)
  Plan received (480 chars)

--- Step 3: Streaming travel summary ---
  Bali is currently enjoying sunny weather with a warm 28 degrees...

------------------------------------------------------------
Flushing telemetry...
Done. Check your collector / backend for traces, logs, and metrics.
```

## Sample Telemetry

Sample output from a single run with
`OTEL_INSTRUMENTATION_GENAI_EMITTERS=span_metric_event`.

### Traces

The run produces **4 traces** containing **5 spans**:

```
Span Name                              Latency   Tokens (In/Out)  Notes
─────────────────────────────────────  ────────  ───────────────  ─────────────────────────────────
embeddings gemini-embedding-001          536ms        8 / —       query embedding (3072 dims)
embeddings gemini-embedding-001          456ms       65 / —       5 destinations batch
generate_content gemini-2.5-flash       3630ms      336 / 96      tool calling + final answer
  └─ execute_tool get_weather              0ms        — / —       child span (tool execution)
generate_content gemini-2.5-flash       3447ms       70 / 64      streaming summary
```

Key span attributes on `generate_content` spans:

| Attribute | Example Value |
|-----------|---------------|
| `gen_ai.operation.name` | `generate_content` |
| `gen_ai.request.model` | `gemini-2.5-flash` |
| `gen_ai.response.model` | `gemini-2.5-flash` |
| `gen_ai.system` | `vertex_ai` |
| `gen_ai.framework` | `google-genai-sdk` |
| `gen_ai.provider.name` | `google` |
| `gen_ai.usage.input_tokens` | `336` |
| `gen_ai.usage.output_tokens` | `96` |
| `gen_ai.response.finish_reasons` | `('stop',)` |
| `gen_ai.input.messages` | Full prompt JSON (when content capture enabled) |
| `gen_ai.output.messages` | Full response JSON (when content capture enabled) |
| `gen_ai.tool.definitions` | Tool schema JSON (when `CAPTURE_TOOL_DEFINITIONS=true`) |

Key span attributes on `execute_tool` span:

| Attribute | Example Value |
|-----------|---------------|
| `gen_ai.operation.name` | `execute_tool` |
| `gen_ai.tool.name` | `get_weather` |
| `gen_ai.tool.type` | `function` |
| `gen_ai.tool.description` | `Get the current weather forecast for a city.` |

Streaming spans additionally include:

| Attribute | Example Value |
|-----------|---------------|
| `gen_ai.request.stream` | `True` |
| `gen_ai.response.time_to_first_chunk` | `3.21` (seconds) |

### Metrics

| Metric | Unit | Description |
|--------|------|-------------|
| `gen_ai.client.token.usage` | `token` | Token count by type (input/output) |
| `gen_ai.client.operation.duration` | `s` | End-to-end operation duration |
| `gen_ai.client.operation.time_to_first_chunk` | `s` | Time to first chunk (streaming only) |

Dimensions: `gen_ai.token.type`, `gen_ai.request.model`, `gen_ai.response.model`,
`gen_ai.operation.name`, `gen_ai.framework`, `gen_ai.provider.name`.

### Log Events

When `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=SPAN_AND_EVENT`:

- **Event**: `gen_ai.client.inference.operation.details`
- **Body**: full `gen_ai.input.messages` and `gen_ai.output.messages` as JSON
- **Correlation**: linked to the parent span via `trace_id` / `span_id`

One log event is emitted per `generate_content` / `generate_content_stream` span.

## Project Structure

```
demo-app/
├── main.py              # Travel planner with manual OTel setup
├── requirements.txt     # Pinned dependencies
├── .env.example         # Environment variable template
├── Dockerfile           # Container build
├── cronjob.yaml         # Kubernetes CronJob spec
└── README.md            # This file
```

## Kubernetes Deployment

```bash
# Build and push
docker build --platform linux/amd64 -t <your-registry>/otel-google-genai-demo-app:latest .
docker push <your-registry>/otel-google-genai-demo-app:latest

# Deploy
kubectl apply -f cronjob.yaml

# Trigger a manual run
kubectl create job --from=cronjob/otel-google-genai-demo-app manual-test -n google-genai-agent
```

## Related Documentation

- [Google GenAI Python SDK](https://googleapis.github.io/python-genai/)
- [Splunk Observability for AI](https://help.splunk.com/en/splunk-observability-cloud/observability-for-ai/set-up-observability-for-ai)
- [OpenTelemetry GenAI Semantic Conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
