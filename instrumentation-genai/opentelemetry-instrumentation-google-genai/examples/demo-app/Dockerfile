FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY instrumentation-genai/opentelemetry-instrumentation-google-genai/examples/demo-app/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the local packages source code
COPY util/opentelemetry-util-genai /tmp/opentelemetry-util-genai
COPY util/opentelemetry-util-genai-evals /tmp/opentelemetry-util-genai-evals
COPY util/opentelemetry-util-genai-evals-deepeval /tmp/opentelemetry-util-genai-evals-deepeval
COPY util/opentelemetry-util-genai-emitters-splunk /tmp/opentelemetry-util-genai-emitters-splunk
COPY instrumentation-genai/opentelemetry-instrumentation-google-genai /tmp/opentelemetry-instrumentation-google-genai

# Install local packages with deps first, then force reinstall without deps
# Install extension packages first, core util-genai last (namespace ordering)
RUN pip install /tmp/opentelemetry-util-genai-evals \
    /tmp/opentelemetry-util-genai-evals-deepeval \
    /tmp/opentelemetry-util-genai-emitters-splunk \
    /tmp/opentelemetry-instrumentation-google-genai
RUN pip install /tmp/opentelemetry-util-genai
RUN pip install --force-reinstall --no-deps \
    /tmp/opentelemetry-util-genai-evals \
    /tmp/opentelemetry-util-genai-evals-deepeval \
    /tmp/opentelemetry-util-genai-emitters-splunk \
    /tmp/opentelemetry-instrumentation-google-genai \
    && pip install --force-reinstall --no-deps /tmp/opentelemetry-util-genai

# Copy application code
COPY instrumentation-genai/opentelemetry-instrumentation-google-genai/examples/demo-app/main.py .

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Run the application
ENTRYPOINT ["python", "main.py"]
