# Copyright The OpenTelemetry Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""End-to-end tests for embed_content instrumentation.

These tests use VCR cassettes to replay recorded Gemini API interactions.
To record new cassettes, run with ``--vcr-record=all`` and valid credentials.
"""

import asyncio
import json
import os
import subprocess

import google.auth
import google.auth.credentials
import google.genai
import pytest
import yaml
from google.genai import types
from vcr.record_mode import RecordMode

from opentelemetry.instrumentation.google_genai import (
    GoogleGenAiSdkInstrumentor,
)
from opentelemetry.util.genai.environment_variables import (
    OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT,
)

from ..common.auth import FakeCredentials
from ..common.otel_mocker import OTelMocker

_FAKE_PROJECT = "test-project"
_FAKE_LOCATION = "test-location"
_FAKE_API_KEY = "test-api-key"
_DEFAULT_REAL_LOCATION = "us-central1"


# ---------------------------------------------------------------------------
# Helper utilities (shared with generate_content/test_e2e.py)
# ---------------------------------------------------------------------------


def _get_project_from_env():
    return (
        os.getenv("GCLOUD_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT") or ""
    )


def _get_project_from_gcloud_cli():
    try:
        gcloud_call_result = subprocess.run(
            "gcloud config get project",
            shell=True,
            capture_output=True,
            check=True,
        )
    except subprocess.CalledProcessError:
        return None
    gcloud_output = gcloud_call_result.stdout.decode()
    return gcloud_output.strip()


def _get_project_from_credentials():
    _, from_creds = google.auth.default()
    return from_creds


def _get_real_project():
    from_env = _get_project_from_env()
    if from_env:
        return from_env
    from_cli = _get_project_from_gcloud_cli()
    if from_cli:
        return from_cli
    return _get_project_from_credentials()


def _get_real_location():
    return (
        os.getenv("GCLOUD_LOCATION")
        or os.getenv("GOOGLE_CLOUD_LOCATION")
        or _DEFAULT_REAL_LOCATION
    )


def _should_redact_header(header_key):
    if header_key.startswith("x-goog"):
        return True
    if header_key.startswith("sec-goog"):
        return True
    if header_key in ["server", "server-timing"]:
        return True
    return False


def _redact_headers(headers):
    for header_key in headers:
        if _should_redact_header(header_key.lower()):
            headers[header_key] = "<REDACTED>"


def _before_record_request(request):
    if request.method:
        request.method = request.method.upper()
    if request.headers:
        _redact_headers(request.headers)
    uri = request.uri
    project = _get_project_from_env()
    if project:
        uri = uri.replace(f"projects/{project}", f"projects/{_FAKE_PROJECT}")
    location = _get_real_location()
    if location:
        uri = uri.replace(
            f"locations/{location}", f"locations/{_FAKE_LOCATION}"
        )
        uri = uri.replace(
            f"//{location}-aiplatform.googleapis.com",
            f"//{_FAKE_LOCATION}-aiplatform.googleapis.com",
        )
    request.uri = uri
    return request


def _before_record_response(response):
    if hasattr(response, "headers") and response.headers:
        _redact_headers(response.headers)
    return response


# ---------------------------------------------------------------------------
# VCR configuration
# ---------------------------------------------------------------------------


@pytest.fixture(name="vcr_config", scope="module")
def fixture_vcr_config():
    return {
        "filter_query_parameters": [
            "key",
            "apiKey",
            "quotaUser",
            "userProject",
            "token",
            "access_token",
            "accessToken",
            "refesh_token",
            "refreshToken",
            "authuser",
            "bearer",
            "bearer_token",
            "bearerToken",
            "userIp",
        ],
        "filter_post_data_parameters": ["apikey", "api_key", "key"],
        "filter_headers": [
            "x-goog-api-key",
            "authorization",
            "server",
            "Server",
            "Server-Timing",
            "Date",
        ],
        "before_record_request": _before_record_request,
        "before_record_response": _before_record_response,
        "ignore_hosts": [
            "oauth2.googleapis.com",
            "iam.googleapis.com",
        ],
    }


class _LiteralBlockScalar(str):
    """Formats the string as a literal block scalar, preserving whitespace and
    without interpreting escape characters"""


def _literal_block_scalar_presenter(dumper, data):
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")


@pytest.fixture(
    name="internal_setup_yaml_pretty_formatting", scope="module", autouse=True
)
def fixture_setup_yaml_pretty_formatting():
    yaml.add_representer(_LiteralBlockScalar, _literal_block_scalar_presenter)


def _process_string_value(string_value):
    try:
        json_data = json.loads(string_value)
        return _LiteralBlockScalar(json.dumps(json_data, indent=2))
    except (ValueError, TypeError):
        if len(string_value) > 80:
            return _LiteralBlockScalar(string_value)
    return string_value


def _convert_body_to_literal(cassette_dict):
    for interaction in cassette_dict.get("interactions", []):
        for key in ["request", "response"]:
            body = interaction.get(key, {}).get("body")
            if isinstance(body, dict) and isinstance(body.get("string"), str):
                body["string"] = _process_string_value(body["string"])
            elif isinstance(body, str):
                interaction[key]["body"] = _process_string_value(body)
    return cassette_dict


class _PrettyPrintJSONBody:
    @staticmethod
    def serialize(cassette_dict):
        cassette_dict = _convert_body_to_literal(cassette_dict)
        return yaml.dump(
            cassette_dict, default_flow_style=False, allow_unicode=True
        )

    @staticmethod
    def deserialize(cassette_string):
        return yaml.load(cassette_string, Loader=yaml.Loader)


@pytest.fixture(name="fully_initialized_vcr", scope="module", autouse=True)
def setup_vcr(vcr):
    vcr.register_serializer("yaml", _PrettyPrintJSONBody)
    vcr.serializer = "yaml"
    return vcr


# ---------------------------------------------------------------------------
# Instrumentation and OTel fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(name="instrumentor")
def fixture_instrumentor():
    return GoogleGenAiSdkInstrumentor()


@pytest.fixture(name="internal_instrumentation_setup", autouse=True)
def fixture_setup_instrumentation(instrumentor):
    instrumentor.instrument()
    yield
    instrumentor.uninstrument()


@pytest.fixture(name="otel_mocker", autouse=True)
def fixture_otel_mocker():
    result = OTelMocker()
    result.install()
    yield result
    result.uninstall()


@pytest.fixture(
    name="setup_content_recording",
    autouse=True,
    params=[
        pytest.param("logcontent", id="content"),
        pytest.param("excludecontent", id="nocontent"),
    ],
)
def fixture_setup_content_recording(request):
    enabled = request.param == "logcontent"
    os.environ[OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT] = str(
        enabled
    )
    yield


# ---------------------------------------------------------------------------
# Auth / client fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(name="vcr_record_mode")
def fixture_vcr_record_mode(vcr):
    return vcr.record_mode


@pytest.fixture(name="in_replay_mode")
def fixture_in_replay_mode(vcr_record_mode):
    return vcr_record_mode == RecordMode.NONE


@pytest.fixture(name="gcloud_project", autouse=True)
def fixture_gcloud_project(in_replay_mode):
    if in_replay_mode:
        return _FAKE_PROJECT
    result = _get_real_project()
    for env_var in ["GCLOUD_PROJECT", "GOOGLE_CLOUD_PROJECT"]:
        os.environ[env_var] = result
    return result


@pytest.fixture(name="gcloud_location")
def fixture_gcloud_location(in_replay_mode):
    if in_replay_mode:
        return _FAKE_LOCATION
    return _get_real_location()


@pytest.fixture(name="gcloud_credentials")
def fixture_gcloud_credentials(in_replay_mode):
    if in_replay_mode:
        return FakeCredentials()
    creds, _ = google.auth.default()
    return google.auth.credentials.with_scopes_if_required(
        creds, ["https://www.googleapis.com/auth/cloud-platform"]
    )


@pytest.fixture(name="gemini_api_key")
def fixture_gemini_api_key(in_replay_mode):
    if in_replay_mode:
        return _FAKE_API_KEY
    return os.getenv("GEMINI_API_KEY")


@pytest.fixture(name="gcloud_api_key", autouse=True)
def fixture_gcloud_api_key(gemini_api_key):
    if "GOOGLE_API_KEY" not in os.environ:
        os.environ["GOOGLE_API_KEY"] = gemini_api_key
    return os.getenv("GOOGLE_API_KEY")


@pytest.fixture(name="vertex_client_factory")
def fixture_vertex_client_factory(
    gcloud_project, gcloud_location, gcloud_credentials
):
    def _factory():
        return google.genai.Client(
            vertexai=True,
            project=gcloud_project,
            location=gcloud_location,
            credentials=gcloud_credentials,
            http_options=types.HttpOptions(
                headers={"accept-encoding": "identity"}
            ),
        )

    return _factory


@pytest.fixture(name="nonvertex_client_factory")
def fixture_nonvertex_client_factory(gemini_api_key):
    def _factory():
        return google.genai.Client(
            api_key=gemini_api_key,
            vertexai=False,
            http_options=types.HttpOptions(
                headers={"accept-encoding": "identity"}
            ),
        )

    return _factory


@pytest.fixture(
    name="genai_sdk_backend", params=[pytest.param("vertexaiapi", id="vertex")]
)
def fixture_genai_sdk_backend(request):
    return request.param


@pytest.fixture(name="use_vertex", autouse=True)
def fixture_use_vertex(genai_sdk_backend):
    result = bool(genai_sdk_backend == "vertexaiapi")
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "1" if result else "0"
    return result


@pytest.fixture(name="client")
def fixture_client(
    vertex_client_factory, nonvertex_client_factory, use_vertex
):
    if use_vertex:
        return vertex_client_factory()
    return nonvertex_client_factory()


# ---------------------------------------------------------------------------
# Embed-specific fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(name="is_async", params=["sync", "async"])
def fixture_is_async(request):
    return request.param == "async"


@pytest.fixture(
    name="model",
    params=[pytest.param("text-embedding-005", id="emb005")],
)
def fixture_model(request):
    return request.param


@pytest.fixture(name="embed_content")
def fixture_embed_content(client, is_async):
    def _sync_impl(*args, **kwargs):
        return client.models.embed_content(*args, **kwargs)

    def _async_impl(*args, **kwargs):
        return asyncio.run(client.aio.models.embed_content(*args, **kwargs))

    if is_async:
        return _async_impl
    return _sync_impl


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.vcr
def test_embed_content(embed_content, model, otel_mocker):
    """Basic embed_content produces a valid response and a span."""
    response = embed_content(
        model=model, contents="What is the meaning of life?"
    )
    assert response is not None
    assert response.embeddings is not None
    assert len(response.embeddings) > 0
    assert len(response.embeddings[0].values) > 0
    otel_mocker.assert_has_span_named(f"embeddings {model}")


@pytest.mark.vcr
def test_embed_content_batch(embed_content, model, otel_mocker):
    """Batch embed_content with multiple texts."""
    response = embed_content(
        model=model,
        contents=[
            "First text to embed",
            "Second text to embed",
        ],
    )
    assert response is not None
    assert response.embeddings is not None
    assert len(response.embeddings) == 2
    otel_mocker.assert_has_span_named(f"embeddings {model}")


@pytest.mark.vcr
def test_embed_content_with_output_dimensionality(
    embed_content, model, otel_mocker
):
    """embed_content with output_dimensionality config."""
    response = embed_content(
        model=model,
        contents="Test with custom dimensions",
        config=types.EmbedContentConfig(output_dimensionality=256),
    )
    assert response is not None
    assert response.embeddings is not None
    assert len(response.embeddings[0].values) == 256

    span = otel_mocker.get_span_named(f"embeddings {model}")
    assert span is not None
    assert span.attributes.get("gen_ai.embeddings.dimension.count") == 256
    assert (
        span.attributes.get("gen_ai.google.request.output_dimensionality")
        == 256
    )


@pytest.mark.vcr
def test_embed_content_span_attributes(embed_content, model, otel_mocker):
    """Verify semconv attributes on the span."""
    response = embed_content(model=model, contents="Hello embeddings")
    assert response is not None

    span = otel_mocker.get_span_named(f"embeddings {model}")
    assert span is not None
    assert span.attributes.get("gen_ai.operation.name") == "embeddings"
    assert span.attributes.get("gen_ai.request.model") == model
    assert span.attributes.get("gen_ai.system") in ("gemini", "vertex_ai")
    assert span.attributes.get("gen_ai.provider.name") == "google"
    assert span.attributes.get("gen_ai.framework") == "google-genai-sdk"
    assert span.attributes.get("gen_ai.embeddings.dimension.count") == len(
        response.embeddings[0].values
    )

    otel_mocker.assert_has_metrics_data_named(
        "gen_ai.client.operation.duration"
    )
