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

import unittest
import unittest.mock
from typing import Optional

import google.genai.types as genai_types
from google.genai.models import AsyncModels, Models

from opentelemetry import context as context_api
from opentelemetry.util.genai.attributes import (
    SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY,
)

from ..common.base import TestCase as CommonTestCaseBase


def create_embed_response(
    values: Optional[list[float]] = None,
    billable_character_count: Optional[int] = None,
) -> genai_types.EmbedContentResponse:
    """Build a minimal EmbedContentResponse for testing."""
    if values is None:
        values = [0.1, 0.2, 0.3]
    embedding = genai_types.ContentEmbedding(values=values)
    metadata = None
    if billable_character_count is not None:
        metadata = genai_types.EmbedContentMetadata(
            billable_character_count=billable_character_count
        )
    return genai_types.EmbedContentResponse(
        embeddings=[embedding],
        metadata=metadata,
    )


class EmbedContentTestCase(CommonTestCaseBase):
    """Shared test infrastructure for embed_content (sync + async)."""

    def setUp(self):
        super().setUp()
        if self.__class__ == EmbedContentTestCase:
            raise unittest.SkipTest("Skipping testcase base.")
        self._embed_mock = None
        self._original_embed_content = Models.embed_content
        self._original_async_embed_content = AsyncModels.embed_content
        self._embed_response = create_embed_response()

    @property
    def embed_mock(self):
        if self._embed_mock is None:
            self._create_and_install_mocks()
        return self._embed_mock

    def configure_embed_response(self, **kwargs):
        self._embed_response = create_embed_response(**kwargs)
        self._create_and_install_mocks()

    def configure_embed_exception(self, exc):
        self._create_and_install_mocks(exc)

    def _create_and_install_mocks(self, exc=None):
        if self._embed_mock is not None:
            return
        self.reset_client()
        self.reset_instrumentation()
        mock = unittest.mock.MagicMock()
        if exc is not None:
            mock.side_effect = exc
        else:
            response = self._embed_response

            def _impl(*args, **kwargs):
                return response

            mock.side_effect = _impl
        self._embed_mock = mock

        async def _async_impl(*args, **kwargs):
            return mock(*args, **kwargs)

        Models.embed_content = mock
        AsyncModels.embed_content = _async_impl

    def tearDown(self):
        super().tearDown()
        Models.embed_content = self._original_embed_content
        AsyncModels.embed_content = self._original_async_embed_content


class EmbedContentSharedTests(EmbedContentTestCase):
    """Shared test cases executed for both sync and async embed_content."""

    def setUp(self):
        super().setUp()
        if self.__class__ == EmbedContentSharedTests:
            raise unittest.SkipTest("Skipping testcase base.")

    def embed_content(self, **kwargs):
        raise NotImplementedError("Subclass must implement embed_content()")

    @property
    def expected_code_function_name(self) -> str:
        raise NotImplementedError

    # ------------------------------------------------------------------ basic

    def test_instrumentation_does_not_break_core_functionality(self):
        self.configure_embed_response(values=[0.1, 0.2, 0.3])
        response = self.embed_content(
            model="gemini-embedding-001",
            contents="Hello world",
        )
        self.assertIsNotNone(response.embeddings)
        self.assertEqual(len(response.embeddings[0].values), 3)

    def test_generates_span(self):
        self.configure_embed_response()
        self.embed_content(
            model="gemini-embedding-001",
            contents="Hello world",
        )
        self.otel.assert_has_span_named("embeddings gemini-embedding-001")

    def test_model_reflected_in_span_name(self):
        self.configure_embed_response()
        self.embed_content(
            model="text-embedding-005",
            contents="Hello world",
        )
        self.otel.assert_has_span_named("embeddings text-embedding-005")

    # ----------------------------------------------------- span attributes

    def test_span_has_genai_system_attribute(self):
        self.configure_embed_response()
        self.embed_content(model="gemini-embedding-001", contents="hi")
        span = self.otel.get_span_named("embeddings gemini-embedding-001")
        self.assertIsNotNone(span)
        self.assertEqual(span.attributes.get("gen_ai.system"), "gemini")

    def test_span_has_operation_name_attribute(self):
        self.configure_embed_response()
        self.embed_content(model="gemini-embedding-001", contents="hi")
        span = self.otel.get_span_named("embeddings gemini-embedding-001")
        self.assertIsNotNone(span)
        self.assertEqual(
            span.attributes.get("gen_ai.operation.name"), "embeddings"
        )

    def test_span_has_request_model_attribute(self):
        self.configure_embed_response()
        self.embed_content(model="gemini-embedding-001", contents="hi")
        span = self.otel.get_span_named("embeddings gemini-embedding-001")
        self.assertIsNotNone(span)
        self.assertEqual(
            span.attributes.get("gen_ai.request.model"), "gemini-embedding-001"
        )

    def test_dimension_count_set_from_response(self):
        self.configure_embed_response(values=[0.1] * 768)
        self.embed_content(model="gemini-embedding-001", contents="hi")
        span = self.otel.get_span_named("embeddings gemini-embedding-001")
        self.assertIsNotNone(span)
        self.assertEqual(
            span.attributes.get("gen_ai.embeddings.dimension.count"), 768
        )

    def test_billable_character_count_vendor_attribute(self):
        self.configure_embed_response(
            values=[0.1, 0.2], billable_character_count=42
        )
        self.embed_content(model="gemini-embedding-001", contents="hi")
        span = self.otel.get_span_named("embeddings gemini-embedding-001")
        self.assertIsNotNone(span)
        self.assertEqual(
            span.attributes.get(
                "gen_ai.google.usage.billable_character_count"
            ),
            42,
        )

    def test_backend_vendor_attribute_gemini(self):
        self.configure_embed_response()
        self.embed_content(model="gemini-embedding-001", contents="hi")
        span = self.otel.get_span_named("embeddings gemini-embedding-001")
        self.assertIsNotNone(span)
        self.assertEqual(
            span.attributes.get("gen_ai.google.request.backend"), "gemini"
        )

    def test_backend_vendor_attribute_vertex(self):
        self.set_use_vertex(True)
        self.configure_embed_response()
        self.embed_content(model="gemini-embedding-001", contents="hi")
        span = self.otel.get_span_named("embeddings gemini-embedding-001")
        self.assertIsNotNone(span)
        self.assertEqual(
            span.attributes.get("gen_ai.google.request.backend"), "vertex_ai"
        )

    def test_span_has_provider_name_attribute(self):
        self.configure_embed_response()
        self.embed_content(model="gemini-embedding-001", contents="hi")
        span = self.otel.get_span_named("embeddings gemini-embedding-001")
        self.assertIsNotNone(span)
        self.assertEqual(span.attributes.get("gen_ai.provider.name"), "google")

    def test_span_has_framework_attribute(self):
        self.configure_embed_response()
        self.embed_content(model="gemini-embedding-001", contents="hi")
        span = self.otel.get_span_named("embeddings gemini-embedding-001")
        self.assertIsNotNone(span)
        self.assertEqual(
            span.attributes.get("gen_ai.framework"), "google-genai-sdk"
        )

    def test_span_has_code_function_name(self):
        self.configure_embed_response()
        self.embed_content(model="gemini-embedding-001", contents="hi")
        span = self.otel.get_span_named("embeddings gemini-embedding-001")
        self.assertIsNotNone(span)
        self.assertEqual(
            span.attributes.get("code.function.name"),
            self.expected_code_function_name,
        )

    # --------------------------------------------------------- metrics

    def test_records_duration_metric(self):
        self.configure_embed_response()
        self.embed_content(model="gemini-embedding-001", contents="hi")
        self.otel.assert_has_metrics_data_named(
            "gen_ai.client.operation.duration"
        )

    # --------------------------------------------------------- content capture

    def test_input_texts_captured_when_content_enabled(self):
        self.configure_embed_response()
        self.embed_content(
            model="gemini-embedding-001", contents="Capture this text"
        )
        span = self.otel.get_span_named("embeddings gemini-embedding-001")
        self.assertIsNotNone(span)
        input_texts = span.attributes.get("gen_ai.embeddings.input.texts")
        self.assertIsNotNone(input_texts)
        self.assertIn("Capture this text", input_texts)

    def test_input_texts_not_captured_when_content_disabled(self):
        from unittest.mock import patch

        self.configure_embed_response()
        with patch.dict(
            "os.environ",
            {"OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT": "false"},
        ):
            self.embed_content(
                model="gemini-embedding-001", contents="Do not capture"
            )
        span = self.otel.get_span_named("embeddings gemini-embedding-001")
        self.assertIsNotNone(span)
        self.assertIsNone(span.attributes.get("gen_ai.embeddings.input.texts"))

    # ------------------------------------------------------------ error path

    def test_error_span_on_exception(self):
        self.configure_embed_exception(RuntimeError("API error"))
        with self.assertRaises(RuntimeError):
            self.embed_content(
                model="gemini-embedding-001", contents="anything"
            )
        span = self.otel.get_span_named("embeddings gemini-embedding-001")
        self.assertIsNotNone(span)
        from opentelemetry.trace import StatusCode

        self.assertEqual(span.status.status_code, StatusCode.ERROR)
        self.assertEqual(span.attributes.get("error.type"), "RuntimeError")

    # ---------------------------------------------------------- suppression

    def test_suppression_key_bypasses_instrumentation(self):
        self.configure_embed_response()
        token = context_api.attach(
            context_api.set_value(
                SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY, True
            )
        )
        try:
            self.embed_content(
                model="gemini-embedding-001", contents="suppressed"
            )
        finally:
            context_api.detach(token)
        self.otel.assert_does_not_have_span_named(
            "embeddings gemini-embedding-001"
        )
