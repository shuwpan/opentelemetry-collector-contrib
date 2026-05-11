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
from unittest.mock import patch

from opentelemetry import context as context_api
from opentelemetry.instrumentation._semconv import (
    _OpenTelemetrySemanticConventionStability,
    _OpenTelemetryStabilitySignalType,
    _StabilityMode,
)
from opentelemetry.instrumentation.google_genai import (
    GENERATE_CONTENT_EXTRA_ATTRIBUTES_CONTEXT_KEY,
)

from .base import TestCase
from .util import create_response


class StreamingTestCase(TestCase):
    # The "setUp" function is defined by "unittest.TestCase" and thus
    # this name must be used. Uncertain why pylint doesn't seem to
    # recognize that this is a unit test class for which this is inherited.
    def setUp(self):  # pylint: disable=invalid-name
        super().setUp()
        if self.__class__ == StreamingTestCase:
            raise unittest.SkipTest("Skipping testcase base.")

    def generate_content(self, *args, **kwargs):
        raise NotImplementedError("Must implement 'generate_content'.")

    @property
    def expected_function_name(self):
        raise NotImplementedError("Must implement 'expected_function_name'.")

    def test_span_has_request_stream_attribute(self):
        self.configure_valid_response(text="hi")
        self.generate_content(model="gemini-2.0-flash", contents="hi")
        span = self.otel.get_span_named("generate_content gemini-2.0-flash")
        self.assertEqual(span.attributes.get("gen_ai.request.stream"), True)

    def test_span_has_time_to_first_chunk(self):
        self.configure_valid_response(text="hello")
        self.generate_content(model="gemini-2.0-flash", contents="hi")
        span = self.otel.get_span_named("generate_content gemini-2.0-flash")
        ttfc = span.attributes.get("gen_ai.response.time_to_first_chunk")
        self.assertIsNotNone(ttfc)
        self.assertIsInstance(ttfc, float)
        self.assertGreaterEqual(ttfc, 0.0)

    def test_instrumentation_does_not_break_core_functionality(self):
        self.configure_valid_response(text="Yep, it works!")
        responses = self.generate_content(
            model="gemini-2.0-flash", contents="Does this work?"
        )
        self.assertEqual(len(responses), 1)
        response = responses[0]
        self.assertEqual(response.text, "Yep, it works!")

    def test_generated_span_has_extra_genai_attributes(self):
        self.configure_valid_response(text="Yep, it works!")
        tok = context_api.attach(
            context_api.set_value(
                GENERATE_CONTENT_EXTRA_ATTRIBUTES_CONTEXT_KEY,
                {"custom_extra_attribute_key": "extra_attribute_value"},
            )
        )
        try:
            self.generate_content(
                model="gemini-2.0-flash", contents="Does this work?"
            )
            self.otel.assert_has_span_named(
                "generate_content gemini-2.0-flash"
            )
            span = self.otel.get_span_named(
                "generate_content gemini-2.0-flash"
            )
            self.assertEqual(
                span.attributes["custom_extra_attribute_key"],
                "extra_attribute_value",
            )
        finally:
            context_api.detach(tok)

    def test_handles_multiple_responses(self):
        self.configure_valid_response(text="First response")
        self.configure_valid_response(text="Second response")
        responses = self.generate_content(
            model="gemini-2.0-flash", contents="Does this work?"
        )
        self.assertEqual(len(responses), 2)
        self.assertEqual(responses[0].text, "First response")
        self.assertEqual(responses[1].text, "Second response")
        self.otel.assert_has_span_named("generate_content gemini-2.0-flash")
        self.otel.assert_has_event_named(
            "gen_ai.client.inference.operation.details"
        )
        # Verify that streaming deltas were merged into a single output
        # message (both chunks share candidate index 0).
        event = self.otel.get_event_named(
            "gen_ai.client.inference.operation.details"
        )
        body = event.body or {}
        output_messages = body.get("gen_ai.output.messages", [])
        self.assertEqual(len(output_messages), 1)

    def test_includes_token_counts_in_span_not_aggregated_from_responses(self):
        # Tokens should not be aggregated in streaming. Cumulative counts are returned on each response.
        self.configure_valid_response(input_tokens=3, output_tokens=5)
        self.configure_valid_response(input_tokens=3, output_tokens=5)
        self.configure_valid_response(input_tokens=3, output_tokens=5)

        self.generate_content(model="gemini-2.0-flash", contents="Some input")

        self.otel.assert_has_span_named("generate_content gemini-2.0-flash")
        span = self.otel.get_span_named("generate_content gemini-2.0-flash")
        self.assertEqual(span.attributes["gen_ai.usage.input_tokens"], 3)
        self.assertEqual(span.attributes["gen_ai.usage.output_tokens"], 5)

    def test_new_semconv_log_has_genai_attributes(self):
        patched_environ = patch.dict(
            "os.environ",
            {
                "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT": "EVENT_ONLY",
                "OTEL_SEMCONV_STABILITY_OPT_IN": "gen_ai_latest_experimental",
            },
        )
        patched_otel_mapping = patch.dict(
            _OpenTelemetrySemanticConventionStability._OTEL_SEMCONV_STABILITY_SIGNAL_MAPPING,
            {
                _OpenTelemetryStabilitySignalType.GEN_AI: _StabilityMode.GEN_AI_LATEST_EXPERIMENTAL
            },
        )
        with patched_environ, patched_otel_mapping:
            self.configure_valid_response(text="Yep, it works!")
            self.generate_content(
                model="gemini-2.0-flash",
                contents="Does this work?",
            )
            self.otel.assert_has_event_named(
                "gen_ai.client.inference.operation.details"
            )
            event = self.otel.get_event_named(
                "gen_ai.client.inference.operation.details"
            )
            self.assertEqual(
                event.attributes["gen_ai.operation.name"],
                "generate_content",
            )
            self.assertEqual(
                event.attributes["gen_ai.request.model"],
                "gemini-2.0-flash",
            )

    # ------------------------------------------------------------- error path

    def test_mid_stream_error_records_error_span(self):
        # Trigger mock creation, then override side_effect to yield one
        # chunk and then raise.
        self.configure_valid_response(text="Partial chunk")

        def _error_after_first(*args, **kwargs):
            yield create_response(
                text="Partial", input_tokens=5, output_tokens=3
            )
            raise RuntimeError("Connection lost")

        self.mock_generate_content_stream.side_effect = _error_after_first

        with self.assertRaises(RuntimeError):
            self.generate_content(model="gemini-2.0-flash", contents="Hello")

        span = self.otel.get_span_named("generate_content gemini-2.0-flash")
        self.assertIsNotNone(span)
        self.assertEqual(span.attributes.get("error.type"), "RuntimeError")
        # Partial token data should still be recorded from the chunk
        # received before the error.
        self.assertEqual(span.attributes.get("gen_ai.usage.input_tokens"), 5)
        self.assertEqual(span.attributes.get("gen_ai.usage.output_tokens"), 3)

    def test_empty_stream_produces_error_span(self):
        # Create mocks, then clear responses so the stream yields nothing.
        self.configure_valid_response(text="dummy")
        self._responses.clear()

        self.generate_content(model="gemini-2.0-flash", contents="Hello")

        span = self.otel.get_span_named("generate_content gemini-2.0-flash")
        self.assertIsNotNone(span)
        self.assertEqual(
            span.attributes.get("error.type"),
            "NoCandidatesError",
        )
