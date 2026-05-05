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
"""Non-streaming Google GenAI tests for the TelemetryHandler / LLMInvocation
migration (HYBIM-663).

Upstream-only event names and span attributes that are not part of the
TelemetryHandler / LLMInvocation model have been removed.
"""

import unittest

import pytest

from .base import TestCase

# pylint: disable=too-many-public-methods


class NonStreamingTestCase(TestCase):
    """Shared assertions for sync + async generate_content (no streaming)."""

    def setUp(self):  # pylint: disable=invalid-name
        super().setUp()
        if self.__class__ == NonStreamingTestCase:
            raise unittest.SkipTest("Skipping testcase base.")

    def generate_content(self, *args, **kwargs):
        raise NotImplementedError("Must implement 'generate_content'.")

    @property
    def expected_function_name(self):
        raise NotImplementedError("Must implement 'expected_function_name'.")

    # ------------------------------------------------------------------ basic

    def test_instrumentation_does_not_break_core_functionality(self):
        self.configure_valid_response(text="Yep, it works!")
        response = self.generate_content(
            model="gemini-2.0-flash", contents="Does this work?"
        )
        self.assertEqual(response.text, "Yep, it works!")

    def test_generates_span(self):
        self.configure_valid_response(text="Yep, it works!")
        response = self.generate_content(
            model="gemini-2.0-flash", contents="Does this work?"
        )
        self.assertEqual(response.text, "Yep, it works!")
        self.otel.assert_has_span_named("generate_content gemini-2.0-flash")

    def test_model_reflected_into_span_name(self):
        self.configure_valid_response(text="Yep, it works!")
        response = self.generate_content(
            model="gemini-1.5-flash", contents="Does this work?"
        )
        self.assertEqual(response.text, "Yep, it works!")
        self.otel.assert_has_span_named("generate_content gemini-1.5-flash")

    # -------------------------------------------------------- span attributes

    def test_generated_span_has_minimal_genai_attributes(self):
        self.configure_valid_response(text="Yep, it works!")
        self.generate_content(
            model="gemini-2.0-flash", contents="Does this work?"
        )
        span = self.otel.get_span_named("generate_content gemini-2.0-flash")
        self.assertIsNotNone(span)
        self.assertEqual(span.attributes["gen_ai.system"], "gemini")
        self.assertEqual(
            span.attributes["gen_ai.operation.name"], "generate_content"
        )
        self.assertEqual(
            span.attributes["gen_ai.request.model"], "gemini-2.0-flash"
        )

    def test_generated_span_has_code_function_name(self):
        self.configure_valid_response(text="Yep, it works!")
        self.generate_content(
            model="gemini-2.0-flash", contents="Does this work?"
        )
        span = self.otel.get_span_named("generate_content gemini-2.0-flash")
        self.assertIsNotNone(span)
        self.assertEqual(
            span.attributes["code.function.name"],
            self.expected_function_name,
        )

    def test_generated_span_has_vertex_ai_system_when_configured(self):
        self.set_use_vertex(True)
        self.configure_valid_response(text="Yep, it works!")
        self.generate_content(
            model="gemini-2.0-flash", contents="Does this work?"
        )
        span = self.otel.get_span_named("generate_content gemini-2.0-flash")
        self.assertIsNotNone(span)
        self.assertEqual(span.attributes["gen_ai.system"], "vertex_ai")
        self.assertEqual(
            span.attributes["gen_ai.operation.name"], "generate_content"
        )

    def test_generated_span_counts_tokens(self):
        self.configure_valid_response(input_tokens=123, output_tokens=456)
        self.generate_content(model="gemini-2.0-flash", contents="Some input")
        span = self.otel.get_span_named("generate_content gemini-2.0-flash")
        self.assertIsNotNone(span)
        self.assertEqual(span.attributes["gen_ai.usage.input_tokens"], 123)
        self.assertEqual(span.attributes["gen_ai.usage.output_tokens"], 456)

    # ------------------------------------------------------------- error path

    def test_span_and_event_still_written_when_response_is_exception(self):
        self.configure_exception(ValueError("Uh oh!"))
        with pytest.raises(ValueError):
            self.generate_content(
                model="gemini-2.0-flash", contents="Does this work?"
            )
        span = self.otel.get_span_named("generate_content gemini-2.0-flash")
        self.assertIsNotNone(span)
        self.assertEqual(span.attributes.get("error.type"), "ValueError")

    # ------------------------------------------------------- inference event

    def test_inference_details_event_emitted(self):
        self.configure_valid_response(text="Some response content")
        self.generate_content(model="gemini-2.0-flash", contents="Some input")
        self.otel.assert_has_event_named(
            "gen_ai.client.inference.operation.details"
        )
        event = self.otel.get_event_named(
            "gen_ai.client.inference.operation.details"
        )
        # The event carries the same gen_ai.* attributes as the span.
        self.assertEqual(
            event.attributes["gen_ai.operation.name"], "generate_content"
        )
        self.assertEqual(
            event.attributes["gen_ai.request.model"], "gemini-2.0-flash"
        )

    # ------------------------------------------------------------- metrics

    def test_records_metrics_data(self):
        self.configure_valid_response(input_tokens=10, output_tokens=20)
        self.generate_content(model="gemini-2.0-flash", contents="Some input")
        self.otel.assert_has_metrics_data_named("gen_ai.client.token.usage")
        self.otel.assert_has_metrics_data_named(
            "gen_ai.client.operation.duration"
        )
