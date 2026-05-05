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


from google.genai import types as genai_types

from .base import TestCase


class FinishReasonsTestCase(TestCase):
    def generate_and_get_span_finish_reasons(self):
        self.client.models.generate_content(
            model="gemini-2.5-flash-001", contents="Some prompt"
        )
        span = self.otel.get_span_named(
            "generate_content gemini-2.5-flash-001"
        )
        assert span is not None
        # ``gen_ai.response.finish_reasons`` may be absent when the
        # response carries no finish reason; util-genai's SpanEmitter
        # does not emit empty finish-reason lists. Treat absent as [].
        return list(span.attributes.get("gen_ai.response.finish_reasons", []))

    def test_single_candidate_with_valid_reason(self):
        self.configure_valid_response(
            candidate=genai_types.Candidate(
                finish_reason=genai_types.FinishReason.STOP
            )
        )
        self.assertEqual(self.generate_and_get_span_finish_reasons(), ["stop"])

    def test_single_candidate_with_safety_reason(self):
        self.configure_valid_response(
            candidate=genai_types.Candidate(
                finish_reason=genai_types.FinishReason.SAFETY
            )
        )
        self.assertEqual(
            self.generate_and_get_span_finish_reasons(), ["content_filter"]
        )

    def test_single_candidate_with_max_tokens_reason(self):
        self.configure_valid_response(
            candidate=genai_types.Candidate(
                finish_reason=genai_types.FinishReason.MAX_TOKENS
            )
        )
        self.assertEqual(
            self.generate_and_get_span_finish_reasons(), ["length"]
        )

    def test_single_candidate_with_no_reason(self):
        self.configure_valid_response(
            candidate=genai_types.Candidate(finish_reason=None)
        )
        self.assertEqual(self.generate_and_get_span_finish_reasons(), [])

    def test_single_candidate_with_unspecified_reason(self):
        self.configure_valid_response(
            candidate=genai_types.Candidate(
                finish_reason=genai_types.FinishReason.FINISH_REASON_UNSPECIFIED
            )
        )
        self.assertEqual(
            self.generate_and_get_span_finish_reasons(), ["error"]
        )

    def test_multiple_candidates_with_valid_reasons(self):
        self.configure_valid_response(
            candidates=[
                genai_types.Candidate(
                    finish_reason=genai_types.FinishReason.MAX_TOKENS
                ),
                genai_types.Candidate(
                    finish_reason=genai_types.FinishReason.STOP
                ),
            ]
        )
        self.assertEqual(
            self.generate_and_get_span_finish_reasons(), ["length", "stop"]
        )

    def test_sorts_finish_reasons(self):
        self.configure_valid_response(
            candidates=[
                genai_types.Candidate(
                    finish_reason=genai_types.FinishReason.STOP
                ),
                genai_types.Candidate(
                    finish_reason=genai_types.FinishReason.MAX_TOKENS
                ),
                genai_types.Candidate(
                    finish_reason=genai_types.FinishReason.SAFETY
                ),
            ]
        )
        self.assertEqual(
            self.generate_and_get_span_finish_reasons(),
            ["content_filter", "length", "stop"],
        )

    def test_blocklist_maps_to_content_filter(self):
        self.configure_valid_response(
            candidate=genai_types.Candidate(
                finish_reason=genai_types.FinishReason.BLOCKLIST
            )
        )
        self.assertEqual(
            self.generate_and_get_span_finish_reasons(), ["content_filter"]
        )

    def test_recitation_maps_to_content_filter(self):
        self.configure_valid_response(
            candidate=genai_types.Candidate(
                finish_reason=genai_types.FinishReason.RECITATION
            )
        )
        self.assertEqual(
            self.generate_and_get_span_finish_reasons(), ["content_filter"]
        )

    def test_spii_maps_to_content_filter(self):
        self.configure_valid_response(
            candidate=genai_types.Candidate(
                finish_reason=genai_types.FinishReason.SPII
            )
        )
        self.assertEqual(
            self.generate_and_get_span_finish_reasons(), ["content_filter"]
        )

    def test_malformed_function_call_maps_to_error(self):
        self.configure_valid_response(
            candidate=genai_types.Candidate(
                finish_reason=genai_types.FinishReason.MALFORMED_FUNCTION_CALL
            )
        )
        self.assertEqual(
            self.generate_and_get_span_finish_reasons(), ["error"]
        )

    def test_unexpected_tool_call_maps_to_error(self):
        self.configure_valid_response(
            candidate=genai_types.Candidate(
                finish_reason=genai_types.FinishReason.UNEXPECTED_TOOL_CALL
            )
        )
        self.assertEqual(
            self.generate_and_get_span_finish_reasons(), ["error"]
        )

    def test_other_maps_to_error(self):
        self.configure_valid_response(
            candidate=genai_types.Candidate(
                finish_reason=genai_types.FinishReason.OTHER
            )
        )
        self.assertEqual(
            self.generate_and_get_span_finish_reasons(), ["error"]
        )

    def test_deduplicates_finish_reasons(self):
        self.configure_valid_response(
            candidates=[
                genai_types.Candidate(
                    finish_reason=genai_types.FinishReason.STOP
                ),
                genai_types.Candidate(
                    finish_reason=genai_types.FinishReason.MAX_TOKENS
                ),
                genai_types.Candidate(
                    finish_reason=genai_types.FinishReason.STOP
                ),
                genai_types.Candidate(
                    finish_reason=genai_types.FinishReason.STOP
                ),
                genai_types.Candidate(
                    finish_reason=genai_types.FinishReason.SAFETY
                ),
                genai_types.Candidate(
                    finish_reason=genai_types.FinishReason.STOP
                ),
                genai_types.Candidate(
                    finish_reason=genai_types.FinishReason.STOP
                ),
                genai_types.Candidate(
                    finish_reason=genai_types.FinishReason.STOP
                ),
            ]
        )
        self.assertEqual(
            self.generate_and_get_span_finish_reasons(),
            ["content_filter", "length", "stop"],
        )
