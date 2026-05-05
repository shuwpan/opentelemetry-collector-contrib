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

"""Coverage for the response-side behavior of ``_apply_response``:

- ``response_model_name`` and ``response_id`` are populated as
  ``gen_ai.response.model`` / ``gen_ai.response.id`` on the span.
- A legitimate zero token count is preserved (not silently dropped).
- Responses with no candidates route through ``fail_llm`` so the span
  shows an ERROR status with the right ``error.type`` (instead of being
  silently dropped by the SpanEmitter's supplemental-attribute filter).
"""

import google.genai.types as genai_types
import pytest
from google.genai.types import (
    BlockedReason,
    GenerateContentResponse,
    GenerateContentResponsePromptFeedback,
)

from .base import TestCase


class ResponseSemconvTestCase(TestCase):
    def _generate(self, response: GenerateContentResponse):
        # Push a single canned response through the same plumbing that
        # ``configure_valid_response`` uses, so the instrumentor's full
        # mock setup runs.
        self.configure_valid_response(text="placeholder")  # installs mocks
        # Replace the queued response with our hand-built one so we can
        # control model_version, response_id, prompt_feedback, etc.
        self._responses.clear()
        self._responses.append(response)
        self.client.models.generate_content(
            model="gemini-2.0-flash", contents="hi"
        )

    # -------------------------------------------- response identity attrs

    def test_response_model_name_set_from_model_version(self):
        self._generate(
            GenerateContentResponse(
                candidates=[
                    genai_types.Candidate(
                        content=genai_types.Content(
                            parts=[genai_types.Part(text="hi back")],
                            role="model",
                        )
                    )
                ],
                model_version="gemini-2.0-flash-001",
            )
        )
        span = self.otel.get_span_named("generate_content gemini-2.0-flash")
        self.assertEqual(
            span.attributes["gen_ai.response.model"], "gemini-2.0-flash-001"
        )

    def test_response_id_set_from_response_id(self):
        self._generate(
            GenerateContentResponse(
                candidates=[
                    genai_types.Candidate(
                        content=genai_types.Content(
                            parts=[genai_types.Part(text="hi back")],
                            role="model",
                        )
                    )
                ],
                response_id="resp-abc-123",
            )
        )
        span = self.otel.get_span_named("generate_content gemini-2.0-flash")
        self.assertEqual(span.attributes["gen_ai.response.id"], "resp-abc-123")

    # ------------------------------------------------ token-count edge case

    def test_zero_input_tokens_preserved(self):
        self._generate(
            GenerateContentResponse(
                candidates=[
                    genai_types.Candidate(
                        content=genai_types.Content(
                            parts=[genai_types.Part(text="hi back")],
                            role="model",
                        )
                    )
                ],
                usage_metadata=genai_types.GenerateContentResponseUsageMetadata(
                    prompt_token_count=0,
                    candidates_token_count=7,
                ),
            )
        )
        span = self.otel.get_span_named("generate_content gemini-2.0-flash")
        # Zero is a valid token count; emitter must record it, not skip it.
        self.assertEqual(span.attributes["gen_ai.usage.input_tokens"], 0)
        self.assertEqual(span.attributes["gen_ai.usage.output_tokens"], 7)

    # --------------------------------------------- blocked-response error

    def test_blocked_response_marks_span_as_error_with_typed_error(self):
        self._generate(
            GenerateContentResponse(
                candidates=[],
                prompt_feedback=GenerateContentResponsePromptFeedback(
                    block_reason=BlockedReason.SAFETY
                ),
            )
        )
        span = self.otel.get_span_named("generate_content gemini-2.0-flash")
        # Goes through fail_llm → SpanEmitter.on_error → ERROR status +
        # error.type = synthetic exception's __qualname__.
        self.assertEqual(span.attributes["error.type"], "BlockedPromptError")

    def test_no_candidates_marks_span_as_error_with_typed_error(self):
        self._generate(
            GenerateContentResponse(
                candidates=[],
                # No prompt_feedback at all → NoCandidatesError path.
            )
        )
        span = self.otel.get_span_named("generate_content gemini-2.0-flash")
        self.assertEqual(span.attributes["error.type"], "NoCandidatesError")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
