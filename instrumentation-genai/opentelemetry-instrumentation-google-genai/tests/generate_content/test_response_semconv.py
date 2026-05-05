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

    # ---------------------------------------- Gemini-specific attributes
    # These vendor-specific attributes (gen_ai.google.*) are populated on the
    # LLMInvocation.attributes dict but are currently dropped by the util-genai
    # SpanEmitter (only allows known semconv keys or custom_* prefixed keys).
    # We test the invocation-level population directly until the SpanEmitter
    # is updated to allow gen_ai.* vendor extensions.

    def test_gemini_model_version_set_on_invocation(self):
        from opentelemetry.instrumentation.google_genai.generate_content import (
            _apply_response,
        )
        from opentelemetry.util.genai.types import LLMInvocation

        inv = LLMInvocation(request_model="gemini-2.0-flash", attributes={})
        _apply_response(
            inv,
            GenerateContentResponse(
                candidates=[
                    genai_types.Candidate(
                        content=genai_types.Content(
                            parts=[genai_types.Part(text="ok")],
                            role="model",
                        )
                    )
                ],
                model_version="gemini-2.0-flash-001",
            ),
        )
        self.assertEqual(
            inv.attributes["gen_ai.google.response.model_version"],
            "gemini-2.0-flash-001",
        )

    def test_gemini_backend_set_on_invocation(self):
        from opentelemetry.instrumentation.google_genai.generate_content import (
            _build_invocation,
        )

        inv = _build_invocation(
            self.client.models,
            "gemini-2.0-flash",
            "hi",
            None,
            None,
        )
        self.assertEqual(
            inv.attributes["gen_ai.google.request.backend"], "gemini"
        )

    # ---------------------------------------- Gemini usage token attributes

    def test_gemini_usage_tokens_set_on_invocation(self):
        """Positive token counts land in invocation.attributes."""
        from opentelemetry.instrumentation.google_genai.generate_content import (
            _apply_response,
        )
        from opentelemetry.util.genai.types import LLMInvocation

        inv = LLMInvocation(request_model="gemini-2.0-flash", attributes={})
        _apply_response(
            inv,
            GenerateContentResponse(
                candidates=[
                    genai_types.Candidate(
                        content=genai_types.Content(
                            parts=[genai_types.Part(text="ok")],
                            role="model",
                        )
                    )
                ],
                usage_metadata=genai_types.GenerateContentResponseUsageMetadata(
                    thoughts_token_count=42,
                    tool_use_prompt_token_count=7,
                    cached_content_token_count=15,
                ),
            ),
        )
        self.assertEqual(
            inv.attributes["gen_ai.google.usage.thought_tokens"], 42
        )
        self.assertEqual(
            inv.attributes["gen_ai.google.usage.tool_use_prompt_tokens"], 7
        )
        self.assertEqual(
            inv.attributes["gen_ai.google.usage.cached_content_tokens"], 15
        )

    def test_gemini_usage_tokens_zero_omitted_from_invocation(self):
        """Zero-value token counts must NOT appear in invocation.attributes."""
        from opentelemetry.instrumentation.google_genai.generate_content import (
            _apply_response,
        )
        from opentelemetry.util.genai.types import LLMInvocation

        inv = LLMInvocation(request_model="gemini-2.0-flash", attributes={})
        _apply_response(
            inv,
            GenerateContentResponse(
                candidates=[
                    genai_types.Candidate(
                        content=genai_types.Content(
                            parts=[genai_types.Part(text="ok")],
                            role="model",
                        )
                    )
                ],
                usage_metadata=genai_types.GenerateContentResponseUsageMetadata(
                    thoughts_token_count=0,
                    tool_use_prompt_token_count=0,
                    cached_content_token_count=0,
                ),
            ),
        )
        self.assertNotIn("gen_ai.google.usage.thought_tokens", inv.attributes)
        self.assertNotIn(
            "gen_ai.google.usage.tool_use_prompt_tokens", inv.attributes
        )
        self.assertNotIn(
            "gen_ai.google.usage.cached_content_tokens", inv.attributes
        )

    # ---------------------------------------- thought-part filtering

    def test_thought_parts_excluded_from_output_messages(self):
        """Thinking-model parts (thought=True) must not appear in telemetry."""
        from opentelemetry.instrumentation.google_genai.message import (
            to_output_messages,
        )

        candidates = [
            genai_types.Candidate(
                content=genai_types.Content(
                    parts=[
                        genai_types.Part(text="thinking...", thought=True),
                        genai_types.Part(text="visible answer"),
                    ],
                    role="model",
                ),
                finish_reason=genai_types.FinishReason.STOP,
            )
        ]
        messages = to_output_messages(candidates=candidates)
        self.assertEqual(len(messages), 1)
        # Only the non-thought part should survive
        self.assertEqual(len(messages[0].parts), 1)
        self.assertEqual(messages[0].parts[0].content, "visible answer")

    # -------------------------------------------- provider / framework

    def test_provider_is_google(self):
        self.configure_valid_response(text="hello")
        self.client.models.generate_content(
            model="gemini-2.0-flash", contents="hi"
        )
        span = self.otel.get_span_named("generate_content gemini-2.0-flash")
        self.assertEqual(span.attributes.get("gen_ai.provider.name"), "google")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
