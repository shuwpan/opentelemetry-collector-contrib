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


from .nonstreaming_base import NonStreamingTestCase
from .streaming_base import StreamingTestCase


class StreamingMixin:
    @property
    def expected_function_name(self):
        return "google.genai.Models.generate_content_stream"

    def generate_content_stream(self, *args, **kwargs):
        result = []
        for response in self.client.models.generate_content_stream(  # pylint: disable=missing-kwoa
            *args, **kwargs
        ):
            result.append(response)
        return result


class TestGenerateContentStreamingWithSingleResult(
    StreamingMixin, NonStreamingTestCase
):
    def generate_content(self, *args, **kwargs):
        responses = self.generate_content_stream(*args, **kwargs)
        self.assertEqual(len(responses), 1)
        return responses[0]


class TestGenerateContentStreamingWithStreamedResults(
    StreamingMixin, StreamingTestCase
):
    def generate_content(self, *args, **kwargs):
        return self.generate_content_stream(*args, **kwargs)


class TestSyncStreamEarlyBreak(StreamingMixin, StreamingTestCase):
    """Tests that verify span finalization when the stream is not fully consumed."""

    def generate_content(self, *args, **kwargs):
        return self.generate_content_stream(*args, **kwargs)

    def test_early_break_finalizes_span_after_cleanup(self):
        self.configure_valid_response(text="First chunk")
        self.configure_valid_response(text="Second chunk")
        stream = self.client.models.generate_content_stream(
            model="gemini-2.0-flash", contents="Hello"
        )
        for chunk in stream:
            break  # consume only the first chunk
        # For regular iterators (not generators), Python does NOT call
        # close() on break.  Span finalization relies on __del__.
        # Force it by dropping the last reference + GC.
        del stream
        import gc

        gc.collect()
        span = self.otel.get_span_named("generate_content gemini-2.0-flash")
        self.assertIsNotNone(span)
        # Span should have been finalized (not left dangling).
        self.assertIsNotNone(span.end_time)

    def test_explicit_close_finalizes_span(self):
        self.configure_valid_response(text="First chunk")
        self.configure_valid_response(text="Second chunk")
        stream = self.client.models.generate_content_stream(
            model="gemini-2.0-flash", contents="Hello"
        )
        first = next(iter(stream))
        self.assertEqual(first.text, "First chunk")
        stream.close()
        span = self.otel.get_span_named("generate_content gemini-2.0-flash")
        self.assertIsNotNone(span)
        self.assertIsNotNone(span.end_time)
        # Partial token data from the consumed chunk should be recorded.
        self.assertIsNotNone(
            span.attributes.get("gen_ai.response.time_to_first_chunk")
        )
