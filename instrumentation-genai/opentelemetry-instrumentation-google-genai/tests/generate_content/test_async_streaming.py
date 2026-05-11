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

import asyncio

from .nonstreaming_base import NonStreamingTestCase
from .streaming_base import StreamingTestCase


class AsyncStreamingMixin:
    @property
    def expected_function_name(self):
        return "google.genai.AsyncModels.generate_content_stream"

    async def _generate_content_stream_helper(self, *args, **kwargs):
        result = []
        async for (
            response
        ) in await self.client.aio.models.generate_content_stream(  # pylint: disable=missing-kwoa
            *args, **kwargs
        ):
            result.append(response)
        return result

    def generate_content_stream(self, *args, **kwargs):
        return asyncio.run(
            self._generate_content_stream_helper(*args, **kwargs)
        )


class TestGenerateContentAsyncStreamingWithSingleResult(
    AsyncStreamingMixin, NonStreamingTestCase
):
    def generate_content(self, *args, **kwargs):
        responses = self.generate_content_stream(*args, **kwargs)
        self.assertEqual(len(responses), 1)
        return responses[0]


class TestGenerateContentAsyncStreamingWithStreamedResults(
    AsyncStreamingMixin, StreamingTestCase
):
    def generate_content(self, *args, **kwargs):
        return self.generate_content_stream(*args, **kwargs)


class TestAsyncStreamEarlyBreak(AsyncStreamingMixin, StreamingTestCase):
    """Tests that verify span finalization when the async stream is not fully consumed."""

    def generate_content(self, *args, **kwargs):
        return self.generate_content_stream(*args, **kwargs)

    def test_early_break_finalizes_span_after_cleanup(self):
        self.configure_valid_response(text="First chunk")
        self.configure_valid_response(text="Second chunk")

        async def _break_after_first():
            stream = await self.client.aio.models.generate_content_stream(
                model="gemini-2.0-flash", contents="Hello"
            )
            async for chunk in stream:
                break  # consume only the first chunk
            # Drop last reference so __del__ fires.
            del stream
            import gc

            gc.collect()

        asyncio.run(_break_after_first())
        span = self.otel.get_span_named("generate_content gemini-2.0-flash")
        self.assertIsNotNone(span)
        self.assertIsNotNone(span.end_time)

    def test_explicit_aclose_finalizes_span(self):
        self.configure_valid_response(text="First chunk")
        self.configure_valid_response(text="Second chunk")

        async def _aclose_after_first():
            stream = await self.client.aio.models.generate_content_stream(
                model="gemini-2.0-flash", contents="Hello"
            )
            first = await stream.__anext__()
            self.assertEqual(first.text, "First chunk")
            await stream.aclose()

        asyncio.run(_aclose_after_first())
        span = self.otel.get_span_named("generate_content gemini-2.0-flash")
        self.assertIsNotNone(span)
        self.assertIsNotNone(span.end_time)
        self.assertIsNotNone(
            span.attributes.get("gen_ai.response.time_to_first_chunk")
        )
