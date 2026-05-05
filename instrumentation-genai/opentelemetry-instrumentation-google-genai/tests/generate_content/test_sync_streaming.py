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


import pytest

from .nonstreaming_base import NonStreamingTestCase
from .streaming_base import StreamingTestCase

# Streaming instrumentation is deferred to PR 2 (HYBIM-665). The
# generate_content_stream / async_generate_content_stream methods are
# currently passthrough — they emit no spans, metrics, or events — so
# any inherited assertions cannot pass.
pytestmark = pytest.mark.skip(
    reason="Streaming instrumentation deferred to PR 2 (HYBIM-665)."
)


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
