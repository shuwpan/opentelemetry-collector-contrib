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

"""Tests for SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY functionality.

When an outer instrumentation (e.g. LangChain) sets the suppression key,
the google-genai wrapper must skip instrumentation and call through directly.
"""

import asyncio

import pytest

from opentelemetry import context as context_api
from opentelemetry.util.genai.attributes import (
    SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY,
)

from .base import TestCase


class SuppressionTestCase(TestCase):
    def test_sync_generate_content_suppressed(self):
        self.configure_valid_response(text="hello")
        token = context_api.attach(
            context_api.set_value(
                SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY, True
            )
        )
        try:
            self.client.models.generate_content(
                model="gemini-2.0-flash", contents="hi"
            )
        finally:
            context_api.detach(token)
        spans = self.otel.get_finished_spans()
        self.assertEqual(
            len(spans),
            0,
            f"Expected no spans under suppression, got {len(spans)}",
        )

    def test_async_generate_content_suppressed(self):
        self.configure_valid_response(text="hello")
        token = context_api.attach(
            context_api.set_value(
                SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY, True
            )
        )
        try:
            asyncio.run(
                self.client.aio.models.generate_content(
                    model="gemini-2.0-flash", contents="hi"
                )
            )
        finally:
            context_api.detach(token)
        spans = self.otel.get_finished_spans()
        self.assertEqual(
            len(spans),
            0,
            f"Expected no spans under async suppression, got {len(spans)}",
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
