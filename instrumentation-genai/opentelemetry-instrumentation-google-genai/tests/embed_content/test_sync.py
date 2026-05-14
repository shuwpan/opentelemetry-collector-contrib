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

from .base import EmbedContentSharedTests


class TestSyncEmbedContent(EmbedContentSharedTests):
    """Run shared embed_content tests against the sync client."""

    @property
    def expected_code_function_name(self):
        return "google.genai.Models.embed_content"

    def embed_content(self, **kwargs):
        return self.client.models.embed_content(**kwargs)
