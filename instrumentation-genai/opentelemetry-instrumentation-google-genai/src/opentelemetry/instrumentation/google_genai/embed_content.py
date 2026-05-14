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

import functools
import logging
from typing import Any, Optional, Union

from google.genai.models import AsyncModels, Models
from google.genai.models import t as transformers
from google.genai.types import (
    EmbedContentConfig,
    EmbedContentConfigDict,
    EmbedContentResponse,
)

from opentelemetry import context as context_api
from opentelemetry.semconv._incubating.attributes import (
    code_attributes,
)
from opentelemetry.util.genai.attributes import (
    SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY,
)
from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.types import (
    EmbeddingInvocation,
    Error,
)

from .generate_content import (
    _determine_genai_system,
    _get_vertexai_system_name,
)

_logger = logging.getLogger(__name__)

_SYNC_CODE_FUNCTION_NAME = "google.genai.Models.embed_content"
_ASYNC_CODE_FUNCTION_NAME = "google.genai.AsyncModels.embed_content"

_VENDOR_ATTR_PREFIXES = ("gen_ai.google.", "gcp.gen_ai.")

EmbedContentConfigOrDict = Union[EmbedContentConfig, EmbedContentConfigDict]


# ---------------------------------------------------------------------------
# Content normalisation
# ---------------------------------------------------------------------------


def _normalise_contents_to_texts(contents: Any) -> list[str]:
    """Convert the SDK's ContentListUnion into a flat list of text strings.

    Attempts to use the SDK transformer first; falls back to a best-effort
    extraction that walks the raw value without raising.
    """
    try:
        normalised = transformers.t_contents(contents)
        texts = []
        for content in normalised:
            for part in content.parts or []:
                if part.text is not None:
                    texts.append(part.text)
        return texts
    except Exception:
        # If the SDK transformer fails (e.g. unsupported type), do a best-effort
        # extraction without crashing instrumentation.
        _logger.debug(
            "Failed to normalise embed_content contents to text list",
            exc_info=True,
        )
        return []


# ---------------------------------------------------------------------------
# Invocation builders
# ---------------------------------------------------------------------------


def _build_embedding_invocation(
    models_object: Union[Models, AsyncModels],
    model: str,
    contents: Any,
    config: Optional[EmbedContentConfigOrDict],
) -> EmbeddingInvocation:
    """Build an EmbeddingInvocation from google-genai embed_content parameters."""
    genai_system = _determine_genai_system(models_object)
    backend = (
        "vertex_ai"
        if genai_system == _get_vertexai_system_name()
        else "gemini"
    )

    input_texts = _normalise_contents_to_texts(contents)

    invocation = EmbeddingInvocation(
        system=genai_system,
        provider="google",
        framework="google-genai-sdk",
        request_model=model,
        input_texts=input_texts,
    )

    invocation.attributes = {
        "gen_ai.google.request.backend": backend,
    }

    # Capture output_dimensionality as a vendor attribute when set.
    if config is not None:
        dim: Optional[int] = None
        if isinstance(config, dict):
            dim = config.get("output_dimensionality")
        else:
            dim = getattr(config, "output_dimensionality", None)
        if dim is not None:
            invocation.attributes[
                "gen_ai.google.request.output_dimensionality"
            ] = dim

    return invocation


def _apply_embed_response(
    invocation: EmbeddingInvocation,
    response: EmbedContentResponse,
) -> None:
    """Populate EmbeddingInvocation fields from the embed_content response."""
    embeddings = getattr(response, "embeddings", None)
    if embeddings:
        first = embeddings[0]
        values = getattr(first, "values", None)
        if values:
            invocation.dimension_count = len(values)

        # Extract input_tokens from Vertex AI ContentEmbedding.statistics.token_count
        total_tokens = 0
        for emb in embeddings:
            stats = getattr(emb, "statistics", None)
            if stats is not None:
                tc = getattr(stats, "token_count", None)
                if tc is not None:
                    total_tokens += tc
        if total_tokens > 0:
            invocation.input_tokens = int(total_tokens)

    metadata = getattr(response, "metadata", None)
    if metadata is not None:
        billable = getattr(metadata, "billable_character_count", None)
        if isinstance(billable, int) and billable > 0:
            invocation.attributes[
                "gen_ai.google.usage.billable_character_count"
            ] = billable


def _set_vendor_attributes_on_span(invocation: EmbeddingInvocation) -> None:
    """Copy vendor-prefixed attributes from invocation.attributes directly
    onto the span so they survive SpanEmitter filtering."""
    span = invocation.span
    if not span or not span.is_recording():
        return
    for key, value in (invocation.attributes or {}).items():
        if any(key.startswith(prefix) for prefix in _VENDOR_ATTR_PREFIXES):
            span.set_attribute(key, value)


# ---------------------------------------------------------------------------
# Snapshot for uninstrument
# ---------------------------------------------------------------------------


class _EmbedMethodsSnapshot:
    def __init__(self):
        self._original_embed_content = Models.embed_content
        self._original_async_embed_content = AsyncModels.embed_content

    @property
    def embed_content(self):
        return self._original_embed_content

    @property
    def async_embed_content(self):
        return self._original_async_embed_content

    def restore(self):
        Models.embed_content = self._original_embed_content
        AsyncModels.embed_content = self._original_async_embed_content


# ---------------------------------------------------------------------------
# Instrumented wrapper factories
# ---------------------------------------------------------------------------


def _create_instrumented_embed_content(
    snapshot: _EmbedMethodsSnapshot,
    handler: TelemetryHandler,
):
    wrapped_func = snapshot.embed_content

    @functools.wraps(wrapped_func)
    def instrumented_embed_content(
        self: Models,
        *,
        model: str,
        contents: Any,
        config: Optional[EmbedContentConfigOrDict] = None,
        **kwargs: Any,
    ) -> EmbedContentResponse:
        # defensive check
        if context_api.get_value(SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY):
            return wrapped_func(
                self,
                model=model,
                contents=contents,
                config=config,
                **kwargs,
            )
        invocation = _build_embedding_invocation(self, model, contents, config)
        handler.start_embedding(invocation)
        if invocation.span and invocation.span.is_recording():
            invocation.span.set_attribute(
                code_attributes.CODE_FUNCTION_NAME,
                _SYNC_CODE_FUNCTION_NAME,
            )
        _set_vendor_attributes_on_span(invocation)
        try:
            response = wrapped_func(
                self,
                model=model,
                contents=contents,
                config=config,
                **kwargs,
            )
        except Exception as error:
            handler.fail_embedding(
                invocation,
                Error(message=str(error), type=type(error)),
            )
            raise
        try:
            _apply_embed_response(invocation, response)
            _set_vendor_attributes_on_span(invocation)
            handler.stop_embedding(invocation)
        except Exception:  # pragma: no cover - defensive
            pass
        return response

    return instrumented_embed_content


def _create_instrumented_async_embed_content(
    snapshot: _EmbedMethodsSnapshot,
    handler: TelemetryHandler,
):
    wrapped_func = snapshot.async_embed_content

    @functools.wraps(wrapped_func)
    async def instrumented_async_embed_content(
        self: AsyncModels,
        *,
        model: str,
        contents: Any,
        config: Optional[EmbedContentConfigOrDict] = None,
        **kwargs: Any,
    ) -> EmbedContentResponse:
        if context_api.get_value(SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY):
            return await wrapped_func(
                self,
                model=model,
                contents=contents,
                config=config,
                **kwargs,
            )
        invocation = _build_embedding_invocation(self, model, contents, config)
        handler.start_embedding(invocation)
        if invocation.span and invocation.span.is_recording():
            invocation.span.set_attribute(
                code_attributes.CODE_FUNCTION_NAME,
                _ASYNC_CODE_FUNCTION_NAME,
            )
        _set_vendor_attributes_on_span(invocation)
        try:
            response = await wrapped_func(
                self,
                model=model,
                contents=contents,
                config=config,
                **kwargs,
            )
        except Exception as error:
            handler.fail_embedding(
                invocation,
                Error(message=str(error), type=type(error)),
            )
            raise
        try:
            _apply_embed_response(invocation, response)
            _set_vendor_attributes_on_span(invocation)
            handler.stop_embedding(invocation)
        except Exception:  # pragma: no cover - defensive
            pass
        return response

    return instrumented_async_embed_content


# ---------------------------------------------------------------------------
# Public instrument / uninstrument API
# ---------------------------------------------------------------------------


def uninstrument_embed_content(snapshot: object) -> None:
    assert isinstance(snapshot, _EmbedMethodsSnapshot)
    snapshot.restore()


def instrument_embed_content(handler: TelemetryHandler) -> object:
    snapshot = _EmbedMethodsSnapshot()
    Models.embed_content = _create_instrumented_embed_content(
        snapshot, handler
    )
    AsyncModels.embed_content = _create_instrumented_async_embed_content(
        snapshot, handler
    )
    return snapshot
