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
import functools
import json
import logging
import os
import timeit
from enum import Enum
from typing import Any, Optional, Union

from google.genai.models import AsyncModels, Models
from google.genai.models import t as transformers
from google.genai.types import (
    BlockedReason,
    Candidate,
    Content,
    ContentListUnion,
    ContentListUnionDict,
    ContentUnion,
    GenerateContentConfig,
    GenerateContentConfigOrDict,
    GenerateContentResponse,
    Tool,
)

from opentelemetry import context as context_api
from opentelemetry.semconv._incubating.attributes import (
    code_attributes,
    gen_ai_attributes,
)
from opentelemetry.util.genai.attributes import (
    GEN_AI_RESPONSE_TIME_TO_FIRST_CHUNK,
    SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY,
)
from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.types import (
    Error,
    ErrorClassification,
    InputMessage,
    LLMInvocation,
)
from opentelemetry.util.genai.utils import should_capture_tool_definitions
from opentelemetry.util.types import AttributeValue

from .allowlist_util import AllowList
from .custom_semconv import GCP_GENAI_OPERATION_CONFIG
from .dict_util import flatten_dict
from .message import (
    _to_finish_reason,
    to_input_messages,
    to_output_messages,
    to_system_instructions,
)
from .tool_call_wrapper import wrapped as wrap_tools

_logger = logging.getLogger(__name__)

_SYNC_CODE_FUNCTION_NAME = "google.genai.Models.generate_content"
_ASYNC_CODE_FUNCTION_NAME = "google.genai.AsyncModels.generate_content"
_SYNC_STREAM_CODE_FUNCTION_NAME = "google.genai.Models.generate_content_stream"
_ASYNC_STREAM_CODE_FUNCTION_NAME = (
    "google.genai.AsyncModels.generate_content_stream"
)

# Constant used for the value of 'gen_ai.operation.name".
_GENERATE_CONTENT_OP_NAME = "generate_content"

GENERATE_CONTENT_EXTRA_ATTRIBUTES_CONTEXT_KEY = context_api.create_key(
    "generate_content_extra_attributes_context_key"
)


# Synthetic exception classes used to drive ``fail_llm`` for responses where
# the SDK returned successfully but the model produced no usable output.
# Their ``__qualname__`` becomes the ``error.type`` span attribute, so the
# class names are deliberately descriptive and stable.
class BlockedPromptError(RuntimeError):
    """Raised internally when the model blocks the prompt (safety, etc.)."""


class NoCandidatesError(RuntimeError):
    """Raised internally when the response carries no candidates and no
    block reason."""


class _MethodsSnapshot:
    def __init__(self):
        self._original_generate_content = Models.generate_content
        self._original_generate_content_stream = Models.generate_content_stream
        self._original_async_generate_content = AsyncModels.generate_content
        self._original_async_generate_content_stream = (
            AsyncModels.generate_content_stream
        )

    @property
    def generate_content(self):
        return self._original_generate_content

    @property
    def generate_content_stream(self):
        return self._original_generate_content_stream

    @property
    def async_generate_content(self):
        return self._original_async_generate_content

    @property
    def async_generate_content_stream(self):
        return self._original_async_generate_content_stream

    def restore(self):
        Models.generate_content = self._original_generate_content
        Models.generate_content_stream = self._original_generate_content_stream
        AsyncModels.generate_content = self._original_async_generate_content
        AsyncModels.generate_content_stream = (
            self._original_async_generate_content_stream
        )


# ---------------------------------------------------------------------------
# GenAI system detection helpers
# ---------------------------------------------------------------------------


def _get_vertexai_system_name():
    return gen_ai_attributes.GenAiSystemValues.VERTEX_AI.value


def _get_gemini_system_name():
    return gen_ai_attributes.GenAiSystemValues.GEMINI.value


def _guess_genai_system_from_env():
    if os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "0").lower() in [
        "true",
        "1",
    ]:
        return _get_vertexai_system_name()
    return _get_gemini_system_name()


def _get_is_vertexai(models_object: Union[Models, AsyncModels]):
    # Since commit 8e561de04965bb8766db87ad8eea7c57c1040442 of "googleapis/python-genai",
    # it is possible to obtain the information using a documented property.
    if hasattr(models_object, "vertexai"):
        vertexai_attr = getattr(models_object, "vertexai")
        if vertexai_attr is not None:
            return vertexai_attr
    # For earlier revisions, it is necessary to deeply inspect the internals.
    if hasattr(models_object, "_api_client"):
        client = getattr(models_object, "_api_client")
        if not client:
            return None
        if hasattr(client, "vertexai"):
            return getattr(client, "vertexai")
    return None


def _determine_genai_system(models_object: Union[Models, AsyncModels]):
    vertexai_attr = _get_is_vertexai(models_object)
    if vertexai_attr is None:
        return _guess_genai_system_from_env()
    if vertexai_attr:
        return _get_vertexai_system_name()
    return _get_gemini_system_name()


# ---------------------------------------------------------------------------
# Config / attribute helpers
# ---------------------------------------------------------------------------


def _to_dict(value: object):
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        try:
            return value.model_dump()
        except TypeError:
            _logger.debug(
                "model_dump() failed for %r; config attributes will be missing",
                value,
            )
            return {}

    return json.loads(json.dumps(value))


def _config_to_system_instruction(
    config: Union[GenerateContentConfigOrDict, None],
) -> Union[ContentUnion, None]:
    if not config:
        return None

    if isinstance(config, dict):
        return GenerateContentConfig.model_validate(config).system_instruction
    return config.system_instruction


def _get_extra_generate_content_attributes() -> dict[str, AttributeValue]:
    attrs = context_api.get_value(
        GENERATE_CONTENT_EXTRA_ATTRIBUTES_CONTEXT_KEY
    )
    return dict(attrs or {})


# Google's GenerateContentConfig field names that have a direct OTel GenAI
# semconv counterpart and are populated as typed LLMInvocation fields below.
# Listed here so the vendor-attribute capture step can skip them.
_STANDARD_CONFIG_FIELDS: tuple[str, ...] = (
    "temperature",
    "top_p",
    "top_k",
    "candidate_count",  # → request_choice_count
    "max_output_tokens",  # → request_max_tokens
    "stop_sequences",
    "frequency_penalty",
    "presence_penalty",
    "seed",
    "response_mime_type",  # → output_type
)


def _output_type_from_mime(mime: Optional[str]) -> Optional[str]:
    if not mime:
        return None
    if mime == "text/plain":
        return "text"
    if mime == "application/json":
        return "json"
    return mime


def _wrap_config_tools(
    config: Optional[GenerateContentConfigOrDict],
    handler: TelemetryHandler,
    system: Optional[str],
    provider: Optional[str] = None,
) -> Optional[GenerateContentConfigOrDict]:
    """Return a copy of config with callable tool functions wrapped for telemetry.

    If config has no tools (or no callable tools), the original config object
    is returned unchanged to avoid unnecessary copies.
    """
    if config is None:
        return config

    if isinstance(config, dict):
        tools = config.get("tools")
        wrapped = wrap_tools(tools, handler, system, provider)
        if wrapped is tools:
            return config
        return {**config, "tools": wrapped}

    tools = getattr(config, "tools", None)
    wrapped = wrap_tools(tools, handler, system, provider)
    if wrapped is tools:
        return config
    config = config.model_copy(update={"tools": wrapped})
    return config


def _capture_vendor_config_attributes(
    config_dict: dict,
    allow_list: AllowList,
) -> dict[str, AttributeValue]:
    """Flatten Google-specific config fields under the gcp.* namespace.

    Standard semconv fields (see ``_STANDARD_CONFIG_FIELDS``) are excluded
    here because they are populated as typed ``LLMInvocation`` fields by
    ``_build_invocation``. Everything else is gated by ``allow_list`` so
    operators control which Google-specific config keys leak into telemetry.
    """
    if not config_dict:
        return {}
    flat = flatten_dict(
        config_dict,
        key_prefix=GCP_GENAI_OPERATION_CONFIG,
        exclude_keys=[
            # System instruction can be overly long for a span attribute;
            # it is also captured as a system message on the invocation.
            f"{GCP_GENAI_OPERATION_CONFIG}.system_instruction",
            # Standard semconv fields are surfaced via typed LLMInvocation
            # fields, not duplicated under the gcp.* namespace.
            *(
                f"{GCP_GENAI_OPERATION_CONFIG}.{name}"
                for name in _STANDARD_CONFIG_FIELDS
            ),
        ],
    )
    return {k: v for k, v in flat.items() if allow_list.allowed(k)}


# ---------------------------------------------------------------------------
# Response property accessor
# ---------------------------------------------------------------------------


def _get_response_property(response: GenerateContentResponse, path: str):
    path_segments = path.split(".")
    current_context = response
    for path_segment in path_segments:
        if current_context is None:
            return None
        if isinstance(current_context, dict):
            current_context = current_context.get(path_segment)
        else:
            current_context = getattr(current_context, path_segment, None)
    return current_context


# ---------------------------------------------------------------------------
# Tool definition helpers
# ---------------------------------------------------------------------------


def _clean_parameters(params: Any) -> Any:
    """Convert parameter objects into plain JSON-serializable dicts."""
    if params is None:
        return None
    if isinstance(params, dict):
        # Recursively convert enum values to strings
        return {
            k: v.value if isinstance(v, Enum) else _clean_parameters(v)
            for k, v in params.items()
        }
    if hasattr(params, "model_dump"):
        return _clean_parameters(params.model_dump(exclude_none=True))
    if isinstance(params, list):
        return [_clean_parameters(item) for item in params]
    if isinstance(params, Enum):
        return params.value
    return params


def _collect_tool_definitions(
    config: Optional[GenerateContentConfigOrDict],
) -> Optional[str]:
    """Extract tool definitions from config and return as JSON string.

    Covers ``Tool`` objects (with ``function_declarations``) and plain
    callable functions.  Returns ``None`` when no tools are present.
    """
    if config is None:
        return None

    if isinstance(config, dict):
        tools = config.get("tools")
    else:
        tools = getattr(config, "tools", None)

    if not tools:
        return None

    definitions: list[dict[str, Any]] = []
    try:
        for tool in tools:
            # Coerce dict-shaped tools (ToolDict) into Tool objects so the
            # isinstance(tool, Tool) branch handles them uniformly.
            if isinstance(tool, dict):
                try:
                    tool = Tool.model_validate(tool)
                except Exception:
                    continue

            if isinstance(tool, Tool):
                for fd in tool.function_declarations or []:
                    entry: dict[str, Any] = {
                        "type": "function",
                        "name": getattr(fd, "name", None) or type(fd).__name__,
                    }
                    desc = getattr(fd, "description", None)
                    if desc:
                        entry["description"] = desc
                    params = getattr(fd, "parameters", None)
                    if params is not None:
                        entry["parameters"] = _clean_parameters(params)
                    definitions.append(entry)
            elif callable(tool):
                entry = {
                    "type": "function",
                    "name": getattr(tool, "__name__", type(tool).__name__),
                }
                doc = getattr(tool, "__doc__", None)
                if doc:
                    entry["description"] = doc.strip()
                definitions.append(entry)

        if not definitions:
            return None
        return json.dumps(definitions)
    except Exception:
        _logger.debug("Failed to collect tool definitions", exc_info=True)
        return None


# ---------------------------------------------------------------------------
# LLMInvocation builder helpers
# ---------------------------------------------------------------------------


def _build_invocation(
    models_object: Union[Models, AsyncModels],
    model: str,
    contents: Union[ContentListUnion, ContentListUnionDict],
    config: Optional[GenerateContentConfigOrDict],
    allow_list: AllowList,
) -> LLMInvocation:
    """Build an LLMInvocation from Google GenAI request parameters."""
    genai_system = _determine_genai_system(models_object)

    # Convert request contents to InputMessage list
    input_messages = to_input_messages(
        contents=transformers.t_contents(contents)
    )

    # Extract system instructions from config and prepend as a system message
    system_content = _config_to_system_instruction(config)
    if system_content:
        try:
            transformed = transformers.t_contents(system_content)
        except (ValueError, TypeError):
            transformed = None
        if transformed:
            system_parts = to_system_instructions(content=transformed[0])
            if system_parts:
                input_messages = [
                    InputMessage(role="system", parts=system_parts),
                    *input_messages,
                ]

    config_dict: dict = _to_dict(config) if config else {}

    # Standard semconv fields → typed LLMInvocation fields. Most Google config
    # field names match semconv directly; a couple use different words (noted
    # in the comments below).
    backend = (
        "vertex_ai"
        if genai_system == _get_vertexai_system_name()
        else "gemini"
    )

    invocation = LLMInvocation(
        system=genai_system,
        provider="google",
        framework="google-genai-sdk",
        request_model=model,
        operation=_GENERATE_CONTENT_OP_NAME,
        input_messages=input_messages,
        request_temperature=config_dict.get("temperature"),
        request_top_p=config_dict.get("top_p"),
        request_top_k=config_dict.get("top_k"),
        # Google calls it `candidate_count`; semconv calls it `choice_count`.
        request_choice_count=config_dict.get("candidate_count"),
        # Google calls it `max_output_tokens`; semconv calls it `max_tokens`.
        request_max_tokens=config_dict.get("max_output_tokens"),
        request_frequency_penalty=config_dict.get("frequency_penalty"),
        request_presence_penalty=config_dict.get("presence_penalty"),
        request_seed=config_dict.get("seed"),
        output_type=_output_type_from_mime(
            config_dict.get("response_mime_type")
        ),
    )
    stop_sequences = config_dict.get("stop_sequences")
    if stop_sequences:
        invocation.request_stop_sequences = list(stop_sequences)

    # Vendor-specific config fields (safety_settings, thinking_config, etc.)
    # land in the freeform attributes dict under the gcp.* namespace, gated
    # by the operator-controlled allow-list.
    invocation.attributes = _capture_vendor_config_attributes(
        config_dict, allow_list
    )
    invocation.attributes.update(_get_extra_generate_content_attributes())

    # Gemini-specific span attribute: which backend is being targeted.
    invocation.attributes["gen_ai.google.request.backend"] = backend

    # Populate gen_ai.tool.definitions (JSON-serialized tool schemas).
    # Gated by env var to avoid serialization cost when the feature is disabled.
    if should_capture_tool_definitions():
        invocation.tool_definitions = _collect_tool_definitions(config)

    return invocation


# Prefixes for Google-specific attributes that the util-genai SpanEmitter
# would filter out (it only passes standard gen_ai.* semconv keys and
# custom_* keys).  These must be set directly on the span.
_VENDOR_ATTR_PREFIXES = ("gen_ai.google.", "gcp.gen_ai.")


def _set_vendor_attributes_on_span(invocation: LLMInvocation) -> None:
    """Copy vendor-prefixed attributes from *invocation.attributes* directly
    onto the span so they survive SpanEmitter filtering."""
    span = invocation.span
    if not span or not span.is_recording():
        return
    for key, value in (invocation.attributes or {}).items():
        if any(key.startswith(prefix) for prefix in _VENDOR_ATTR_PREFIXES):
            span.set_attribute(key, value)


def _apply_response(
    invocation: LLMInvocation,
    response: GenerateContentResponse,
) -> Optional[Error]:
    """Apply response data to an existing LLMInvocation.

    Returns an ``Error`` when the response carries no usable output (blocked
    prompt or no candidates). The caller is expected to route through
    ``handler.fail_llm`` in that case so ``error.type`` reaches the span via
    the ``on_error`` path. Returns ``None`` for a normal response.
    """
    # Output messages
    if response.candidates:
        invocation.output_messages = to_output_messages(
            candidates=response.candidates
        )

    # Finish reasons — use _to_finish_reason() for OTel semconv mapping
    finish_reasons = set()
    if response.candidates:
        for candidate in response.candidates:
            if candidate.finish_reason is not None:
                reason_str = _to_finish_reason(candidate.finish_reason)
                if reason_str:
                    finish_reasons.add(reason_str)
    invocation.response_finish_reasons = sorted(finish_reasons)

    # Response identity (semconv: gen_ai.response.model, gen_ai.response.id)
    model_version = getattr(response, "model_version", None)
    if model_version:
        invocation.response_model_name = model_version
        # Gemini-specific: actual resolved model version string
        invocation.attributes["gen_ai.google.response.model_version"] = (
            model_version
        )
    response_id = getattr(response, "response_id", None)
    if response_id:
        invocation.response_id = response_id

    # Gemini-specific usage attributes (set only when > 0)
    thoughts_tokens = _get_response_property(
        response, "usage_metadata.thoughts_token_count"
    )
    if isinstance(thoughts_tokens, int) and not isinstance(
        thoughts_tokens, bool
    ):
        if thoughts_tokens > 0:
            invocation.attributes["gen_ai.google.usage.thought_tokens"] = (
                thoughts_tokens
            )

    tool_use_prompt_tokens = _get_response_property(
        response, "usage_metadata.tool_use_prompt_token_count"
    )
    if isinstance(tool_use_prompt_tokens, int) and not isinstance(
        tool_use_prompt_tokens, bool
    ):
        if tool_use_prompt_tokens > 0:
            invocation.attributes[
                "gen_ai.google.usage.tool_use_prompt_tokens"
            ] = tool_use_prompt_tokens

    cached_content_tokens = _get_response_property(
        response, "usage_metadata.cached_content_token_count"
    )
    if isinstance(cached_content_tokens, int) and not isinstance(
        cached_content_tokens, bool
    ):
        if cached_content_tokens > 0:
            # Vendor-specific attribute (set on span via _set_vendor_attributes_on_span).
            invocation.attributes[
                "gen_ai.google.usage.cached_content_tokens"
            ] = cached_content_tokens
            # Standard semconv key — set directly on span so it survives
            # SpanEmitter filtering (which only passes gen_ai.* semconv keys).
            if invocation.span and invocation.span.is_recording():
                invocation.span.set_attribute(
                    gen_ai_attributes.GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS,
                    cached_content_tokens,
                )

    # Token counts. Guard against bool (isinstance(True, int) is True).
    input_tokens = _get_response_property(
        response, "usage_metadata.prompt_token_count"
    )
    output_tokens = _get_response_property(
        response, "usage_metadata.candidates_token_count"
    )
    if isinstance(input_tokens, int) and not isinstance(input_tokens, bool):
        invocation.input_tokens = input_tokens
    if isinstance(output_tokens, int) and not isinstance(output_tokens, bool):
        invocation.output_tokens = output_tokens

    # Error path for responses with no candidates. We surface these via
    # ``fail_llm`` so the span gets a proper ERROR status + ``error.type``;
    # ``stop_llm`` would silently drop the error.type because the
    # SpanEmitter filters non-``gen_ai.*`` supplemental attributes.
    if not response.candidates:
        if (
            (not response.prompt_feedback)
            or (not response.prompt_feedback.block_reason)
            or (
                response.prompt_feedback.block_reason
                == BlockedReason.BLOCKED_REASON_UNSPECIFIED
            )
        ):
            return Error(
                message="Response carried no candidates.",
                type=NoCandidatesError,
            )
        block_reason = response.prompt_feedback.block_reason.name.upper()
        return Error(
            message=f"Prompt was blocked: {block_reason}.",
            type=BlockedPromptError,
        )
    return None


# ---------------------------------------------------------------------------
# Streaming accumulation helpers
# ---------------------------------------------------------------------------


def _merge_candidates_by_index(
    candidates: list[Candidate],
) -> list[Candidate]:
    """Group streaming candidates by index, concatenating content parts.

    In Gemini streaming each chunk yields one ``Candidate`` at a given
    ``index`` with a partial text delta.  This helper merges all deltas
    for the same index into a single ``Candidate`` so that downstream
    ``to_output_messages`` emits one ``OutputMessage`` per choice index
    — matching non-streaming behaviour.

    ``finish_reason`` and ``safety_ratings`` are taken from the **last**
    candidate for each index (that is when the SDK provides them).
    """
    if not candidates:
        return []

    groups: dict[int, list] = {}
    for candidate in candidates:
        idx = candidate.index if candidate.index is not None else 0
        groups.setdefault(idx, []).append(candidate)

    merged: list = []
    for idx in sorted(groups):
        group = groups[idx]
        all_parts: list = []
        for c in group:
            if c.content and c.content.parts:
                all_parts.extend(c.content.parts)
        last = group[-1]
        merged_content = (
            Content(parts=all_parts, role="model") if all_parts else None
        )
        merged.append(
            Candidate(
                index=idx,
                content=merged_content,
                finish_reason=last.finish_reason,
                safety_ratings=getattr(last, "safety_ratings", None),
            )
        )
    return merged


def _build_accumulated_response(
    chunks: list[GenerateContentResponse],
) -> GenerateContentResponse:
    """Build a synthetic response from accumulated streaming chunks.

    * Candidates from all chunks are grouped by ``candidate.index`` and
      merged — content parts are concatenated, ``finish_reason`` and
      ``safety_ratings`` are taken from the last chunk for each index.
      This produces one ``OutputMessage`` per choice index, matching
      non-streaming telemetry shape.
    * ``usage_metadata`` is taken from the **last** chunk that carries it
      (the SDK provides cumulative counts on every chunk; the last one is
      the most complete).
    * ``model_version`` and ``response_id`` are taken from the **first**
      chunk that provides them.
    * ``prompt_feedback`` is taken from the **first** chunk that has it
      (a blocked-prompt signal is emitted immediately).
    """
    if not chunks:
        return GenerateContentResponse(candidates=[])

    all_candidates: list = []
    model_version = None
    response_id = None
    last_usage = None
    prompt_feedback = None

    for chunk in chunks:
        if chunk.candidates:
            all_candidates.extend(chunk.candidates)
        if model_version is None:
            mv = getattr(chunk, "model_version", None)
            if mv:
                model_version = mv
        if response_id is None:
            rid = getattr(chunk, "response_id", None)
            if rid:
                response_id = rid
        usage = getattr(chunk, "usage_metadata", None)
        if usage is not None:
            last_usage = usage
        if prompt_feedback is None:
            pf = getattr(chunk, "prompt_feedback", None)
            if pf:
                prompt_feedback = pf

    all_candidates = _merge_candidates_by_index(all_candidates)

    kwargs: dict[str, Any] = {
        "candidates": all_candidates if all_candidates else None,
        "usage_metadata": last_usage,
        "model_version": model_version,
        "response_id": response_id,
    }
    if prompt_feedback is not None:
        kwargs["prompt_feedback"] = prompt_feedback
    return GenerateContentResponse(**kwargs)


def _classify_error(error: BaseException) -> ErrorClassification:
    """Map an exception to the appropriate ``ErrorClassification``."""
    if isinstance(error, KeyboardInterrupt):
        return ErrorClassification.INTERRUPT
    if isinstance(error, (asyncio.CancelledError, GeneratorExit)):
        return ErrorClassification.CANCELLATION
    return ErrorClassification.REAL_ERROR


class _StreamFinalizer:
    """Shared finalization logic for sync and async stream wrappers.

    Subclasses must set ``self._invocation``, ``self._handler``, and
    ``self._chunks`` before any iteration begins.  The ``_record_ttfc``
    helper should be called on the first received chunk.
    """

    _invocation: LLMInvocation
    _handler: TelemetryHandler
    _chunks: list[GenerateContentResponse]
    _finished: bool
    _first_chunk_received: bool

    def _init_finalizer(
        self,
        invocation: LLMInvocation,
        handler: TelemetryHandler,
    ):
        self._invocation = invocation
        self._handler = handler
        self._chunks: list[GenerateContentResponse] = []
        self._finished = False
        self._first_chunk_received = False

    def _record_ttfc(self):
        """Record time-to-first-chunk on the first received chunk."""
        if not self._first_chunk_received:
            self._first_chunk_received = True
            self._invocation.attributes[
                GEN_AI_RESPONSE_TIME_TO_FIRST_CHUNK
            ] = timeit.default_timer() - self._invocation.start_time

    def __del__(self):
        if not getattr(self, "_finished", True):
            try:
                self._finalize()
            except Exception:  # pragma: no cover
                # __del__ may run from GC on an arbitrary thread/task where
                # ContextVar access is invalid.  Swallow to avoid noise.
                _logger.debug(
                    "Stream finalizer failed in __del__", exc_info=True
                )

    def _finalize(self):
        if self._finished:
            return
        self._finished = True
        try:
            accumulated = _build_accumulated_response(self._chunks)
            response_error = _apply_response(self._invocation, accumulated)
            _set_vendor_attributes_on_span(self._invocation)
            if response_error is None:
                self._handler.stop_llm(self._invocation)
            else:
                self._handler.fail_llm(self._invocation, response_error)
        except Exception:  # pragma: no cover - defensive
            pass

    def _handle_error(self, error):
        if self._finished:
            return
        self._finished = True
        if self._chunks:
            try:
                accumulated = _build_accumulated_response(self._chunks)
                # Note: _apply_response return value (Optional[Error]) is
                # intentionally discarded — the real exception takes
                # precedence over synthetic response errors.
                _apply_response(self._invocation, accumulated)
                _set_vendor_attributes_on_span(self._invocation)
            except Exception:
                _logger.debug(
                    "Failed to apply partial response on error",
                    exc_info=True,
                )
        classification = _classify_error(error)
        self._handler.fail_llm(
            self._invocation,
            Error(
                message=str(error),
                type=type(error),
                classification=classification,
            ),
        )


class _SyncStreamWrapper(_StreamFinalizer):
    """Wraps a sync streaming iterator, accumulating chunks for telemetry.

    Each ``__next__`` call transparently forwards the chunk to the caller
    while recording it for later aggregation.  On normal exhaustion
    (``StopIteration``), on error, or on early exit (``close``), the
    accumulated data is applied to the ``LLMInvocation`` and the span is
    finalised via the handler.
    """

    def __init__(
        self,
        stream,
        invocation: LLMInvocation,
        handler: TelemetryHandler,
    ):
        self._stream = iter(stream)
        self._init_finalizer(invocation, handler)

    def __iter__(self):
        return self

    def __next__(self) -> GenerateContentResponse:
        try:
            chunk = next(self._stream)
            self._record_ttfc()
            self._chunks.append(chunk)
            return chunk
        except StopIteration:
            self._finalize()
            raise
        except BaseException as error:
            self._handle_error(error)
            raise

    def close(self):
        """Finalize the span and delegate to the underlying stream."""
        self._finalize()
        if hasattr(self._stream, "close"):
            try:
                self._stream.close()
            except Exception:
                pass


class _AsyncStreamWrapper(_StreamFinalizer):
    """Wraps an async streaming iterator, accumulating chunks for telemetry.

    Async counterpart of ``_SyncStreamWrapper``.
    """

    def __init__(
        self,
        stream,
        invocation: LLMInvocation,
        handler: TelemetryHandler,
    ):
        self._stream = stream.__aiter__()
        self._init_finalizer(invocation, handler)

    def __aiter__(self):
        return self

    async def __anext__(self) -> GenerateContentResponse:
        try:
            chunk = await self._stream.__anext__()
            self._record_ttfc()
            self._chunks.append(chunk)
            return chunk
        except StopAsyncIteration:
            self._finalize()
            raise
        except BaseException as error:
            self._handle_error(error)
            raise

    async def aclose(self):
        """Finalize the span and delegate to the underlying stream."""
        self._finalize()
        if hasattr(self._stream, "aclose"):
            try:
                await self._stream.aclose()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Instrumented wrapper functions (sync + async)
# ---------------------------------------------------------------------------


def _create_instrumented_generate_content(
    snapshot: _MethodsSnapshot,
    handler: TelemetryHandler,
    generate_content_config_key_allowlist: AllowList,
):
    wrapped_func = snapshot.generate_content

    @functools.wraps(wrapped_func)
    def instrumented_generate_content(
        self: Models,
        *,
        model: str,
        contents: Union[ContentListUnion, ContentListUnionDict],
        config: Optional[GenerateContentConfigOrDict] = None,
        **kwargs: Any,
    ) -> GenerateContentResponse:
        if context_api.get_value(SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY):
            return wrapped_func(
                self,
                model=model,
                contents=contents,
                config=config,
                **kwargs,
            )
        invocation = _build_invocation(
            self,
            model,
            contents,
            config,
            generate_content_config_key_allowlist,
        )
        handler.start_llm(invocation)
        if invocation.span and invocation.span.is_recording():
            invocation.span.set_attribute(
                code_attributes.CODE_FUNCTION_NAME,
                _SYNC_CODE_FUNCTION_NAME,
            )
        _set_vendor_attributes_on_span(invocation)
        config = _wrap_config_tools(
            config, handler, invocation.system, invocation.provider
        )
        try:
            response = wrapped_func(
                self,
                model=model,
                contents=contents,
                config=config,
                **kwargs,
            )
        except Exception as error:
            handler.fail_llm(
                invocation,
                Error(message=str(error), type=type(error)),
            )
            raise
        try:
            response_error = _apply_response(invocation, response)
            _set_vendor_attributes_on_span(invocation)
            if response_error is None:
                handler.stop_llm(invocation)
            else:
                handler.fail_llm(invocation, response_error)
        except Exception:  # pragma: no cover - defensive
            pass
        return response

    return instrumented_generate_content


def _create_instrumented_generate_content_stream(
    snapshot: _MethodsSnapshot,
    handler: TelemetryHandler,
    generate_content_config_key_allowlist: AllowList,
):
    wrapped_func = snapshot.generate_content_stream

    @functools.wraps(wrapped_func)
    def instrumented_generate_content_stream(
        self: Models,
        *,
        model: str,
        contents: Union[ContentListUnion, ContentListUnionDict],
        config: Optional[GenerateContentConfigOrDict] = None,
        **kwargs: Any,
    ):
        if context_api.get_value(SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY):
            return wrapped_func(
                self,
                model=model,
                contents=contents,
                config=config,
                **kwargs,
            )
        invocation = _build_invocation(
            self,
            model,
            contents,
            config,
            generate_content_config_key_allowlist,
        )
        invocation.request_stream = True
        handler.start_llm(invocation)
        if invocation.span and invocation.span.is_recording():
            invocation.span.set_attribute(
                code_attributes.CODE_FUNCTION_NAME,
                _SYNC_STREAM_CODE_FUNCTION_NAME,
            )
        _set_vendor_attributes_on_span(invocation)
        config = _wrap_config_tools(
            config, handler, invocation.system, invocation.provider
        )
        try:
            stream = wrapped_func(
                self,
                model=model,
                contents=contents,
                config=config,
                **kwargs,
            )
        except Exception as error:
            handler.fail_llm(
                invocation,
                Error(message=str(error), type=type(error)),
            )
            raise
        return _SyncStreamWrapper(stream, invocation, handler)

    return instrumented_generate_content_stream


def _create_instrumented_async_generate_content(
    snapshot: _MethodsSnapshot,
    handler: TelemetryHandler,
    generate_content_config_key_allowlist: AllowList,
):
    wrapped_func = snapshot.async_generate_content

    @functools.wraps(wrapped_func)
    async def instrumented_generate_content(
        self: AsyncModels,
        *,
        model: str,
        contents: Union[ContentListUnion, ContentListUnionDict],
        config: Optional[GenerateContentConfigOrDict] = None,
        **kwargs: Any,
    ) -> GenerateContentResponse:
        if context_api.get_value(SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY):
            return await wrapped_func(
                self,
                model=model,
                contents=contents,
                config=config,
                **kwargs,
            )
        invocation = _build_invocation(
            self,
            model,
            contents,
            config,
            generate_content_config_key_allowlist,
        )
        handler.start_llm(invocation)
        if invocation.span and invocation.span.is_recording():
            invocation.span.set_attribute(
                code_attributes.CODE_FUNCTION_NAME,
                _ASYNC_CODE_FUNCTION_NAME,
            )
        _set_vendor_attributes_on_span(invocation)
        config = _wrap_config_tools(
            config, handler, invocation.system, invocation.provider
        )
        try:
            response = await wrapped_func(
                self,
                model=model,
                contents=contents,
                config=config,
                **kwargs,
            )
        except Exception as error:
            handler.fail_llm(
                invocation,
                Error(message=str(error), type=type(error)),
            )
            raise
        try:
            response_error = _apply_response(invocation, response)
            _set_vendor_attributes_on_span(invocation)
            if response_error is None:
                handler.stop_llm(invocation)
            else:
                handler.fail_llm(invocation, response_error)
        except Exception:  # pragma: no cover - defensive
            pass
        return response

    return instrumented_generate_content


def _create_instrumented_async_generate_content_stream(
    snapshot: _MethodsSnapshot,
    handler: TelemetryHandler,
    generate_content_config_key_allowlist: AllowList,
):
    wrapped_func = snapshot.async_generate_content_stream

    @functools.wraps(wrapped_func)
    async def instrumented_async_generate_content_stream(
        self: AsyncModels,
        *,
        model: str,
        contents: Union[ContentListUnion, ContentListUnionDict],
        config: Optional[GenerateContentConfigOrDict] = None,
        **kwargs: Any,
    ):
        if context_api.get_value(SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY):
            return await wrapped_func(
                self,
                model=model,
                contents=contents,
                config=config,
                **kwargs,
            )
        invocation = _build_invocation(
            self,
            model,
            contents,
            config,
            generate_content_config_key_allowlist,
        )
        invocation.request_stream = True
        handler.start_llm(invocation)
        if invocation.span and invocation.span.is_recording():
            invocation.span.set_attribute(
                code_attributes.CODE_FUNCTION_NAME,
                _ASYNC_STREAM_CODE_FUNCTION_NAME,
            )
        _set_vendor_attributes_on_span(invocation)
        config = _wrap_config_tools(
            config, handler, invocation.system, invocation.provider
        )
        try:
            stream = await wrapped_func(
                self,
                model=model,
                contents=contents,
                config=config,
                **kwargs,
            )
        except Exception as error:
            handler.fail_llm(
                invocation,
                Error(message=str(error), type=type(error)),
            )
            raise
        return _AsyncStreamWrapper(stream, invocation, handler)

    return instrumented_async_generate_content_stream


# ---------------------------------------------------------------------------
# Public API: instrument / uninstrument
# ---------------------------------------------------------------------------


def uninstrument_generate_content(snapshot: object):
    assert isinstance(snapshot, _MethodsSnapshot)
    snapshot.restore()


def instrument_generate_content(
    handler: TelemetryHandler,
    generate_content_config_key_allowlist: Optional[AllowList] = None,
) -> object:
    allow_list = generate_content_config_key_allowlist or AllowList()
    snapshot = _MethodsSnapshot()
    Models.generate_content = _create_instrumented_generate_content(
        snapshot, handler, allow_list
    )
    Models.generate_content_stream = (
        _create_instrumented_generate_content_stream(
            snapshot, handler, allow_list
        )
    )
    AsyncModels.generate_content = _create_instrumented_async_generate_content(
        snapshot, handler, allow_list
    )
    AsyncModels.generate_content_stream = (
        _create_instrumented_async_generate_content_stream(
            snapshot, handler, allow_list
        )
    )
    return snapshot
