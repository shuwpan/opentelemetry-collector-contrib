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
import json
import logging
import os
from typing import Any, Optional, Union

from google.genai.models import AsyncModels, Models
from google.genai.models import t as transformers
from google.genai.types import (
    BlockedReason,
    ContentListUnion,
    ContentListUnionDict,
    ContentUnion,
    GenerateContentConfig,
    GenerateContentConfigOrDict,
    GenerateContentResponse,
)

from opentelemetry import context as context_api
from opentelemetry.semconv._incubating.attributes import (
    code_attributes,
    gen_ai_attributes,
)
from opentelemetry.util.genai.attributes import (
    SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY,
)
from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.types import (
    Error,
    InputMessage,
    LLMInvocation,
)
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

_logger = logging.getLogger(__name__)

_SYNC_CODE_FUNCTION_NAME = "google.genai.Models.generate_content"
_ASYNC_CODE_FUNCTION_NAME = "google.genai.AsyncModels.generate_content"

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
        system_parts = to_system_instructions(
            content=transformers.t_contents(system_content)[0]
        )
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

    return invocation


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
            invocation.attributes[
                "gen_ai.google.usage.cached_content_tokens"
            ] = cached_content_tokens

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
# Instrumented wrapper functions (sync + async, no streaming yet)
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
            if response_error is None:
                handler.stop_llm(invocation)
            else:
                handler.fail_llm(invocation, response_error)
        except Exception:
            _logger.exception("Failed to apply response telemetry")
            try:
                handler.stop_llm(invocation)
            except Exception:
                pass
        return response

    return instrumented_generate_content


def _create_instrumented_generate_content_stream(
    snapshot: _MethodsSnapshot,
    handler: TelemetryHandler,
    generate_content_config_key_allowlist: AllowList,
):
    # Streaming is deferred (HYBIM-665). Wrap so the log fires per call.
    original = snapshot.generate_content_stream

    @functools.wraps(original)
    def passthrough(self, *args, **kwargs):
        _logger.debug("generate_content_stream not instrumented (HYBIM-665)")
        return original(self, *args, **kwargs)

    return passthrough


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
            if response_error is None:
                handler.stop_llm(invocation)
            else:
                handler.fail_llm(invocation, response_error)
        except Exception:
            _logger.exception("Failed to apply response telemetry")
            try:
                handler.stop_llm(invocation)
            except Exception:
                pass
        return response

    return instrumented_generate_content


def _create_instrumented_async_generate_content_stream(
    snapshot: _MethodsSnapshot,
    handler: TelemetryHandler,
    generate_content_config_key_allowlist: AllowList,
):
    # Streaming is deferred (HYBIM-665). Wrap so the log fires per call.
    original = snapshot.async_generate_content_stream

    @functools.wraps(original)
    async def passthrough(self, *args, **kwargs):
        _logger.debug(
            "async_generate_content_stream not instrumented (HYBIM-665)"
        )
        return await original(self, *args, **kwargs)

    return passthrough


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
