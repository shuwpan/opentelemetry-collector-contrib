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

from __future__ import annotations

import logging
from enum import Enum

from google.genai import types as genai_types

from opentelemetry.util.genai.types import (
    FinishReason,
    InputMessage,
    MessagePart,
    OutputMessage,
    Text,
    ToolCallResponse,
)


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


_logger = logging.getLogger(__name__)


def to_input_messages(
    *,
    contents: list[genai_types.Content],
) -> list[InputMessage]:
    return [_to_input_message(content) for content in contents]


def to_output_messages(
    *,
    candidates: list[genai_types.Candidate],
) -> list[OutputMessage]:
    def content_to_output_message(
        candidate: genai_types.Candidate,
    ) -> OutputMessage | None:
        if not candidate.content:
            return None

        message = _to_input_message(candidate.content)
        return OutputMessage(
            finish_reason=_to_finish_reason(candidate.finish_reason),
            role=message.role,
            parts=message.parts,
        )

    messages = (
        content_to_output_message(candidate) for candidate in candidates
    )
    return [message for message in messages if message is not None]


def to_system_instructions(
    *,
    content: genai_types.Content,
) -> list[MessagePart]:
    parts = (
        _to_part(part, idx) for idx, part in enumerate(content.parts or [])
    )
    return [part for part in parts if part is not None]


def _to_input_message(
    content: genai_types.Content,
) -> InputMessage:
    parts = (
        _to_part(part, idx) for idx, part in enumerate(content.parts or [])
    )
    return InputMessage(
        role=_to_role(content.role),
        # filter Nones
        parts=[part for part in parts if part is not None],
    )


def _to_part(part: genai_types.Part, idx: int) -> MessagePart | None:
    # Thinking-model parts (e.g. Gemini 2.5) have `thought=True`.
    # These are internal chain-of-thought and should not appear in telemetry.
    if getattr(part, "thought", False):
        return None

    def tool_call_id(name: str | None) -> str:
        if name:
            return f"{name}_{idx}"
        return f"{idx}"

    if (text := part.text) is not None:
        return Text(content=text)

    # Blob (inline_data), Uri (file_data), and ToolCallRequest (function_call)
    # types are not yet available in util-genai. Skipped until HYBIM-604.
    if part.inline_data:
        _logger.debug("Skipping inline_data part (Blob type not available)")
        return None

    if part.file_data:
        _logger.debug("Skipping file_data part (Uri type not available)")
        return None

    if part.function_call:
        _logger.debug(
            "Skipping function_call part (ToolCallRequest type not available)"
        )
        return None

    if response := part.function_response:
        return ToolCallResponse(
            id=response.id or tool_call_id(response.name),
            response=response.response,
        )

    _logger.info("Unknown part dropped from telemetry %s", part)
    return None


def _to_role(role: str | None) -> str:
    if role == "user":
        return Role.USER.value
    if role == "model":
        return Role.ASSISTANT.value
    return ""


_CONTENT_FILTER_REASONS: frozenset[genai_types.FinishReason] = frozenset(
    r
    for name in (
        "SAFETY",
        "IMAGE_SAFETY",
        "BLOCKLIST",
        "PROHIBITED_CONTENT",
        "IMAGE_PROHIBITED_CONTENT",
        "SPII",
        "RECITATION",
        "IMAGE_RECITATION",
        "LANGUAGE",
    )
    if (r := getattr(genai_types.FinishReason, name, None)) is not None
)

_ERROR_REASONS: frozenset[genai_types.FinishReason] = frozenset(
    r
    for name in (
        "FINISH_REASON_UNSPECIFIED",
        "OTHER",
        "IMAGE_OTHER",
        "UNEXPECTED_TOOL_CALL",
        "MALFORMED_FUNCTION_CALL",
        "NO_IMAGE",
    )
    if (r := getattr(genai_types.FinishReason, name, None)) is not None
)


def _to_finish_reason(
    finish_reason: genai_types.FinishReason | None,
) -> FinishReason | str:
    if finish_reason is None:
        return ""
    if finish_reason in _ERROR_REASONS:
        return "error"
    if finish_reason is genai_types.FinishReason.STOP:
        return "stop"
    if finish_reason is genai_types.FinishReason.MAX_TOKENS:
        return "length"
    if finish_reason in _CONTENT_FILTER_REASONS:
        return "content_filter"
    # If there is no 1:1 mapping to an OTel preferred enum value, use the exact reason
    return finish_reason.name.lower()
