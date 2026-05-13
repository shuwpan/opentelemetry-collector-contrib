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
import inspect
import json
import logging
from typing import Any, Callable, Optional, Union

from google.genai.types import (
    ToolListUnion,
    ToolListUnionDict,
    ToolOrDict,
)

from opentelemetry import trace
from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.types import Error, ToolCall
from opentelemetry.util.genai.utils import get_content_capturing_mode

_logger = logging.getLogger(__name__)

ToolFunction = Callable[..., Any]


def _is_primitive(value):
    return isinstance(value, (str, int, bool, float))


def _to_otel_value(python_value):
    """Coerces parameters to something representable with Open Telemetry."""
    if python_value is None or _is_primitive(python_value):
        return python_value
    if isinstance(python_value, list):
        return [_to_otel_value(x) for x in python_value]
    if isinstance(python_value, dict):
        return {
            key: _to_otel_value(val) for (key, val) in python_value.items()
        }
    if hasattr(python_value, "model_dump"):
        return python_value.model_dump()
    if hasattr(python_value, "__dict__"):
        return _to_otel_value(python_value.__dict__)
    return repr(python_value)


def _is_homogenous_primitive_list(value):
    if not isinstance(value, list):
        return False
    if not value:
        return True
    if not _is_primitive(value[0]):
        return False
    first_type = type(value[0])
    for entry in value[1:]:
        if not isinstance(entry, first_type):
            return False
    return True


def _to_otel_attribute(python_value):
    otel_value = _to_otel_value(python_value)
    if _is_primitive(otel_value) or _is_homogenous_primitive_list(otel_value):
        return otel_value
    return json.dumps(otel_value)


def _is_capture_content_enabled() -> bool:
    from opentelemetry.util.genai.types import ContentCapturingMode

    mode = get_content_capturing_mode()
    return mode in (
        ContentCapturingMode.SPAN_ONLY,
        ContentCapturingMode.SPAN_AND_EVENT,
    )


def _record_function_call_argument(
    span, param_name, param_value, include_values
):
    attribute_prefix = f"code.function.parameters.{param_name}"
    span.set_attribute(f"{attribute_prefix}.type", type(param_value).__name__)
    if include_values:
        span.set_attribute(
            f"{attribute_prefix}.value", _to_otel_attribute(param_value)
        )


def _record_function_call_arguments(
    tool_function, function_args, function_kwargs
):
    """Records function invocation details as span attributes on the current span."""
    include_values = _is_capture_content_enabled()
    span = trace.get_current_span()
    signature = inspect.signature(tool_function)
    params = list(signature.parameters.values())
    for index, entry in enumerate(function_args):
        param_name = (
            params[index].name if index < len(params) else f"args[{index}]"
        )
        _record_function_call_argument(span, param_name, entry, include_values)
    for key, value in function_kwargs.items():
        _record_function_call_argument(span, key, value, include_values)


def _record_function_call_result(result):
    """Records function return value details as span attributes on the current span."""
    include_values = _is_capture_content_enabled()
    span = trace.get_current_span()
    span.set_attribute("code.function.return.type", type(result).__name__)
    if include_values:
        span.set_attribute(
            "code.function.return.value", _to_otel_attribute(result)
        )


def _build_tool_call(
    tool_function: ToolFunction,
    system: Optional[str],
    provider: Optional[str] = None,
) -> ToolCall:
    """Build a ToolCall invocation from a Python function."""
    tool_call = ToolCall(
        name=tool_function.__name__,
        tool_type="function",
        system=system,
        provider=provider,
    )
    if tool_function.__doc__:
        tool_call.tool_description = tool_function.__doc__
    tool_call.attributes["code.function.name"] = tool_function.__name__
    tool_call.attributes["code.module"] = tool_function.__module__
    return tool_call


def _wrap_sync_tool_function(
    tool_function: ToolFunction,
    handler: TelemetryHandler,
    system: Optional[str] = None,
    provider: Optional[str] = None,
):
    @functools.wraps(tool_function)
    def wrapped_function(*args, **kwargs):
        tool_call = _build_tool_call(tool_function, system, provider)
        tool_call.attributes["code.args.positional.count"] = len(args)
        tool_call.attributes["code.args.keyword.count"] = len(kwargs)
        handler.start_tool_call(tool_call)
        try:
            _record_function_call_arguments(tool_function, args, kwargs)
            result = tool_function(*args, **kwargs)
            _record_function_call_result(result)
            handler.stop_tool_call(tool_call)
            return result
        except Exception as error:
            handler.fail_tool_call(
                tool_call, Error(message=str(error), type=type(error))
            )
            raise

    return wrapped_function


def _wrap_async_tool_function(
    tool_function: ToolFunction,
    handler: TelemetryHandler,
    system: Optional[str] = None,
    provider: Optional[str] = None,
):
    @functools.wraps(tool_function)
    async def wrapped_function(*args, **kwargs):
        tool_call = _build_tool_call(tool_function, system, provider)
        tool_call.attributes["code.args.positional.count"] = len(args)
        tool_call.attributes["code.args.keyword.count"] = len(kwargs)
        handler.start_tool_call(tool_call)
        try:
            _record_function_call_arguments(tool_function, args, kwargs)
            result = await tool_function(*args, **kwargs)
            _record_function_call_result(result)
            handler.stop_tool_call(tool_call)
            return result
        except Exception as error:
            handler.fail_tool_call(
                tool_call, Error(message=str(error), type=type(error))
            )
            raise

    return wrapped_function


def _wrap_tool_function(
    tool_function: ToolFunction,
    handler: TelemetryHandler,
    system: Optional[str] = None,
    provider: Optional[str] = None,
):
    if inspect.iscoroutinefunction(tool_function):
        return _wrap_async_tool_function(
            tool_function, handler, system, provider
        )
    return _wrap_sync_tool_function(tool_function, handler, system, provider)


def wrapped(
    tool_or_tools: Optional[
        Union[ToolFunction, ToolOrDict, ToolListUnion, ToolListUnionDict]
    ],
    handler: TelemetryHandler,
    system: Optional[str] = None,
    provider: Optional[str] = None,
):
    if tool_or_tools is None:
        return None
    if isinstance(tool_or_tools, list):
        result = [
            wrapped(item, handler, system, provider) for item in tool_or_tools
        ]
        # Return the original list when nothing changed so the caller's
        # ``if wrapped is tools`` identity check can short-circuit.
        if all(r is o for r, o in zip(result, tool_or_tools)):
            return tool_or_tools
        return result
    # Check callable before dict so that callable objects that also
    # satisfy isinstance(x, dict) are wrapped as functions, not recursed.
    # Note: ToolDict items inside a list still hit the dict branch below
    # and are walked harmlessly (their values are not callable, so they
    # pass through unchanged).
    if callable(tool_or_tools):
        return _wrap_tool_function(tool_or_tools, handler, system, provider)
    if isinstance(tool_or_tools, dict):
        return {
            key: wrapped(value, handler, system, provider)
            for (key, value) in tool_or_tools.items()
        }
    return tool_or_tools
