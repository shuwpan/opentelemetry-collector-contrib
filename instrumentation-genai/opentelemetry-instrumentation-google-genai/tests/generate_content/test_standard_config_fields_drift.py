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

"""Drift-detection test for the standard-config field set.

``_STANDARD_CONFIG_FIELDS`` in ``generate_content.py`` lists every Google
``GenerateContentConfig`` field name that we surface as a typed
``LLMInvocation`` field. The same names must be referenced inside
``_build_invocation`` via ``config_dict.get("<name>")`` (or
``stop_sequences = config_dict.get("stop_sequences")``).

If the two drift apart, the vendor-attribute capture step would either:

* duplicate a standard semconv field under the ``gcp.*`` namespace (forgot
  to add the name to ``_STANDARD_CONFIG_FIELDS``), or
* exclude a field nobody actually populates as a typed semconv field
  (forgot to remove the name).

This test parses ``_build_invocation`` with ``ast`` and compares the two
sets so future contributors get a loud failure instead of silent drift.
"""

import ast
import inspect

from opentelemetry.instrumentation.google_genai import generate_content


def _config_dict_get_keys(func) -> set[str]:
    """Return the literal keys passed to ``config_dict.get(...)`` in ``func``."""
    source = inspect.getsource(func)
    tree = ast.parse(inspect.cleandoc(source))
    keys: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        callee = node.func
        if (
            isinstance(callee, ast.Attribute)
            and callee.attr == "get"
            and isinstance(callee.value, ast.Name)
            and callee.value.id == "config_dict"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            keys.add(node.args[0].value)
    return keys


def test_standard_config_fields_match_build_invocation_lookups():
    declared = set(generate_content._STANDARD_CONFIG_FIELDS)
    referenced = _config_dict_get_keys(generate_content._build_invocation)

    missing_from_declared = referenced - declared
    missing_from_referenced = declared - referenced

    assert not missing_from_declared, (
        "These config keys are read inside _build_invocation but are not "
        "listed in _STANDARD_CONFIG_FIELDS, so vendor-attribute capture "
        "will duplicate them under the gcp.* namespace: "
        f"{sorted(missing_from_declared)}"
    )
    assert not missing_from_referenced, (
        "These config keys are listed in _STANDARD_CONFIG_FIELDS but are "
        "not read inside _build_invocation, so vendor-attribute capture "
        "is excluding fields nobody populates as typed semconv fields: "
        f"{sorted(missing_from_referenced)}"
    )
