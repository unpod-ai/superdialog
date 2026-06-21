"""Deterministic tool serialization for prompt caching.

The toolcall engine marks the tools array as a cacheable prefix, so the tool
JSON must be byte-identical turn-to-turn. ``_descriptors_to_openai_tools``
canonicalizes key order (preserving author-meaningful ``properties`` order).
"""

from __future__ import annotations

import json

from superdialog.machine.adapters.toolcall_adapter import (
    _canonical_json_schema,
    _descriptors_to_openai_tools,
)
from superdialog.machine.models import ToolDescriptor


def _desc() -> ToolDescriptor:
    return ToolDescriptor(
        id="book_slot",
        description="Book a tee time.",
        input_schema={
            "type": "object",
            "required": ["city", "date"],
            "properties": {
                "city": {"type": "string", "description": "City name"},
                "date": {"type": "string"},
            },
        },
    )


def test_canonical_sorts_metadata_keys() -> None:
    out = _canonical_json_schema(
        {"type": "object", "required": ["a"], "properties": {"x": {"type": "string"}}}
    )
    # top-level keys sorted: properties, required, type
    assert list(out.keys()) == ["properties", "required", "type"]


def test_canonical_preserves_property_order() -> None:
    """Property declaration order is author-meaningful and must NOT be sorted."""
    out = _canonical_json_schema(
        {"properties": {"zebra": {"type": "string"}, "apple": {"type": "string"}}}
    )
    assert list(out["properties"].keys()) == ["zebra", "apple"]  # not alphabetized


def test_canonical_is_idempotent_and_stable() -> None:
    schema = _desc().input_schema
    once = _canonical_json_schema(schema)
    twice = _canonical_json_schema(once)
    assert json.dumps(once) == json.dumps(twice)


def test_tools_byte_identical_across_builds() -> None:
    """Same descriptors -> byte-identical tool JSON (the cache prefix is stable)."""
    descriptors = [_desc()]
    a = _descriptors_to_openai_tools(descriptors)
    b = _descriptors_to_openai_tools(descriptors)
    assert json.dumps(a) == json.dumps(b)


def test_tools_byte_stable_against_shuffled_schema_keys() -> None:
    """Two descriptors with the same schema but different dict key insertion
    order produce identical tool JSON after canonicalization."""
    d1 = ToolDescriptor(
        id="t",
        description="d",
        input_schema={"type": "object", "properties": {"a": {"type": "string"}}},
    )
    d2 = ToolDescriptor(
        id="t",
        description="d",
        input_schema={"properties": {"a": {"type": "string"}}, "type": "object"},
    )
    assert json.dumps(_descriptors_to_openai_tools([d1])) == json.dumps(
        _descriptors_to_openai_tools([d2])
    )


def test_tool_shape_preserved() -> None:
    tool = _descriptors_to_openai_tools([_desc()])[0]
    assert tool["type"] == "function"
    assert tool["function"]["name"] == "book_slot"
    assert tool["function"]["description"] == "Book a tee time."
    # properties retained, order preserved
    assert list(tool["function"]["parameters"]["properties"].keys()) == ["city", "date"]
