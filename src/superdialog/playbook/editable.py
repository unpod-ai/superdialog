"""EditableDoc: prose-only mutation surface over playbook documents.

The optimizer's reflect step returns targeted ``{address, new_text}`` edits.
Applying them through this module enforces the prose whitelist by
construction: a non-whitelisted address raises ``MutationError`` and the
document is recompiled (re-validated) after every apply.
"""

from __future__ import annotations

import copy
import re
from typing import Any

import yaml
from pydantic import BaseModel

# _YamlLoader subclasses yaml.SafeLoader and only overrides boolean
# resolution (YAML 1.2: on/off/yes/no stay strings). Loading with it is a
# safe load — it cannot construct arbitrary Python objects — and mirrors
# what Playbook.from_yaml itself does.
from .models import Playbook, _YamlLoader
from .simple import is_simple_playbook, simple_to_playbook


class MutationError(ValueError):
    """An edit addressed a frozen field or carried a bad payload."""


class Edit(BaseModel):
    """One targeted prose edit, addressed into the source document."""

    address: str
    new_text: str | list[str]


class FieldRef(BaseModel):
    """An editable field: its address and current text."""

    address: str
    text: str | list[str]


_RULE_RE = re.compile(r"^advance_when\[(\d+)\]\.when$")


class FullDoc:
    """Editable view over a full-format playbook document."""

    def __init__(self, doc: dict[str, Any]) -> None:
        self._doc = doc
        self.compile()  # validate eagerly; raises on a broken document

    @classmethod
    def from_text(cls, text: str) -> "FullDoc":
        """Parse full-format YAML/JSON text (models' YAML-1.2 loader)."""
        return cls(yaml.load(text, Loader=_YamlLoader))

    def compile(self) -> Playbook:
        """Validate and return the runtime Playbook."""
        return Playbook.model_validate(self._doc)

    def emit(self) -> str:
        """Serialize the document back to YAML, preserving key order."""
        return yaml.safe_dump(self._doc, sort_keys=False, allow_unicode=True)

    def fields(self) -> list[FieldRef]:
        """Enumerate every whitelisted (editable) field with current text."""
        refs = [FieldRef(address="persona", text=self._doc.get("persona", ""))]
        for jname, journey in self._doc.get("journeys", {}).items():
            for cp in journey.get("checkpoints", []):
                base = f"journeys.{jname}.checkpoints.{cp['id']}"
                refs.append(
                    FieldRef(address=f"{base}.guidance", text=cp.get("guidance", ""))
                )
                refs.append(FieldRef(address=f"{base}.goal", text=cp.get("goal", "")))
                refs.append(
                    FieldRef(address=f"{base}.never_say", text=cp.get("never_say", []))
                )
                if cp.get("say_verbatim") is not None:
                    refs.append(
                        FieldRef(
                            address=f"{base}.say_verbatim", text=cp["say_verbatim"]
                        )
                    )
                for slot_name, spec in (cp.get("slots") or {}).items():
                    refs.append(
                        FieldRef(
                            address=f"{base}.slots.{slot_name}.description",
                            text=(spec or {}).get("description", ""),
                        )
                    )
                for i, rule in enumerate(cp.get("advance_when", [])):
                    if rule.get("judge", "llm") == "llm":
                        refs.append(
                            FieldRef(
                                address=f"{base}.advance_when[{i}].when",
                                text=rule.get("when", ""),
                            )
                        )
        return refs

    def apply(self, edits: list[Edit]) -> "FullDoc":
        """Return a new FullDoc with edits applied; reject frozen addresses."""
        allowed = {f.address: f.text for f in self.fields()}
        new = copy.deepcopy(self._doc)
        for edit in edits:
            if edit.address not in allowed:
                raise MutationError(f"address not editable: {edit.address}")
            _check_payload(edit, allowed[edit.address])
            _set_full(new, edit.address, edit.new_text)
        return FullDoc(new)


def _check_payload(edit: Edit, current: str | list[str]) -> None:
    """List fields take grow-only lists of strings; others take strings."""
    if isinstance(current, list):
        if not isinstance(edit.new_text, list) or not all(
            isinstance(s, str) for s in edit.new_text
        ):
            raise MutationError(f"{edit.address}: expected a list of strings")
        if len(edit.new_text) < len(current):
            raise MutationError(f"{edit.address}: entries may not be removed")
    elif not isinstance(edit.new_text, str):
        raise MutationError(f"{edit.address}: expected a string")


def _set_full(doc: dict[str, Any], address: str, value: str | list[str]) -> None:
    """Write `value` at a (pre-validated) FullDoc address inside the dict."""
    if address == "persona":
        doc["persona"] = value
        return
    parts = address.split(".")
    # journeys.<j>.checkpoints.<id>.<rest...>
    journey = doc["journeys"][parts[1]]
    cp = next(c for c in journey["checkpoints"] if c["id"] == parts[3])
    rest = parts[4:]
    if rest[0] == "slots":
        cp["slots"][rest[1]]["description"] = value
        return
    rule_match = _RULE_RE.match(".".join(rest))
    if rule_match:
        cp["advance_when"][int(rule_match.group(1))]["when"] = value
        return
    cp[rest[0]] = value


class SimpleDoc:
    """Editable view over a simple-format playbook document."""

    def __init__(self, doc: dict[str, Any]) -> None:
        self._doc = doc
        self.compile()  # validate eagerly

    @classmethod
    def from_text(cls, text: str) -> "SimpleDoc":
        """Parse simple-format YAML/JSON text (same loader as load_simple)."""
        return cls(yaml.safe_load(text))

    def compile(self) -> Playbook:
        """Lower to the validated runtime Playbook."""
        return simple_to_playbook(self._doc)

    def emit(self) -> str:
        """Serialize back to simple-format YAML, preserving key order."""
        return yaml.safe_dump(self._doc, sort_keys=False, allow_unicode=True)

    def fields(self) -> list[FieldRef]:
        """Enumerate the simple-format prose whitelist."""
        persona = self._doc.get("persona") or {}
        refs = [
            FieldRef(address="opening", text=self._doc.get("opening", "")),
            FieldRef(address="closing", text=self._doc.get("closing", "")),
            FieldRef(address="persona.identity", text=persona.get("identity", "")),
            FieldRef(
                address="persona.voice_style", text=persona.get("voice_style", "")
            ),
        ]
        for step in self._doc.get("playbook", []):
            base = f"steps.{step['id']}"
            refs.append(FieldRef(address=f"{base}.say", text=step.get("say", "")))
            refs.append(
                FieldRef(address=f"{base}.done_when", text=step.get("done_when", ""))
            )
            refs.append(
                FieldRef(address=f"{base}.purpose", text=step.get("purpose", ""))
            )
        return refs

    def apply(self, edits: list[Edit]) -> "SimpleDoc":
        """Return a new SimpleDoc with edits applied; reject frozen addresses."""
        allowed = {f.address: f.text for f in self.fields()}
        new = copy.deepcopy(self._doc)
        for edit in edits:
            if edit.address not in allowed:
                raise MutationError(f"address not editable: {edit.address}")
            _check_payload(edit, allowed[edit.address])
            _set_simple(new, edit.address, edit.new_text)
        return SimpleDoc(new)


def _set_simple(doc: dict[str, Any], address: str, value: str | list[str]) -> None:
    """Write `value` at a (pre-validated) SimpleDoc address inside the dict."""
    if address.startswith("steps."):
        _, step_id, field = address.split(".")
        step = next(s for s in doc["playbook"] if s["id"] == step_id)
        step[field] = value
    elif address.startswith("persona."):
        doc.setdefault("persona", {})[address.split(".", 1)[1]] = value
    else:
        doc[address] = value


EditableDoc = FullDoc | SimpleDoc


def make_editable(text: str) -> EditableDoc:
    """Detect the source format and wrap it in the matching editable doc."""
    probe = yaml.safe_load(text)
    if is_simple_playbook(probe):
        return SimpleDoc(probe)
    return FullDoc.from_text(text)
