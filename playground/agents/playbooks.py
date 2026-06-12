"""Playbook registry — discover superdialog Playbooks and build a playbook machine.

Every runnable playbook under the playbooks directory (``PLAYBOOKS_DIR`` env, else
the sibling superdialog ``examples/playbooks``) is exposed to the UI. The worker
runs a single-playbook :class:`DialogMachine` in ``engine="playbook"`` mode for the
chosen file — the framework's default compound Talker/Director runtime.

Playbooks are keyed by filename stem; that id is what the UI sends on the wire.
Persona suites (``*.personas.yaml`` — eval fixtures, a top-level list) and any file
that is not a valid playbook are skipped, so only runnable playbooks are listed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from superdialog import DialogMachine
from superdialog.playbook import Playbook

_PLAYBOOK_GLOBS = ("*.yaml", "*.yml")

# Per-file parse cache: path -> (mtime, raw_doc, playbook | None). The directory
# is re-globbed on every call so newly added/edited/removed playbooks appear
# without restarting the harness; the mtime guard avoids re-parsing large
# unchanged YAMLs (the golf playbook is ~168 KB).
_parse_cache: dict[Path, tuple[float, dict[str, Any], "Playbook | None"]] = {}


@dataclass(frozen=True)
class PlaybookInfo:
    """Display metadata for one playbook (consumed by the UI sidebar)."""

    id: str
    label: str
    goal: str
    journeys: int
    checkpoints: int
    initial: str
    description: str


def _playbooks_dir() -> Path:
    """Resolve the playbooks directory (env override, else superdialog examples)."""
    env = os.getenv("PLAYBOOKS_DIR")
    if env:
        return Path(env).expanduser()
    # This package lives at <repo>/playground/agents/, so examples/ is two up.
    return Path(__file__).resolve().parents[2] / "examples" / "playbooks"


def _drafts_dir() -> Path:
    """Resolve the local drafts overlay dir (env override, else playground/.drafts).

    Drafts are unvalidated working copies the editor saves; a call runs the draft
    in preference to the committed example until it is published.
    """
    env = os.getenv("PLAYBOOK_DRAFTS_DIR")
    if env:
        return Path(env).expanduser()
    # This package lives at <repo>/playground/agents/, so .drafts/ is one up.
    return Path(__file__).resolve().parents[1] / ".drafts"


def draft_path(playbook_id: str) -> Path:
    """Path the local draft for ``playbook_id`` would occupy (may not exist)."""
    return _drafts_dir() / f"{playbook_id}.yaml"


def canonical_path(playbook_id: str) -> Path | None:
    """The committed example file for ``playbook_id`` (None if unknown)."""
    entry = _scan().get(playbook_id)
    return entry[0] if entry else None


def effective_path(playbook_id: str) -> Path | None:
    """Draft if one exists, else the canonical file (None if unknown)."""
    dp = draft_path(playbook_id)
    if dp.exists():
        return dp
    return canonical_path(playbook_id)


def _label(stem: str) -> str:
    """Turn a filename stem into a human-readable label."""
    words = stem.replace(".", " ").replace("_", " ").split()
    return " ".join(w.capitalize() for w in words) or stem


def _raw_doc(path: Path) -> dict[str, Any]:
    """Best-effort raw YAML mapping (name/goal live here in the simple format)."""
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _parse(path: Path) -> tuple[dict[str, Any], "Playbook | None"]:
    """Return ``(raw_doc, playbook)`` for ``path``, re-parsing only on mtime change.

    ``playbook`` is ``None`` when the file is not a runnable playbook (no
    ``playbook``/``journeys`` key) or fails to load — the caller skips those.
    """
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return {}, None
    cached = _parse_cache.get(path)
    if cached is not None and cached[0] == mtime:
        return cached[1], cached[2]
    doc = _raw_doc(path)
    playbook: Playbook | None = None
    if doc and ("playbook" in doc or "journeys" in doc):
        try:
            playbook = Playbook.load(str(path))
        except Exception:
            playbook = None  # malformed — skip rather than crash the listing
    _parse_cache[path] = (mtime, doc, playbook)
    return doc, playbook


def _scan() -> dict[str, tuple[Path, dict[str, Any], Playbook]]:
    """Map runnable-playbook id (stem) → ``(path, raw_doc, playbook)``, id-ordered.

    Re-globs ``PLAYBOOKS_DIR`` every call so added/edited/removed playbooks are
    reflected live (no harness restart). Persona suites (``*.personas.yaml`` —
    eval fixtures, a top-level list) and non-playbook files are skipped.
    """
    out: dict[str, tuple[Path, dict[str, Any], Playbook]] = {}
    directory = _playbooks_dir()
    if not directory.exists():
        return out
    for pattern in _PLAYBOOK_GLOBS:
        for path in sorted(directory.glob(pattern)):
            if ".personas" in path.name:
                continue
            doc, playbook = _parse(path)
            if playbook is not None:
                out[path.stem] = (path, doc, playbook)
    return out


def _first_checkpoint_goal(playbook: Playbook) -> str:
    """First non-empty checkpoint goal across journeys, for a description fallback."""
    for journey in (getattr(playbook, "journeys", {}) or {}).values():
        for checkpoint in getattr(journey, "checkpoints", []) or []:
            goal = (getattr(checkpoint, "goal", "") or "").strip()
            if goal:
                return goal
    return ""


def playbook_registry() -> list[PlaybookInfo]:
    """Return display metadata for every runnable playbook, in id order."""
    infos: list[PlaybookInfo] = []
    for pid, (_, doc, playbook) in _scan().items():
        journeys = getattr(playbook, "journeys", {}) or {}
        checkpoints = sum(
            len(getattr(j, "checkpoints", []) or []) for j in journeys.values()
        )
        goal = (doc.get("goal") or "").strip() or _first_checkpoint_goal(playbook)
        persona = getattr(playbook, "persona", "")
        persona_text = " ".join(persona.split()) if isinstance(persona, str) else ""
        description = (goal or persona_text)[:140]
        try:
            initial = playbook.initial_checkpoint_id
        except Exception:
            initial = ""
        infos.append(
            PlaybookInfo(
                id=pid,
                label=(doc.get("name") or _label(pid)).strip(),
                goal=goal[:140],
                journeys=len(journeys),
                checkpoints=checkpoints,
                initial=initial,
                description=description,
            )
        )
    return infos


def build(llm: str, playbook_id: str) -> Any:
    """Build a single-playbook ``DialogMachine`` (engine='playbook') for ``id``.

    Loads the effective source (draft if present, else canonical) fresh from
    disk so a call always runs the latest saved playbook.
    """
    src = effective_path(playbook_id)
    if src is None:
        raise KeyError(f"unknown playbook {playbook_id!r}")
    return DialogMachine(Playbook.load(str(src)), llm=llm, engine="playbook")
