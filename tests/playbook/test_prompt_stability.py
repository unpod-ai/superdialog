"""KV-cache invariants: canonical serialization + byte-stable prompt prefixes.

Provider prompt caches key on byte-identical prefixes. These tests pin the
two properties that make cache hits happen by construction (the Shepherd
lesson: measure the cache, never hand-place it):

1. anything embedded in a prompt serializes canonically (same value ⇒ same
   bytes, regardless of dict insertion order), and
2. the Director's cache prefix is session-constant, and its full system
   prompt is byte-identical across turns when nothing state-visible changed.
"""

import json

from superdialog.llm.prompt_cache import CACHE_PREFIX_KEY
from superdialog.playbook._canon import canonical_json
from superdialog.playbook.models import Playbook
from superdialog.playbook.runtime import PlaybookRuntime
from superdialog.playbook.supervisor import Supervisor
from tests.playbook.test_models import MINIMAL_YAML
from tests.playbook.test_supervisor import SUP_YAML, _VerdictLLM
from tests.playbook.test_toolexec import FakeHttp

_IDLE = {"slots": {}, "advance": None, "note": None}


class RecordingLLM:
    """Director LLM that records every prompt and answers a canned verdict."""

    def __init__(self) -> None:
        self.prompts: list[list[dict[str, str]]] = []

    async def complete(self, messages: list[dict[str, str]], **kw: object) -> str:
        self.prompts.append(messages)
        return json.dumps(_IDLE)


def test_canonical_json_is_order_independent() -> None:
    assert canonical_json({"b": 1, "a": 2}) == canonical_json({"a": 2, "b": 1})
    assert canonical_json({"b": 1, "a": 2}) == '{"a":2,"b":1}'


def test_canonical_json_is_ascii_safe_and_compact() -> None:
    out = canonical_json({"name": "पुणे", "n": 1})
    assert out == '{"n":1,"name":"\\u092a\\u0941\\u0923\\u0947"}'


async def test_director_prompt_prefix_is_byte_stable_across_turns() -> None:
    llm = RecordingLLM()
    rt = PlaybookRuntime(
        Playbook.from_yaml(MINIMAL_YAML), director_llm=llm, http=FakeHttp([])
    )
    await rt.start()
    await rt.on_user_text("hello there")
    await rt.on_user_text("just thinking")  # idle turns: no slots, no advance
    assert len(llm.prompts) == 2
    sys1, sys2 = llm.prompts[0][0], llm.prompts[1][0]
    # The session-constant cache prefix must be byte-identical every turn.
    assert sys1[CACHE_PREFIX_KEY] == sys2[CACHE_PREFIX_KEY]
    # And with no state-visible change, the WHOLE system prompt must be too —
    # any drift here is a silent full-cache invalidation on every turn.
    assert sys1["content"] == sys2["content"]


async def test_director_cache_prefix_survives_slot_writes() -> None:
    llm = RecordingLLM()
    rt = PlaybookRuntime(
        Playbook.from_yaml(MINIMAL_YAML), director_llm=llm, http=FakeHttp([])
    )
    await rt.start()
    await rt.on_user_text("hello")
    from superdialog.playbook.events import SlotWriteEvent

    rt.log.append(
        SlotWriteEvent(key="city", value="Pune", status="confirmed", by="director")
    )
    await rt.on_user_text("Pune please")
    sys1, sys2 = llm.prompts[0][0], llm.prompts[1][0]
    assert sys1[CACHE_PREFIX_KEY] == sys2[CACHE_PREFIX_KEY]
    assert sys1["content"] != sys2["content"]  # volatile tail moved, as designed


def test_supervisor_cache_prefix_is_playbook_constant() -> None:
    pb = Playbook.from_yaml(SUP_YAML)
    sup = Supervisor(_VerdictLLM({}), pb)
    from superdialog.playbook.events import EventLog, UtteranceEvent
    from superdialog.playbook.state import ConversationState

    log1, log2 = EventLog(), EventLog()
    log2.append(UtteranceEvent(role="user", text="different conversation"))
    p1 = sup._prompt(ConversationState.fold(log1), log1, ["repair_streak"])
    p2 = sup._prompt(ConversationState.fold(log2), log2, ["degraded", "oscillation"])
    assert p1[0][CACHE_PREFIX_KEY] == p2[0][CACHE_PREFIX_KEY]
    assert p1[0]["content"].startswith(p1[0][CACHE_PREFIX_KEY])
