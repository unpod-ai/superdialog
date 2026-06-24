import textwrap

from superdialog.playbook.events import (
    AdvanceEvent,
    EventLog,
    SlotWriteEvent,
    SteeringNoteEvent,
    SummaryEvent,
    ToolResultEvent,
    UtteranceEvent,
)
from superdialog.llm.prompt_cache import CACHE_PREFIX_KEY
from superdialog.playbook.models import Playbook
from superdialog.playbook.render import estimate_tokens, render_view
from superdialog.playbook.state import ConversationState
from tests.playbook.test_models import MINIMAL_YAML


def _setup() -> tuple[Playbook, ConversationState]:
    pb = Playbook.from_yaml(MINIMAL_YAML)
    log = EventLog()
    log.append(
        AdvanceEvent(from_checkpoint=None, to_checkpoint="booking.collect", rule="init")
    )
    log.append(SummaryEvent(text="Caller is a returning member."))
    log.append(SteeringNoteEvent(text="Don't re-ask the city.", kind="steer"))
    log.append(
        SlotWriteEvent(key="city", value="Pune", status="confirmed", by="director")
    )
    for i in range(30):
        log.append(UtteranceEvent(role="user", text=f"user line {i}"))
        log.append(UtteranceEvent(role="assistant", text=f"agent line {i}"))
    return pb, ConversationState.fold(log, playbook=pb)


def test_view_contains_priority_sections() -> None:
    pb, state = _setup()
    view = render_view(pb, state, token_budget=10_000)
    system = view.messages[0]["content"]
    assert "booking assistant" in system  # persona
    assert "Collect naturally." in system  # guidance
    assert "Don't re-ask the city." in system  # steering note
    assert "city: Pune" in system  # slots
    assert view.spoke_from_version == state.version


def test_system_message_cache_prefix_is_persona() -> None:
    """The system message is annotated with the persona as its stable prefix."""
    pb, state = _setup()
    view = render_view(pb, state, token_budget=10_000)
    sys_msg = view.messages[0]
    content = sys_msg["content"]
    # (a) content stays a bare string at the assembler.
    assert isinstance(content, str)
    # (b) the annotated prefix is a true leading substring of content.
    assert CACHE_PREFIX_KEY in sys_msg
    assert content.startswith(sys_msg[CACHE_PREFIX_KEY])
    # (c) the prefix contains the persona (now larger: persona + static block + anchor).
    assert pb.persona.strip() in sys_msg[CACHE_PREFIX_KEY]


def test_budget_drops_old_transcript_before_guidance() -> None:
    pb, state = _setup()
    from superdialog.playbook.models import GuidelineConfig
    pb.guidelines = GuidelineConfig(channel="text")
    view = render_view(pb, state, token_budget=300)
    system = view.messages[0]["content"]
    assert "Collect naturally." in system  # guidance survives
    texts = [m["content"] for m in view.messages[1:]]
    assert any("line 29" in t for t in texts)  # newest turns survive
    assert not any("line 0" in t for t in texts)  # oldest dropped
    assert "returning member" in system  # summary survives


def test_env_never_rendered() -> None:
    pb, state = _setup()
    state.env["ACCESS_TOKEN"] = "secret-xyz"
    view = render_view(pb, state, token_budget=10_000)
    joined = " ".join(m["content"] for m in view.messages)
    assert "secret-xyz" not in joined


def test_views_cannot_leak_env() -> None:
    yaml_text = textwrap.dedent("""
        persona: "Assistant."
        views:
          tok: "env.ACCESS_TOKEN"
        journeys:
          j:
            checkpoints:
              - id: start
                guidance: "Greet."
    """)
    pb = Playbook.from_yaml(yaml_text)
    log = EventLog()
    log.append(AdvanceEvent(from_checkpoint=None, to_checkpoint="j.start", rule="init"))
    state = ConversationState.fold(log)
    state.env["ACCESS_TOKEN"] = "secret-xyz"
    view = render_view(pb, state, token_budget=10_000)
    joined = " ".join(m["content"] for m in view.messages)
    assert "secret-xyz" not in joined


def test_template_errors_degrade_to_raw_text() -> None:
    pb, state = _setup()
    cp = pb.checkpoint("booking.collect")

    # ChainableUndefined: missing roots render empty (never raw, never raise)
    cp.guidance = "Hi {{ env.X }} there"
    view = render_view(pb, state, token_budget=10_000)
    assert "Hi  there" in view.messages[0]["content"]
    assert "{{" not in view.messages[0]["content"]

    # chained access through missing results defers to |default
    cp.guidance = "Hello {{ results.missing.data.name|default('friend') }}"
    view = render_view(pb, state, token_budget=10_000)
    assert "Hello friend" in view.messages[0]["content"]

    cp.guidance = "broken {{ slots."  # TemplateSyntaxError -> raw text
    view = render_view(pb, state, token_budget=10_000)
    assert "broken {{ slots." in view.messages[0]["content"]


def test_guidance_is_jinja_rendered_with_views() -> None:
    yaml_text = textwrap.dedent("""
        persona: "Assistant."
        views:
          slot_times: "pluck(results.availability_result.data.slots, 'time')"
        journeys:
          j:
            checkpoints:
              - id: present
                guidance: "Offer one of {{ views.slot_times }} to {{ slots.name }}."
    """)
    pb = Playbook.from_yaml(yaml_text)
    log = EventLog()
    log.append(
        AdvanceEvent(from_checkpoint=None, to_checkpoint="j.present", rule="init")
    )
    log.append(
        SlotWriteEvent(key="name", value="Ravi", status="confirmed", by="director")
    )
    log.append(
        ToolResultEvent(
            tool="avail",
            store_as="availability_result",
            ok=True,
            data={"slots": [{"time": "09:00"}, {"time": "10:00"}]},
        )
    )
    state = ConversationState.fold(log)
    view = render_view(pb, state, token_budget=10_000)
    system = view.messages[0]["content"]
    assert "09:00" in system and "Ravi" in system
    assert "{{" not in system


def test_ssti_payloads_contained() -> None:
    """SSTI payloads in guidance must never execute (sandboxed Jinja)."""
    pb, state = _setup()
    cp = pb.checkpoint("booking.collect")

    cp.guidance = "{{ ''.__class__ }}"
    view = render_view(pb, state, token_budget=10_000)
    system = view.messages[0]["content"]
    assert "<class" not in system  # not executed; degraded or empty

    cp.guidance = (
        "{{ cycler.__init__.__globals__.__builtins__"
        ".__import__('os').popen('id').read() }}"
    )
    view = render_view(pb, state, token_budget=10_000)
    system = view.messages[0]["content"]
    assert "uid=" not in system  # shell command must not run
    assert "cycler.__init__" in system  # degraded to raw template text


_KB_YAML = textwrap.dedent("""
    persona: "Booking assistant."
    knowledge_base: "A 2BHK is priced at 50 lakh. Office hours are 9 to 6."
    journeys:
      j:
        checkpoints:
          - id: collect
            goal: "Collect the name."
            guidance: "Collect naturally."
""")


def _kb_state(pb: Playbook) -> ConversationState:
    log = EventLog()
    log.append(AdvanceEvent(from_checkpoint=None, to_checkpoint="j.collect", rule="init"))
    return ConversationState.fold(log, playbook=pb)


def test_knowledge_base_injected_with_return_to_goal_directive() -> None:
    pb = Playbook.from_yaml(_KB_YAML)
    state = _kb_state(pb)
    system = render_view(pb, state, token_budget=10_000).messages[0]["content"]
    assert "## Knowledge base" in system
    assert "50 lakh" in system  # KB content is present
    # the aside-then-resume directive
    assert "answer briefly from the Knowledge base" in system
    assert "do not abandon" in system
    # the grounding line is softened to allow the KB as a fact source
    assert "Known information, Reference data, or the Knowledge base" in system


def test_empty_knowledge_base_is_byte_identical_grounding() -> None:
    """Regression guard: with no KB the system block is unchanged from before —
    the original grounding line, no KB section. Proves existing playbooks are
    unaffected by the additive field (the shared-package safety property)."""
    pb, state = _setup()  # MINIMAL_YAML has no knowledge_base
    system = render_view(pb, state, token_budget=10_000).messages[0]["content"]
    assert "## Knowledge base" not in system
    assert (
        "Only state facts present in Known information or Reference data; "
        "if asked something not there, say you are checking." in system
    )
    assert "or the Knowledge base" not in system


def test_knowledge_base_jinja_degrades_to_raw_text() -> None:
    """A KB authoring typo must not crash the speaking path (degrade to raw)."""
    yaml_text = _KB_YAML.replace(
        'knowledge_base: "A 2BHK is priced at 50 lakh. Office hours are 9 to 6."',
        'knowledge_base: "Price is {{ slots."',  # broken template
    )
    pb = Playbook.from_yaml(yaml_text)
    state = _kb_state(pb)
    system = render_view(pb, state, token_budget=10_000).messages[0]["content"]
    assert "Price is {{ slots." in system  # raw, un-rendered — no crash


def test_devanagari_budget_estimate() -> None:
    """Byte-based estimate keeps Devanagari from voiding the token budget."""
    assert estimate_tokens("मुझे कल सुबह दस बजे का स्लॉट चाहिए") >= 22


def test_voice_guideline_block_present_and_cache_prefix_extends() -> None:
    from superdialog.llm.prompt_cache import CACHE_PREFIX_KEY
    pb, state = _setup()  # MINIMAL_YAML: default guidelines (voice)
    view = render_view(pb, state, token_budget=10_000)
    sys_msg = view.messages[0]
    content = sys_msg["content"]
    assert "live phone call" in content.lower()
    assert "CONVERSATIONAL LEADERSHIP" in content
    prefix = sys_msg[CACHE_PREFIX_KEY]
    assert content.startswith(prefix)
    assert pb.persona.strip() in prefix
    assert "live phone call" in prefix.lower()       # static block in prefix
    assert "CURRENT DATE & TIME" in prefix            # anchor in prefix


def test_text_channel_has_no_voice_block() -> None:
    from superdialog.playbook.models import GuidelineConfig
    pb = Playbook.from_yaml(MINIMAL_YAML).model_copy(
        update={"guidelines": GuidelineConfig(channel="text")})
    log = EventLog()
    log.append(AdvanceEvent(from_checkpoint=None, to_checkpoint="booking.collect", rule="init"))
    state = ConversationState.fold(log, playbook=pb)
    system = render_view(pb, state, token_budget=10_000).messages[0]["content"]
    assert "live phone call" not in system.lower()


def test_handover_checkpoint_injects_handover_block() -> None:
    import textwrap
    yaml_text = textwrap.dedent('''
        persona: "Assistant."
        journeys:
          j:
            checkpoints:
              - id: transfer
                handover: true
                guidance: "Connect to a human."
              - id: done
                terminal: true
    ''')
    pb = Playbook.from_yaml(yaml_text)
    log = EventLog()
    log.append(AdvanceEvent(from_checkpoint=None, to_checkpoint="j.transfer", rule="init"))
    state = ConversationState.fold(log, playbook=pb)
    system = render_view(pb, state, token_budget=10_000).messages[0]["content"]
    assert "Handover" in system
    assert "summary" in system.lower()


def test_render_edges() -> None:
    # (a) tiny budget: system survives, transcript dropped; placeholder injected
    # so message list is never system-only (provider compatibility).
    pb, state = _setup()
    view = render_view(pb, state, token_budget=1)
    assert len(view.messages) == 2
    assert view.messages[0]["role"] == "system"
    assert view.messages[1]["role"] == "user"

    # (b) no checkpoint: persona + grounding render, no "Current step"
    pb2 = Playbook.from_yaml(MINIMAL_YAML)
    state2 = ConversationState.fold(EventLog())
    assert state2.checkpoint_id is None
    view2 = render_view(pb2, state2, token_budget=10_000)
    system2 = view2.messages[0]["content"]
    assert "booking assistant" in system2  # persona
    assert "Only state facts" in system2  # grounding rule
    assert "Current step" not in system2

    # (c) repair steering yields the "Correction" label
    log = EventLog()
    log.append(
        AdvanceEvent(from_checkpoint=None, to_checkpoint="booking.collect", rule="init")
    )
    log.append(SteeringNoteEvent(text="Apologise for the mixup.", kind="repair"))
    state3 = ConversationState.fold(log, playbook=pb2)
    view3 = render_view(pb2, state3, token_budget=10_000)
    system3 = view3.messages[0]["content"]
    assert "Correction" in system3
    assert "Direction" not in system3
