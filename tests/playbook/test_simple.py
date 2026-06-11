import textwrap


from superdialog.playbook.models import Playbook
from superdialog.playbook.simple import (
    SimplePlaybook,
    SimpleStep,
    is_simple_playbook,
    simple_to_playbook,
)

SIMPLE = textwrap.dedent("""
    name: "Tiny Booking Bot"
    goal: "Book a haircut and confirm it."
    persona:
      name: Mira
      language: English
      voice_style: "Warm and brief. One question at a time."
      identity: "You are Mira, a booking assistant for Glow Studio."
    opening: "Greet the caller warmly."
    closing: "Thank them and say goodbye."
    playbook:
      - id: greet
        purpose: "Open the call."
        say: "Greet the caller and ask how you can help."
        done_when: "Caller is ready to book."
      - id: collect
        purpose: "Get the booking details."
        say: "Ask for their name and preferred service."
        collect: [name, service]
        done_when: "Name and service are captured."
      - id: confirm
        purpose: "Confirm and close."
        say: "Read back the booking and confirm."
        done_when: "Caller has confirmed."
    facts:
      services: [haircut, massage, facial]
      canonical_pricing:
        haircut: "₹400"
        massage: "₹900"
    objections:
      - trigger: "Caller says it's too expensive."
        handle: "Acknowledge and mention the value; offer the cheapest option."
    boundaries:
      - "NEVER invent prices."
    fallback_actions:
      callback: "Offer to call back at a convenient time."
""")


def test_detection_simple_vs_playbook() -> None:
    import yaml

    assert is_simple_playbook(yaml.safe_load(SIMPLE)) is True
    assert is_simple_playbook({"journeys": {"j": {"checkpoints": []}}}) is False
    assert is_simple_playbook({"nodes": [], "initial_node": "a"}) is False
    assert is_simple_playbook("not a dict") is False
    assert is_simple_playbook({"playbook": []}) is False


def test_compiles_to_valid_playbook_with_expected_checkpoints() -> None:
    import yaml

    pb = simple_to_playbook(yaml.safe_load(SIMPLE))
    assert isinstance(pb, Playbook)
    ids = pb.checkpoint_ids()
    assert ids == {"main.greet", "main.collect", "main.confirm"}
    assert pb.initial_checkpoint_id == "main.greet"


def test_steps_chain_to_next_and_last_is_terminal() -> None:
    import yaml

    pb = simple_to_playbook(yaml.safe_load(SIMPLE))
    greet = pb.checkpoint("main.greet")
    assert [r.to for r in greet.advance_when] == ["main.collect"]
    assert greet.advance_when[0].judge == "llm"
    assert greet.advance_when[0].when == "Caller is ready to book."
    collect = pb.checkpoint("main.collect")
    assert collect.advance_when[0].to == "main.confirm"
    confirm = pb.checkpoint("main.confirm")
    assert confirm.terminal is True
    assert confirm.outcome == "closed"
    assert confirm.advance_when == []


def test_collect_maps_to_str_slots_and_requires() -> None:
    import yaml

    pb = simple_to_playbook(yaml.safe_load(SIMPLE))
    collect = pb.checkpoint("main.collect")
    assert set(collect.slots) == {"name", "service"}
    assert all(s.type == "str" for s in collect.slots.values())
    assert collect.advance_when[0].requires == ["name", "service"]
    assert pb.checkpoint("main.greet").advance_when[0].requires == []


def test_guidance_is_the_say_prose() -> None:
    import yaml

    pb = simple_to_playbook(yaml.safe_load(SIMPLE))
    assert pb.checkpoint("main.greet").guidance == (
        "Greet the caller and ask how you can help."
    )


def test_persona_folds_facts_objections_boundaries_fallbacks_closing() -> None:
    import yaml

    pb = simple_to_playbook(yaml.safe_load(SIMPLE))
    persona = pb.persona
    assert "Mira" in persona
    assert "Voice & manner: Warm and brief" in persona
    assert "Overall goal: Book a haircut" in persona
    assert "## Reference facts" in persona
    assert "canonical_pricing" in persona and "₹400" in persona
    assert "## Objection handling" in persona
    assert "If Caller says it's too expensive. ->" in persona
    assert "## Hard boundaries" in persona and "NEVER invent prices." in persona
    assert "## Fallback actions" in persona and "callback:" in persona
    assert "## Closing line" in persona and "Thank them and say goodbye." in persona


def test_facts_not_in_env() -> None:
    import yaml

    pb = simple_to_playbook(yaml.safe_load(SIMPLE))
    assert pb.env == {}


def test_opening_seeds_first_guidance_only_when_say_missing() -> None:
    import yaml

    doc = yaml.safe_load(SIMPLE)
    doc["playbook"][0].pop("say")
    pb = simple_to_playbook(doc)
    assert pb.checkpoint("main.greet").guidance == "Greet the caller warmly."


def test_empty_done_when_defaults_to_step_complete() -> None:
    import yaml

    doc = yaml.safe_load(SIMPLE)
    doc["playbook"][0].pop("done_when")
    pb = simple_to_playbook(doc)
    assert pb.checkpoint("main.greet").advance_when[0].when == "step complete"


def test_simpleplaybook_model_round_trips_keys() -> None:
    import yaml

    sp = SimplePlaybook.model_validate(yaml.safe_load(SIMPLE))
    assert sp.name == "Tiny Booking Bot"
    assert sp.persona.identity.startswith("You are Mira")
    assert [s.id for s in sp.playbook] == ["greet", "collect", "confirm"]
    assert isinstance(sp.playbook[1], SimpleStep)
    assert sp.playbook[1].collect == ["name", "service"]


def test_compiled_playbook_round_trips_through_from_yaml() -> None:
    import yaml

    pb = simple_to_playbook(yaml.safe_load(SIMPLE))
    dumped = yaml.safe_dump(pb.model_dump(mode="json"), sort_keys=False)
    reloaded = Playbook.from_yaml(dumped)
    assert reloaded.checkpoint_ids() == pb.checkpoint_ids()
    assert reloaded.persona == pb.persona


def test_single_step_playbook_is_terminal_with_no_rules() -> None:
    pb = simple_to_playbook(
        {
            "persona": {"identity": "Solo."},
            "playbook": [
                {"id": "only", "purpose": "p", "say": "Say hi.", "done_when": "done"}
            ],
        }
    )
    only = pb.checkpoint("main.only")
    assert only.terminal is True and only.advance_when == []
