import textwrap
from pathlib import Path


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


FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "playbooks"
EXAMPLES = Path(__file__).resolve().parents[2] / "examples" / "playbooks"


def test_golden_fixture_compiles_and_validates() -> None:
    from superdialog.playbook.simple import load_simple

    pb = load_simple(str(FIXTURES / "simple_booking.yaml"))
    ids = pb.checkpoint_ids()
    assert ids == {
        "main.greeting",
        "main.collect_details",
        "main.present_price",
        "main.confirm_booking",
    }
    cd = pb.checkpoint("main.collect_details")
    assert set(cd.slots) == {"name", "service"}
    assert cd.advance_when[0].requires == ["name", "service"]
    assert cd.advance_when[0].to == "main.present_price"
    assert pb.checkpoint("main.confirm_booking").terminal is True
    assert "## Reference facts" in pb.persona
    assert "canonical_pricing" in pb.persona and "₹400" in pb.persona
    assert "## Objection handling" in pb.persona
    assert "If Caller says the price is too high. ->" in pb.persona
    assert "## Hard boundaries" in pb.persona
    assert "NEVER invent prices" in pb.persona
    assert pb.env == {}


def test_golden_fixture_round_trips_through_from_yaml() -> None:
    import yaml

    from superdialog.playbook.simple import load_simple

    pb = load_simple(str(FIXTURES / "simple_booking.yaml"))
    dumped = yaml.safe_dump(pb.model_dump(mode="json"), sort_keys=False)
    reloaded = Playbook.from_yaml(dumped)
    assert reloaded.checkpoint_ids() == pb.checkpoint_ids()


def test_realestate_simple_example_compiles() -> None:
    from superdialog.playbook.simple import load_simple

    pb = load_simple(str(EXAMPLES / "realestate_site_visit.simple.yaml"))
    assert isinstance(pb, Playbook)
    assert "main.deliver_closing" in pb.checkpoint_ids()
    assert pb.checkpoint("main.deliver_closing").terminal is True
    assert "Hard boundaries" in pb.persona


def test_public_exports() -> None:
    from superdialog.playbook import (
        is_simple_playbook,
        load_simple,
        simple_to_playbook,
    )

    assert callable(is_simple_playbook)
    assert callable(load_simple)
    assert callable(simple_to_playbook)
