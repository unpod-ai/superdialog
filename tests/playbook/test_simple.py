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


def test_supervisor_flag_defaults_none_and_passes_through() -> None:
    import yaml

    default = simple_to_playbook(yaml.safe_load(SIMPLE))
    # None ⇒ resolved from the continuity mode at agent construction
    assert default.guidelines.supervisor is None
    doc = yaml.safe_load(SIMPLE)
    doc["supervisor"] = True
    assert simple_to_playbook(doc).guidelines.supervisor is True
    doc["supervisor"] = False
    assert simple_to_playbook(doc).guidelines.supervisor is False


def test_step_gate_defaults_hard_and_passes_soft_through() -> None:
    import yaml

    pb = simple_to_playbook(yaml.safe_load(SIMPLE))
    cps = {c.id: c for j in pb.journeys.values() for c in j.checkpoints}
    assert cps["greet"].gate == "hard"  # byte-compatible default
    doc = yaml.safe_load(SIMPLE)
    doc["playbook"][0]["gate"] = "soft"  # pure-talk step opts out of barrier
    pb2 = simple_to_playbook(doc)
    cps2 = {c.id: c for j in pb2.journeys.values() for c in j.checkpoints}
    assert cps2["greet"].gate == "soft"
    assert cps2["collect"].gate == "hard"  # others untouched


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
    # collect slots are required + soft-gated, and the llm advance rule
    # requires them: the Director may not advance past unfilled slots, but
    # soft means filled (not confirmed) suffices — provisional verdict writes
    # pass the same turn, avoiding the re-ask loop hard inheritance caused.
    assert all(s.required and s.gate == "soft" for s in collect.slots.values())
    # rule 0: companion expr rule — all-slots-filled advances with no LLM call
    expr_rule, llm_rule = collect.advance_when
    assert expr_rule.judge == "expr"
    assert expr_rule.when == ("slots.name is not None and slots.service is not None")
    assert llm_rule.judge == "llm"
    assert llm_rule.requires == ["name", "service"]
    # no collect -> no expr rule, no requires
    greet_rules = pb.checkpoint("main.greet").advance_when
    assert [r.judge for r in greet_rules] == ["llm"]
    assert greet_rules[0].requires == []


def test_non_terminal_steps_get_default_turn_budget() -> None:
    import yaml

    pb = simple_to_playbook(yaml.safe_load(SIMPLE))
    assert pb.checkpoint("main.collect").turn_budget == 4
    assert pb.checkpoint("main.confirm").turn_budget is None  # terminal: no budget
    # author override
    doc = yaml.safe_load(SIMPLE)
    doc["playbook"][1]["turn_budget"] = 7
    assert simple_to_playbook(doc).checkpoint("main.collect").turn_budget == 7


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
    # collect slots gate the llm advance (soft: filled suffices, no re-ask
    # loop); the preceding expr rule advances for free once both are filled.
    assert cd.advance_when[-1].requires == ["name", "service"]
    assert cd.advance_when[-1].to == "main.present_price"
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


def test_persona_language_is_folded() -> None:
    import yaml

    pb = simple_to_playbook(yaml.safe_load(SIMPLE))
    assert "Default conversation language: English" in pb.persona


def test_persona_name_folds_only_when_identity_lacks_it() -> None:
    import yaml

    doc = yaml.safe_load(SIMPLE)
    # SIMPLE's identity already says "You are Mira..." -> no duplicate line
    pb = simple_to_playbook(doc)
    assert "Your name is Mira." not in pb.persona

    doc["persona"]["identity"] = "You are a booking assistant for Glow Studio."
    pb2 = simple_to_playbook(doc)
    assert "Your name is Mira." in pb2.persona


def test_blank_name_and_language_fold_nothing() -> None:
    import yaml

    doc = yaml.safe_load(SIMPLE)
    doc["persona"]["name"] = ""
    doc["persona"]["language"] = ""
    pb = simple_to_playbook(doc)
    assert "Your name is" not in pb.persona
    assert "Default conversation language" not in pb.persona


def test_language_accepts_codes_and_lists() -> None:
    import yaml

    doc = yaml.safe_load(SIMPLE)
    doc["persona"]["language"] = ["en", "hi"]
    persona = simple_to_playbook(doc).persona
    assert "Default conversation language: English." in persona
    assert "Also speaks: Hindi." in persona

    doc["persona"]["language"] = "hi"
    assert "Default conversation language: Hindi." in simple_to_playbook(doc).persona


def test_language_names_pass_through_unmapped() -> None:
    import yaml

    doc = yaml.safe_load(SIMPLE)
    doc["persona"]["language"] = ["en", "Marathi"]
    persona = simple_to_playbook(doc).persona
    assert "Default conversation language: English." in persona
    assert "Also speaks: Marathi." in persona


def test_empty_language_list_folds_nothing() -> None:
    import yaml

    doc = yaml.safe_load(SIMPLE)
    doc["persona"]["language"] = []
    assert "Default conversation language" not in simple_to_playbook(doc).persona


def test_language_map_covers_common_languages() -> None:
    from superdialog.playbook.simple import _LANG_NAMES

    spot = {
        "ja": "Japanese",
        "zh": "Chinese",
        "ar": "Arabic",
        "pt": "Portuguese",
        "sw": "Swahili",
        "cy": "Welsh",
        "no": "Norwegian",
        "ml": "Malayalam",
    }
    for code, name in spot.items():
        assert _LANG_NAMES.get(code) == name, code
    assert len(_LANG_NAMES) >= 59  # the documented common-language set


def test_interrupts_compile_to_interrupt_specs() -> None:
    import yaml

    doc = yaml.safe_load(SIMPLE)
    doc["interrupts"] = [
        {"when": "Caller says goodbye.", "to": "main.confirm"},
        {"id": "busy", "when": "Caller is busy.", "to": "main.confirm"},
    ]
    pb = simple_to_playbook(doc)
    assert len(pb.interrupts) == 2
    auto, named = pb.interrupts
    assert auto.id == "interrupt_0" and auto.judge == "llm" and auto.resume is False
    assert named.id == "busy" and named.to == "main.confirm"


def test_interrupt_with_dangling_target_is_rejected() -> None:
    import pytest as _pytest
    import yaml

    doc = yaml.safe_load(SIMPLE)
    doc["interrupts"] = [{"when": "Caller says goodbye.", "to": "main.nope"}]
    with _pytest.raises(ValueError):
        simple_to_playbook(doc)


def test_simple_maps_memory_and_followup_toggles() -> None:
    import yaml
    from superdialog.playbook.simple import simple_to_playbook

    doc = yaml.safe_load(SIMPLE)
    # defaults: both off
    assert simple_to_playbook(doc).guidelines.memory_enabled is False
    assert simple_to_playbook(doc).guidelines.followup_enabled is False

    doc2 = yaml.safe_load(SIMPLE)
    doc2["memory_enabled"] = True
    doc2["followup_enabled"] = True
    pb = simple_to_playbook(doc2)
    assert pb.guidelines.memory_enabled is True
    assert pb.guidelines.followup_enabled is True


def test_simple_maps_guideline_fields() -> None:
    import yaml
    from superdialog.playbook.simple import simple_to_playbook

    doc = yaml.safe_load(SIMPLE)
    doc["channel"] = "voice"
    doc["tone"] = "casual"
    doc["call_type"] = "support"
    doc["timezone"] = "Asia/Kolkata"
    doc["persona"] = {**doc.get("persona", {}), "language": "hi"}
    pb = simple_to_playbook(doc)
    assert pb.guidelines.tone == "casual"
    assert pb.guidelines.call_type == "support"
    assert pb.guidelines.timezone == "Asia/Kolkata"
    assert pb.guidelines.language == "hi"  # lifted from persona.language


def test_branchy_steps_do_not_require_all_collected_slots() -> None:
    import yaml

    doc = yaml.safe_load(SIMPLE)
    # a qualify-style step collecting many per-category alternatives
    doc["playbook"][1]["collect"] = ["a", "b", "c", "d", "e"]
    pb = simple_to_playbook(doc)
    cp = pb.checkpoint("main.collect")
    # requiring all 5 would deadlock the step -> auto heuristic requires none
    assert cp.advance_when[-1].requires == []
    assert [r.judge for r in cp.advance_when] == ["llm"]  # no expr rule either
    assert all(not s.required for s in cp.slots.values())


def test_explicit_require_overrides_heuristic() -> None:
    import yaml

    doc = yaml.safe_load(SIMPLE)
    doc["playbook"][1]["collect"] = ["a", "b", "c", "d", "e"]
    doc["playbook"][1]["require"] = ["a", "b"]
    pb = simple_to_playbook(doc)
    cp = pb.checkpoint("main.collect")
    assert cp.advance_when[-1].requires == ["a", "b"]
    assert cp.advance_when[0].judge == "expr"
    assert "slots.a is not None and slots.b is not None" == cp.advance_when[0].when
    assert cp.slots["a"].required and not cp.slots["c"].required


def _doc(steps: list[dict]) -> dict:
    return {"playbook": steps}


def test_then_overrides_linear_successor() -> None:
    pb = simple_to_playbook(
        _doc(
            [
                {"id": "a", "done_when": "done", "then": "close"},
                {"id": "b", "done_when": "done"},
                {"id": "close"},
            ]
        )
    )
    a = pb.checkpoint("main.a")
    assert [r.to for r in a.advance_when if r.judge == "llm"] == ["main.close"]


def test_terminal_mid_list_ends_the_chain() -> None:
    pb = simple_to_playbook(
        _doc(
            [
                {"id": "a", "done_when": "done"},
                {"id": "bye", "terminal": True, "outcome": "callback_scheduled"},
                {"id": "z"},
            ]
        )
    )
    bye = pb.checkpoint("main.bye")
    assert bye.terminal is True
    assert bye.outcome == "callback_scheduled"
    assert bye.advance_when == []  # terminal steps get no advance rules


def test_terminal_default_outcome_is_closed() -> None:
    pb = simple_to_playbook(
        _doc(
            [
                {"id": "a", "done_when": "done"},
                {"id": "bye", "terminal": True},
                {"id": "z"},
            ]
        )
    )
    assert pb.checkpoint("main.bye").outcome == "closed"


def test_then_to_unknown_step_is_rejected() -> None:
    import pytest as _pytest

    # Playbook target validation (need_cp) rejects the dangling ref
    with _pytest.raises(ValueError):
        simple_to_playbook(
            _doc(
                [
                    {"id": "a", "done_when": "done", "then": "nope"},
                    {"id": "b"},
                ]
            )
        )


def test_then_targeting_itself_is_rejected() -> None:
    import pytest as _pytest

    with _pytest.raises(ValueError, match="cannot target itself"):
        simple_to_playbook(
            _doc(
                [
                    {"id": "a", "collect": ["x"], "then": "a"},
                    {"id": "b"},
                ]
            )
        )


def test_branches_compile_to_llm_rules_in_author_order() -> None:
    pb = simple_to_playbook(
        _doc(
            [
                {
                    "id": "pitch",
                    "done_when": "customer responded",
                    "branches": [
                        {"when": "customer accepts the visit", "to": "book"},
                        {"when": "customer firmly declines", "to": "fallback"},
                    ],
                },
                {"id": "clarify", "done_when": "done"},
                {"id": "book", "done_when": "done"},
                {"id": "fallback", "terminal": True},
            ]
        )
    )
    rules = [r for r in pb.checkpoint("main.pitch").advance_when if r.judge == "llm"]
    # branches first (author order), then the done_when default to the successor
    assert [r.to for r in rules] == ["main.book", "main.fallback", "main.clarify"]
    assert rules[0].when == "customer accepts the visit"


def test_branch_requires_gates_the_branch() -> None:
    pb = simple_to_playbook(
        _doc(
            [
                {
                    "id": "a",
                    "collect": ["day"],
                    "branches": [
                        {"when": "slot given", "to": "c", "requires": ["day"]}
                    ],
                },
                {"id": "b"},
                {"id": "c"},
            ]
        )
    )
    branch = pb.checkpoint("main.a").advance_when[-2]  # before default rule
    assert branch.requires == ["day"]


def test_branch_to_unknown_step_is_rejected() -> None:
    import pytest as _pytest

    # Playbook target validation (need_cp) rejects the dangling ref
    with _pytest.raises(ValueError):
        simple_to_playbook(
            _doc(
                [
                    {"id": "a", "branches": [{"when": "x", "to": "ghost"}]},
                    {"id": "b"},
                ]
            )
        )


def test_branch_targeting_itself_is_rejected() -> None:
    import pytest as _pytest

    with _pytest.raises(ValueError, match="cannot target itself"):
        simple_to_playbook(
            _doc(
                [
                    {"id": "a", "branches": [{"when": "x", "to": "a"}]},
                    {"id": "b"},
                ]
            )
        )


def test_terminal_step_with_branches_is_rejected() -> None:
    import pytest as _pytest

    with _pytest.raises(ValueError, match="cannot have branches"):
        simple_to_playbook(
            _doc(
                [
                    {"id": "a", "done_when": "done"},
                    {
                        "id": "bye",
                        "terminal": True,
                        "branches": [{"when": "x", "to": "a"}],
                    },
                ]
            )
        )


def test_last_step_with_branches_is_rejected() -> None:
    import pytest as _pytest

    # last step without a then: is implicitly terminal -> same rejection
    with _pytest.raises(ValueError, match="cannot have branches"):
        simple_to_playbook(
            _doc(
                [
                    {"id": "a", "done_when": "done"},
                    {"id": "z", "branches": [{"when": "x", "to": "a"}]},
                ]
            )
        )


def test_linear_compile_unchanged_without_new_fields() -> None:
    pb = simple_to_playbook(
        _doc(
            [
                {"id": "a", "done_when": "done"},
                {"id": "b"},
            ]
        )
    )
    a = pb.checkpoint("main.a")
    assert [r.to for r in a.advance_when] == ["main.b"]
    assert pb.checkpoint("main.b").terminal is True  # last step still terminal


def test_kb_false_step_compiles_uses_kb_false() -> None:
    import yaml

    doc = yaml.safe_load(SIMPLE)
    doc["playbook"][1]["kb"] = False
    pb = simple_to_playbook(doc)
    assert pb.checkpoint("main.collect").uses_kb is False
    assert pb.checkpoint("main.greet").uses_kb is None  # legacy default


# -- strict unknown-key validation --------------------------------------------


def test_unknown_top_level_keys_raise() -> None:
    import pytest as _pytest

    doc = {"playbook": [{"id": "a"}], "language_lock": {}, "agent_runtime": {}}
    with _pytest.raises(ValueError, match="language_lock"):
        simple_to_playbook(doc)


def test_unknown_step_keys_raise_with_path() -> None:
    import pytest as _pytest

    doc = {"playbook": [{"id": "a", "done_wehn": "typo"}]}
    with _pytest.raises(ValueError, match=r"playbook\[0\].*done_wehn"):
        simple_to_playbook(doc)


def test_unknown_branch_keys_raise_with_path() -> None:
    import pytest as _pytest

    # `require` (missing s) would silently drop the branch slot gate
    doc = {
        "playbook": [
            {
                "id": "a",
                "branches": [{"when": "x", "to": "c", "require": ["day"]}],
            },
            {"id": "b"},
            {"id": "c"},
        ]
    }
    with _pytest.raises(ValueError, match=r"playbook\[0\]\.branches\[0\].*require"):
        simple_to_playbook(doc)


def test_non_string_top_level_key_reports_cleanly() -> None:
    import pytest as _pytest

    doc = {"playbook": [{"id": "a"}], 2024: "foo"}
    with _pytest.raises(ValueError, match="2024"):
        simple_to_playbook(doc)


def test_strict_false_warns_and_loads(recwarn) -> None:
    doc = {"playbook": [{"id": "a"}], "latency_optimization": {}}
    pb = simple_to_playbook(doc, strict=False)
    assert pb.checkpoint("main.a")
    assert any("latency_optimization" in str(w.message) for w in recwarn.list)


def test_clean_doc_no_warning(recwarn) -> None:
    simple_to_playbook({"playbook": [{"id": "a"}]})
    assert not recwarn.list


def test_load_simple_passes_strict_through(tmp_path: Path) -> None:
    import pytest as _pytest

    from superdialog.playbook.simple import load_simple

    p = tmp_path / "pb.yaml"
    p.write_text("playbook:\n  - id: a\nlanguage_lock: {}\n", encoding="utf-8")
    with _pytest.raises(ValueError, match="language_lock"):
        load_simple(str(p))
    with _pytest.warns(UserWarning, match="language_lock"):
        assert load_simple(str(p), strict=False).checkpoint("main.a")


# -- compile-time guards from review -------------------------------------------


def test_branch_targeting_default_next_is_rejected() -> None:
    import pytest as _pytest

    # branch to the linear successor shadows the done_when slot gate
    with _pytest.raises(ValueError, match="default next step"):
        simple_to_playbook(
            _doc(
                [
                    {"id": "a", "branches": [{"when": "x", "to": "b"}]},
                    {"id": "b"},
                ]
            )
        )
    # same rejection when the default comes from then:
    with _pytest.raises(ValueError, match="default next step"):
        simple_to_playbook(
            _doc(
                [
                    {
                        "id": "a",
                        "then": "c",
                        "branches": [{"when": "x", "to": "c"}],
                    },
                    {"id": "b"},
                    {"id": "c"},
                ]
            )
        )


def test_duplicate_branch_targets_rejected() -> None:
    import pytest as _pytest

    with _pytest.raises(ValueError, match="duplicate branch target"):
        simple_to_playbook(
            _doc(
                [
                    {
                        "id": "a",
                        "branches": [
                            {"when": "x", "to": "c"},
                            {"when": "y", "to": "c"},
                        ],
                    },
                    {"id": "b"},
                    {"id": "c"},
                ]
            )
        )


def test_outcome_on_non_terminal_step_rejected() -> None:
    import pytest as _pytest

    with _pytest.raises(ValueError, match="only used on terminal steps"):
        simple_to_playbook(
            _doc(
                [
                    {"id": "a", "outcome": "callback_scheduled"},
                    {"id": "b"},
                ]
            )
        )


def test_then_and_branch_to_are_whitespace_stripped() -> None:
    pb = simple_to_playbook(
        _doc(
            [
                {
                    "id": "a",
                    "done_when": "done",
                    "then": " close ",
                    "branches": [{"when": "x", "to": " alt "}],
                },
                {"id": "alt"},
                {"id": "close"},
            ]
        )
    )
    rules = pb.checkpoint("main.a").advance_when
    assert [r.to for r in rules] == ["main.alt", "main.close"]


def test_branch_empty_when_rejected() -> None:
    import pytest as _pytest

    with _pytest.raises(ValueError, match="when"):
        simple_to_playbook(
            _doc(
                [
                    {"id": "a", "branches": [{"when": "   ", "to": "c"}]},
                    {"id": "b"},
                    {"id": "c"},
                ]
            )
        )


def test_then_say_compiles_to_exit_say() -> None:
    pb = simple_to_playbook(
        _doc(
            [
                {
                    "id": "a",
                    "collect": ["city"],
                    "then_say": "Pitch connectivity for {{slots.city}}.",
                },
                {"id": "b"},
            ]
        )
    )
    assert pb.checkpoint("main.a").exit_say.startswith("Pitch connectivity")


def test_then_say_on_terminal_step_rejected() -> None:
    import pytest as _pytest

    # A terminal step never advances OUT, so its exit_say would never fire.
    with _pytest.raises(ValueError, match="then_say"):
        simple_to_playbook(
            _doc(
                [
                    {"id": "a"},
                    {"id": "b", "then_say": "never spoken"},
                ]
            )
        )


# -- Westgate end-to-end fixture (proves the authored intent is expressible) ----


def test_westgate_fixture_compiles_and_routes() -> None:
    from superdialog.playbook.simple import load_simple

    pb = load_simple(str(FIXTURES / "westgate_routed.yaml"))
    # Every fallback step routes to the closing instead of falling into
    # whatever step happens to follow it in list order.
    for fb in [
        "busy_reschedule",
        "whatsapp_only",
        "wrong_number_or_dnc",
        "advisor_callback",
        "out_of_scope",
        "user_silence",
    ]:
        targets = [r.to for r in pb.checkpoint(f"main.{fb}").advance_when]
        assert targets and set(targets) == {"main.deliver_closing"}, fb
    assert pb.checkpoint("main.summary_before_closing").advance_when[-1].to == (
        "main.deliver_closing"
    )
    assert pb.checkpoint("main.deliver_closing").terminal
    # Visit pitch: accept rides done_when to BPCL confirm; firm decline
    # branches to the advisor-callback fallback.
    pitch = [
        r.to
        for r in pb.checkpoint("main.pitch_site_visit").advance_when
        if r.judge == "llm"
    ]
    assert pitch == [
        "main.advisor_callback",
        "main.confirm_qualification_bpcl",
    ]
    # Early decline routes to the reference ask, not accidental list order.
    nie = [r.to for r in pb.checkpoint("main.not_interested_early").advance_when]
    assert set(nie) == {"main.ask_for_reference"}
    # The connectivity pitch rides the advance as exit_say.
    assert "connectivity" in pb.checkpoint("main.qualify_location").exit_say


def test_legacy_continuity_is_inert_and_warns(caplog) -> None:
    import logging

    import yaml

    with caplog.at_level(logging.WARNING):
        assert simple_to_playbook(yaml.safe_load(SIMPLE)).legacy_continuity is False
    assert "legacy_continuity" not in caplog.text  # default: silent
    doc = yaml.safe_load(SIMPLE)
    doc["legacy_continuity"] = True
    with caplog.at_level(logging.WARNING):
        pb = simple_to_playbook(doc)
    assert pb.legacy_continuity is False  # deprecated: no longer propagated
    assert "legacy_continuity is deprecated and ignored" in caplog.text
