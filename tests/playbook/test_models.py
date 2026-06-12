import textwrap

import pytest
from pydantic import ValidationError

from superdialog.playbook.models import Playbook, RetrySpec

MINIMAL_YAML = textwrap.dedent("""
    persona: "You are a booking assistant."
    journeys:
      booking:
        checkpoints:
          - id: collect
            goal: "Have city and date"
            slots:
              city: {type: str, required: true, invalidates: [course_id]}
              date: {type: date, required: true}
            guidance: "Collect naturally."
            advance_when:
              - {when: "details complete", judge: llm, to: booking.confirm,
                 requires: [city, date]}
          - id: confirm
            gate: hard
            say_verbatim: "Your booking is held."
            pipeline: confirm_and_hold
            slots:
              price: {type: float, authoritative: true}
            advance_when:
              - {when: "pipeline.ok", judge: expr, to: booking.close}
              - {when: "pipeline.failed", judge: expr, to: booking.collect,
                 set: {error_context: booking_confirm_failed}}
          - id: close
            terminal: true
            outcome: confirmed
    tools:
      - id: hold_slot
        method: POST
        url: "{{ env.API_BASE_URL }}/slots/hold"
        store_response_as: hold_result
        timeout: 30
    pipelines:
      - id: confirm_and_hold
        steps:
          - tool: hold_slot
            on: {ok: continue, failed: {retry: 1, on_exhaust: booking.collect}}
    interrupts:
      - {id: goodbye, when: "caller says goodbye", judge: llm,
         to: booking.close, resume: false}
    policies:
      silence: {max_prompts: 2, prompts: ["Can you hear me?", "Are you there?"],
                then: booking.close}
""")


def test_load_yaml_and_address_checkpoints() -> None:
    pb = Playbook.from_yaml(MINIMAL_YAML)
    cp = pb.checkpoint("booking.collect")
    assert cp.slots["city"].invalidates == ["course_id"]
    assert pb.checkpoint("booking.confirm").gate == "hard"
    assert pb.initial_checkpoint_id == "booking.collect"
    assert [r.judge for r in pb.checkpoint("booking.confirm").advance_when] == [
        "expr",
        "expr",
    ]


def test_validation_rejects_dangling_rule_target() -> None:
    bad = MINIMAL_YAML.replace("to: booking.close}", "to: booking.nope}")
    with pytest.raises(ValueError, match="booking.nope"):
        Playbook.from_yaml(bad)


def test_unknown_pipeline_rejected() -> None:
    bad = MINIMAL_YAML.replace("pipeline: confirm_and_hold", "pipeline: missing_pipe")
    with pytest.raises(ValueError, match="missing_pipe"):
        Playbook.from_yaml(bad)


def test_silence_policy_target_validated() -> None:
    bad = MINIMAL_YAML.replace("then: booking.close", "then: booking.nowhere")
    with pytest.raises(ValueError, match="booking.nowhere"):
        Playbook.from_yaml(bad)


def test_pipeline_on_block_survives_yaml() -> None:
    pb = Playbook.from_yaml(MINIMAL_YAML)
    step = pb.pipeline("confirm_and_hold").steps[0]
    assert step.on["ok"] == "continue"
    failed = step.on["failed"]
    assert isinstance(failed, RetrySpec)
    assert failed.retry == 1
    assert failed.on_exhaust == "booking.collect"
    # booleans must still resolve as booleans
    assert pb.checkpoint("booking.collect").slots["city"].required is True


def test_dangling_on_exhaust_rejected_from_yaml() -> None:
    bad = MINIMAL_YAML.replace(
        "on_exhaust: booking.collect", "on_exhaust: booking.ghost"
    )
    with pytest.raises(ValueError, match="booking.ghost"):
        Playbook.from_yaml(bad)


def test_handler_loads_from_yaml() -> None:
    yaml_text = MINIMAL_YAML + textwrap.dedent("""
        handlers:
          - {id: h1, on: webhook.payment_captured, pipeline: confirm_and_hold}
    """)
    pb = Playbook.from_yaml(yaml_text)
    assert pb.handlers[0].on == "webhook.payment_captured"


def test_middleware_refresh_tool_validated() -> None:
    yaml_text = MINIMAL_YAML + textwrap.dedent("""
        middleware: {on_status: 401, refresh_with: ghost_tool}
    """)
    with pytest.raises(ValueError, match="ghost_tool"):
        Playbook.from_yaml(yaml_text)


def test_empty_journeys_rejected() -> None:
    with pytest.raises(ValueError):
        Playbook(journeys={})
    with pytest.raises(ValueError):
        Playbook.from_yaml(
            textwrap.dedent("""
                journeys:
                  booking:
                    checkpoints: []
            """)
        )


def test_duplicate_ids_rejected() -> None:
    dup_checkpoint = MINIMAL_YAML.replace(
        "- id: close\n        terminal: true",
        "- id: collect\n        terminal: true",
    )
    assert dup_checkpoint != MINIMAL_YAML
    with pytest.raises(ValueError, match="collect"):
        Playbook.from_yaml(dup_checkpoint)
    dup_tool = MINIMAL_YAML.replace(
        "pipelines:",
        '  - id: hold_slot\n    method: GET\n    url: "x"\npipelines:',
    )
    with pytest.raises(ValueError, match="hold_slot"):
        Playbook.from_yaml(dup_tool)


def test_retry_spec_capped() -> None:
    with pytest.raises(ValidationError):
        RetrySpec(retry=11)
    assert RetrySpec(retry=10).retry == 10


def test_requires_keys_validated() -> None:
    bad = MINIMAL_YAML.replace("requires: [city, date]", "requires: [city, datex]")
    with pytest.raises(ValueError, match="datex"):
        Playbook.from_yaml(bad)
    # a requires key set by the SAME rule's `set:` is allowed
    ok = textwrap.dedent("""
        journeys:
          j:
            checkpoints:
              - id: a
                advance_when:
                  - {when: "true", judge: expr, to: j.b,
                     requires: [flag], set: {flag: true}}
              - id: b
                terminal: true
    """)
    pb = Playbook.from_yaml(ok)
    assert pb.checkpoint("j.a").advance_when[0].requires == ["flag"]


def test_dotted_journey_name_rejected() -> None:
    with pytest.raises(ValueError, match=r"a\.b"):
        Playbook.from_yaml(
            textwrap.dedent("""
                journeys:
                  a.b:
                    checkpoints:
                      - id: only
                        terminal: true
            """)
        )


def test_reserved_pipeline_store_key_rejected() -> None:
    bad = MINIMAL_YAML.replace(
        "store_response_as: hold_result", "store_response_as: pipeline"
    )
    with pytest.raises(ValueError, match="reserved"):
        Playbook.from_yaml(bad)


def test_policies_hold_timeout_default_and_yaml_override() -> None:
    pb = Playbook.from_yaml(MINIMAL_YAML)
    assert pb.policies.hold_timeout == 4.0  # voice default: short post-filler wait
    tuned = Playbook.from_yaml(
        MINIMAL_YAML.replace("policies:", "policies:\n  hold_timeout: 2.5")
    )
    assert tuned.policies.hold_timeout == 2.5


def test_policies_hold_timeout_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        Playbook.from_yaml(
            MINIMAL_YAML.replace("policies:", "policies:\n  hold_timeout: 0")
        )


def test_from_yaml_auto_detects_simple_format() -> None:
    from tests.playbook.test_simple import SIMPLE

    pb = Playbook.from_yaml(SIMPLE)
    assert pb.checkpoint("main.collect").guidance.startswith("Ask for their name")
    assert "₹400" in pb.persona  # folded reference facts came through


def test_from_yaml_simple_routing_matches_simple_to_playbook() -> None:
    import yaml

    from superdialog.playbook.simple import simple_to_playbook
    from tests.playbook.test_simple import SIMPLE

    via_loader = Playbook.from_yaml(SIMPLE)
    via_compiler = simple_to_playbook(yaml.safe_load(SIMPLE))
    assert via_loader.model_dump() == via_compiler.model_dump()


def test_load_auto_detects_simple_yaml_and_json_files(tmp_path) -> None:
    import json

    import yaml

    from tests.playbook.test_simple import SIMPLE

    y = tmp_path / "simple.yaml"
    y.write_text(SIMPLE)
    assert Playbook.load(str(y)).checkpoint("main.greet").terminal is False

    j = tmp_path / "simple.json"
    j.write_text(json.dumps(yaml.safe_load(SIMPLE)))
    assert Playbook.load(str(j)).checkpoint("main.confirm").terminal is True


def test_from_yaml_rejects_neither_format() -> None:
    with pytest.raises(ValidationError):
        Playbook.from_yaml("name: not-a-playbook-of-either-format\n")
