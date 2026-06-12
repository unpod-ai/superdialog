"""Tests for the flow→playbook compiler (FlowIndex node classification)."""

import json
from pathlib import Path

from superdialog.flow.models import ConversationFlow, Edge, FlowNode
from superdialog.playbook.compiler import (
    FlowIndex,
    _rewrite_template,
    compile_edge_condition,
    compile_flow,
    coverage_report,
    union_slot_schemas,
)

GOLF = Path(__file__).parents[1] / "fixtures" / "flow" / "golf_booking.json"

EXPECTED_SYSTEM = {
    "check_booking_status",
    "payment_expired_handler",
    "webhook_booking_confirm",
    "not_registered_close",
}
EXPECTED_COMPUTATIONAL = {
    "player_id_check",
    "greeting_details",
    "token_refresh",
    "resolve_course_name",
    "check_course_availability",
    "profile_check",
    "create_player_profile",
    "hold_slot_payment",
    "confirm_booking",
    "confirm_booking_retry",
    "profile_check_waitlist",
    "create_player_for_waitlist",
}
# slot_taken has static_text and no tools → auto-advancing conversational checkpoint


def _flow() -> ConversationFlow:
    return ConversationFlow.model_validate(json.loads(GOLF.read_text()))


def test_classification_matches_derived_ground_truth() -> None:
    idx = FlowIndex(_flow())
    kinds = {n.id: idx.classify(n) for n in idx.flow.nodes}
    assert {n for n, k in kinds.items() if k == "system"} == EXPECTED_SYSTEM
    assert {
        n for n, k in kinds.items() if k == "computational"
    } == EXPECTED_COMPUTATIONAL
    assert kinds["collect_booking_details"] == "conversational"
    assert kinds["silence_check"] == "conversational"


def test_reverse_edges_and_node_lookup() -> None:
    idx = FlowIndex(_flow())
    inbound = idx.reverse_edges["collect_booking_details"]
    assert len(inbound) == 15
    assert all(isinstance(src, str) and isinstance(eid, str) for src, eid in inbound)
    node = idx.node("collect_booking_details")
    assert node.id == "collect_booking_details"
    assert idx.node("token_refresh").id == "token_refresh"


def test_data_predicate_conditions_become_expr_rules() -> None:
    rule = compile_edge_condition(
        "availability_result.success == true",
        store_keys={"availability_result"},
        target="main.present",
    )
    assert rule.judge == "expr"
    assert rule.when == "results.availability_result.ok"
    assert rule.to == "main.present"


def test_status_code_predicates_translate() -> None:
    rule = compile_edge_condition(
        "hold_result.status == 409", store_keys={"hold_result"}, target="main.taken"
    )
    assert rule.judge == "expr"
    assert rule.when == "results.hold_result.status == 409"


def test_intent_conditions_become_llm_rules() -> None:
    rule = compile_edge_condition(
        "caller wants to cancel an existing booking",
        store_keys=set(),
        target="main.cancel",
    )
    assert rule.judge == "llm"
    assert rule.when == "caller wants to cancel an existing booking"


def test_untranslatable_data_conditions_stay_llm() -> None:
    # mentions a store key but the shape isn't confidently translatable
    rule = compile_edge_condition(
        "availability_result has slots near the preferred time",
        store_keys={"availability_result"},
        target="main.present",
    )
    assert rule.judge == "llm"


def test_negated_and_prose_success_forms() -> None:
    keys = {"hold_result", "booking_confirm_result"}
    negated = compile_edge_condition(
        "hold_result.success == false", store_keys=keys, target="main.t"
    )
    assert (negated.judge, negated.when) == ("expr", "not results.hold_result.ok")
    not_form = compile_edge_condition(
        "not hold_result.success", store_keys=keys, target="main.t"
    )
    assert (not_form.judge, not_form.when) == ("expr", "not results.hold_result.ok")
    # legacy prose spelling used by the golf flow, with an em-dash gloss
    prose = compile_edge_condition(
        "booking_confirm_result.success is false — route to retry attempt",
        store_keys=keys,
        target="main.retry",
    )
    assert prose.judge == "expr"
    assert prose.when == "not results.booking_confirm_result.ok"
    # a gloss that qualifies the predicate must NOT be dropped
    qualified = compile_edge_condition(
        "hold_result.success is true — unless the caller already paid",
        store_keys=keys,
        target="main.t",
    )
    assert qualified.judge == "llm"


def test_unknown_store_key_and_compounds_stay_llm() -> None:
    unknown = compile_edge_condition(
        "mystery_result.success == true", store_keys={"hold_result"}, target="main.t"
    )
    assert unknown.judge == "llm"
    assert unknown.when == "mystery_result.success == true"
    compound = compile_edge_condition(
        "hold_result.success == true and availability_result.success == true",
        store_keys={"hold_result", "availability_result"},
        target="main.t",
    )
    assert compound.judge == "llm"


def test_union_schemas_with_per_rule_requires() -> None:
    flow = _flow()
    node = next(n for n in flow.nodes if n.id == "collect_booking_details")
    slots, requires_by_edge = union_slot_schemas(node)
    edges_with_schema = {e.id for e in node.edges if e.input_schema}
    assert set(requires_by_edge) == edges_with_schema
    for req in requires_by_edge.values():
        assert set(req) <= set(slots)  # every required is declared
    assert all(not s.required for s in slots.values())  # union: all optional


def test_json_schema_type_mapping() -> None:
    flow = _flow()
    # find an array-typed property anywhere in the flow
    found_array = False
    for node in flow.nodes:
        for e in node.edges:
            if not e.input_schema:
                continue
            slots, _ = union_slot_schemas(node)
            assert isinstance(e.input_schema, dict)
            for key, prop in (e.input_schema.get("properties") or {}).items():
                if prop.get("type") == "array":
                    assert slots[key].type == "array"
                    found_array = True
    assert found_array  # the golf flow has 2 array fields


def test_enum_description_and_first_declaration_wins() -> None:
    node = FlowNode(
        id="n",
        name="n",
        edges=[
            Edge(
                id="e1",
                condition="c1",
                target_node_id="x",
                input_schema={
                    "type": "object",
                    "properties": {
                        "tee_period": {
                            "type": "string",
                            "enum": ["morning", "afternoon"],
                            "description": "Preferred period",
                        },
                        "count": {"type": "integer"},
                    },
                    "required": ["tee_period"],
                },
            ),
            Edge(
                id="e2",
                condition="c2",
                target_node_id="y",
                input_schema={
                    "type": "object",
                    "properties": {"tee_period": {"type": "string"}},
                    "required": [],
                },
            ),
        ],
    )
    slots, requires_by_edge = union_slot_schemas(node)
    assert slots["tee_period"].type == "enum"  # first declaration wins
    assert slots["tee_period"].values == ["morning", "afternoon"]
    assert slots["tee_period"].description == "Preferred period"
    assert slots["count"].type == "int"
    assert requires_by_edge == {"e1": ["tee_period"], "e2": []}


# -- full compile_flow ---------------------------------------------------------


def test_golf_flow_compiles_to_valid_playbook() -> None:
    pb = compile_flow(_flow())  # Playbook validators must all pass
    assert pb.persona  # system_prompt carried over
    assert len(pb.tools) == 25  # all global_actions
    assert pb.handlers  # webhook + timer system nodes
    assert pb.interrupts  # global_goodbye
    assert pb.policies.silence  # silence nodes -> policy
    assert pb.middleware is not None  # token_refresh -> middleware
    assert pb.middleware.refresh_with == "action-auth-refresh"
    assert pb.initial_checkpoint_id == "main.greeting"


def test_every_flow_construct_is_mapped() -> None:
    flow = _flow()
    report = coverage_report(flow, compile_flow(flow))
    assert report.unmapped_nodes == []
    assert report.unmapped_edges == []
    assert report.unmapped_actions == []


def test_conversational_nodes_become_checkpoints() -> None:
    pb = compile_flow(_flow())
    ids = pb.checkpoint_ids()
    assert any(i.endswith(".collect_booking_details") for i in ids)
    assert any(i.endswith(".other_query_handler") for i in ids)  # it speaks
    assert not any(i.endswith(".hold_slot_payment") for i in ids)  # computational
    assert not any(i.endswith(".greeting_details") for i in ids)  # hub router
    assert not any(i.endswith(".silence_check") for i in ids)  # silence policy
    assert not any(i.endswith(".token_refresh") for i in ids)  # middleware


def test_template_rewriting() -> None:
    pb = compile_flow(_flow())
    hold = pb.tool("action-slots-hold")
    assert hold.url.startswith("{{env.API_BASE_URL}}")
    assert any("env.ACCESS_TOKEN" in v for v in hold.headers.values())
    assert hold.body["slot_id"] == "{{slots.slot_id}}"
    assert hold.body["player_id"] == "{{env.player_id}}"  # env_updates key
    assert hold.env_updates == {"hold_id": "data.hold_id"}
    assert set(hold.args) == {"slot_id", "player_id"}
    get_player = pb.tool("action-players-get")
    assert get_player.when == "env.player_id"  # condition {{player_id}}
    assert get_player.run_once is True


def test_rewrite_template_helper() -> None:
    env, res = {"ACCESS_TOKEN", "player_id"}, {"availability_result"}
    rw = lambda t: _rewrite_template(t, env, res)  # noqa: E731
    assert rw("{{ACCESS_TOKEN}}") == "{{env.ACCESS_TOKEN}}"
    assert rw("{{city}}") == "{{slots.city}}"
    assert (
        rw("{{availability_result.data.slots}}")
        == "{{results.availability_result.data.slots}}"
    )
    # filters: only the leading variable token is rewritten
    assert rw("{{city|default('')}}") == "{{slots.city|default('')}}"
    # kwargs and string literals stay untouched
    assert (
        rw("{{xs|selectattr('a','equalto',city)|map(attribute='b')|list}}")
        == "{{slots.xs|selectattr('a','equalto',slots.city)|map(attribute='b')|list}}"
    )
    # result .success maps to the executor's .ok field
    assert (
        rw("{{availability_result.success|default(false)}}")
        == "{{results.availability_result.ok|default(false)}}"
    )
    # statement blocks are rewritten too; keywords are preserved
    assert (
        rw("{% if availability_result.data.slots %}x{% endif %}")
        == "{% if results.availability_result.data.slots %}x{% endif %}"
    )
    assert rw("plain text without templates") == "plain text without templates"


AVAIL_REF = "main.collect_booking_details__collect_to_check_availability"
MONEY_REF = "main.collect_booking_confirmation_details__booking_details_to_hold"


def test_happy_path_rules_exist() -> None:
    pb = compile_flow(_flow())
    greeting = pb.checkpoint("main.greeting")
    assert greeting.advance_when  # hub rules merged into the source
    assert any(r.to == "main.collect_booking_details" for r in greeting.advance_when)
    # collect routes into the availability INTERMEDIATE, not past it
    collect = pb.checkpoint("main.collect_booking_details")
    assert any(r.to == AVAIL_REF for r in collect.advance_when)
    avail = pb.checkpoint(AVAIL_REF)
    assert avail.pipeline is not None
    assert [s.tool for s in pb.pipeline(avail.pipeline).steps] == [
        "action-courses-availability"
    ]
    routes = {(r.when, r.judge, r.to) for r in avail.advance_when}
    assert ("pipeline.ok", "expr", "main.present_available_slot") in routes
    assert ("pipeline.failed", "expr", "main.offer_waitlist") in routes
    # the nearest-slot branch is preserved as an llm rule on the intermediate
    assert any(
        r.judge == "llm" and r.to == "main.offer_nearest_slot"
        for r in avail.advance_when
    )
    present = pb.checkpoint("main.present_available_slot")
    targets = {r.to for r in present.advance_when}
    assert "main.ask_profile_details" in targets  # new caller branch
    assert "main.collect_booking_confirmation_details" in targets  # registered
    assert AVAIL_REF in targets  # recheck reuses the shared intermediate
    assert present.on_enter == []  # no path-wise chain tools
    confirm = pb.checkpoint("main.collect_booking_confirmation_details")
    assert any(r.to == MONEY_REF for r in confirm.advance_when)
    close = pb.checkpoint("main.booking_close")
    assert close.terminal and close.outcome == "booking_close"
    assert close.on_enter == []  # money tools live in the pipeline only


def test_no_premature_money_tools() -> None:
    """Money tools never fire as on_enter side effects of any checkpoint."""
    pb = compile_flow(_flow())
    money = {"action-slots-hold", "action-bookings-confirm"}
    for journey in pb.journeys.values():
        for cp in journey.checkpoints:
            assert not money & set(cp.on_enter), f"{cp.id} fires {cp.on_enter}"
    pipeline_tools = {s.tool for p in pb.pipelines for s in p.steps}
    assert money <= pipeline_tools  # they run ONLY as pipeline steps


def test_money_chain_is_pipeline() -> None:
    """hold→confirm compiles to a pipeline whose results gate booking_close."""
    pb = compile_flow(_flow())
    money = pb.checkpoint(MONEY_REF)
    assert money.pipeline is not None
    steps = pb.pipeline(money.pipeline).steps
    assert [s.tool for s in steps] == [
        "action-slots-hold",
        "action-bookings-confirm",
    ]
    # a failed hold routes to slot_taken (auto checkpoint that speaks the message),
    # which then re-runs availability via the shared intermediate
    assert steps[0].on["failed"] == "main.slot_taken"
    slot_taken_cp = pb.checkpoint("main.slot_taken")
    assert slot_taken_cp.auto is True
    assert any(r.to == AVAIL_REF for r in slot_taken_cp.advance_when)
    # a failed confirm routes to the retry intermediate (legacy retry node)
    retry_ref = steps[1].on["failed"]
    assert retry_ref == "main.confirm_booking__confirm_to_retry"
    retry = pb.checkpoint(retry_ref)
    assert retry.pipeline is not None
    retry_steps = pb.pipeline(retry.pipeline).steps
    assert [s.tool for s in retry_steps] == ["action-bookings-confirm"]
    assert retry_steps[0].on["failed"] == "main.other_query_handler"
    # routing is judged on the pipeline RESULT, after the tools ran
    money_routes = {(r.when, r.judge, r.to) for r in money.advance_when}
    assert ("pipeline.ok", "expr", "main.booking_close") in money_routes
    assert any(w == "pipeline.failed" and j == "expr" for w, j, _ in money_routes)
    retry_routes = {(r.when, r.judge, r.to) for r in retry.advance_when}
    assert ("pipeline.ok", "expr", "main.booking_close") in retry_routes
    assert ("pipeline.failed", "expr", "main.other_query_handler") in retry_routes


def test_chain_intermediates_are_dedup_shared() -> None:
    """Every entry into the availability chain shares ONE intermediate."""
    pb = compile_flow(_flow())
    sources = {
        "main.collect_booking_details",
        "main.returning_rebook_same",
        "main.present_available_slot",
        "main.offer_nearest_slot",
        "main.offer_waitlist",
    }
    for ref in sources:
        cp = pb.checkpoint(ref)
        assert any(r.to == AVAIL_REF for r in cp.advance_when), ref
    # exactly one availability intermediate exists
    avail_pipes = [
        p
        for p in pb.pipelines
        if [s.tool for s in p.steps] == ["action-courses-availability"]
    ]
    assert len(avail_pipes) == 1


def test_silence_policy_compiled_from_silence_chain() -> None:
    pb = compile_flow(_flow())
    silence = pb.policies.silence
    assert silence is not None
    assert silence.prompts == [
        "Hello, can you hear me?",
        "I am unable to hear you. Are you there?",
    ]
    assert silence.max_prompts == 2
    assert silence.then == "main.call_end"


def test_handlers_wired_to_pipelines() -> None:
    pb = compile_flow(_flow())
    ons = {h.on for h in pb.handlers}
    assert ons == {"webhook.payment_captured", "timer.hold_expired"}
    webhook = next(h for h in pb.handlers if h.on.startswith("webhook."))
    steps = pb.pipeline(webhook.pipeline).steps
    assert [s.tool for s in steps] == ["action-bookings-confirm"]
    timer = next(h for h in pb.handlers if h.on.startswith("timer."))
    assert [s.tool for s in pb.pipeline(timer.pipeline).steps] == [
        "action-slots-release"
    ]


def test_dispatch_compiled_from_hub_router() -> None:
    pb = compile_flow(_flow())
    assert len(pb.dispatch) == 13  # 14 hub edges minus the silence target
    ids = pb.checkpoint_ids()
    assert all(d.to in ids for d in pb.dispatch)
    courses = next(d for d in pb.dispatch if d.to == "main.list_courses_in_city")
    assert courses.requires == ["city"]


def test_coverage_report_buckets() -> None:
    flow = _flow()
    report = coverage_report(flow, compile_flow(flow))
    assert report.orphans == ["check_booking_status", "not_registered_close"]
    assert "silence_check" in report.dropped["silence_policy"]
    assert "still_silent_check" in report.dropped["silence_policy"]
    assert "token_refresh" in report.dropped["middleware"]
    assert "greeting_details" in report.dropped["hubs"]
    chains = report.dropped["computational_chains"]
    assert {"hold_slot_payment", "confirm_booking", "check_course_availability"} <= set(
        chains
    )
    assert any("intermediate pipeline checkpoints" in n for n in report.notes)
    # branches the pipeline cannot route deterministically are noted
    assert any("not deterministically routable" in n for n in report.notes)


def test_interrupt_from_global_goodbye() -> None:
    pb = compile_flow(_flow())
    goodbye = next(i for i in pb.interrupts if i.id == "global_goodbye")
    assert goodbye.to == "main.call_end"
    assert goodbye.judge == "llm"
    assert goodbye.resume is False


def test_env_passes_through() -> None:
    flow = _flow()
    pb = compile_flow(flow)
    assert pb.env == flow.environment_variables
    assert pb.env["API_BASE_URL"].startswith("https://")


def test_orphan_system_nodes_become_checkpoints() -> None:
    pb = compile_flow(_flow())
    status = pb.checkpoint("main.check_booking_status")
    assert status.on_enter == ["action-booking-status-poll"]
    assert {r.to for r in status.advance_when} == {
        "main.booking_confirmed_close",
        "main.booking_close_pending",
    }
    closed = pb.checkpoint("main.not_registered_close")
    assert [r.to for r in closed.advance_when] == ["main.call_end"]


def test_neutral_gloss_is_not_stripped() -> None:
    # allowlist: only narration glosses ("route to ...") are stripped
    rule = compile_edge_condition(
        "hold_result.success is true — caller verified",
        store_keys={"hold_result"},
        target="main.t",
    )
    assert rule.judge == "llm"
