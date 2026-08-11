import logging

import pytest

from superdialog.playbook import toolexec
from superdialog.playbook.events import (
    EventLog,
    SlotWriteEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from superdialog.playbook.models import SlotSpec, ToolSpec
from superdialog.playbook.state import ConversationState
from superdialog.playbook.toolexec import ToolExecutor, coerce_args


class FakeHttp:
    def __init__(self, responses: list[tuple[int, dict]]) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []

    async def __call__(
        self,
        *,
        method: str,
        url: str,
        headers: dict,
        body: dict | None,
        timeout: float,
    ) -> tuple[int, dict]:
        self.calls.append(
            {"method": method, "url": url, "headers": headers, "body": body}
        )
        return self.responses.pop(0)


def _state(**slots) -> ConversationState:
    log = EventLog()
    for k, v in slots.items():
        log.append(SlotWriteEvent(key=k, value=v, status="confirmed", by="director"))
    state = ConversationState.fold(log)
    state.env["API_BASE_URL"] = "https://api.test"
    state.env["ACCESS_TOKEN"] = "tok-1"
    return state


HOLD = ToolSpec(
    id="hold_slot",
    method="POST",
    url="{{ env.API_BASE_URL }}/slots/hold",
    headers={"Authorization": "Bearer {{ env.ACCESS_TOKEN }}"},
    body={"slot_id": "{{ slots.slot_id }}", "players": "{{ slots.players }}"},
    store_response_as="hold_result",
    env_updates={"hold_id": "data.hold_id"},
)


async def test_executes_and_stores_result_and_env() -> None:
    http = FakeHttp([(200, {"data": {"hold_id": "h-77"}})])
    ex = ToolExecutor(http=http)
    events = await ex.execute(HOLD, _state(slot_id="s1", players=4))
    kinds = [type(e).__name__ for e in events]
    assert kinds == ["ToolCallEvent", "ToolResultEvent", "EnvWriteEvent"]
    assert http.calls[0]["url"] == "https://api.test/slots/hold"
    assert http.calls[0]["headers"]["Authorization"] == "Bearer tok-1"
    result = events[1]
    assert (
        isinstance(result, ToolResultEvent)
        and result.ok
        and result.store_as == "hold_result"
    )
    assert events[2].key == "hold_id" and events[2].value == "h-77"


async def test_when_predicate_skips() -> None:
    spec = HOLD.model_copy(update={"when": "slots.player_id"})
    ex = ToolExecutor(http=FakeHttp([]))
    events = await ex.execute(spec, _state(slot_id="s1"))  # no player_id
    assert events == []


async def test_run_once_skips_second_execution() -> None:
    spec = HOLD.model_copy(update={"run_once": True})
    state = _state(slot_id="s1", players=2)
    state.tool_call_counts["hold_slot"] = 1  # already ran once
    ex = ToolExecutor(http=FakeHttp([]))
    assert await ex.execute(spec, state) == []


async def test_http_error_yields_failed_result() -> None:
    ex = ToolExecutor(http=FakeHttp([(503, {"error": "upstream"})]))
    events = await ex.execute(HOLD, _state(slot_id="s1", players=2))
    result = events[1]
    assert result.ok is False and result.status == 503


async def test_template_error_yields_failed_result() -> None:
    # Broken template syntax: degrade to a failed result, never raise.
    bad = HOLD.model_copy(update={"url": "{{ env.API_BASE_URL "})
    http = FakeHttp([])
    ex = ToolExecutor(http=http)
    events = await ex.execute(bad, _state(slot_id="s1", players=2))
    kinds = [type(e).__name__ for e in events]
    assert kinds == ["ToolCallEvent", "ToolResultEvent"]
    result = events[1]
    assert isinstance(result, ToolResultEvent) and result.ok is False
    assert http.calls == []  # bad template never reaches HTTP
    assert "template error" in (result.error or "")

    # SSTI payload: the sandbox blocks the attribute walk; the payload is
    # NOT executed (a plain Environment would render a function repr with
    # a memory address like "at 0x...").
    # Keep a valid absolute base so this stays an SSTI assertion, not an SSRF one.
    ssti = HOLD.model_copy(
        update={"url": "{{ env.API_BASE_URL }}/{{ cycler.__init__ }}"}
    )
    http2 = FakeHttp([(200, {})])
    events2 = await ToolExecutor(http=http2).execute(
        ssti, _state(slot_id="s1", players=2)
    )
    assert http2.calls[0]["url"] == "https://api.test/"  # unsafe attr rendered empty
    assert "0x" not in http2.calls[0]["url"]
    assert [type(e).__name__ for e in events2] == [
        "ToolCallEvent",
        "ToolResultEvent",
    ]


async def test_python_tool_and_arg_coercion() -> None:
    spec = ToolSpec(
        id="score",
        type="python",
        store_response_as="score_result",
        args={"n": SlotSpec(type="int")},
    )
    seen: list[object] = []

    async def score_fn(args: dict, state: ConversationState) -> dict:
        seen.append(args["n"])
        assert isinstance(args["n"], int) and args["n"] == 7
        return {"score": args["n"] * 2}

    ex = ToolExecutor(http=FakeHttp([]), python_tools={"score": score_fn})
    events = await ex.execute(spec, _state(), args={"n": "7"})
    kinds = [type(e).__name__ for e in events]
    assert kinds == ["ToolCallEvent", "ToolResultEvent"]
    result = events[1]
    assert isinstance(result, ToolResultEvent) and result.ok
    assert result.data == {"score": 14}
    assert seen == [7]

    async def boom_fn(args: dict, state: ConversationState) -> dict:
        raise RuntimeError("python tool exploded")

    spec_boom = spec.model_copy(update={"id": "boom"})
    ex2 = ToolExecutor(http=FakeHttp([]), python_tools={"boom": boom_fn})
    events2 = await ex2.execute(spec_boom, _state())
    result2 = events2[1]
    assert isinstance(result2, ToolResultEvent) and result2.ok is False
    assert "exploded" in (result2.error or "")


async def test_python_tool_applies_env_and_slot_updates() -> None:
    # Previously env_updates/slot_updates only fired on the type:http path,
    # forcing a python tool that needed to write a slot into a redundant
    # second HTTP call whose only purpose was to reach that code path (see
    # the golf playbook's action-slot-exact-check, which re-fetched
    # availability from scratch a second time purely to get slot_id written
    # via slot_updates). A python tool's own return value must flow through
    # the same field-update machinery as an http response.
    # A python tool's return value IS the `data` payload directly -- there is
    # no separate envelope to dig through the way an http response has one
    # (that "data.x" shape is this test suite's REST convention, not a
    # framework rule), so the update paths key straight off the return dict.
    spec = ToolSpec(
        id="pick_slot",
        type="python",
        store_response_as="pick_result",
        env_updates={"picked_course": "course_id"},
        slot_updates={"slot_id": "matches.0.slot_id"},
    )

    async def pick_fn(args: dict, state: ConversationState) -> dict:
        return {"course_id": "course_1", "matches": [{"slot_id": "slot_abc"}]}

    ex = ToolExecutor(http=FakeHttp([]), python_tools={"pick_slot": pick_fn})
    events = await ex.execute(spec, _state())
    kinds = [type(e).__name__ for e in events]
    assert kinds == [
        "ToolCallEvent",
        "ToolResultEvent",
        "EnvWriteEvent",
        "SlotWriteEvent",
    ]
    assert events[2].key == "picked_course" and events[2].value == "course_1"
    assert events[3].key == "slot_id" and events[3].value == "slot_abc"


async def test_python_tool_failure_skips_field_updates() -> None:
    # A failed python tool must not write env/slots from a data shape that
    # was never actually produced.
    spec = ToolSpec(
        id="pick_slot_boom",
        type="python",
        store_response_as="pick_result",
        slot_updates={"slot_id": "data.slot_id"},
    )

    async def boom_fn(args: dict, state: ConversationState) -> dict:
        raise RuntimeError("no availability data")

    ex = ToolExecutor(http=FakeHttp([]), python_tools={"pick_slot_boom": boom_fn})
    events = await ex.execute(spec, _state())
    kinds = [type(e).__name__ for e in events]
    assert kinds == ["ToolCallEvent", "ToolResultEvent"]
    assert events[1].ok is False


async def test_unregistered_python_tool_degrades() -> None:
    spec = ToolSpec(id="ghost", type="python", store_response_as="ghost_result")
    ex = ToolExecutor(http=FakeHttp([]))  # no python tools registered
    events = await ex.execute(spec, _state())
    kinds = [type(e).__name__ for e in events]
    assert kinds == ["ToolCallEvent", "ToolResultEvent"]
    result = events[1]
    assert isinstance(result, ToolResultEvent) and result.ok is False
    assert "not registered" in (result.error or "")
    assert result.store_as == "ghost_result"


async def test_error_strings_carry_type() -> None:
    async def slow_fn(args: dict, state: ConversationState) -> dict:
        raise TimeoutError()  # str(exc) would be ""

    spec = ToolSpec(id="slow", type="python", store_response_as="slow_result")
    ex = ToolExecutor(http=FakeHttp([]), python_tools={"slow": slow_fn})
    events = await ex.execute(spec, _state())
    result = events[1]
    assert isinstance(result, ToolResultEvent) and result.ok is False
    assert (result.error or "").startswith("TimeoutError")


async def test_coercion_failure_shape() -> None:
    spec = ToolSpec(
        id="score",
        type="python",
        store_response_as="score_result",
        args={"n": SlotSpec(type="int")},
    )
    ex = ToolExecutor(http=FakeHttp([]))
    events = await ex.execute(spec, _state(), args={"n": "7.5"})
    kinds = [type(e).__name__ for e in events]
    assert kinds == ["ToolCallEvent", "ToolResultEvent"]
    result = events[1]
    assert isinstance(result, ToolResultEvent)
    assert result.ok is False and result.status is None
    assert (result.error or "").startswith("bad args")


async def test_template_body_parses_to_json() -> None:
    # A compiler '_template' body renders to a JSON document that becomes
    # the REAL request body — never a literal {"_template": "..."} dict.
    spec = ToolSpec(
        id="confirm_booking",
        method="POST",
        url="{{ env.API_BASE_URL }}/bookings/confirm",
        body={"_template": '{"hold_id": {{ slots.hold_id|tojson }}, "n": 2}'},
        store_response_as="confirm_result",
    )
    http = FakeHttp([(200, {"data": {"booking_id": "b1"}})])
    events = await ToolExecutor(http=http).execute(spec, _state(hold_id="h-9"))
    assert http.calls[0]["body"] == {"hold_id": "h-9", "n": 2}
    assert "_template" not in http.calls[0]["body"]
    result = events[1]
    assert isinstance(result, ToolResultEvent) and result.ok


async def test_template_body_invalid_json_fails() -> None:
    spec = ToolSpec(
        id="confirm_booking",
        method="POST",
        url="https://api.test/bookings/confirm",
        body={"_template": "hold={{ slots.hold_id }} (not json)"},
        store_response_as="confirm_result",
    )
    http = FakeHttp([])
    events = await ToolExecutor(http=http).execute(spec, _state(hold_id="h-9"))
    kinds = [type(e).__name__ for e in events]
    assert kinds == ["ToolCallEvent", "ToolResultEvent"]
    result = events[1]
    assert isinstance(result, ToolResultEvent) and result.ok is False
    assert "not valid JSON" in (result.error or "")
    assert http.calls == []  # a broken body never reaches HTTP


async def test_template_key_with_other_fields_is_not_parsed() -> None:
    # Only an EXACT {"_template": <str>} body is the whole-document form.
    spec = ToolSpec(
        id="odd",
        method="POST",
        url="https://api.test/odd",
        body={"_template": '{"a": 1}', "extra": "x"},
        store_response_as="odd_result",
    )
    http = FakeHttp([(200, {})])
    await ToolExecutor(http=http).execute(spec, _state())
    assert http.calls[0]["body"] == {"_template": '{"a": 1}', "extra": "x"}


async def test_env_update_missing_path_skipped() -> None:
    # env_updates wants data.hold_id, but the response has no hold_id.
    http = FakeHttp([(200, {"data": {}})])
    ex = ToolExecutor(http=http)
    events = await ex.execute(HOLD, _state(slot_id="s1", players=2))
    kinds = [type(e).__name__ for e in events]
    assert kinds == ["ToolCallEvent", "ToolResultEvent"]  # no EnvWriteEvent


async def test_secret_body_keys_redacted() -> None:
    spec = ToolSpec(
        id="auth",
        method="POST",
        url="{{ env.API_BASE_URL }}/auth",
        body={"client_secret": "{{ env.CS }}", "city": "x"},
        store_response_as="auth_result",
    )
    state = _state()
    state.env["CS"] = "s3cr3t"
    http = FakeHttp([(200, {})])
    ex = ToolExecutor(http=http)
    events = await ex.execute(spec, state)
    call = events[0]
    assert isinstance(call, ToolCallEvent)
    assert call.args["body"]["client_secret"] == "***"  # event log masked
    assert call.args["body"]["city"] == "x"  # non-secret keys intact
    assert http.calls[0]["body"]["client_secret"] == "s3cr3t"  # real body sent


async def test_redaction_recursive_and_broad() -> None:
    spec = ToolSpec(
        id="nested",
        method="POST",
        url="{{ env.API_BASE_URL }}/nested",
        body={"auth": {"client_secret": "s"}, "items": [{"jwt": "x", "city": "Pune"}]},
        store_response_as="nested_result",
    )
    http = FakeHttp([(200, {})])
    ex = ToolExecutor(http=http)
    events = await ex.execute(spec, _state())
    call = events[0]
    assert isinstance(call, ToolCallEvent)
    assert call.args["body"]["auth"] == "***"  # broad denylist: auth masked
    assert call.args["body"]["items"][0]["jwt"] == "***"  # recursed into list
    assert call.args["body"]["items"][0]["city"] == "Pune"  # non-secret intact
    assert http.calls[0]["body"] == {  # real body sent unmasked
        "auth": {"client_secret": "s"},
        "items": [{"jwt": "x", "city": "Pune"}],
    }


async def test_url_redaction() -> None:
    spec = ToolSpec(
        id="geo",
        method="GET",
        url="https://u:p@api.test/x?api_key={{ env.K }}&city=pune",
        store_response_as="geo_result",
    )
    state = _state()
    state.env["K"] = "sek"
    http = FakeHttp([(200, {})])
    ex = ToolExecutor(http=http)
    events = await ex.execute(spec, state)
    call = events[0]
    assert isinstance(call, ToolCallEvent)
    recorded = call.args["url"]
    assert "u:p@" not in recorded  # userinfo stripped
    assert "sek" not in recorded
    assert "api_key=***" in recorded  # secret param masked, key kept
    assert "city=pune" in recorded  # non-secret param intact
    # the REAL url (userinfo + secrets) still went to http
    assert http.calls[0]["url"] == "https://u:p@api.test/x?api_key=sek&city=pune"


def test_bool_coercion_false() -> None:
    assert coerce_args({"f": "false"}, {"f": SlotSpec(type="bool")})["f"] is False


# --- Idempotency keys on side-effecting tool calls (capability tool-call-idempotency) ---


async def test_post_tool_carries_idempotency_key() -> None:
    http = FakeHttp([(200, {"data": {}})])
    await ToolExecutor(http=http).execute(HOLD, _state(slot_id="s1", players=2))
    key = http.calls[0]["headers"].get("Idempotency-Key")
    assert isinstance(key, str) and len(key) == 64  # sha256 hex digest


async def test_idempotency_key_deterministic_for_same_request() -> None:
    # A retry re-runs execute with identical inputs → identical key, so the
    # server de-dupes the duplicate side effect.
    http = FakeHttp([(200, {"data": {}}), (200, {"data": {}})])
    ex = ToolExecutor(http=http)
    await ex.execute(HOLD, _state(slot_id="s1", players=2))
    await ex.execute(HOLD, _state(slot_id="s1", players=2))
    assert (
        http.calls[0]["headers"]["Idempotency-Key"]
        == http.calls[1]["headers"]["Idempotency-Key"]
    )


async def test_idempotency_key_differs_for_different_request() -> None:
    http = FakeHttp([(200, {"data": {}}), (200, {"data": {}})])
    ex = ToolExecutor(http=http)
    await ex.execute(HOLD, _state(slot_id="s1", players=2))
    await ex.execute(HOLD, _state(slot_id="s2", players=4))
    assert (
        http.calls[0]["headers"]["Idempotency-Key"]
        != http.calls[1]["headers"]["Idempotency-Key"]
    )


async def test_get_tool_has_no_idempotency_key() -> None:
    spec = ToolSpec(
        id="lookup",
        method="GET",
        url="{{ env.API_BASE_URL }}/lookup",
        store_response_as="lookup_result",
    )
    http = FakeHttp([(200, {})])
    await ToolExecutor(http=http).execute(spec, _state())
    assert "Idempotency-Key" not in http.calls[0]["headers"]


async def test_author_supplied_idempotency_key_preserved() -> None:
    spec = HOLD.model_copy(
        update={"headers": {**HOLD.headers, "Idempotency-Key": "author-key"}}
    )
    http = FakeHttp([(200, {"data": {}})])
    await ToolExecutor(http=http).execute(spec, _state(slot_id="s1", players=2))
    assert http.calls[0]["headers"]["Idempotency-Key"] == "author-key"


async def test_idempotency_key_independent_of_auth_header() -> None:
    # A 401→refresh→replay changes only the Authorization token; the same
    # logical operation must reuse the same key so the replay is de-duped.
    http = FakeHttp([(200, {"data": {}}), (200, {"data": {}})])
    ex = ToolExecutor(http=http)
    await ex.execute(HOLD, _state(slot_id="s1", players=2))  # ACCESS_TOKEN=tok-1
    s2 = _state(slot_id="s1", players=2)
    s2.env["ACCESS_TOKEN"] = "tok-2-refreshed"
    await ex.execute(HOLD, s2)
    assert (
        http.calls[0]["headers"]["Authorization"]
        != http.calls[1]["headers"]["Authorization"]
    )
    assert (
        http.calls[0]["headers"]["Idempotency-Key"]
        == http.calls[1]["headers"]["Idempotency-Key"]
    )


# Secret-named tokens in the rendered request/response must never reach logs or
# stdout — regression guard for the fixed print() leak (rendered URL+body and
# refresh-minted tokens were printed unredacted three lines after the event log
# deliberately redacted them).
LEAKY = ToolSpec(
    id="refresh_tool",
    method="POST",
    url="{{ env.API_BASE_URL }}/refresh?api_key={{ slots.req_secret }}",
    body={"password": "{{ slots.req_secret }}", "user": "{{ slots.user }}"},
    store_response_as="refresh_result",
)


async def test_no_secret_reaches_logs_or_stdout(caplog, capsys) -> None:
    caplog.set_level(logging.DEBUG)
    resp_token = "tokLIVESECRET"
    http = FakeHttp([(200, {"data": {"access_token": resp_token}})])
    await ToolExecutor(http=http).execute(
        LEAKY, _state(req_secret="hunter2SECRET", user="alice")
    )
    # The real request DID carry the secret (behaviour unchanged)...
    assert http.calls[0]["url"].endswith("/refresh?api_key=hunter2SECRET")
    assert http.calls[0]["body"]["password"] == "hunter2SECRET"
    # ...but neither the request secret nor the response token may appear in logs.
    assert "hunter2SECRET" not in caplog.text
    assert resp_token not in caplog.text
    # Redaction actually happened, and the non-secret parts are still logged.
    assert "***" in caplog.text
    assert "api.test/refresh" in caplog.text
    assert "user=alice" not in caplog.text or "***" in caplog.text
    # Nothing goes to stdout any more.
    captured = capsys.readouterr()
    assert captured.out == ""


async def test_tool_exception_type_logged_not_url(caplog, capsys) -> None:
    # httpx exceptions embed the full request URL incl. query secrets; only the
    # exception type is logged at WARNING, never the message at INFO/WARNING.
    class Boom:
        async def __call__(self, **_: object) -> tuple[int, dict]:
            raise RuntimeError(
                "connect to https://api.test/refresh?api_key=hunter2SECRET"
            )

    caplog.set_level(logging.INFO)
    events = await ToolExecutor(http=Boom()).execute(
        LEAKY, _state(req_secret="hunter2SECRET", user="alice")
    )
    assert events[-1].ok is False
    warning_text = "\n".join(
        r.message for r in caplog.records if r.levelno >= logging.WARNING
    )
    assert "hunter2SECRET" not in warning_text
    assert "RuntimeError" in warning_text
    assert capsys.readouterr().out == ""


class CaptureHttp:
    """Records the timeout and headers of each call; returns a fixed response."""

    def __init__(self, response: tuple[int, dict] = (200, {"data": {}})) -> None:
        self.response = response
        self.timeouts: list[float] = []
        self.headers: list[dict] = []

    async def __call__(self, *, method, url, headers, body, timeout):
        self.timeouts.append(timeout)
        self.headers.append(dict(headers))
        return self.response


def _get(url: str) -> ToolSpec:
    return ToolSpec(id="fetch", method="GET", url=url, store_response_as="r")


# --- task 2: timeout clamp -------------------------------------------------


async def test_timeout_clamped_to_bounds() -> None:
    cap = CaptureHttp()
    ex = ToolExecutor(http=cap)
    for declared, expected in [(9999.0, 300.0), (0.0, 0.1), (-5.0, 0.1), (30.0, 30.0)]:
        spec = HOLD.model_copy(update={"timeout": declared})
        await ex.execute(spec, _state(slot_id="s", players=1))
        assert cap.timeouts[-1] == expected


def test_out_of_range_timeout_still_constructs() -> None:
    # No Field bound on ToolSpec.timeout: a persisted playbook with an
    # out-of-range value must LOAD (clamp happens at execute, not validation).
    spec = ToolSpec(id="x", method="GET", url="https://api.test/x", timeout=3600.0)
    assert spec.timeout == 3600.0


# --- task 3: transport retry ----------------------------------------------


async def test_transport_retry_then_success(monkeypatch) -> None:
    monkeypatch.setattr(toolexec, "_backoff", lambda a: 0.0)

    class Flaky:
        def __init__(self) -> None:
            self.calls = 0
            self.keys: list[str | None] = []

        async def __call__(self, *, method, url, headers, body, timeout):
            self.calls += 1
            self.keys.append(headers.get("Idempotency-Key"))
            if self.calls < 3:
                raise ConnectionError("reset")
            return (200, {"data": {"hold_id": "h"}})

    flaky = Flaky()
    events = await ToolExecutor(http=flaky).execute(
        HOLD, _state(slot_id="s", players=1)
    )
    assert flaky.calls == 3
    # Exactly one ToolCallEvent despite three transport attempts (retries stay
    # inside one execute — no per-attempt event, no RetrySpec multiplication).
    assert [type(e).__name__ for e in events].count("ToolCallEvent") == 1
    assert events[1].ok is True
    # Byte-identical request each attempt ⇒ same idempotency key.
    assert len(set(flaky.keys)) == 1 and flaky.keys[0] is not None


async def test_transport_retries_exhausted_yields_failed(monkeypatch) -> None:
    monkeypatch.setattr(toolexec, "_backoff", lambda a: 0.0)

    class AlwaysFail:
        def __init__(self) -> None:
            self.calls = 0

        async def __call__(self, **_):
            self.calls += 1
            raise ConnectionError("down")

    h = AlwaysFail()
    events = await ToolExecutor(http=h).execute(HOLD, _state(slot_id="s", players=1))
    assert h.calls == 3  # 1 + _TRANSPORT_RETRIES
    assert events[-1].ok is False
    assert "ConnectionError" in (events[-1].error or "")


async def test_timeout_is_terminal_not_retried(monkeypatch) -> None:
    monkeypatch.setattr(toolexec, "_backoff", lambda a: 0.0)

    class ReadTimeout(Exception):  # name contains "Timeout" (httpx-style)
        pass

    for exc_type in (TimeoutError, ReadTimeout):

        class TimeoutHttp:
            def __init__(self) -> None:
                self.calls = 0

            async def __call__(self, **_):
                self.calls += 1
                raise exc_type("timed out")

        h = TimeoutHttp()
        events = await ToolExecutor(http=h).execute(
            HOLD, _state(slot_id="s", players=1)
        )
        assert h.calls == 1, f"{exc_type.__name__} was retried"
        assert events[-1].ok is False


async def test_non_2xx_status_single_shot() -> None:
    cap = CaptureHttp(response=(503, {"error": "upstream"}))
    events = await ToolExecutor(http=cap).execute(HOLD, _state(slot_id="s", players=1))
    assert len(cap.timeouts) == 1  # a returned status is never retried
    assert events[1].ok is False and events[1].status == 503


# --- task 4: response cap + shared client ---------------------------------


async def test_response_cap_rejects_via_content_length(monkeypatch) -> None:
    import httpx

    def handler(_request):
        return httpx.Response(200, content=b"x" * (2 * 1024 * 1024))

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(toolexec, "_client", client)
    with pytest.raises(ValueError, match="too large"):
        await toolexec.httpx_http(
            method="GET", url="https://api.test/big", headers={}, body=None, timeout=5.0
        )
    await client.aclose()


async def test_response_cap_rejects_chunked_midstream(monkeypatch) -> None:
    import httpx

    async def agen():
        for _ in range(3):
            yield b"y" * (512 * 1024)  # 1.5MB total, streamed, no content-length

    def handler(_request):
        return httpx.Response(200, content=agen())

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(toolexec, "_client", client)
    with pytest.raises(ValueError, match="too large"):
        await toolexec.httpx_http(
            method="GET", url="https://api.test/s", headers={}, body=None, timeout=5.0
        )
    await client.aclose()


async def test_response_under_cap_returns_json(monkeypatch) -> None:
    import httpx

    def handler(_request):
        return httpx.Response(200, json={"ok": True})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(toolexec, "_client", client)
    status, data = await toolexec.httpx_http(
        method="GET", url="https://api.test/x", headers={}, body=None, timeout=5.0
    )
    assert status == 200 and data == {"ok": True}
    await client.aclose()


# --- task 5: SSRF guard on the rendered URL -------------------------------


async def test_ssrf_blocks_metadata_via_slot_interpolation() -> None:
    ex = ToolExecutor(http=FakeHttp([(200, {})]))
    events = await ex.execute(
        _get("http://{{ slots.host }}/latest/meta-data"),
        _state(host="169.254.169.254"),
    )
    assert events[-1].ok is False
    assert "blocked tool URL" in (events[-1].error or "")
    # The blocked request never reached HTTP (FakeHttp response untouched).


async def test_ssrf_blocks_obfuscated_ip_spellings() -> None:
    for host in ["2130706433", "0177.0.0.1", "127.1"]:  # decimal / octal / short
        ex = ToolExecutor(http=FakeHttp([(200, {})]))
        events = await ex.execute(_get("http://{{ slots.host }}/x"), _state(host=host))
        assert events[-1].ok is False, f"{host} not blocked"
    # IPv4-mapped IPv6 (bracketed literal so urlsplit finds the host).
    ex = ToolExecutor(http=FakeHttp([(200, {})]))
    events = await ex.execute(_get("http://[::ffff:169.254.169.254]/x"), _state())
    assert events[-1].ok is False


async def test_ssrf_allows_public_host_under_strict_default() -> None:
    ex = ToolExecutor(http=FakeHttp([(200, {"data": {}})]))
    events = await ex.execute(_get("https://api.test/x"), _state())
    assert events[-1].ok is True  # a DNS name is not a literal IP → passes


async def test_ssrf_opt_out_permits_localhost() -> None:
    ex = ToolExecutor(http=FakeHttp([(200, {"data": {}})]), allow_private_hosts=True)
    events = await ex.execute(_get("http://localhost:8000/x"), _state())
    assert events[-1].ok is True
