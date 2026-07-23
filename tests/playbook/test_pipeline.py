from superdialog.playbook.events import EnvWriteEvent, EventLog, SlotWriteEvent
from superdialog.playbook.models import (
    MiddlewareSpec,
    PipelineSpec,
    PipelineStep,
    Playbook,
    RetrySpec,
    ToolSpec,
)
from superdialog.playbook.pipeline import PipelineRunner
from superdialog.playbook.state import ConversationState
from superdialog.playbook.toolexec import ToolExecutor
from tests.playbook.test_toolexec import FakeHttp


def _pb(
    steps: list[PipelineStep], middleware: MiddlewareSpec | None = None
) -> Playbook:
    return Playbook(
        journeys={
            "j": {
                "checkpoints": [
                    {"id": "a"},
                    {"id": "fallback"},
                    {"id": "done", "terminal": True},
                ]
            }
        },
        tools=[
            ToolSpec(
                id="hold",
                method="POST",
                url="http://t/hold",
                headers={"Authorization": "Bearer {{ env.ACCESS_TOKEN }}"},
                store_response_as="hold_result",
            ),
            ToolSpec(
                id="confirm",
                method="POST",
                url="http://t/confirm",
                store_response_as="confirm_result",
            ),
            ToolSpec(
                id="refresh",
                method="POST",
                url="http://t/auth",
                env_updates={"ACCESS_TOKEN": "token"},
            ),
            ToolSpec(
                id="maybe",
                method="POST",
                url="http://t/maybe",
                store_response_as="maybe_result",
                when="slots.nonexistent",
            ),
            ToolSpec(
                id="confirm_hold",
                method="POST",
                url="http://t/confirm/{{ results.hold_result.data.hold_id }}",
                store_response_as="confirm_result",
            ),
        ],
        pipelines=[PipelineSpec(id="p", steps=steps)],
        middleware=middleware,
    )


def _state() -> ConversationState:
    log = EventLog()
    log.append(
        SlotWriteEvent(key="slot_id", value="s1", status="confirmed", by="director")
    )
    state = ConversationState.fold(log)
    state.env["ACCESS_TOKEN"] = "tok-OLD"
    return state


async def test_happy_path_runs_all_steps() -> None:
    pb = _pb(
        [
            PipelineStep(tool="hold", on={"ok": "continue"}),
            PipelineStep(tool="confirm", on={"ok": "j.done"}),
        ]
    )
    runner = PipelineRunner(pb, ToolExecutor(http=FakeHttp([(200, {}), (200, {})])))
    result = await runner.run("p", _state())
    assert result.advance_to == "j.done" and result.ok


async def test_http_code_branch() -> None:
    pb = _pb([PipelineStep(tool="hold", on={"ok": "j.done", "http_409": "j.fallback"})])
    runner = PipelineRunner(
        pb, ToolExecutor(http=FakeHttp([(409, {"error": "taken"})]))
    )
    result = await runner.run("p", _state())
    assert result.advance_to == "j.fallback" and not result.ok


async def test_retry_then_exhaust() -> None:
    pb = _pb(
        [
            PipelineStep(
                tool="confirm",
                on={
                    "ok": "j.done",
                    "failed": RetrySpec(retry=1, on_exhaust="j.fallback"),
                },
            )
        ]
    )
    http = FakeHttp([(503, {}), (503, {})])
    runner = PipelineRunner(pb, ToolExecutor(http=http))
    result = await runner.run("p", _state())
    assert len(http.calls) == 2  # original + 1 retry
    assert result.advance_to == "j.fallback"
    assert result.error_slot == {"error_context": "p:confirm"}


async def test_middleware_refresh_and_replay_on_401() -> None:
    pb = _pb(
        [PipelineStep(tool="hold", on={"ok": "j.done"})],
        middleware=MiddlewareSpec(on_status=401, refresh_with="refresh"),
    )
    http = FakeHttp([(401, {}), (200, {"token": "tok-NEW"}), (200, {})])
    runner = PipelineRunner(pb, ToolExecutor(http=http))
    result = await runner.run("p", _state())
    assert result.advance_to == "j.done"
    assert [c["url"] for c in http.calls] == [
        "http://t/hold",
        "http://t/auth",
        "http://t/hold",
    ]
    # the replayed call must render with the REFRESHED token
    assert http.calls[2]["headers"]["Authorization"] == "Bearer tok-NEW"
    assert http.calls[0]["headers"]["Authorization"] == "Bearer tok-OLD"


async def test_middleware_does_not_loop() -> None:
    pb = _pb(
        [PipelineStep(tool="hold", on={"ok": "j.done"})],
        middleware=MiddlewareSpec(on_status=401, refresh_with="refresh"),
    )
    http = FakeHttp([(401, {}), (200, {"token": "t2"}), (401, {})])
    runner = PipelineRunner(pb, ToolExecutor(http=http))
    result = await runner.run("p", _state())
    assert len(http.calls) == 3  # no second refresh
    assert not result.ok  # replayed 401 has no branch -> failed stop


async def test_failure_without_branch_stops() -> None:
    pb = _pb([PipelineStep(tool="hold", on={"ok": "j.done"})])
    runner = PipelineRunner(pb, ToolExecutor(http=FakeHttp([(503, {})])))
    result = await runner.run("p", _state())
    assert not result.ok and result.advance_to is None
    assert result.error_slot == {"error_context": "p:hold"}


async def test_skipped_step_continues() -> None:
    # Middleware is configured: a skipped step (no ToolResultEvent) must not
    # trigger a refresh, and the pipeline must proceed to the next step.
    pb = _pb(
        [
            PipelineStep(tool="maybe", on={"ok": "continue"}),
            PipelineStep(tool="confirm", on={"ok": "j.done"}),
        ],
        middleware=MiddlewareSpec(on_status=401, refresh_with="refresh"),
    )
    http = FakeHttp([(200, {})])
    runner = PipelineRunner(pb, ToolExecutor(http=http))
    result = await runner.run("p", _state())
    assert result.ok and result.advance_to == "j.done"
    assert len(http.calls) == 1  # only "confirm"; skip is silent


async def test_refresh_env_event_in_result_events() -> None:
    # The refreshed token must survive the run: result.events carries the
    # EnvWriteEvent so the caller's real log retains it.
    pb = _pb(
        [PipelineStep(tool="hold", on={"ok": "j.done"})],
        middleware=MiddlewareSpec(on_status=401, refresh_with="refresh"),
    )
    http = FakeHttp([(401, {}), (200, {"token": "tok-NEW"}), (200, {})])
    runner = PipelineRunner(pb, ToolExecutor(http=http))
    result = await runner.run("p", _state())
    assert any(
        isinstance(e, EnvWriteEvent)
        and e.key == "ACCESS_TOKEN"
        and e.value == "tok-NEW"
        for e in result.events
    )


async def test_cross_step_result_visibility() -> None:
    # Step 2's url template reads step 1's stored result via the _refold
    # overlay in the main loop.
    pb = _pb(
        [
            PipelineStep(tool="hold", on={"ok": "continue"}),
            PipelineStep(tool="confirm_hold", on={"ok": "j.done"}),
        ]
    )
    http = FakeHttp([(200, {"hold_id": "h9"}), (200, {})])
    runner = PipelineRunner(pb, ToolExecutor(http=http))
    result = await runner.run("p", _state())
    assert result.ok and result.advance_to == "j.done"
    assert "h9" in http.calls[1]["url"]


async def test_retry_then_success() -> None:
    pb = _pb(
        [
            PipelineStep(
                tool="confirm",
                on={
                    "ok": "j.done",
                    "failed": RetrySpec(retry=2, on_exhaust="j.fallback"),
                },
            )
        ]
    )
    http = FakeHttp([(503, {}), (200, {})])
    runner = PipelineRunner(pb, ToolExecutor(http=http))
    result = await runner.run("p", _state())
    assert result.ok and result.advance_to == "j.done"
    assert len(http.calls) == 2  # one failure + one success, no extras


async def test_transport_retry_does_not_multiply_with_retryspec(monkeypatch) -> None:
    # A transient RAISED exception is absorbed by the executor's transport retry
    # inside ONE execute() — so the pipeline RetrySpec (which only fires on a
    # returned failure branch) never rounds, and the two layers don't compound.
    from superdialog.playbook import toolexec

    monkeypatch.setattr(toolexec, "_backoff", lambda a: 0.0)

    class RaiseThenOk:
        def __init__(self, fails: int) -> None:
            self.fails = fails
            self.calls = 0

        async def __call__(self, **_):
            self.calls += 1
            if self.calls <= self.fails:
                raise ConnectionError("blip")
            return (200, {"data": {}})

    pb = _pb(
        [
            PipelineStep(
                tool="confirm",
                on={
                    "ok": "j.done",
                    "failed": RetrySpec(retry=2, on_exhaust="j.fallback"),
                },
            )
        ]
    )
    http = RaiseThenOk(fails=2)
    result = await PipelineRunner(pb, ToolExecutor(http=http)).run("p", _state())
    assert http.calls == 3  # 1 + 2 transport retries, all in one execute
    assert result.advance_to == "j.done"  # RetrySpec never fired → no fallback
