"""Engine selection + Playbook-mode behavior of the unified DialogMachine."""

import pytest

from superdialog import Flow
from superdialog.dialog_machine import _select_engine
from superdialog.playbook import Playbook
from tests.playbook.test_models import MINIMAL_YAML


def _flow_obj() -> Flow:
    return Flow.model_validate(
        {
            "id": "t",
            "system_prompt": "s",
            "initial_node": "n",
            "nodes": [{"id": "n", "name": "N", "edges": [], "is_final": True}],
        }
    )


def test_flow_object_auto_selects_graph() -> None:
    assert _select_engine(_flow_obj(), "auto") == "graph"


def test_playbook_object_auto_selects_playbook() -> None:
    assert _select_engine(Playbook.from_yaml(MINIMAL_YAML), "auto") == "playbook"


def test_str_path_and_dict_auto_select_playbook() -> None:
    assert _select_engine("booking.yaml", "auto") == "playbook"
    assert _select_engine({"playbook": [{"id": "a"}]}, "auto") == "playbook"


def test_explicit_engine_overrides() -> None:
    assert _select_engine(_flow_obj(), "playbook") == "playbook"  # compile flow
    assert _select_engine("flow.json", "flow") == "graph"


def test_engine_flow_on_playbook_object_is_error() -> None:
    with pytest.raises(ValueError, match="no graph runtime"):
        _select_engine(Playbook.from_yaml(MINIMAL_YAML), "flow")


from superdialog.dialog_machine import _python_tools_from  # noqa: E402
from superdialog.tools import PythonTool  # noqa: E402


async def test_tool_bridge_invokes_execute() -> None:
    calls = {}

    async def impl(city: str) -> dict:
        calls["city"] = city
        return {"ok": True, "city": city}

    tool = PythonTool.of(impl, name="lookup")
    bridged = _python_tools_from([tool])
    assert set(bridged) == {tool.id}
    fn = bridged[tool.id]
    out = await fn({"city": "Pune"}, None)  # state unused by this bridge
    assert out == {"ok": True, "city": "Pune"}
    assert calls["city"] == "Pune"


def test_tool_bridge_empty_and_none() -> None:
    assert _python_tools_from(None) == {}
    assert _python_tools_from([]) == {}


from superdialog import DialogMachine  # noqa: E402
from tests.playbook.test_director import CannedLLM  # noqa: E402
from tests.playbook.test_talker import StreamLLM  # noqa: E402


async def test_playbook_mode_from_path(tmp_path) -> None:
    p = tmp_path / "play.yaml"
    p.write_text(MINIMAL_YAML)
    dm = DialogMachine(str(p), llm="openai/gpt-4o-mini")
    assert dm._engine == "playbook"  # selected, not yet built
    # inject fakes so no network: build the backend, then patch its seams
    dm._talker_override = StreamLLM(["Hi", " there"])
    dm._director_override = CannedLLM({"slots": {}, "advance": None, "note": None})
    result = await dm.turn("hello")
    assert hasattr(result, "text")  # TurnResult carries .text/.metadata


async def test_flow_object_still_graph_engine() -> None:
    dm = DialogMachine(_flow_obj(), llm="openai/gpt-4o-mini")
    assert dm._engine == "graph"


def test_playbook_mode_requires_llm(tmp_path) -> None:
    p = tmp_path / "play.yaml"
    p.write_text(MINIMAL_YAML)
    with pytest.raises(ValueError, match="needs an llm"):
        DialogMachine(str(p))  # no llm, no director_llm
