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
