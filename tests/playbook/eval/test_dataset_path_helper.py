# tests/playbook/eval/test_dataset_path_helper.py
"""Regression test: _dataset_path() must not silently drop version segments
from multi-dot playbook filenames (e.g. "foo.v2.yaml" -> "foo.v2.evalcases.yaml",
NOT "foo.evalcases.yaml"). A prior implementation using chained Path.with_suffix()
calls silently dropped the ".v2" segment for exactly this shape of filename --
the same shape as the harness's own default playbook
(~/Downloads/flow_golf_ai_updated_full.v2.yaml).
"""

from pathlib import Path

from tests.playbook.eval import test_scenarios as ts_module


def test_dataset_path_preserves_multi_dot_version_segment(monkeypatch):
    monkeypatch.setattr(
        ts_module, "PB_PATH", Path("/tmp/flow_golf_ai_updated_full.v2.yaml")
    )
    result = ts_module._dataset_path()
    assert result == Path("/tmp/flow_golf_ai_updated_full.v2.evalcases.yaml")


def test_dataset_path_simple_filename_unaffected(monkeypatch):
    monkeypatch.setattr(ts_module, "PB_PATH", Path("/tmp/simple.yaml"))
    result = ts_module._dataset_path()
    assert result == Path("/tmp/simple.evalcases.yaml")
