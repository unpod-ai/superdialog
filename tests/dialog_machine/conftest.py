"""Per-suite collection rules for the ported dialog_machine tests.

These tests are a direct copy from ``super/tests/core/voice/dialog_machine/``;
several of them depend on modules that were intentionally left behind in
``super.core.voice.dialog_machine.eval`` and related sub-packages (they
target the eval / RL / corpus-generator surface that lives outside the
slim superdialog port). We skip those at collection time rather than
delete them — keeping them around makes the eventual eval port a simpler
copy/paste exercise.
"""

from __future__ import annotations

collect_ignore_glob = [
    # Depend on superdialog.machine.eval.* (not ported)
    "test_engine_advisor.py",
    "test_failure_classifier.py",
    "test_flow_optimizer.py",
    "test_graph_analysis.py",
    "test_multi_model.py",
    "test_path_traversal.py",
    "test_rl_loop.py",
    # Depends on superdialog.machine.engine (not ported)
    "test_engine_resolver.py",
    # Depends on super.core.voice.schema / lite_v2.state
    "test_dialog_machine_e2e.py",
    # Depend on superdialog.machine.adapters.simple_agent (not ported)
    "test_simple_flow_agent.py",
    "test_scope_build_invariant.py",
    # Depends on superdialog.machine.adapters.livekit_bridge (not ported)
    "test_livekit_bridge.py",
    "test_gated_traversal_e2e.py",
    # Depend on hardcoded flow JSON fixtures absent in this tree
    "test_bob_card_e2e.py",
    "test_sample_flow.py",
    "test_custom_tool_e2e.py",
]


import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--flow",
        default=None,
        help="Path to a flow JSON file for real-flow eval tests.",
    )


@pytest.fixture
def flow_under_test(request: pytest.FixtureRequest):
    from superdialog.flow.models import ConversationFlow
    path = request.config.getoption("--flow")
    if path is None:
        pytest.skip("pass --flow <path> to run against a real flow")
    return ConversationFlow.from_json_file(path)
