from __future__ import annotations

import os
import sys
from pathlib import Path  # noqa: F401 — used in fixtures
from unittest.mock import MagicMock

import pytest

for _mod in [
    "livekit.agents",
    "livekit.agents.llm",
    "livekit.agents.llm.tool_context",
    "livekit.agents.voice",
    "livekit.api",
]:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()


def pytest_addoption(parser: pytest.Parser) -> None:
    # --flow is registered once in the root tests/conftest.py (shared with
    # tests/dialog_machine); registering it again here collides at collection.
    parser.addoption("--traversal", default=None, help="Path to traversal JSON file")
    parser.addoption("--model", default="gpt-4.1-mini", help="OpenAI model for eval LLM")
    # --corpus is consumed by test_run_eval's corpus_path fixture; pytest only
    # honors pytest_addoption from a conftest/plugin, not from a test module, so
    # it must be registered here (was previously declared in the test module and
    # never took effect -> "no option named '--corpus'").
    parser.addoption("--corpus", default=None, help="Path to pre-generated corpus JSON")


@pytest.fixture
def flow_path(request: pytest.FixtureRequest) -> str:
    # These eval tests are manual integration harnesses (they hit a live LLM and
    # need a real flow file). Skip unless an explicit --flow is provided, rather
    # than falling back to a developer-machine path — so the default suite is
    # deterministic and portable instead of running live whenever a hardcoded
    # personal file happens to exist.
    path = request.config.getoption("--flow")
    if not path:
        pytest.skip("--flow not provided")
    if not Path(path).exists():
        pytest.skip(f"flow file not found: {path}")
    return path


@pytest.fixture
def traversal_path(request: pytest.FixtureRequest) -> str:
    path = request.config.getoption("--traversal")
    if not path:
        pytest.skip("--traversal not provided")
    if not Path(path).exists():
        pytest.skip(f"traversal file not found: {path}")
    return path


@pytest.fixture
def eval_model(request: pytest.FixtureRequest) -> str:
    return request.config.getoption("--model")


@pytest.fixture
def llm_fn(eval_model: str):
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set")
    import openai
    client = openai.AsyncOpenAI(api_key=api_key)

    async def _call(messages):
        resp = await client.chat.completions.create(model=eval_model, messages=messages)
        return resp.choices[0].message.content

    return _call


@pytest.fixture
def flow(flow_path: str):
    from superdialog.flow import load_flow
    return load_flow(flow_path)
