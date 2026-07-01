import pytest

from superdialog.eval.endpoints.base import ConversationEndpoint
from superdialog.eval.endpoints.in_process import InProcessVanilla
from tests.eval.fakes import FakeProvider


def _vanilla() -> ConversationEndpoint:
    return InProcessVanilla(playbook_text="bot", llm=FakeProvider({"*": "ok"}))


@pytest.mark.parametrize("factory", [_vanilla])
async def test_endpoint_satisfies_protocol_and_lifecycle(factory):
    ep = factory()
    assert isinstance(ep, ConversationEndpoint)
    assert isinstance(await ep.start(), str)
    assert isinstance(await ep.turn("hello"), str)
    ep.reset()
    assert isinstance(await ep.turn("again"), str)
