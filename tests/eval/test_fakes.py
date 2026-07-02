from tests.eval.fakes import FakeProvider, FakeSpeaksUser


async def test_fake_provider_scripts_by_marker():
    llm = FakeProvider({"greet": "Hi there!", "*": "ok"})
    res = await llm.complete([{"role": "user", "content": "please greet"}])
    assert "Hi there" in res


async def test_fake_provider_stream_yields_words():
    llm = FakeProvider({"*": "one two three"})
    out = "".join([c async for c in llm.stream([{"role": "user", "content": "x"}])])
    assert out == "one two three"


async def test_fake_speaks_user_cycles_scripted_lines():
    user = FakeSpeaksUser(["hello", "my name is Sam"])
    assert await user.complete([]) == "hello"
    assert await user.complete([]) == "my name is Sam"
    assert await user.complete([]) == ""  # exhausted
