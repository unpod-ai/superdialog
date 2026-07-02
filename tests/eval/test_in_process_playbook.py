import superdialog.eval.endpoints.in_process as ip


class _FakeMachine:
    def __init__(self, *a, **k):
        self.started = False

    async def start(self):
        from superdialog.stream import Turn

        self.started = True
        return Turn(text="Welcome to the spa!")

    async def turn(self, text, **k):
        from superdialog.stream import Turn

        return Turn(text=f"you said: {text}", metadata={"checkpoint": "collect"})


async def test_playbook_endpoint_delegates_to_dialogmachine(monkeypatch):
    monkeypatch.setattr(ip, "DialogMachine", _FakeMachine)
    ep = ip.InProcessPlaybook(playbook_path="x.yaml", agent_model="openai/gpt-4.1-mini")
    assert "Welcome" in await ep.start()
    assert await ep.turn("hi") == "you said: hi"
    ep.reset()  # rebuilds machine; must not raise
