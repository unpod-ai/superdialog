from superdialog.eval.endpoints.base import Transcript, TurnRecord


def test_transcript_records_and_serializes_to_messages():
    t = Transcript()
    t.add("assistant", "Hi, how can I help?", latency_ms=12.0)
    t.add("user", "book a haircut")
    t.add("assistant", "Sure, what day?", latency_ms=20.0)
    msgs = t.to_messages()
    assert msgs[0] == {"role": "assistant", "content": "Hi, how can I help?"}
    assert [m["role"] for m in msgs] == ["assistant", "user", "assistant"]
    assert t.assistant_latencies_ms() == [12.0, 20.0]
    assert t.turn_count() == 1  # one user turn
    assert TurnRecord("user", "x").metadata == {}
