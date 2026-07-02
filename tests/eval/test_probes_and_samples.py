from superdialog.eval.dataset.models import EvalCase, Probe
from superdialog.eval.endpoints.base import Transcript
from superdialog.eval.endpoints.in_process import InProcessVanilla
from superdialog.eval.runner import run_probes, samples_from_run
from superdialog.playbook.eval.models import PersonaSpec
from tests.eval.fakes import FakeProvider


async def test_run_probes_collects_responses():
    ep = InProcessVanilla(
        playbook_text="bot", llm=FakeProvider({"hours": "9 to 5", "*": "no"})
    )
    probes = [
        Probe(
            kind="factual",
            utterance="your hours?",
            expect="9 to 5",
            reference_contexts=["Open 9-5"],
        )
    ]
    results = await run_probes(ep, probes)
    assert results[0][1] == "9 to 5"


def test_samples_from_run_makes_conversation_and_probe_samples():
    t = Transcript()
    t.add("assistant", "hi")
    t.add("user", "book")
    t.add("assistant", "ok")
    case = EvalCase(
        id="c1",
        playbook="p",
        reference="book haircut",
        persona=PersonaSpec(name="s", traits="", goal="book"),
        probes=[
            Probe(
                kind="factual",
                utterance="hours?",
                expect="9-5",
                reference_contexts=["9-5"],
            )
        ],
    )
    probe_results = [(case.probes[0], "9-5")]
    samples = samples_from_run(case, "playbook", t, probe_results)
    assert samples[0].kind == "conversation"
    assert samples[0].reference == "book haircut"
    assert samples[1].kind == "probe" and samples[1].response == "9-5"
    assert samples[1].metadata["mode"] == "playbook"
