"""superdialog CLI: generate / chat / optimize / playbook / flow / eval.

``generate`` creates a playbook from a prompt (the default creation path)
and ``chat`` runs any artifact — full playbook, simple format, or legacy
flow JSON — on the Playbook engine by default (``--mode flow`` opts into
the legacy DialogMachine). The CLI is intentionally thin -- it defers all
real work to the public API so each command behaves the same as a Python
caller would.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path
from typing import Any, AsyncIterator, Sequence, cast

import yaml
from dotenv import load_dotenv

from .. import DialogMachine, Flow, create_dialog_flow
from ..stream import StreamChunk


def _run_chat_repl(flow: "Flow", llm: str, adapter: str = "llm") -> None:
    """Blocking interactive REPL. Separated for testability."""
    machine = DialogMachine(flow=flow, llm=llm, adapter=adapter)

    async def _loop() -> None:
        result = await machine.start()
        if result.text:
            print(result.text)
        while True:
            try:
                user = input("> ")
            except (EOFError, KeyboardInterrupt):
                print()
                return
            if user.strip() in {"quit", "exit"}:
                return
            if not user.strip():
                continue
            _t0 = time.monotonic()
            turn = await machine.turn(user)
            _ms = int((time.monotonic() - _t0) * 1000)
            if turn.text:
                print(turn.text)
            print(f"[{_ms}ms]", file=sys.stderr)
            if machine.is_complete:
                return

    asyncio.run(_loop())


def _looks_like_simple_playbook(path: str) -> bool:
    """True when ``path`` parses to a simple playbook (top-level ``playbook`` list).

    Tolerant by design: any read/parse failure means "not a simple playbook" so
    the caller falls back to the flow loader, which then reports the real error.
    """
    from ..playbook.simple import is_simple_playbook

    try:
        text = Path(path).read_text(encoding="utf-8")
        doc = yaml.safe_load(text)
    except (OSError, yaml.YAMLError):
        return False
    return is_simple_playbook(doc)


def _drive_agent(agent: Any, speak_first: bool = False) -> None:
    """Run the shared async REPL loop over a PlaybookAgent.

    ``speak_first=True`` simulates outbound-call UX where the agent speaks
    the opening greeting before waiting for user input.
    """

    async def _loop() -> None:
        lines = await agent.runtime.start()
        for line in lines:
            print(line)
        if speak_first and not lines:
            # Agent greets before the user speaks (outbound call connect).
            async for chunk in agent.greet():
                if chunk.text:
                    print(chunk.text, end="", flush=True)
            print()
        while True:
            try:
                user = input("> ")
            except (EOFError, KeyboardInterrupt):
                print()
                return
            if user.strip() in {"quit", "exit"}:
                return
            if not user.strip():
                continue
            chunks = cast(
                AsyncIterator[StreamChunk], await agent.turn(user, stream=True)
            )
            async for chunk in chunks:
                if chunk.text:
                    print(chunk.text, end="", flush=True)
            print()
            state = agent.runtime.state
            print(
                f"[checkpoint={state.checkpoint_id} ended={state.ended}]",
                file=sys.stderr,
            )
            if state.ended:
                return

    asyncio.run(_loop())


def _build_playbook_agent(
    playbook: Any,
    llm: str,
    barrier_timeout: float = 0.4,
    token_budget: int = 4000,
) -> Any:
    """Build a PlaybookAgent for ``playbook`` using a single resolved model."""
    from ..llm.resolver import resolve_llm
    from ..playbook import PlaybookAgent, httpx_http, provider_adapters

    provider = resolve_llm(llm)
    director, talker = provider_adapters(provider)
    return PlaybookAgent(
        playbook=playbook,
        talker_llm=talker,
        director_llm=director,
        http=httpx_http,
        barrier_timeout=barrier_timeout,
        token_budget=token_budget,
    )


def _run_playbook_repl(playbook_path: str, llm: str) -> None:
    """Blocking interactive REPL driving a Playbook. Separated for testability.

    One model drives both the Director and the Talker here — fine for a dev
    REPL. Production splits them (a cheap streaming Talker, a stronger
    Director); see :func:`superdialog.playbook.provider_adapters`.
    """
    from ..playbook import Playbook

    agent = _build_playbook_agent(Playbook.load(playbook_path), llm)
    _drive_agent(agent)


def _run_simple_repl(simple_path: str, llm: str) -> None:
    """Blocking interactive REPL driving a compiled simple playbook."""
    from ..playbook.simple import load_simple

    # barrier_timeout=2.0: gives Director enough time to complete before
    # Talker plays filler (default 0.4 s is too short for typical LLM latency).
    # token_budget=8000: large personas can exceed 4 k tokens, leaving no room
    # for transcript; 8 k keeps both the system block and conversation visible.
    agent = _build_playbook_agent(
        load_simple(simple_path), llm, barrier_timeout=2.0, token_budget=8000
    )
    _drive_agent(agent, speak_first=True)


def _run_generate_playbook(prompt: str, llm: str) -> str:
    """Generate simple-format playbook YAML from a description (LLM-backed)."""
    from ..llm.resolver import resolve_llm
    from ..playbook import provider_adapters
    from ..playbook.generate import generate_simple_playbook

    director, _ = provider_adapters(resolve_llm(llm))
    return asyncio.run(generate_simple_playbook(prompt, director))


def _cmd_generate_playbook(args: argparse.Namespace) -> int:
    """Generate a playbook (simple format) — the default creation path."""
    load_dotenv()
    prompt = args.prompt
    if not prompt and getattr(args, "from_file", None):
        prompt = Path(args.from_file).read_text(encoding="utf-8")
    if not prompt or not prompt.strip():
        print(
            'Provide an agent description: superdialog generate "..." (or --from FILE)',
            file=sys.stderr,
        )
        return 1
    text = _run_generate_playbook(prompt.strip(), args.llm)
    Path(args.output).write_text(text, encoding="utf-8")
    print(f"Wrote {args.output}")
    print(f"Try it: superdialog chat --playbook {args.output}")
    return 0


def _cmd_chat(args: argparse.Namespace) -> int:
    """Interactive REPL over a flow or a playbook (auto-detected or explicit)."""
    load_dotenv()

    llm = getattr(args, "llm", "openai/gpt-4.1-mini") or "openai/gpt-4.1-mini"

    playbook_path = getattr(args, "playbook", None)
    if playbook_path:
        if not Path(playbook_path).exists():
            print(f"No playbook found at: {playbook_path}", file=sys.stderr)
            return 1
        return _chat_playbook(playbook_path, llm)

    simple_path = getattr(args, "simple", None)
    if simple_path:
        if not Path(simple_path).exists():
            print(f"No simple playbook found at: {simple_path}", file=sys.stderr)
            return 1
        return _chat_simple(simple_path, llm)

    flow_path = getattr(args, "flow", None)
    if not flow_path:  # default: prefer a playbook, fall back to a flow
        for candidate in ("playbook.yaml", "flow.json"):
            if Path(candidate).exists():
                flow_path = candidate
                break
    if not flow_path or not Path(flow_path).exists():
        print(
            f"No {flow_path or 'playbook.yaml (or flow.json)'} found.\n"
            'Create one: superdialog generate "describe your agent" '
            "--output playbook.yaml",
            file=sys.stderr,
        )
        return 1

    mode = getattr(args, "mode", "playbook") or "playbook"
    if mode == "flow":  # explicit opt-in to the legacy DialogMachine engine
        try:
            flow = Flow.load(flow_path)
        except Exception as exc:  # malformed JSON / schema -> clean exit
            print(f"Could not load flow {flow_path}: {exc}", file=sys.stderr)
            return 1
        adapter = getattr(args, "adapter", "llm") or "llm"
        _run_chat_repl(flow, llm, adapter)
        return 0

    if _looks_like_simple_playbook(flow_path):
        return _chat_simple(flow_path, llm)

    # Default: the Playbook engine runs everything — full playbooks load
    # directly; flow JSON is compiled losslessly by the unified loader.
    return _chat_playbook(flow_path, llm)


def _chat_playbook(path: str, llm: str) -> int:
    """Validate-load a playbook (clean error on failure) then run its REPL."""
    from ..playbook import Playbook

    try:
        Playbook.load(path)  # pre-flight: surface schema errors as one line
    except Exception as exc:
        print(f"Invalid playbook {path}: {exc}", file=sys.stderr)
        return 1
    _run_playbook_repl(path, llm)
    return 0


def _chat_simple(path: str, llm: str) -> int:
    """Validate-compile a simple playbook (clean error on failure) then run it."""
    from ..playbook.simple import load_simple

    try:
        load_simple(path)  # pre-flight: surface compile errors as one line
    except Exception as exc:
        print(f"Invalid simple playbook {path}: {exc}", file=sys.stderr)
        return 1
    _run_simple_repl(path, llm)
    return 0


def _run_optimize(
    playbook_path: str,
    *,
    rounds: int,
    n: int,
    personas_path: str | None,
    llm: str,
    candidate_llm: str | None,
    user_llm: str | None,
) -> tuple[str, list[str]]:
    """Run the optimize loop against real providers; return (yaml, trace lines)."""
    from ..llm.resolver import resolve_llm
    from ..playbook import (
        Playbook,
        PlaybookAgent,
        httpx_http,
        make_editable,
        optimize,
        provider_adapters,
    )
    from ..playbook.personas import (
        derive_default_persona,
        generate_personas,
        load_personas,
        persona_cache_path,
        save_personas,
    )

    doc = make_editable(Path(playbook_path).read_text(encoding="utf-8"))
    director, talker = provider_adapters(resolve_llm(llm))
    cand = (
        provider_adapters(resolve_llm(candidate_llm))[0] if candidate_llm else director
    )
    user = provider_adapters(resolve_llm(user_llm))[0] if user_llm else director

    def agent_factory(pb: "Playbook") -> "PlaybookAgent":
        return PlaybookAgent(
            playbook=pb, talker_llm=talker, director_llm=director, http=httpx_http
        )

    notes: list[str] = []

    async def _go() -> Any:
        playbook = doc.compile()
        cache = persona_cache_path(playbook_path)
        if personas_path:
            personas = load_personas(personas_path)
        elif Path(cache).exists():
            personas = load_personas(cache)
            notes.append(f"personas: loaded cache {cache}")
        else:
            try:
                personas = await generate_personas(playbook, cand)
                save_personas(personas, cache)
                notes.append(f"personas: generated suite -> {cache} (review it)")
            except ValueError as exc:
                personas = [derive_default_persona(playbook)]
                notes.append(
                    f"personas: generation failed ({exc}); using one derived persona"
                )
        return await optimize(
            doc,
            personas=personas,
            candidate_llm=cand,
            user_llm=user,
            agent_factory=agent_factory,
            rounds=rounds,
            n=n,
        )

    report = asyncio.run(_go())
    lines = list(notes)
    for t in report.trace:
        if t.candidate_breakdown is None:
            lines.append(f"round {t.round_no}: {t.detail}")
        else:
            n_edits = len(t.edits)
            verdict = (
                f"accepted ({n_edits} edit{'s' if n_edits != 1 else ''})"
                if t.accepted
                else "rejected"
            )
            lines.append(
                f"round {t.round_no}: incumbent "
                f"{t.incumbent_breakdown.objective:.2f} vs candidate "
                f"{t.candidate_breakdown.objective:.2f} - {verdict}"
            )
    lines.append(
        f"objective: {report.initial_breakdown.objective:.2f} -> "
        f"{report.final_breakdown.objective:.2f}"
    )
    return report.final_yaml, lines


def _cmd_optimize(args: argparse.Namespace) -> int:
    """Validate inputs, run the loop, write the improved playbook."""
    path = args.playbook
    if not Path(path).exists():
        print(f"Playbook not found: {path}", file=sys.stderr)
        return 1
    try:  # pre-flight: the unified loader accepts full and simple formats
        from ..playbook import Playbook

        Playbook.load(path)
    except Exception as exc:
        print(f"Invalid playbook {path}: {exc}", file=sys.stderr)
        return 1
    out = args.out or str(Path(path).parent / f"improved.{Path(path).name}")
    final_yaml, lines = _run_optimize(
        path,
        rounds=args.rounds,
        n=args.n,
        personas_path=args.personas,
        llm=args.llm,
        candidate_llm=args.candidate_llm,
        user_llm=args.user_llm,
    )
    Path(out).write_text(final_yaml, encoding="utf-8")
    for line in lines:
        print(line)
    print(f"Wrote {out}")
    return 0


def _cmd_lint(args: argparse.Namespace) -> int:
    """Validate edge target references; exit non-zero on broken refs."""
    flow = Flow.load(args.flow)
    issues = _lint_flow(flow)
    if not issues:
        print("OK")
        return 0
    for issue in issues:
        print(issue)
    return 1


def _lint_flow(flow: Any) -> list[str]:
    """Return a list of human-readable issues (empty list = clean flow)."""
    issues: list[str] = []
    node_ids = {n.id for n in flow.nodes}
    for node in flow.nodes:
        for edge in node.edges or []:
            target = edge.target_node_id
            if target and target not in node_ids:
                issues.append(
                    f"node {node.id!r}: edge {edge.id!r} -> unknown target {target!r}"
                )
    for gedge in getattr(flow, "global_edges", []) or []:
        target = gedge.target_node_id
        if target and target not in node_ids:
            issues.append(f"global edge {gedge.id!r} -> unknown target {target!r}")

    # Warn when a required criteria key is not found in any edge input_schema property
    for node in flow.nodes:
        criteria = getattr(node, "completion_criteria", None) or []
        all_edge_schema_keys: set[str] = set()
        for edge in node.edges or []:
            schema = getattr(edge, "input_schema", None)
            if isinstance(schema, dict):
                props = schema.get("properties", {})
                if isinstance(props, dict):
                    all_edge_schema_keys.update(props.keys())
        for criterion in criteria:
            required = getattr(criterion, "required", True)
            if not required:
                continue
            key = getattr(criterion, "key", None)
            if key and all_edge_schema_keys and key not in all_edge_schema_keys:
                issues.append(
                    f"node {node.id!r}: criteria key {key!r} is required but not found "
                    f"in any edge input_schema - the LLM may never extract this value"
                )

    return issues


def _cmd_draw(args: argparse.Namespace) -> int:
    """Emit a Mermaid ``graph TD`` rendering of the flow's edges."""
    flow = Flow.load(args.flow)
    for line in _draw_mermaid(flow):
        print(line)
    return 0


def _draw_mermaid(flow: Any) -> list[str]:
    lines = ["graph TD"]
    for node in flow.nodes:
        for edge in node.edges or []:
            if edge.target_node_id:
                lines.append(f"  {node.id} -->|{edge.id}| {edge.target_node_id}")
    for gedge in getattr(flow, "global_edges", []) or []:
        if gedge.target_node_id:
            lines.append(f"  * -->|{gedge.id}| {gedge.target_node_id}")
    return lines


def _cmd_generate(args: argparse.Namespace) -> int:
    """Generate a flow JSON from a natural-language prompt or description file."""
    load_dotenv()

    # Resolve description text
    from_file = getattr(args, "from_file", None)
    if from_file and getattr(args, "prompt", None):
        print("Warning: --from provided; ignoring positional prompt", file=sys.stderr)
    if from_file:
        p = Path(from_file)
        if not p.exists():
            print(f"Error: description file not found: {from_file}", file=sys.stderr)
            return 1
        prompt = p.read_text()
    else:
        prompt = getattr(args, "prompt", None)

    if not prompt or not prompt.strip():
        print("Error: provide a prompt or --from <file>", file=sys.stderr)
        return 1

    output = getattr(args, "output", "flow.json") or "flow.json"
    llm = getattr(args, "llm", "openai/gpt-4.1-mini") or "openai/gpt-4.1-mini"

    print(f"Generating flow using {llm}...", flush=True)
    flow = asyncio.run(create_dialog_flow(prompt=prompt.strip(), llm=llm))
    flow.save(output)

    node_count = len(flow.nodes)
    edge_count = sum(len(n.edges or []) for n in flow.nodes)
    print(f"Saved: {output}  ({node_count} nodes, {edge_count} edges)")

    # Auto-lint: run checks immediately after generation
    issues = _lint_flow(flow)
    if issues:
        print(f"Lint warnings ({len(issues)}):")
        for issue in issues:
            print(f"  warning: {issue}")
        print(f"Run 'superdialog flow lint {output}' to re-check after edits.")
    else:
        print("Lint: OK")

    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    """Run eval against a flow — audit mode (with --traversal) or synthetic mode."""
    load_dotenv()
    from superdialog.machine.eval import run_eval
    from superdialog.machine.eval.models import AuditReport

    traversal = getattr(args, "traversal", None)
    model = getattr(args, "model", "gpt-4.1-mini") or "gpt-4.1-mini"

    print(f"Running eval: flow={args.flow} model={model}", file=sys.stderr)
    if traversal:
        print(f"Mode: audit  traversal={traversal}", file=sys.stderr)
    else:
        print("Mode: synthetic (generating corpus)", file=sys.stderr)

    report = asyncio.run(
        run_eval(
            flow_path=args.flow,
            traversal_path=traversal,
            model=model,
        )
    )

    if isinstance(report, AuditReport):
        print(report.to_markdown())
        return 0 if report.overall_score >= 0.0 else 1

    # EvalReport
    for score in report.models:
        print(f"\n{'=' * 50}")
        print(f"Model:              {score.model_id}")
        print(f"Edge accuracy:      {score.edge_accuracy:.1%}")
        print(f"Persona completion: {score.persona_completion:.1%}")
        if score.failures:
            print(f"\nFailures ({len(score.failures)}):")
            for f in score.failures:
                print(f"  - {f}")
    return 0


def _compile_flow_to_yaml(flow_path: str) -> str:
    """Compile a ConversationFlow JSON to Playbook YAML string."""
    from superdialog.flow.models import ConversationFlow
    from superdialog.playbook import compile_flow

    flow = ConversationFlow.load(flow_path)
    pb = compile_flow(flow)
    return yaml.safe_dump(
        pb.model_dump(mode="json", exclude_defaults=True),
        sort_keys=False,
        allow_unicode=True,
        width=80,
    )


def _cmd_playbook_compile(args: argparse.Namespace) -> int:
    """Compile a flow JSON to Playbook YAML; emit to stdout or --output file."""
    text = _compile_flow_to_yaml(args.flow)
    output = getattr(args, "output", None)
    if output:
        Path(output).write_text(text)
        print(f"Written: {output}", file=sys.stderr)
    else:
        sys.stdout.write(text)
    return 0


async def _playbook_chat_loop(playbook_path: str, model: str) -> None:
    import anyio

    from superdialog.llm.litellm_provider import LitellmProvider
    from superdialog.playbook import Playbook, httpx_http
    from superdialog.playbook.runtime import PlaybookRuntime
    from superdialog.playbook.talker import Talker

    class _DirectorLLM:
        def __init__(self, provider: LitellmProvider) -> None:
            self._p = provider

        async def complete(self, messages: list[dict[str, str]], **kw: Any) -> str:
            return (await self._p.complete(list(messages))).text

    class _TalkerLLM:
        def __init__(self, provider: LitellmProvider) -> None:
            self._p = provider

        async def stream(
            self, messages: list[dict[str, str]], **kw: Any
        ) -> AsyncIterator[str]:
            async for chunk in self._p.stream(list(messages)):
                if chunk.text:
                    yield chunk.text

    provider = LitellmProvider(model)
    playbook = Playbook.load(playbook_path)
    runtime = PlaybookRuntime(
        playbook, director_llm=_DirectorLLM(provider), http=httpx_http
    )
    talker = Talker(playbook, llm=_TalkerLLM(provider))

    async def _speak() -> None:
        print("agent> ", end="", flush=True)
        async for chunk in talker.speak(runtime.state):
            if chunk.text:
                print(chunk.text, end="", flush=True)
        print()

    print(f"Playbook demo on {model} — type 'quit' to exit.\n")
    pass_through = await runtime.start()
    for line in pass_through:
        print(f"agent> {line}")
    await _speak()

    while True:
        try:
            text = await anyio.to_thread.run_sync(input, "you> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if text.strip().lower() in {"quit", "exit"}:
            break
        if not text.strip():
            continue
        cp_before = runtime.state.checkpoint_id
        pass_through = await runtime.on_user_text(text)
        cp_after = runtime.state.checkpoint_id
        print(
            f"[DEBUG] {cp_before} → {cp_after}  ended={runtime.state.ended}",
            file=sys.stderr,
        )
        for line in pass_through:
            print(f"agent> {line}")
        await _speak()
        if runtime.state.ended:
            print(f"[session ended — outcome: {runtime.state.outcome}]")
            break


def _cmd_playbook_chat(args: argparse.Namespace) -> int:
    """Run the playbook REPL against an existing YAML file."""
    load_dotenv()
    playbook_path = args.playbook
    if not Path(playbook_path).exists():
        print(f"Playbook not found: {playbook_path}", file=sys.stderr)
        return 1
    model = getattr(args, "model", "openai/gpt-4.1-mini") or "openai/gpt-4.1-mini"
    import anyio

    anyio.run(_playbook_chat_loop, playbook_path, model)
    return 0


def _cmd_playbook_run(args: argparse.Namespace) -> int:
    """Compile a flow JSON to a playbook YAML, then start the REPL."""
    load_dotenv()
    import tempfile

    model = getattr(args, "model", "openai/gpt-4.1-mini") or "openai/gpt-4.1-mini"

    # If arg is already a .yaml/.yml file, skip compile
    src = args.flow
    if src.endswith((".yaml", ".yml")):
        playbook_path = src
    else:
        text = _compile_flow_to_yaml(src)
        output = getattr(args, "output", None)
        if output:
            Path(output).write_text(text)
            playbook_path = output
            print(f"Compiled → {output}", file=sys.stderr)
        else:
            # Write to a temp file so the REPL can load it
            tmp = tempfile.NamedTemporaryFile(
                suffix=".yaml", delete=False, mode="w", encoding="utf-8"
            )
            tmp.write(text)
            tmp.close()
            playbook_path = tmp.name
            print(f"Compiled → {playbook_path} (temp)", file=sys.stderr)

    import anyio

    anyio.run(_playbook_chat_loop, playbook_path, model)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="superdialog")
    sub = parser.add_subparsers(dest="cmd", required=True)

    chat = sub.add_parser("chat", help="Interactive REPL against a flow or playbook")
    chat.add_argument(
        "--flow",
        default=None,
        help="Path to a playbook or flow file (default: ./playbook.yaml, "
        "then ./flow.json); any format is auto-detected and runs on the "
        "Playbook engine",
    )
    chat.add_argument(
        "--playbook",
        default=None,
        help="Path to a playbook (YAML/JSON); forces the playbook REPL",
    )
    chat.add_argument(
        "--simple",
        default=None,
        help="Path to a simple-format playbook (YAML/JSON); compiles then runs",
    )
    chat.add_argument("--llm", default="openai/gpt-4.1-mini")
    chat.add_argument(
        "--mode",
        default="playbook",
        choices=["playbook", "flow"],
        help="Engine: 'playbook' (default) runs everything on the Playbook "
        "engine, compiling flow JSON automatically; 'flow' runs flow JSON "
        "on the legacy DialogMachine",
    )
    chat.add_argument(
        "--adapter",
        default="toolcall",
        choices=["llm", "toolcall"],
        help="DialogMachine adapter (--mode flow only): 'toolcall' (default, "
        "1 LLM call/turn, mirrors production) or 'llm' (2 LLM calls/turn)",
    )
    chat.set_defaults(fn=_cmd_chat)

    gen = sub.add_parser(
        "generate",
        help="Generate a playbook (simple format) from a natural-language "
        "prompt — the default creation path (legacy: flow generate)",
    )
    gen.add_argument(
        "prompt",
        nargs="?",
        default=None,
        help="Inline agent description (omit if using --from)",
    )
    gen.add_argument(
        "--from",
        dest="from_file",
        metavar="FILE",
        help="Path to a description file (alternative to positional prompt)",
    )
    gen.add_argument(
        "--output",
        default="playbook.yaml",
        help="Output path (default: playbook.yaml)",
    )
    gen.add_argument("--llm", default="openai/gpt-4.1-mini")
    gen.set_defaults(fn=_cmd_generate_playbook)

    flow = sub.add_parser("flow", help="Inspect / manipulate flow files")
    flow_sub = flow.add_subparsers(dest="subcmd", required=True)

    lint = flow_sub.add_parser("lint", help="Validate edge target references")
    lint.add_argument("flow")
    lint.set_defaults(fn=_cmd_lint)

    draw = flow_sub.add_parser("draw", help="Print a Mermaid graph of the flow")
    draw.add_argument("flow")
    draw.set_defaults(fn=_cmd_draw)

    generate = flow_sub.add_parser(
        "generate", help="Generate a flow JSON from a natural-language prompt"
    )
    generate.add_argument(
        "prompt",
        nargs="?",
        default=None,
        help="Inline description string (omit if using --from)",
    )
    generate.add_argument(
        "--from",
        dest="from_file",
        metavar="FILE",
        help="Path to description file (alternative to positional prompt)",
    )
    generate.add_argument(
        "--output",
        default="flow.json",
        help="Output path for flow JSON (default: flow.json)",
    )
    generate.add_argument("--llm", default="openai/gpt-4o-mini")
    generate.set_defaults(fn=_cmd_generate)

    opt = sub.add_parser(
        "optimize", help="Reflectively improve a playbook's prose via self-play"
    )
    opt.add_argument(
        "--playbook",
        required=True,
        help="Path to a playbook (full or simple format)",
    )
    opt.add_argument("--rounds", type=int, default=3)
    opt.add_argument(
        "--n", type=int, default=1, help="Eval sessions per persona per side"
    )
    opt.add_argument(
        "--personas", default=None, help="Path to a PersonaSpec list (YAML/JSON)"
    )
    opt.add_argument("--llm", default="openai/gpt-4o-mini")
    opt.add_argument(
        "--candidate-llm",
        default=None,
        help="Override the reflecting LLM (default: --llm)",
    )
    opt.add_argument(
        "--user-llm",
        default=None,
        help="Override the caller-simulator LLM (default: --llm)",
    )
    opt.add_argument(
        "--out",
        default=None,
        help="Output path (default: improved.<name>, same format)",
    )
    opt.set_defaults(fn=_cmd_optimize)

    # -- playbook subcommand group ------------------------------------------
    pb = sub.add_parser("playbook", help="Compile and run Playbook engine sessions")
    pb_sub = pb.add_subparsers(dest="pb_subcmd", required=True)

    pb_compile = pb_sub.add_parser(
        "compile", help="Compile a flow JSON to Playbook YAML"
    )
    pb_compile.add_argument("flow", help="Path to flow JSON")
    pb_compile.add_argument(
        "--output", "-o", default=None, help="Output YAML file (default: stdout)"
    )
    pb_compile.set_defaults(fn=_cmd_playbook_compile)

    pb_chat = pb_sub.add_parser(
        "chat", help="Interactive REPL against an existing Playbook YAML"
    )
    pb_chat.add_argument("--playbook", required=True, help="Path to playbook YAML")
    pb_chat.add_argument(
        "--model", default="openai/gpt-4.1-mini", help="LiteLLM model string"
    )
    pb_chat.set_defaults(fn=_cmd_playbook_chat)

    pb_run = pb_sub.add_parser(
        "run", help="Compile a flow JSON and immediately start a Playbook REPL"
    )
    pb_run.add_argument(
        "flow", help="Path to flow JSON (or existing .yaml to skip compile)"
    )
    pb_run.add_argument(
        "--output",
        "-o",
        default=None,
        help="Save compiled YAML here (default: temp file)",
    )
    pb_run.add_argument(
        "--model", default="openai/gpt-4.1-mini", help="LiteLLM model string"
    )
    pb_run.set_defaults(fn=_cmd_playbook_run)

    # -- eval subcommand ----------------------------------------------------
    eval_cmd = sub.add_parser(
        "eval",
        help="Eval a flow: audit real session (--traversal) or run synthetic corpus",
    )
    eval_cmd.add_argument("--flow", required=True, help="Path to flow JSON")
    eval_cmd.add_argument(
        "--traversal",
        default=None,
        help="Path to traversal JSON (audit mode). Omit for synthetic eval.",
    )
    eval_cmd.add_argument(
        "--model",
        default="gpt-4.1-mini",
        help="OpenAI model for eval LLM (default: gpt-4.1-mini)",
    )
    eval_cmd.set_defaults(fn=_cmd_eval)

    bench = sub.add_parser(
        "benchmark",
        help="RAGAS + deterministic benchmark over a dataset (raw vs SuperDialog)",
    )
    bench.add_argument("--data", default="universal",
                       help="dataset: short name (e.g. universal) or path to a .jsonl")
    bench.add_argument("--flow", default=None,
                       help="playbook YAML to run (default: from dataset's playbook field)")
    bench.add_argument("--prompt", default=None,
                       help="raw-LLM system-prompt .txt (needed to run the raw baseline)")
    bench.add_argument("--models", default=",".join(_BENCH_ALL),
                       help="comma-separated models (aliases or litellm ids). "
                            f"default: {','.join(_BENCH_ALL)}")
    bench.add_argument("--sd-only", action="store_true", help="only with-SuperDialog")
    bench.add_argument("--raw-only", action="store_true", help="only raw LLM")
    bench.add_argument("--no-ragas", action="store_true",
                       help="deterministic metrics only (no judge, fast/free)")
    bench.add_argument("--out", default=None, help="write the big-table report to this path")
    bench.set_defaults(fn=_cmd_benchmark)

    return parser


_BENCH_ALIASES = {
    "gpt-4o-mini": "gpt-4o-mini",
    "gpt-4.1-mini": "gpt-4.1-mini",
    "claude-haiku": "anthropic/claude-haiku-4-5-20251001",
}
_BENCH_ALL = ["gpt-4o-mini", "gpt-4.1-mini", "claude-haiku"]


def _cmd_benchmark(args: argparse.Namespace) -> int:
    """Run the RAGAS + deterministic benchmark over a dataset; print one big table.

    Dataset-based (unlike ``eval``'s session audit): replays each dataset's user
    turns at the model(s), scores raw and/or with-SuperDialog against the
    ground truth, single fixed judge. Defers to ``superdialog.benchmark``.
    """
    from ..benchmark.loader import load_dataset, load_named
    from ..benchmark.panel import render_big_table
    from ..benchmark.ragas_scorer import DEFAULT_JUDGES
    from ..benchmark.runner import run_raw_mode, run_sd_mode

    samples = (
        load_dataset(args.data)
        if args.data.endswith(".jsonl")
        else load_named(args.data)
    )
    playbook = args.flow or next((s.playbook for s in samples if s.playbook), None)
    if not args.raw_only and not playbook:
        print("error: no playbook (dataset has none; pass --flow)", file=sys.stderr)
        return 2

    run_ragas = not args.no_ragas
    sd_only = args.sd_only or not args.prompt  # raw needs a system-prompt file
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    reports = []
    for m in models:
        ms = _BENCH_ALIASES.get(m, m)
        print(f"\n# model: {ms}", file=sys.stderr)
        if not sd_only:
            reports.append(
                run_raw_mode(samples, ms, args.prompt,
                             label=f"Raw LLM ({ms})", run_ragas=run_ragas)
            )
        if not args.raw_only:
            reports.append(
                run_sd_mode(samples, ms, playbook,
                            label=f"With SuperDialog ({ms})", run_ragas=run_ragas)
            )

    judge = None if args.no_ragas else DEFAULT_JUDGES[0]
    pb_name = Path(playbook).name if playbook else None
    table = render_big_table(reports, dataset=args.data, playbook=pb_name, judge=judge)
    print(table)

    if args.out:
        out = Path(args.out)
        header = (
            f"# SuperDialog Benchmark - {args.data}\n\n"
            f"- judge: {judge or 'n/a (RAGAS off)'}\n\n## Results\n\n"
        )
        out.write_text(header + table + "\n", encoding="utf-8")
        print(f"\n# report written: {out}", file=sys.stderr)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Returns the process exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    rc = args.fn(args)
    return int(rc or 0)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
