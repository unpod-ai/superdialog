"""Observer protocol, NullObserver, LangfuseObserver, TracingProvider, build_observer."""

from __future__ import annotations

import logging
import os
from typing import Any, AsyncIterator, Protocol, runtime_checkable

from ..llm.provider import CompletionResult, LLMProvider, StreamChunk

logger = logging.getLogger(__name__)


@runtime_checkable
class Observer(Protocol):
    """Backend-agnostic sink for session / LLM observability."""

    def on_session_start(self, session_id: str, metadata: dict[str, Any]) -> str:
        """Create or open a session trace. Returns trace_id."""
        ...

    def on_generation_start(
        self, trace_id: str, name: str, input_messages: list[dict[str, Any]]
    ) -> str:
        """Record the start of an LLM generation. Returns observation_id."""
        ...

    def on_generation_end(
        self,
        observation_id: str,
        output: str,
        tool_calls: list[dict[str, Any]],
        metadata: dict[str, Any],
    ) -> None: ...

    def on_tool_call(
        self, trace_id: str, name: str, args: dict[str, Any], result: Any
    ) -> None: ...

    def on_flow_node(
        self,
        trace_id: str,
        node_id: str,
        slots: dict[str, Any],
        *,
        prev_node: str | None = None,
    ) -> None: ...

    def on_voice_turn(
        self,
        trace_id: str,
        metrics: dict[str, Any],
    ) -> None: ...

    def on_session_end(self, trace_id: str, output: str) -> None: ...


class NullObserver:
    """No-op observer — the safe default, zero external dependencies."""

    def on_session_start(self, session_id: str, metadata: dict[str, Any]) -> str:
        return session_id

    def on_generation_start(
        self, trace_id: str, name: str, input_messages: list[dict[str, Any]], **_: Any
    ) -> str:
        return ""

    def on_generation_end(
        self,
        observation_id: str,
        output: str,
        tool_calls: list[dict[str, Any]],
        metadata: dict[str, Any],
    ) -> None:
        return None

    def on_tool_call(
        self, trace_id: str, name: str, args: dict[str, Any], result: Any
    ) -> None:
        return None

    def on_flow_node(
        self,
        trace_id: str,
        node_id: str,
        slots: dict[str, Any],
        *,
        prev_node: str | None = None,
    ) -> None:
        return None

    def on_voice_turn(
        self,
        trace_id: str,
        metrics: dict[str, Any],
    ) -> None:
        return None

    def on_session_end(self, trace_id: str, output: str) -> None:
        return None

    def on_error(self, trace_id: str, message: str, metadata: dict[str, Any]) -> None:
        return None

    def flush(self) -> None:
        return None


class LangfuseObserver:
    """Langfuse backend — wraps an injected Langfuse client. Best-effort, never raises."""

    def __init__(self, client: Any) -> None:
        self._client = client
        self._traces: dict[str, Any] = {}
        self._pending: dict[str, tuple[Any, str]] = {}  # obs_id -> (gen, trace_id)

    def on_session_start(self, session_id: str, metadata: dict[str, Any]) -> str:
        try:
            playbook_id = metadata.get("playbook")
            trace_name = (
                f"voice_session:{playbook_id}"
                if playbook_id
                else f"voice_session:{metadata.get('agent_id') or session_id}"
            )
            trace = self._client.trace(
                id=session_id,
                name=trace_name,
                metadata=metadata,
            )
            self._traces[trace.id] = trace
            return trace.id
        except Exception as exc:
            logger.debug("langfuse on_session_start skipped: %s", exc)
            return session_id

    def on_generation_start(
        self,
        trace_id: str,
        name: str,
        input_messages: list[dict[str, Any]],
        *,
        model: str | None = None,
        model_parameters: dict[str, Any] | None = None,
    ) -> str:
        try:
            gen_kwargs: dict[str, Any] = {
                "trace_id": trace_id,
                "name": name,
                "input": input_messages,
            }
            if model is not None:
                gen_kwargs["model"] = model
            if model_parameters is not None:
                gen_kwargs["model_parameters"] = model_parameters
            gen = self._client.generation(**gen_kwargs)
            self._pending[gen.id] = (gen, trace_id)
            print(f"[LANGFUSE-DIALOG] generation queued name={name!r} gen_id={gen.id!r} trace_id={trace_id!r}", flush=True)
            return gen.id
        except Exception as exc:
            print(f"[LANGFUSE-DIALOG] generation FAILED name={name!r} exc={type(exc).__name__}: {exc}", flush=True)
            logger.debug("langfuse on_generation_start skipped: %s", exc)
            return ""

    def on_generation_end(
        self,
        observation_id: str,
        output: str,
        tool_calls: list[dict[str, Any]],
        metadata: dict[str, Any],
    ) -> None:
        entry = self._pending.pop(observation_id, None)
        if entry is None:
            return
        gen, trace_id = entry
        try:
            gen.end(
                output=output,
                usage={
                    "input": metadata.get("prompt_tokens") or metadata.get("input_tokens", 0),
                    "output": metadata.get("completion_tokens") or metadata.get("output_tokens", 0),
                },
                metadata=metadata,
            )
        except Exception as exc:
            logger.debug("langfuse on_generation_end skipped: %s", exc)
        for tc in tool_calls or []:
            name = tc.get("name") or tc.get("function", {}).get("name") or "tool"
            self.on_tool_call(trace_id, name, tc, tc.get("result"))

    def on_tool_call(
        self, trace_id: str, name: str, args: dict[str, Any], result: Any
    ) -> None:
        try:
            span = self._client.span(
                trace_id=trace_id,
                name=f"tool:{name}",
                input=args,
            )
            span.end(output=str(result))
        except Exception as exc:
            logger.debug("langfuse on_tool_call skipped: %s", exc)

    def on_flow_node(
        self,
        trace_id: str,
        node_id: str,
        slots: dict[str, Any],
        *,
        prev_node: str | None = None,
    ) -> None:
        # Meta-events (speech pipeline) pass prev_node=None and carry no graph
        # position.  Skip flow_node + node_transition for them — they are already
        # logged as user_turn / agent_turn / voice:opening_turn / voice:interruption
        # spans by VoiceObserver.  Only real dialog graph nodes emit traversal spans.
        _META = {"user_turn", "agent_turn", "opening_turn", "interruption"}
        if node_id in _META:
            return

        try:
            span = self._client.span(
                trace_id=trace_id,
                name="flow_node",
                input={
                    "node_id": node_id,
                    "prev_node": prev_node,
                    "transition": f"{prev_node} → {node_id}" if prev_node else f"START → {node_id}",
                    "slots": slots,
                },
                metadata={
                    "node_id": node_id,
                    "prev_node": prev_node,
                    "is_first_node": prev_node is None,
                },
            )
            span.end(output={"arrived_at": node_id, "from": prev_node, "slots_count": len(slots)})
        except Exception as exc:
            logger.debug("langfuse on_flow_node skipped: %s", exc)

        if prev_node is not None:
            try:
                span = self._client.span(
                    trace_id=trace_id,
                    name="node_transition",
                    input={
                        "from_node": prev_node,
                        "to_node": node_id,
                        "transition": f"{prev_node} → {node_id}",
                        "slots_at_transition": slots,
                    },
                    metadata={
                        "from_node": prev_node,
                        "to_node": node_id,
                        "layer": "superdialog",
                    },
                )
                span.end(output={"from_node": prev_node, "to_node": node_id})
            except Exception as exc:
                logger.debug("langfuse node_transition skipped: %s", exc)

    def on_voice_turn(
        self,
        trace_id: str,
        metrics: dict[str, Any],
    ) -> None:
        try:
            span = self._client.span(
                trace_id=trace_id,
                name="voice_turn",
                input={},
                metadata=metrics,
            )
            span.end()
        except Exception as exc:
            logger.debug("langfuse on_voice_turn skipped: %s", exc)

    def on_session_end(self, trace_id: str, output: str) -> None:
        try:
            trace = self._traces.pop(trace_id, None)
            if trace is not None:
                trace.update(output=output)
        except Exception as exc:
            logger.debug("langfuse session_end update skipped: %s", exc)
        self.flush()

    def on_error(self, trace_id: str, message: str, metadata: dict[str, Any]) -> None:
        try:
            span = self._client.span(
                trace_id=trace_id,
                name="error",
                level="ERROR",
                input={"error": message, **{k: v for k, v in (metadata or {}).items()}},
            )
            span.end(output={"error": message})
        except Exception as exc:
            logger.debug("langfuse on_error skipped: %s", exc)

    def flush(self) -> None:
        try:
            self._client.flush()
            print(f"[LANGFUSE-DIALOG] flush ok pending={len(self._pending)}", flush=True)
        except Exception as exc:
            print(f"[LANGFUSE-DIALOG] flush FAILED exc={type(exc).__name__}: {exc}", flush=True)
            logger.debug("langfuse flush skipped: %s", exc)


# Alias: superdialog's full-featured observer — exported so tracing.py in
# supervoice can import it as SuperdialogObserver without a separate class.
SuperdialogObserver = LangfuseObserver


class TracingProvider:
    """Wraps any LLMProvider and records generations via an Observer."""

    def __init__(
        self,
        inner: LLMProvider,
        observer: Observer,
        trace_id: str,
        model_uri: str | None = None,
        role: str = "llm",
    ) -> None:
        self._inner = inner
        self._observer = observer
        self._trace_id = trace_id
        # Optional generation tagging. ``role`` (e.g. "talker"/"director")
        # prefixes the generation name so multiple LLM roles in one turn are
        # distinguishable; it defaults to "llm". ``model_uri`` is recorded as
        # ``metadata['model']`` on generation end so the trace shows which model
        # produced the output. Both keep the original 3-arg construction valid.
        self._model_uri = model_uri
        self._role = role

    def _gen_name(self, base: str) -> str:
        """Prefix the generation name with the role when one is set."""
        return f"{self._role}:{base}" if self._role else base

    def _tag_model(self, metadata: dict[str, Any]) -> dict[str, Any]:
        """Record the model uri in the end metadata without clobbering a model
        the provider already reported."""
        if self._model_uri and "model" not in metadata:
            return {**metadata, "model": self._model_uri}
        return metadata

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **opts: Any,
    ) -> CompletionResult:
        obs_id = self._observer.on_generation_start(
            self._trace_id, self._gen_name("complete"), messages
        )
        text_out: str = ""
        tool_calls_out: list[dict[str, Any]] = []
        metadata_out: dict[str, Any] = {}
        try:
            result = await self._inner.complete(messages, tools, **opts)
            text_out = result.text
            tool_calls_out = result.tool_calls
            metadata_out = result.metadata
            return result
        finally:
            self._observer.on_generation_end(
                obs_id, text_out, tool_calls_out, self._tag_model(metadata_out)
            )

    async def stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **opts: Any,
    ) -> AsyncIterator[StreamChunk]:
        import time
        obs_id = self._observer.on_generation_start(
            self._trace_id, self._gen_name("stream"), messages
        )
        buffer: list[str] = []
        final_metadata: dict[str, Any] = {}
        t0 = time.perf_counter()
        try:
            async for chunk in self._inner.stream(messages, tools, **opts):
                if chunk.text:
                    buffer.append(chunk.text)
                if chunk.usage:
                    final_metadata.update(chunk.usage)
                yield chunk
        finally:
            latency_ms = (time.perf_counter() - t0) * 1000
            final_metadata.setdefault("latency_ms", latency_ms)
            self._observer.on_generation_end(
                obs_id, "".join(buffer), [], self._tag_model(final_metadata)
            )


def build_observer(
    public_key: str | None = None,
    secret_key: str | None = None,
    host: str | None = None,
    *,
    enable_tracing: bool | None = None,
) -> "LangfuseObserver | NullObserver":
    """Return a LangfuseObserver if tracing is enabled, else NullObserver.

    Enable/disable priority:
      1. ``enable_tracing`` kwarg (explicit caller override)
      2. ``SUPERDIALOG_TRACING`` env var: ``"1"``/``"true"``/``"on"`` → enable,
         ``"0"``/``"false"``/``"off"`` → disable, unset → auto-detect from keys
      3. Auto-detect: enabled when keys are present

    Key resolution order (first non-empty wins):
      explicit kwargs → LANGFUSE_DIALOG_PUBLIC_KEY / LANGFUSE_DIALOG_SECRET_KEY
      → LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY
    """
    # ── enable/disable gate ───────────────────────────────────────────────
    if enable_tracing is None:
        _env = os.environ.get("SUPERDIALOG_TRACING", "").strip().lower()
        if _env in ("0", "false", "off", "no"):
            logger.info("superdialog tracing disabled via SUPERDIALOG_TRACING=%s", _env)
            return NullObserver()
        if _env in ("1", "true", "on", "yes"):
            enable_tracing = True
        # else: auto-detect below

    # ── key resolution ────────────────────────────────────────────────────
    pk = (
        public_key
        or os.environ.get("LANGFUSE_DIALOG_PUBLIC_KEY")
        or os.environ.get("LANGFUSE_PUBLIC_KEY", "")
    )
    sk = (
        secret_key
        or os.environ.get("LANGFUSE_DIALOG_SECRET_KEY")
        or os.environ.get("LANGFUSE_SECRET_KEY", "")
    )
    if not pk or not sk:
        if enable_tracing:
            logger.warning(
                "superdialog tracing: SUPERDIALOG_TRACING=true but no Langfuse keys found "
                "(set LANGFUSE_DIALOG_PUBLIC_KEY + LANGFUSE_DIALOG_SECRET_KEY); "
                "using NullObserver"
            )
        else:
            logger.debug("superdialog tracing: no keys — using NullObserver")
        return NullObserver()

    lf_host = host or os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com")
    try:
        from langfuse import Langfuse

        client = Langfuse(public_key=pk, secret_key=sk, host=lf_host)
        logger.info(
            "superdialog tracing: LangfuseObserver active host=%s pk=%.8s…",
            lf_host, pk,
        )
        return LangfuseObserver(client)
    except Exception as exc:
        logger.warning("superdialog tracing: langfuse unavailable (%s); using NullObserver", exc)
        return NullObserver()


__all__ = [
    "LangfuseObserver",
    "NullObserver",
    "Observer",
    "TracingProvider",
    "build_observer",
]
