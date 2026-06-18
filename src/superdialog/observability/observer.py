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
        self, trace_id: str, node_id: str, slots: dict[str, Any]
    ) -> None: ...

    def on_voice_turn(
        self,
        trace_id: str,
        ttfa_ms: float,
        asr_final_ms: float,
        tts_ttfb_ms: float,
    ) -> None: ...

    def on_session_end(self, trace_id: str, output: str) -> None: ...


class NullObserver:
    """No-op observer — the safe default, zero external dependencies."""

    def on_session_start(self, session_id: str, metadata: dict[str, Any]) -> str:
        return session_id

    def on_generation_start(
        self, trace_id: str, name: str, input_messages: list[dict[str, Any]]
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
        self, trace_id: str, node_id: str, slots: dict[str, Any]
    ) -> None:
        return None

    def on_voice_turn(
        self,
        trace_id: str,
        ttfa_ms: float,
        asr_final_ms: float,
        tts_ttfb_ms: float,
    ) -> None:
        return None

    def on_session_end(self, trace_id: str, output: str) -> None:
        return None


class LangfuseObserver:
    """Langfuse backend — wraps an injected Langfuse client. Best-effort, never raises."""

    def __init__(self, client: Any) -> None:
        self._client = client
        self._traces: dict[str, Any] = {}
        self._pending: dict[str, Any] = {}

    def on_session_start(self, session_id: str, metadata: dict[str, Any]) -> str:
        try:
            trace = self._client.trace(
                id=session_id,
                name="dialog-session",
                metadata=metadata,
            )
            self._traces[trace.id] = trace
            return trace.id
        except Exception as exc:
            logger.debug("langfuse on_session_start skipped: %s", exc)
            return session_id

    def on_generation_start(
        self, trace_id: str, name: str, input_messages: list[dict[str, Any]]
    ) -> str:
        try:
            gen = self._client.generation(
                trace_id=trace_id,
                name=name,
                input=input_messages,
            )
            self._pending[gen.id] = gen
            return gen.id
        except Exception as exc:
            logger.debug("langfuse on_generation_start skipped: %s", exc)
            return ""

    def on_generation_end(
        self,
        observation_id: str,
        output: str,
        tool_calls: list[dict[str, Any]],
        metadata: dict[str, Any],
    ) -> None:
        gen = self._pending.pop(observation_id, None)
        if gen is None:
            return
        try:
            gen.end(
                output=output,
                usage={
                    "input": metadata.get("prompt_tokens", 0),
                    "output": metadata.get("completion_tokens", 0),
                },
                metadata=metadata,
            )
        except Exception as exc:
            logger.debug("langfuse on_generation_end skipped: %s", exc)

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
        self, trace_id: str, node_id: str, slots: dict[str, Any]
    ) -> None:
        try:
            span = self._client.span(
                trace_id=trace_id,
                name="flow_node",
                input={"node_id": node_id, "slots": slots},
            )
            span.end()
        except Exception as exc:
            logger.debug("langfuse on_flow_node skipped: %s", exc)

    def on_voice_turn(
        self,
        trace_id: str,
        ttfa_ms: float,
        asr_final_ms: float,
        tts_ttfb_ms: float,
    ) -> None:
        try:
            span = self._client.span(
                trace_id=trace_id,
                name="voice_turn",
                input={},
                metadata={
                    "ttfa_ms": ttfa_ms,
                    "asr_final_ms": asr_final_ms,
                    "tts_ttfb_ms": tts_ttfb_ms,
                },
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
        try:
            self._client.flush()
        except Exception as exc:
            logger.debug("langfuse flush skipped: %s", exc)


class TracingProvider:
    """Wraps any LLMProvider and records generations via an Observer."""

    def __init__(
        self, inner: LLMProvider, observer: Observer, trace_id: str
    ) -> None:
        self._inner = inner
        self._observer = observer
        self._trace_id = trace_id

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **opts: Any,
    ) -> CompletionResult:
        obs_id = self._observer.on_generation_start(
            self._trace_id, "complete", messages
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
                obs_id, text_out, tool_calls_out, metadata_out
            )

    async def stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **opts: Any,
    ) -> AsyncIterator[StreamChunk]:
        import time
        obs_id = self._observer.on_generation_start(
            self._trace_id, "stream", messages
        )
        buffer: list[str] = []
        final_metadata: dict[str, Any] = {}
        t0 = time.perf_counter()
        try:
            async for chunk in self._inner.stream(messages, tools, **opts):
                if chunk.text:
                    buffer.append(chunk.text)
                # Capture metadata from chunk if available (some providers attach usage on final chunk)
                if hasattr(chunk, "metadata") and chunk.metadata:
                    final_metadata.update(chunk.metadata)
                yield chunk
        finally:
            latency_ms = (time.perf_counter() - t0) * 1000
            final_metadata.setdefault("latency_ms", latency_ms)
            self._observer.on_generation_end(
                obs_id, "".join(buffer), [], final_metadata
            )


def build_observer(
    public_key: str | None = None,
    secret_key: str | None = None,
    host: str | None = None,
) -> "LangfuseObserver | NullObserver":
    """Return a LangfuseObserver if keys are available, else NullObserver.

    Reads LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_HOST from env
    when not provided explicitly. Lazy-imports langfuse so the package is
    optional.
    """
    pk = public_key or os.environ.get("LANGFUSE_PUBLIC_KEY", "")
    sk = secret_key or os.environ.get("LANGFUSE_SECRET_KEY", "")
    if not pk or not sk:
        return NullObserver()
    lf_host = host or os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com")
    try:
        from langfuse import Langfuse

        client = Langfuse(public_key=pk, secret_key=sk, host=lf_host)
        return LangfuseObserver(client)
    except Exception as exc:
        logger.warning("langfuse unavailable (%s); using NullObserver", exc)
        return NullObserver()


__all__ = [
    "LangfuseObserver",
    "NullObserver",
    "Observer",
    "TracingProvider",
    "build_observer",
]
