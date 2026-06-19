"""Observer protocol, NullObserver, LangfuseObserver, TracingProvider, build_observer."""

from __future__ import annotations

import os
import time
from typing import Any, AsyncIterator, Protocol, runtime_checkable

from loguru import logger

from ..llm.provider import CompletionResult, LLMProvider, StreamChunk


@runtime_checkable
class Observer(Protocol):
    """Backend-agnostic sink for session / LLM observability."""

    def on_session_start(self, session_id: str, metadata: dict[str, Any]) -> str:
        """Create or open a session trace. Returns trace_id."""
        ...

    def on_generation_start(
        self,
        trace_id: str,
        name: str,
        input_messages: list[dict[str, Any]],
        *,
        model: str | None = None,
        model_parameters: dict[str, Any] | None = None,
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
        metrics: dict[str, Any],
    ) -> None: ...

    def on_session_end(self, trace_id: str, output: str) -> None: ...

    def on_error(
        self, trace_id: str, message: str, metadata: dict[str, Any]
    ) -> None: ...

    def flush(self) -> None: ...


class NullObserver:
    """No-op observer — the safe default, zero external dependencies."""

    def on_session_start(self, session_id: str, metadata: dict[str, Any]) -> str:
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
        metrics: dict[str, Any],
    ) -> None:
        return None

    def on_session_end(self, trace_id: str, output: str) -> None:
        return None

    def on_error(
        self, trace_id: str, message: str, metadata: dict[str, Any]
    ) -> None:
        return None

    def flush(self) -> None:
        return None


def _ctx(session_meta: dict[str, Any]) -> dict[str, Any]:
    """Shared session context block added to every span's metadata."""
    return {
        "call_id": session_meta.get("call_id"),
        "agent_name": session_meta.get("agent"),
        "agent_id": session_meta.get("agent_id"),
        "mode": session_meta.get("mode"),
        "playbook_id": session_meta.get("playbook"),
        "flow_id": session_meta.get("flow"),
        "llm_model": session_meta.get("llm"),
        "voice_profile_id": session_meta.get("voice_profile"),
        "source": session_meta.get("source") or "playground",
    }


class LangfuseObserver:
    """Langfuse backend — wraps an injected Langfuse client. Best-effort, never raises."""

    def __init__(self, client: Any) -> None:
        self._client = client
        self._traces: dict[str, Any] = {}
        # obs_id -> (gen_object, trace_id, start_perf, gen_name)
        self._pending: dict[str, tuple[Any, str, float, str]] = {}
        # trace_id -> ordered list of {"role", "text", "ts"}
        self._conversations: dict[str, list[dict[str, Any]]] = {}
        # trace_id -> session metadata dict
        self._session_meta: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------------ #
    # Session lifecycle                                                    #
    # ------------------------------------------------------------------ #

    def on_session_start(self, session_id: str, metadata: dict[str, Any]) -> str:
        print(f"[LANGFUSE-DEBUG] on_session_start session_id={session_id}", flush=True)
        try:
            agent_name = metadata.get("agent") or "unknown-agent"
            mode = metadata.get("mode") or "unknown"

            tags = ["playground", f"mode:{mode}", f"agent:{agent_name}"]
            if metadata.get("playbook"):
                tags.append(f"playbook:{metadata['playbook']}")
            if metadata.get("llm"):
                tags.append(f"llm:{metadata['llm']}")
            if metadata.get("voice_profile"):
                tags.append(f"voice_profile:{metadata['voice_profile']}")

            trace = self._client.trace(
                id=session_id,
                name=f"dialog-session:{agent_name}",
                # Langfuse user_id / session_id for filtering in the UI
                user_id=metadata.get("call_id") or session_id,
                session_id=session_id,
                tags=tags,
                # input = who/what started this session
                input={
                    "call_id": metadata.get("call_id") or session_id,
                    "agent_name": agent_name,
                    "agent_id": metadata.get("agent_id"),
                    "mode": mode,
                    "playbook_id": metadata.get("playbook"),
                    "flow_id": metadata.get("flow"),
                    "llm_model": metadata.get("llm"),
                    "voice_profile_id": metadata.get("voice_profile"),
                    "source": metadata.get("source") or "playground",
                    "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "caller_data": metadata.get("caller_data") or {},
                },
                metadata={
                    "description": (
                        f"Voice dialog session for agent '{agent_name}' "
                        f"in '{mode}' mode"
                        + (f", playbook '{metadata.get('playbook')}'" if metadata.get("playbook") else "")
                        + f". LLM: {metadata.get('llm')}. Voice profile: {metadata.get('voice_profile')}."
                    ),
                },
            )
            self._traces[trace.id] = trace
            self._conversations[trace.id] = []
            self._session_meta[trace.id] = metadata
            print(f"[LANGFUSE-DEBUG] trace created id={trace.id} agent={agent_name}", flush=True)
            logger.info(
                "langfuse trace created id={} agent={} mode={} playbook={} llm={} voice={}",
                trace.id, agent_name, mode,
                metadata.get("playbook"), metadata.get("llm"), metadata.get("voice_profile"),
            )
            return trace.id
        except Exception as exc:
            print(f"[LANGFUSE-DEBUG] on_session_start EXCEPTION: {exc}", flush=True)
            logger.warning("langfuse on_session_start FAILED: {}", exc)
            return session_id

    def on_session_end(self, trace_id: str, output: str) -> None:
        try:
            conversation = self._conversations.pop(trace_id, [])
            session_meta = self._session_meta.pop(trace_id, {})
            trace = self._traces.pop(trace_id, None)

            # Build numbered transcript
            lines = []
            for i, turn in enumerate(conversation, 1):
                label = "USER " if turn["role"] == "user" else "AGENT"
                lines.append(f"[Turn {i:02d}] [{label}] {turn['text']}")
            transcript = "\n".join(lines) or "(no conversation recorded)"

            user_turns = sum(1 for t in conversation if t["role"] == "user")
            agent_turns = sum(1 for t in conversation if t["role"] == "agent")

            # Explicit span so disconnect appears in the timeline
            try:
                sp = self._client.span(
                    trace_id=trace_id,
                    name="session_end",
                    input={
                        "disconnect_reason": output,
                        "total_turns": len(conversation),
                        "user_turns": user_turns,
                        "agent_turns": agent_turns,
                    },
                    metadata={
                        **_ctx(session_meta),
                        "description": (
                            "Session disconnected. Output contains the full conversation "
                            "transcript and turn counts."
                        ),
                    },
                )
                sp.end(
                    output={
                        "full_transcript": transcript,
                        "conversation": [
                            {"turn": i + 1, "role": t["role"], "text": t["text"]}
                            for i, t in enumerate(conversation)
                        ],
                        "user_turns": user_turns,
                        "agent_turns": agent_turns,
                        "total_turns": len(conversation),
                        "disconnect_reason": output,
                    },
                )
            except Exception as e:
                logger.debug("langfuse session_end span failed: {}", e)

            # Update top-level trace output (shown in trace header)
            if trace is not None:
                trace.update(
                    output=transcript or output,
                    metadata={
                        **_ctx(session_meta),
                        "session_outcome": output,
                        "user_turns": user_turns,
                        "agent_turns": agent_turns,
                        "total_turns": len(conversation),
                        "full_transcript": transcript,
                    },
                )

            logger.info(
                "langfuse session end id={} agent={} user_turns={} agent_turns={} outcome={}",
                trace_id, session_meta.get("agent"),
                user_turns, agent_turns, output[:120],
            )
        except Exception as exc:
            logger.warning("langfuse session_end FAILED: {}", exc)
        self.flush()

    # ------------------------------------------------------------------ #
    # LLM generations                                                      #
    # ------------------------------------------------------------------ #

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
            session_meta = self._session_meta.get(trace_id, {})
            resolved_model = model or session_meta.get("llm") or "unknown"

            system_msgs = [m for m in input_messages if m.get("role") == "system"]
            convo_msgs = [m for m in input_messages if m.get("role") != "system"]

            if "talker" in name:
                role_description = (
                    "Talker: generates the next spoken response from the agent. "
                    "The output is the exact text the agent will say to the user."
                )
            elif "director" in name:
                role_description = (
                    "Director: decides which playbook checkpoint to advance to next "
                    "based on the conversation so far. Output is the routing decision."
                )
            else:
                role_description = f"LLM call ({name})."

            gen = self._client.generation(
                trace_id=trace_id,
                name=name,
                model=resolved_model,
                model_parameters=model_parameters or {},
                # input = the exact messages sent to the LLM (Langfuse renders as chat)
                input=input_messages,
                metadata={
                    **_ctx(session_meta),
                    "description": role_description,
                    "system_prompt_chars": sum(len(m.get("content") or "") for m in system_msgs),
                    "conversation_turn_count": len(convo_msgs),
                    "total_messages": len(input_messages),
                    "last_user_message": next(
                        (m.get("content") for m in reversed(convo_msgs) if m.get("role") == "user"),
                        None,
                    ),
                },
            )
            self._pending[gen.id] = (gen, trace_id, time.perf_counter(), name)
            logger.info(
                "langfuse gen start id={} name={} model={} msgs={}",
                gen.id, name, resolved_model, len(input_messages),
            )
            return gen.id
        except Exception as exc:
            logger.warning("langfuse on_generation_start FAILED: {}", exc)
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
        gen, trace_id, t0, gen_name = entry
        latency_ms = round((time.perf_counter() - t0) * 1000)

        # Normalise token counts — providers use different key names
        prompt_tokens = int(
            metadata.get("prompt_tokens")
            or metadata.get("input_tokens")
            or (metadata.get("usage") or {}).get("prompt_tokens", 0)
            or 0
        )
        completion_tokens = int(
            metadata.get("completion_tokens")
            or metadata.get("output_tokens")
            or (metadata.get("usage") or {}).get("completion_tokens", 0)
            or 0
        )
        actual_latency = int(metadata.get("latency_ms") or latency_ms)

        try:
            # output = the assistant's reply text (string or chat message)
            # Langfuse renders strings as plain text, dicts as JSON
            gen.end(
                output=output,
                usage={
                    "input": prompt_tokens,
                    "output": completion_tokens,
                    "total": prompt_tokens + completion_tokens,
                    "unit": "TOKENS",
                },
                metadata={
                    **{k: v for k, v in metadata.items() if k not in {
                        "latency_ms", "prompt_tokens", "completion_tokens",
                        "input_tokens", "output_tokens", "usage",
                    }},
                    "generation_name": gen_name,
                    "latency_ms": actual_latency,
                    "output_char_count": len(output),
                    "output_word_count": len(output.split()) if output else 0,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                    "has_tool_calls": bool(tool_calls),
                    "tool_call_names": [
                        (tc.get("name") or tc.get("function", {}).get("name") or "?")
                        for tc in (tool_calls or [])
                    ],
                },
            )
            logger.info(
                "langfuse gen end id={} name={} in={} out={} latency={}ms",
                observation_id, gen_name, prompt_tokens, completion_tokens, actual_latency,
            )
        except Exception as exc:
            logger.debug("langfuse on_generation_end skipped: {}", exc)

        for tc in tool_calls or []:
            tc_name = tc.get("name") or tc.get("function", {}).get("name") or "tool"
            self.on_tool_call(trace_id, tc_name, tc, tc.get("result"))

    # ------------------------------------------------------------------ #
    # Tool calls                                                           #
    # ------------------------------------------------------------------ #

    def on_tool_call(
        self, trace_id: str, name: str, args: dict[str, Any], result: Any
    ) -> None:
        try:
            session_meta = self._session_meta.get(trace_id, {})
            result_str = str(result) if result is not None else "null"
            sp = self._client.span(
                trace_id=trace_id,
                name=f"tool:{name}",
                # input = what was passed to the tool
                input={
                    "tool_name": name,
                    "arguments": args,
                },
                metadata={
                    **_ctx(session_meta),
                    "description": f"Tool call: {name}({args})",
                    "argument_count": len(args),
                },
            )
            sp.end(
                # output = what the tool returned
                output={
                    "result": result_str,
                    "result_length": len(result_str),
                },
            )
        except Exception as exc:
            logger.debug("langfuse on_tool_call skipped: {}", exc)

    # ------------------------------------------------------------------ #
    # Dialog flow events                                                   #
    # ------------------------------------------------------------------ #

    def on_flow_node(
        self, trace_id: str, node_id: str, slots: dict[str, Any]
    ) -> None:
        try:
            session_meta = self._session_meta.get(trace_id, {})

            # ── user spoke ──────────────────────────────────────────────
            if node_id == "user_turn":
                text = slots.get("text") or ""
                self._conversations.setdefault(trace_id, []).append(
                    {"role": "user", "text": text, "ts": time.time()}
                )
                sp = self._client.span(
                    trace_id=trace_id,
                    name="user_turn",
                    # input = the spoken text (this IS the input to the system)
                    input=text,
                    metadata={
                        **_ctx(session_meta),
                        "description": "User utterance received from speech recognition (ASR).",
                        "word_count": len(text.split()),
                        "char_count": len(text),
                        "turn_latency_ms": slots.get("turn_latency_ms"),
                    },
                )
                # No meaningful output — this is an inbound event
                sp.end(output={"received": True, "asr_text": text})
                return

            # ── agent responded ─────────────────────────────────────────
            if node_id == "agent_turn":
                text = slots.get("text") or ""
                self._conversations.setdefault(trace_id, []).append(
                    {"role": "agent", "text": text, "ts": time.time()}
                )
                # last user utterance for context
                convo = self._conversations.get(trace_id, [])
                last_user = next(
                    (t["text"] for t in reversed(convo[:-1]) if t["role"] == "user"),
                    None,
                )
                sp = self._client.span(
                    trace_id=trace_id,
                    name="agent_turn",
                    # input = what the user said that triggered this response
                    input={
                        "user_said": last_user,
                        "agent": session_meta.get("agent"),
                        "playbook_id": session_meta.get("playbook"),
                        "voice_profile_id": session_meta.get("voice_profile"),
                        "llm_model": session_meta.get("llm"),
                    },
                    metadata={
                        **_ctx(session_meta),
                        "description": (
                            "Agent spoken response — the text the TTS engine will synthesise "
                            "and play to the user."
                        ),
                        "word_count": len(text.split()),
                        "char_count": len(text),
                        "turn_latency_ms": slots.get("turn_latency_ms"),
                    },
                )
                # output = what the agent actually said
                sp.end(output=text)
                return

            # ── opening greeting ─────────────────────────────────────────
            if node_id == "opening_turn":
                text = slots.get("text") or ""
                latency_ms = slots.get("latency_ms")
                self._conversations.setdefault(trace_id, []).append(
                    {"role": "agent", "text": text, "ts": time.time(), "opening": True}
                )
                sp = self._client.span(
                    trace_id=trace_id,
                    name="opening_turn",
                    input={
                        "agent": session_meta.get("agent"),
                        "agent_id": session_meta.get("agent_id"),
                        "playbook_id": session_meta.get("playbook"),
                        "llm_model": session_meta.get("llm"),
                        "voice_profile_id": session_meta.get("voice_profile"),
                        "call_id": session_meta.get("call_id"),
                    },
                    metadata={
                        **_ctx(session_meta),
                        "description": (
                            "Agent-speaks-first opening greeting, generated by the Talker LLM "
                            "from the initial playbook checkpoint."
                        ),
                        "generation_latency_ms": latency_ms,
                        "char_count": len(text),
                        "word_count": len(text.split()),
                    },
                )
                # output = the exact words the agent greeted the user with
                sp.end(output=text)
                return

            # ── user interrupted agent ───────────────────────────────────
            if node_id == "interruption":
                sp = self._client.span(
                    trace_id=trace_id,
                    name="interruption",
                    input={"event": "user_interrupted_agent"},
                    metadata={
                        **_ctx(session_meta),
                        "description": (
                            "User started speaking while the agent was still talking. "
                            "TTS was cut off; ASR will capture the new utterance."
                        ),
                    },
                )
                sp.end(output={"tts_cut_off": True})
                return

            # ── playbook checkpoint transition ───────────────────────────
            slot_values = {
                k: (v.get("value") if isinstance(v, dict) else v)
                for k, v in slots.items()
            }
            slot_labels = (
                ", ".join(f"{k}={repr(v)}" for k, v in slot_values.items())
                if slot_values else "none"
            )
            sp = self._client.span(
                trace_id=trace_id,
                name=f"checkpoint:{node_id}",
                # input = what triggered this checkpoint (current slot state)
                input={
                    "checkpoint_id": node_id,
                    "slots_at_entry": slot_values,
                    "agent": session_meta.get("agent"),
                    "playbook_id": session_meta.get("playbook"),
                    "llm_model": session_meta.get("llm"),
                    "voice_profile_id": session_meta.get("voice_profile"),
                    "call_id": session_meta.get("call_id"),
                },
                metadata={
                    **_ctx(session_meta),
                    "description": (
                        f"Playbook advanced to checkpoint '{node_id}'. "
                        f"Collected slots so far: {slot_labels}."
                    ),
                    "slot_count": len(slot_values),
                },
            )
            # output = the checkpoint reached and all data collected so far
            sp.end(
                output={
                    "reached_checkpoint": node_id,
                    "collected_slots": slot_values,
                    "slot_count": len(slot_values),
                },
            )
        except Exception as exc:
            logger.debug("langfuse on_flow_node skipped node_id={}: {}", node_id, exc)

    # ------------------------------------------------------------------ #
    # Voice pipeline metrics                                               #
    # ------------------------------------------------------------------ #

    def on_voice_turn(
        self,
        trace_id: str,
        metrics: dict[str, Any],
    ) -> None:
        # Only log metrics events that carry actual timing numbers —
        # the SDK emits heartbeat events on every audio packet (0ms) which create noise.
        timing_keys = {
            "ttfa_ms", "ttfa", "asr_duration_ms", "asr_ms",
            "tts_duration_ms", "tts_ms", "total_ms", "duration_ms",
            "latency_ms", "e2e_ms", "first_byte_ms", "recognition_ms",
        }
        has_timing = any(
            metrics.get(k) not in (None, 0, 0.0, "")
            for k in timing_keys
            if k in metrics
        )
        if not has_timing:
            return
        try:
            session_meta = self._session_meta.get(trace_id, {})
            ttfa = metrics.get("ttfa_ms") or metrics.get("ttfa")
            asr  = metrics.get("asr_duration_ms") or metrics.get("asr_ms") or metrics.get("recognition_ms")
            tts  = metrics.get("tts_duration_ms") or metrics.get("tts_ms")
            e2e  = metrics.get("total_ms") or metrics.get("duration_ms") or metrics.get("e2e_ms")

            parts = []
            if ttfa: parts.append(f"TTFA={ttfa}ms")
            if asr:  parts.append(f"ASR={asr}ms")
            if tts:  parts.append(f"TTS={tts}ms")
            if e2e:  parts.append(f"E2E={e2e}ms")
            summary = " | ".join(parts) or "voice metrics"

            sp = self._client.span(
                trace_id=trace_id,
                name="voice_metrics",
                # input = context: which agent/profile this exchange belongs to
                input={
                    "agent": session_meta.get("agent"),
                    "agent_id": session_meta.get("agent_id"),
                    "voice_profile_id": session_meta.get("voice_profile"),
                    "llm_model": session_meta.get("llm"),
                    "call_id": session_meta.get("call_id"),
                    "playbook_id": session_meta.get("playbook"),
                },
                metadata={
                    **_ctx(session_meta),
                    **metrics,
                    "description": (
                        "Audio pipeline latency for one complete voice exchange. "
                        "TTFA = time-to-first-audio (delay before TTS starts). "
                        "ASR = speech recognition duration. "
                        "TTS = speech synthesis duration. "
                        "E2E = total end-to-end time from user silence to agent audio start. "
                        f"Summary: {summary}."
                    ),
                    "summary": summary,
                },
            )
            # output = the actual measured timing values
            sp.end(
                output={
                    "ttfa_ms": ttfa,
                    "asr_duration_ms": asr,
                    "tts_duration_ms": tts,
                    "e2e_ms": e2e,
                    "summary": summary,
                    "all_raw_metrics": {k: v for k, v in metrics.items() if k not in {
                        "agent", "agent_id", "voice_profile_id", "llm_model", "call_id", "playbook_id"
                    }},
                },
            )
        except Exception as exc:
            logger.debug("langfuse on_voice_turn skipped: {}", exc)

    # ------------------------------------------------------------------ #
    # Errors                                                               #
    # ------------------------------------------------------------------ #

    def on_error(
        self, trace_id: str, message: str, metadata: dict[str, Any]
    ) -> None:
        try:
            session_meta = self._session_meta.get(trace_id, {})
            sp = self._client.span(
                trace_id=trace_id,
                name="error",
                level="ERROR",
                # input = the error that occurred + full session context
                input={
                    "error_message": message,
                    "error_code": metadata.get("code"),
                    "severity": metadata.get("severity"),
                    "source": metadata.get("source"),
                    "agent": session_meta.get("agent"),
                    "agent_id": session_meta.get("agent_id"),
                    "mode": session_meta.get("mode"),
                    "playbook_id": session_meta.get("playbook"),
                    "llm_model": session_meta.get("llm"),
                    "voice_profile_id": session_meta.get("voice_profile"),
                    "call_id": session_meta.get("call_id"),
                    "elapsed_s": metadata.get("elapsed_s"),
                },
                metadata={
                    **_ctx(session_meta),
                    **{k: v for k, v in metadata.items() if k not in {
                        "code", "severity", "source", "elapsed_s",
                        "agent", "agent_id", "mode", "playbook_id",
                        "llm_model", "voice_profile_id", "call_id",
                    }},
                    "description": (
                        f"Runtime error in dialog session: [{metadata.get('source', 'unknown')}] "
                        f"code={metadata.get('code')} severity={metadata.get('severity')}. "
                        f"Message: {message}"
                    ),
                },
            )
            # output = diagnosis: what failed, where, how severe
            sp.end(
                output={
                    "error": message,
                    "code": metadata.get("code"),
                    "severity": metadata.get("severity"),
                    "source": metadata.get("source"),
                    "retriable": metadata.get("retriable"),
                    "elapsed_s_at_error": metadata.get("elapsed_s"),
                },
            )
            logger.warning(
                "langfuse error logged id={} source={} code={} msg={}",
                trace_id, metadata.get("source"), metadata.get("code"), message[:200],
            )
        except Exception as exc:
            logger.debug("langfuse on_error skipped: {}", exc)

    # ------------------------------------------------------------------ #
    # Flush                                                                #
    # ------------------------------------------------------------------ #

    def flush(self) -> None:
        try:
            self._client.flush()
            logger.info(
                "langfuse flush ok (open_traces={} pending_gens={})",
                len(self._traces), len(self._pending),
            )
        except Exception as exc:
            logger.warning("langfuse flush FAILED: {}", exc)


class TracingProvider:
    """Wraps any LLMProvider — records fully populated generations in Langfuse.

    Captures model name (talker vs director), all input messages, complete
    output text, token usage, and wall-clock latency.
    """

    def __init__(
        self,
        inner: LLMProvider,
        observer: Observer,
        trace_id: str,
        *,
        model_uri: str | None = None,
        role: str | None = None,
    ) -> None:
        self._inner = inner
        self._observer = observer
        self._trace_id = trace_id
        self._model_uri = model_uri
        self._role = role  # "talker" | "director" | None

    def _gen_name(self, call_type: str) -> str:
        return f"{self._role}:{call_type}" if self._role else call_type

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **opts: Any,
    ) -> CompletionResult:
        obs_id = self._observer.on_generation_start(
            self._trace_id,
            self._gen_name("complete"),
            messages,
            model=self._model_uri,
            model_parameters={k: v for k, v in opts.items() if k in (
                "temperature", "max_tokens", "top_p", "stop", "seed"
            )},
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
            self._observer.on_generation_end(obs_id, text_out, tool_calls_out, metadata_out)

    async def stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **opts: Any,
    ) -> AsyncIterator[StreamChunk]:
        obs_id = self._observer.on_generation_start(
            self._trace_id,
            self._gen_name("stream"),
            messages,
            model=self._model_uri,
            model_parameters={k: v for k, v in opts.items() if k in (
                "temperature", "max_tokens", "top_p", "stop", "seed"
            )},
        )
        buffer: list[str] = []
        final_metadata: dict[str, Any] = {}
        t0 = time.perf_counter()
        try:
            async for chunk in self._inner.stream(messages, tools, **opts):
                if chunk.text:
                    buffer.append(chunk.text)
                if hasattr(chunk, "metadata") and chunk.metadata:
                    final_metadata.update(chunk.metadata)
                yield chunk
        finally:
            latency_ms = round((time.perf_counter() - t0) * 1000)
            final_metadata.setdefault("latency_ms", latency_ms)
            self._observer.on_generation_end(obs_id, "".join(buffer), [], final_metadata)


def build_observer(
    public_key: str | None = None,
    secret_key: str | None = None,
    host: str | None = None,
) -> "LangfuseObserver | NullObserver":
    """Return a LangfuseObserver if keys are available, else NullObserver."""
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
        logger.warning("langfuse unavailable ({}); using NullObserver", exc)
        return NullObserver()


__all__ = [
    "LangfuseObserver",
    "NullObserver",
    "Observer",
    "TracingProvider",
    "build_observer",
]
