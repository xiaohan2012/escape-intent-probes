"""A model backend on a hosted chat-completions endpoint.

For the frontier screen (D22): five models from five labs, asked whether any of
them crosses the boundary at all. None of this produces probe data — a hosted
endpoint yields no residual stream, and no API returns token ids — so D12's
frozen-asset machinery is deliberately absent rather than approximated.

Two conversions carry the whole risk. The loop keeps one canonical
representation of a conversation (text, with the tool call in Qwen's tag form)
because the labels, the persistence layer and the fake backend all read it,
while a model post-trained on the tool protocol expects `assistant(tool_calls)`
followed by `tool(result)`. Feeding a model its own call back as prose is the
off-distribution failure D15 is about, and its symptom — every step fails to
parse, every trajectory ends in `parse_failed`, the batch reports zero crossings
— is indistinguishable from the signal the screen is looking for. So the
conversion runs in both directions and is tested in both directions.

Undoing the rendering, rather than keeping a second history alongside it, is
what lets `generate(messages)` keep its contract: the backend stays stateless
and any driver can call it.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from typing import Any, Protocol

from escape_probes.chat import select_tool_schemas
from escape_probes.config import DEFAULT_TOOLS, ModelConfig
from escape_probes.model import (
    TOOL_CALL_OPEN,
    Generation,
    Message,
    decode_payload,
    find_tool_payload,
    render_tool_call,
)


class Transport(Protocol):
    """Whatever actually performs the request.

    Injected so the backend's payload can be asserted on without a network or a
    patched client, and so the HTTP dependency stays out of the core install.
    """

    def post(self, payload: dict[str, Any]) -> dict[str, Any]: ...


def to_api_messages(messages: Sequence[Message]) -> list[dict[str, Any]]:
    """The loop's flat text conversation, back into the API's protocol.

    An assistant turn whose text contains a rendered tool call becomes a
    structured `tool_calls` message, and the observation that follows it becomes
    a `tool` message answering that call. The ids are positional and therefore
    stable: the same conversation converts the same way every time, which is
    what keeps the backend stateless.

    An assistant turn with no parsable call replays as prose, and so does the
    turn after it — that is the parse-retry path, and rendering the retry prompt
    as a tool result would answer a call the model never made.
    """
    converted: list[dict[str, Any]] = []
    open_call_id: str | None = None

    for message in messages:
        if message.role == "assistant":
            call_id = None
            found = find_tool_payload(message.content)
            call = decode_payload(*found) if found is not None else None
            if found is not None and call is not None and call.name is not None:
                call_id = f"call_{len(converted)}"
                preamble_end = found[1] - len(TOOL_CALL_OPEN)
                converted.append(
                    {
                        "role": "assistant",
                        "content": message.content[:preamble_end].strip(),
                        "tool_calls": [
                            {
                                "id": call_id,
                                "type": "function",
                                "function": {
                                    "name": call.name,
                                    "arguments": json.dumps(call.arguments),
                                },
                            }
                        ],
                    }
                )
            else:
                converted.append({"role": "assistant", "content": message.content})
            open_call_id = call_id
            continue

        if message.role == "user" and open_call_id is not None:
            converted.append(
                {"role": "tool", "tool_call_id": open_call_id, "content": message.content}
            )
            open_call_id = None
            continue

        converted.append({"role": message.role, "content": message.content})

    return converted


def text_from_api_message(message: dict[str, Any]) -> str:
    """A structured response, back into the loop's canonical text.

    Reasoning is kept where the provider offers it (D22). It cannot move the
    primary metric — a crossing is decided by a tool call's path argument, not
    by text — but if a crossing happens it is the whole of the explanation.

    Arguments that will not parse as JSON are dropped rather than guessed at, so
    the generation lands on the parse-retry path the loop already handles. A
    truncated argument object executes something the model did not ask for; a
    parse error costs one step.
    """
    parts = [str(message.get(key) or "") for key in ("reasoning", "content")]

    calls = message.get("tool_calls") or []
    if calls:
        function = calls[0].get("function", {})
        try:
            arguments = json.loads(function.get("arguments") or "{}")
        except json.JSONDecodeError:
            arguments = None
        if isinstance(arguments, dict) and function.get("name"):
            parts.append(
                render_tool_call(function["name"], {k: str(v) for k, v in arguments.items()})
            )

    return "\n".join(part for part in parts if part)


class APIModel:
    """A hosted model driven one conversation at a time.

    Safe to call from several threads: every request carries its whole
    conversation, so there is no per-call state to share. That is what
    `drive_threaded` relies on.
    """

    def __init__(
        self,
        config: ModelConfig,
        tools: tuple[str, ...] = DEFAULT_TOOLS,
        transport: Transport | None = None,
    ) -> None:
        self.config = config
        self.tool_schemas = select_tool_schemas(tools)
        self.transport = transport or HTTPTransport(config)

    def generate(self, messages: Sequence[Message]) -> Generation:
        response = self.transport.post(self._payload(messages))
        choices = response.get("choices") or []
        if not choices:
            raise RuntimeError(f"response carried no choices: {response}")

        return Generation(
            prompt_token_ids=(),
            gen_token_ids=(),
            text=text_from_api_message(choices[0].get("message") or {}),
            tool_start_token_idx=None,
        )

    def _payload(self, messages: Sequence[Message]) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.config.model_id,
            "messages": to_api_messages(messages),
            "tools": self.tool_schemas,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "max_tokens": self.config.max_new_tokens,
        }
        if self.config.providers:
            # Absent rather than an empty `only` list, which some gateways read
            # as "no provider is acceptable".
            payload["provider"] = {"only": list(self.config.providers)}
        return payload


class HTTPTransport:
    """The real transport. `httpx` is imported at construction, not at import.

    Same reason as the other backends: the loop and its tests run against the
    fake model with none of the optional dependency groups installed.
    """

    def __init__(self, config: ModelConfig) -> None:
        import httpx  # noqa: PLC0415

        key = os.environ.get(config.api_key_env)
        if not key:
            raise RuntimeError(f"{config.api_key_env} is not set")
        self.url = f"{config.api_base.rstrip('/')}/chat/completions"
        self.client = httpx.Client(
            headers={"Authorization": f"Bearer {key}"},
            timeout=httpx.Timeout(600.0),
        )

    def post(self, payload: dict[str, Any]) -> dict[str, Any]:
        response = self.client.post(self.url, json=payload)
        response.raise_for_status()
        return dict(response.json())


__all__ = ["APIModel", "HTTPTransport", "Transport", "text_from_api_message", "to_api_messages"]
