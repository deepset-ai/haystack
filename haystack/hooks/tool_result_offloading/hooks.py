# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import json
from typing import Any

from haystack import logging
from haystack.components.agents.state.state import State
from haystack.components.agents.state.state_utils import replace_values
from haystack.core.serialization import default_from_dict, default_to_dict
from haystack.dataclasses import ChatMessage, TextContent
from haystack.dataclasses.chat_message import ToolCallResultContentT
from haystack.hooks.tool_result_offloading.types import OffloadPolicy, ToolResultStore
from haystack.utils.deserialization import deserialize_component_inplace

logger = logging.getLogger(__name__)

# Meta key marking an already-offloaded tool-result message (its value is the store reference). The offloaded pointer
# is itself a tool result in the trailing block the hook scans, so this marker stops a second offload hook registered
# under `after_tool` from offloading the pointer text again and writing a junk file.
_OFFLOADED_META_KEY = "tool_result_offloaded"

# Key under which a per-run store override may be supplied via the Agent's `hook_context` (e.g. a request-scoped
# sandbox filesystem).
RESULT_STORE_CONTEXT_KEY = "tool_result_store"


def _result_store_key(tool_name: str, tool_call_id: str | None, step: int, index: int) -> str:
    """
    Build a per-result store key that is stable and unique within a run.

    Combining the step, tool name, and tool call id keeps results from different tools and different steps from
    colliding while staying deterministic (so a re-run produces the same key). When the tool call carries no id
    (it is optional and not every generator sets it), the result's position in the step's batch is used instead, so
    two id-less calls to the same tool in the same step do not collide.

    :param tool_name: The name of the tool that produced the result.
    :param tool_call_id: The id of the originating tool call, or None when the call carried no id.
    :param step: The Agent's current step count.
    :param index: The result's position within this step's batch of tool results, used when `tool_call_id` is None.
    :returns: A file-name-like key for the store, e.g. `2_web_search_call-123.txt`.
    """
    return f"{step}_{tool_name}_{tool_call_id or f'call{index}'}.txt"


def _fresh_tool_results_start(messages: list[ChatMessage]) -> int:
    """
    Return the index at which the trailing run of tool-result messages begins.

    The Agent appends the current step's tool results to the end of the conversation, so the trailing contiguous
    block of tool-result messages is exactly the freshly produced batch; everything before it is history the hook
    must not touch (results from earlier steps or ones the caller passed in).

    :param messages: The conversation, oldest to newest.
    :returns: The index of the first message in the trailing tool-result block, or `len(messages)` when the last
        message is not a tool result (no fresh results to offload).
    """
    start = len(messages)
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].tool_call_result is None:
            break
        start = index
    return start


def _offloadable_text(content: ToolCallResultContentT) -> str | None:
    """
    Return the text of a tool result if it can be offloaded as text, otherwise None.

    A plain string is returned as-is; a non-empty sequence made up entirely of `TextContent` blocks is concatenated
    into a single string. Anything else (e.g. a result containing image or file content) returns None and is left in
    context.

    :param content: The tool result content to inspect.
    :returns: The offloadable text, or None when the content is not purely text.
    """
    if isinstance(content, str):
        return content
    texts = [block.text for block in content if isinstance(block, TextContent)]
    if texts and len(texts) == len(content):
        return "".join(texts)
    return None


def _serialize_offload_strategies(strategies: dict[str | tuple[str, ...], OffloadPolicy]) -> dict[str, Any]:
    """
    Serialize an offload-strategies mapping to a plain, mapping-key-safe dictionary.

    Mapping keys must be strings, so a tuple of tool names (one policy shared across several tools) is encoded as a
    JSON-array string (e.g. `("a", "b")` -> `'["a", "b"]'`); a single tool name or the `"*"` wildcard is kept as-is.
    Each policy is serialized via its own `to_dict`, which embeds its type so it can be reconstructed regardless of
    its concrete class.

    :param strategies: Mapping of tool name (or a tuple of tool names, or `"*"`) to its `OffloadPolicy`.
    :returns: The same mapping with string keys and each policy serialized to a dictionary.
    """
    return {
        (json.dumps(list(key)) if isinstance(key, tuple) else key): policy.to_dict()
        for key, policy in strategies.items()
    }


def _deserialize_offload_strategies(data: dict[str, Any]) -> dict[str | tuple[str, ...], OffloadPolicy]:
    """
    Deserialize an offload-strategies mapping from its serialized form.

    Reverses `_serialize_offload_strategies`: each policy is rebuilt from its stored type via
    `deserialize_component_inplace`, and keys that were encoded as JSON-array strings become tuples of tool names
    (single tool-name and `"*"` keys are kept as-is).

    :param data: Raw dictionary of serialized offload strategies, keyed by tool name(s).
    :returns: The offload strategies with their original key and policy types restored.
    """
    for raw_key in list(data):
        deserialize_component_inplace(data, key=raw_key)
    return {
        (tuple(json.loads(raw_key)) if isinstance(raw_key, str) and raw_key.startswith("[") else raw_key): policy
        for raw_key, policy in data.items()
    }


class ToolResultOffloadHook:
    """
    Offload tool results to a `ToolResultStore`, replacing them in the conversation with a compact pointer.

    This `after_tool` Agent hook writes the full result to the store so the next LLM call sees a reference instead of
    the full result. Register it on an `Agent` under the `after_tool` hook point. Which tools offload, and under what
    condition, is controlled per tool by `offload_strategies`:

    ```python
    from haystack.components.agents import Agent
    from haystack.components.generators.chat import OpenAIChatGenerator
    from haystack.hooks.tool_result_offloading import (
        AlwaysOffload,
        FileSystemToolResultStore,
        NeverOffload,
        OffloadOverChars,
        ToolResultOffloadHook,
    )

    hook = ToolResultOffloadHook(
        store=FileSystemToolResultStore(root="tool_results"),
        offload_strategies={
            "web_search": AlwaysOffload(),          # force offload
            "get_time": NeverOffload(),             # opt out
            ("read_file", "list_dir"): OffloadOverChars(4000),  # tuple key: shared policy
            "*": OffloadOverChars(8000),            # wildcard default for any unlisted tool
        },
    )
    agent = Agent(
        chat_generator=OpenAIChatGenerator(model="gpt-5.4-nano"),
        tools=[web_search, get_time, read_file, list_dir],
        hooks={"after_tool": [hook]},
    )
    ```

    A key may be a single tool name, a tuple of tool names sharing one policy, or the wildcard `"*"` which applies to
    any tool without a more specific entry. More specific keys win. A tool with no matching key (and no `"*"`) is not
    offloaded.

    Only successful, text tool output is offloaded. Error results (including `before_tool` human-in-the-loop
    rejections) are always left in context. Non-text results (image or file content) are also left in context, and a
    warning is logged when such a result has a matching offload policy; supporting only text is a deliberate choice
    for now. Each result is offloaded at most once, even though the hook runs on every tool step.

    For server settings where the store is request-scoped (e.g. an isolated sandbox filesystem), pass it per run via
    the Agent's `hook_context` under the key `RESULT_STORE_CONTEXT_KEY`
    (`agent.run(messages=[...], hook_context={RESULT_STORE_CONTEXT_KEY: my_store})`); it overrides the store the hook
    was constructed with.
    """

    allowed_hook_points = ("after_tool",)

    def __init__(
        self,
        store: ToolResultStore,
        offload_strategies: dict[str | tuple[str, ...], OffloadPolicy],
        *,
        preview_chars: int = 200,
    ) -> None:
        """
        Initialize the hook with a store and per-tool offload strategies.

        :param store: Where offloaded results are written. Can be overridden per run via `hook_context`.
        :param offload_strategies: Mapping of tool name (or a tuple of tool names, or the wildcard `"*"`) to the
            `OffloadPolicy` that decides whether that tool's results are offloaded.
        :param preview_chars: Number of leading characters of the original result to include in the pointer left in
            the conversation, so the model knows roughly what was offloaded.
        """
        self.store = store
        self.offload_strategies = offload_strategies
        self.preview_chars = preview_chars

    def run(self, state: State) -> None:
        """
        Offload the freshly produced tool results in `state.data["messages"]` according to `offload_strategies`.

        Considers only the trailing block of tool-result messages (the current step's results); earlier history is
        left untouched. Offloads each of those messages its policy opts in for, and writes the rewritten conversation
        back to `messages` only if at least one message changed.

        :param state: The Agent's live `State`. Reads the per-run store override from `hook_context` and rewrites the
            offloaded tool-result messages back into `messages`.
        :returns: None. The hook mutates `state` in place.
        """
        messages = state.data.get("messages") or []
        start = _fresh_tool_results_start(messages)
        if start >= len(messages):
            return
        store = self._resolve_store(state)
        rewritten: list[ChatMessage] = list(messages[:start])
        changed = False
        for index, message in enumerate(messages[start:]):
            new_message = self._maybe_offload(message, store, state, index)
            rewritten.append(new_message)
            changed = changed or new_message is not message
        if changed:
            state.set("messages", rewritten, handler_override=replace_values)

    def _resolve_store(self, state: State) -> ToolResultStore:
        """
        Return the store to write to for this run.

        :param state: The Agent's live `State`, whose `hook_context` may carry a per-run store override under
            `RESULT_STORE_CONTEXT_KEY`.
        :returns: The per-run store from `hook_context` if provided, otherwise the store the hook was built with.
        """
        context = state.data.get("hook_context") or {}
        return context.get(RESULT_STORE_CONTEXT_KEY, self.store)

    def _policy_for(self, tool_name: str) -> OffloadPolicy | None:
        """
        Resolve the offload policy that applies to a tool, most specific first.

        Lookup order: an exact tool-name key, then any tuple key that contains the tool name, then the `"*"` wildcard.

        :param tool_name: The name of the tool whose policy to resolve.
        :returns: The matching `OffloadPolicy`, or None when no key (and no `"*"`) applies.
        """
        strategies = self.offload_strategies
        if tool_name in strategies:
            return strategies[tool_name]
        for key, policy in strategies.items():
            if isinstance(key, tuple) and tool_name in key:
                return policy
        return strategies.get("*")

    def _maybe_offload(self, message: ChatMessage, store: ToolResultStore, state: State, index: int) -> ChatMessage:
        """
        Offload a single tool-result message if its policy opts in, otherwise return it unchanged.

        A message is left as-is when it is not a tool result, when the result is an error (including `before_tool`
        human-in-the-loop rejections), when it was already offloaded (e.g. another offload hook under `after_tool`
        handled it), when no policy applies, when the result is non-text (contains image or file content), or when the
        policy declines to offload.

        Otherwise the result text is written to `store` and the message is rebuilt with a pointer in place of the full
        result, preserving its origin and error flag and marking it offloaded.

        :param message: The message to consider offloading.
        :param store: The store to write the result to.
        :param state: The Agent's live `State`, passed to the policy and used to derive the store key.
        :param index: The message's position within this step's batch of tool results, used to build the store key.
        :returns: An offloaded copy of the message, or the original message when it is not offloaded.
        """
        result = message.tool_call_result
        # Only successful tool output is offloaded - never errors, before_tool human-in-the-loop rejections, or a
        # result already offloaded (guards against a second offload hook re-offloading the first one's pointer).
        if result is None or result.error or message.meta.get(_OFFLOADED_META_KEY):
            return message

        tool_name = result.origin.tool_name
        policy = self._policy_for(tool_name)

        # If no policy applies, leave the result in context
        if policy is None:
            return message

        # A policy matched, so an offload was wanted. Offloading only supports text results (a string or a sequence
        # of TextContent) for now, by design; leave image/file content in context and warn since the intent was to
        # offload it.
        text = _offloadable_text(result.result)
        if text is None:
            logger.warning(
                "Tool '{tool}' produced a non-text result; leaving it in context. Result offloading currently "
                "supports text results only.",
                tool=tool_name,
            )
            return message

        # If the policy declines to offload, leave the result in context
        if not policy.should_offload(tool_name, text, state):
            return message

        key = _result_store_key(tool_name, result.origin.id, state.data.get("step_count", 0), index)
        reference = store.write(key=key, content=text)
        return ChatMessage.from_tool(
            tool_result=self._pointer(reference, text),
            origin=result.origin,
            error=result.error,
            meta={**message.meta, _OFFLOADED_META_KEY: reference},
        )

    def _pointer(self, reference: str, result: str) -> str:
        """
        Build the compact pointer that replaces a full result in the conversation.

        :param reference: The store reference the result was written to.
        :param result: The original result string, used for its length and a leading preview.
        :returns: A one-line pointer carrying the reference, the result length, and a `preview_chars`-long preview.
        """
        ellip = "..." if len(result) > self.preview_chars else ""
        preview = result[: self.preview_chars]
        return f"Tool result offloaded to '{reference}' ({len(result)} characters). Preview: {preview}{ellip}"

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize the hook, including its store and per-tool offload strategies.

        :returns: A dictionary representation of the hook.
        """
        return default_to_dict(
            self,
            store=self.store.to_dict(),
            offload_strategies=_serialize_offload_strategies(self.offload_strategies),
            preview_chars=self.preview_chars,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ToolResultOffloadHook":
        """
        Deserialize the hook, reconstructing its store and offload strategies.

        :param data: A dictionary representation produced by `to_dict`.
        :returns: The deserialized `ToolResultOffloadHook`.
        """
        init_params = data.get("init_parameters", {})
        if init_params.get("store") is not None:
            deserialize_component_inplace(init_params, key="store")
        if init_params.get("offload_strategies") is not None:
            init_params["offload_strategies"] = _deserialize_offload_strategies(init_params["offload_strategies"])
        return default_from_dict(cls, data)
