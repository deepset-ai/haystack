# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import json
from typing import Any

from haystack import logging
from haystack.components.agents.state.state import State
from haystack.components.agents.state.state_utils import replace_values
from haystack.core.serialization import default_from_dict, default_to_dict
from haystack.dataclasses import ChatMessage
from haystack.hooks.tool_result_offloading.types import OffloadPolicy, ToolResultStore
from haystack.utils.base_serialization import deserialize_class_instance, serialize_class_instance

logger = logging.getLogger(__name__)

# Meta key marking an already-offloaded tool-result message, so re-running the hook on later steps does not offload
# the same result twice.
_OFFLOADED_META_KEY = "tool_result_offloaded"

# Key under which a per-run store override may be supplied via the Agent's `hook_context` (e.g. a request-scoped
# sandbox filesystem). When absent, the hook uses the store it was constructed with.
RESULT_STORE_CONTEXT_KEY = "tool_result_store"


class ToolResultOffloadHook:
    """
    Offload tool results to a `ToolResultStore`, replacing them in the conversation with a compact pointer.

    This `after_tool` Agent hook writes the full result to the store so the next LLM call sees a reference instead of
    the full result. Register it on an `Agent` under the `after_tool` hook point. Which tools offload, and under what
    condition, is controlled per tool by `offload_strategies`:

    ```python
    from haystack.components.agents import Agent
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
    agent = Agent(chat_generator=..., tools=[...], hooks={"after_tool": [hook]})
    ```

    A key may be a single tool name, a tuple of tool names sharing one policy, or the wildcard `"*"` which applies to
    any tool without a more specific entry. More specific keys win. A tool with no matching key (and no `"*"`) is not
    offloaded.

    Only real, successful tool output is offloaded — results flagged as errors (including `before_tool` rejections)
    are always left in context. Each result is offloaded at most once, even though the hook runs on every tool step.

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

        :param state: The Agent's live `State`. Reads the per-run store override from `hook_context` and rewrites the
            offloaded tool-result messages back into `messages`.
        """
        messages = state.data.get("messages") or []
        store = self._resolve_store(state)
        rewritten: list[ChatMessage] = []
        changed = False
        for message in messages:
            new_message = self._maybe_offload(message, store, state)
            rewritten.append(new_message)
            changed = changed or new_message is not message
        if changed:
            state.set("messages", rewritten, handler_override=replace_values)

    def _resolve_store(self, state: State) -> ToolResultStore:
        """Return the per-run store from `hook_context` if provided, otherwise the store the hook was built with."""
        context = state.data.get("hook_context") or {}
        return context.get(RESULT_STORE_CONTEXT_KEY, self.store)

    def _policy_for(self, tool_name: str) -> OffloadPolicy | None:
        """Resolve the policy for `tool_name`: exact match, then tuple keys, then the `"*"` wildcard, else None."""
        strategies = self.offload_strategies
        if tool_name in strategies:
            return strategies[tool_name]
        for key, policy in strategies.items():
            if isinstance(key, tuple) and tool_name in key:
                return policy
        return strategies.get("*")

    def _maybe_offload(self, message: ChatMessage, store: ToolResultStore, state: State) -> ChatMessage:
        """Return an offloaded copy of a tool-result message when its policy opts in, otherwise the message as-is."""
        result = message.tool_call_result
        # Only real, successful tool output is offloaded — never errors or before_tool rejections, and never a
        # result already offloaded in an earlier step.
        if result is None or result.error or message.meta.get(_OFFLOADED_META_KEY):
            return message
        tool_name = result.origin.tool_name
        policy = self._policy_for(tool_name)
        if policy is None:
            return message
        # Offloading supports text results only today; leave multimodal content (images, files) in context.
        if not isinstance(result.result, str):
            logger.warning(
                "Tool '{tool}' produced a non-string result; leaving it in context. Result offloading currently "
                "supports text results only.",
                tool=tool_name,
            )
            return message
        if not policy.should_offload(tool_name, result.result, state):
            return message
        reference = store.write(key=self._result_key(tool_name, result.origin.id, state), content=result.result)
        return ChatMessage.from_tool(
            tool_result=self._pointer(reference, result.result),
            origin=result.origin,
            error=result.error,
            meta={**message.meta, _OFFLOADED_META_KEY: reference},
        )

    @staticmethod
    def _result_key(tool_name: str, tool_call_id: str | None, state: State) -> str:
        """Build a per-result store key that is stable and unique within a run (step + tool + call id)."""
        step = state.data.get("step_count", 0)
        return f"{step}_{tool_name}_{tool_call_id or 'result'}.txt"

    def _pointer(self, reference: str, result: str) -> str:
        """Build the compact pointer (reference + size + preview) that replaces the full result in the conversation."""
        ellipsis = "…" if len(result) > self.preview_chars else ""
        preview = result[: self.preview_chars]
        return f"Tool result offloaded to '{reference}' ({len(result)} characters). Preview: {preview}{ellipsis}"

    def to_dict(self) -> dict[str, Any]:
        """Serialize the hook, including its store and per-tool offload strategies."""
        return default_to_dict(
            self,
            store=serialize_class_instance(self.store),
            offload_strategies=self._serialize_strategies(self.offload_strategies),
            preview_chars=self.preview_chars,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ToolResultOffloadHook":
        """Deserialize the hook, reconstructing its store and offload strategies."""
        init_params = data.get("init_parameters", {})
        if init_params.get("store") is not None:
            init_params["store"] = deserialize_class_instance(init_params["store"])
        if init_params.get("offload_strategies") is not None:
            init_params["offload_strategies"] = cls._deserialize_strategies(init_params["offload_strategies"])
        return default_from_dict(cls, data)

    @staticmethod
    def _serialize_strategies(strategies: dict[str | tuple[str, ...], OffloadPolicy]) -> dict[str, Any]:
        """Serialize the strategies map: tuple keys become JSON-array strings, each policy is serialized by type."""
        return {
            (json.dumps(list(key)) if isinstance(key, tuple) else key): serialize_class_instance(policy)
            for key, policy in strategies.items()
        }

    @staticmethod
    def _deserialize_strategies(data: dict[str, Any]) -> dict[str | tuple[str, ...], OffloadPolicy]:
        """Reverse of `_serialize_strategies`: array-string keys become tuples, policies deserialized by type."""
        decoded: dict[str | tuple[str, ...], OffloadPolicy] = {}
        for raw_key, raw_policy in data.items():
            key = tuple(json.loads(raw_key)) if isinstance(raw_key, str) and raw_key.startswith("[") else raw_key
            decoded[key] = deserialize_class_instance(raw_policy)
        return decoded
