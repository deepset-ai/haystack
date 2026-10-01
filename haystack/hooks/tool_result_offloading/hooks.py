# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import base64
import json
import mimetypes
from pathlib import Path
from typing import Any

from haystack import logging
from haystack.components.agents.state.state import State
from haystack.components.agents.state.state_utils import replace_values
from haystack.core.serialization import default_from_dict, default_to_dict
from haystack.dataclasses import ChatMessage, FileContent, ImageContent, TextContent
from haystack.hooks.tool_result_offloading.types import OffloadPolicy, ToolResultStore
from haystack.utils.deserialization import deserialize_component_inplace

# Extension used for a binary block whose MIME type is unknown or maps to no known extension.
_FALLBACK_EXTENSION = ".bin"

logger = logging.getLogger(__name__)

# Meta key marking an already-offloaded tool-result message; its value is the list of store references written.
# Stops a second `after_tool` offload hook from offloading the pointer text again, since the pointer is itself a
# tool result.
_OFFLOADED_META_KEY = "tool_result_offloaded"

# Key under which a per-run store override may be supplied via the Agent's `hook_context` (e.g. a request-scoped
# sandbox filesystem).
RESULT_STORE_CONTEXT_KEY = "tool_result_store"


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
    index = len(messages)
    while index > 0 and messages[index - 1].tool_call_result is not None:
        index -= 1
    return index


def _content_block_payload(content_block: TextContent | ImageContent | FileContent) -> str:
    """
    Return the string a content block contributes to the conversation.

    For an image or a file this is the base64 payload, which is what actually occupies the context window.

    :param content_block: The content block to inspect.
    :returns: The content block's text or base64 payload.
    """
    if isinstance(content_block, TextContent):
        return content_block.text
    return content_block.base64_image if isinstance(content_block, ImageContent) else content_block.base64_data


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

    <!-- test-concept -->
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

    Only successful tool output is offloaded; error results are always left in context. Each part of a result is
    written to its own store entry and the pointer says where each one went. Image and file content is only offloaded
    to a store that sets `supports_binary_content`; with a text-only store the result stays in context and a warning
    is logged. Each result is offloaded at most once, even though the hook runs on every tool step.

    The hook keeps no mutable state, so a single instance can be shared across concurrent runs. The constructor
    `store`, however, is shared by every run that does not override it — fine for single-user or local use, but in a
    multi-user server give each run its own isolated store (a per-session directory or sandbox) via `hook_context`
    under the key `RESULT_STORE_CONTEXT_KEY`
    (`agent.run(messages=[...], hook_context={RESULT_STORE_CONTEXT_KEY: per_request_store})`); it overrides the
    constructor store for that run. Isolating the store per run keeps concurrent users from colliding on store keys or
    reading each other's offloaded results — important especially when a bash/read tool is scoped to the store.
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
        :param preview_chars: Number of leading characters of each offloaded text to include in the pointer left in
            the conversation, so the model knows roughly what was offloaded. Image and file blocks are described by
            their MIME type and size instead.
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

        Results are written to the store this run resolves to: a per-run store passed in `state`'s `hook_context`
        under `RESULT_STORE_CONTEXT_KEY` if present, otherwise the store the hook was constructed with. Supply the
        per-run store when calling the Agent, e.g.
        `agent.run(messages=[...], hook_context={RESULT_STORE_CONTEXT_KEY: per_request_store})`. In a multi-user
        server, pass an isolated store per run this way so concurrent users write to separate locations and never
        read each other's results.

        The hook keeps no mutable state, so a single instance is safe to share across concurrent runs; isolation
        comes entirely from giving each run its own store via `hook_context`.

        :param state: The Agent's live `State`. Reads the per-run store from `hook_context` and rewrites the offloaded
            tool-result messages back into `messages`.
        :returns: None. The hook mutates `state` in place.
        """
        messages = state.data.get("messages") or []
        start = _fresh_tool_results_start(messages=messages)
        if start == len(messages):
            return

        # The hook instance is shared across concurrent runs, so a run isolates itself by carrying its own store in
        # `hook_context`.
        hook_context = state.data.get("hook_context") or {}
        store = hook_context.get(RESULT_STORE_CONTEXT_KEY, self.store)

        rewritten: list[ChatMessage] = list(messages[:start])
        changed = False
        for index, message in enumerate(messages[start:]):
            new_message = self._maybe_offload(message=message, store=store, state=state, index=index)
            rewritten.append(new_message)
            changed = changed or new_message is not message

        # Only write back to state when at least one message changed
        if changed:
            state.set(key="messages", value=rewritten, handler_override=replace_values)

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
        handled it), when no policy applies, when the result is empty (no content, or nothing but empty text), when
        the result carries image or file content that `store` cannot store, or when the policy declines to offload.

        Otherwise the result is written to `store` and the message is rebuilt with a pointer in place of the full
        result, preserving its origin and error flag and marking it offloaded. Each part of the result goes to its own
        store entry.

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
        policy = self._policy_for(tool_name=tool_name)

        # If no policy applies, leave the result in context
        if policy is None:
            return message

        # A plain string result is handled as a single text block, so everything below has one shape to work with.
        content_blocks: list[TextContent | ImageContent | FileContent] = (
            [TextContent(text=result.result)] if isinstance(result.result, str) else list(result.result)
        )
        # A result made up of nothing but empty text has nothing worth storing, so it stays in context. `all` also
        # covers a result with no content blocks at all.
        if all(isinstance(content_block, TextContent) and not content_block.text for content_block in content_blocks):
            return message

        # Check whether the store can store binary content before offloading an image or file result. A text-only store
        # leaves the result in context and logs a warning.
        if not getattr(store, "supports_binary_content", False) and not all(
            isinstance(content_block, TextContent) for content_block in content_blocks
        ):
            logger.warning(
                "Tool '{tool}' produced a result with image or file content, but {store} does not support binary "
                "content; leaving the result in context.",
                tool=tool_name,
                store=type(store).__name__,
            )
            return message

        # The policy sizes up the result by the string it occupies in the conversation, which for an image or a file
        # block is its base64 payload.
        payload = "".join(_content_block_payload(content_block=content_block) for content_block in content_blocks)
        if not policy.should_offload(tool_name=tool_name, result=payload, state=state):
            return message

        # Step, tool name, and call id keep results from different tools and steps from colliding. A tool call id is
        # optional, so an id-less call falls back to its position in this step's batch.
        step = state.data.get("step_count", 0)
        prefix = f"{step}_{tool_name}_{result.origin.id or f'call{index}'}"
        references, pointer = self._offload_content_blocks(content_blocks=content_blocks, store=store, prefix=prefix)

        return ChatMessage.from_tool(
            tool_result=pointer,
            origin=result.origin,
            error=result.error,
            meta={**message.meta, _OFFLOADED_META_KEY: references},
        )

    def _offload_content_blocks(
        self, content_blocks: list[TextContent | ImageContent | FileContent], store: ToolResultStore, prefix: str
    ) -> tuple[list[str], str]:
        """
        Write a result's content blocks to the store and build the pointer that replaces them in the conversation.

        Every content block goes to its own store entry. A single block keeps `prefix` as its key and gets a one-line
        pointer; several blocks get position-suffixed keys and one numbered pointer line each.

        :param content_blocks: The result's content blocks, in order.
        :param store: The store to write to.
        :param prefix: The result's store key prefix, as described above.
        :returns: The store references written, and the pointer text for the conversation.
        """
        references: list[str] = []
        descriptions: list[str] = []
        single = len(content_blocks) == 1
        for position, content_block in enumerate(content_blocks):
            reference, description = self._offload_content_block(
                content_block=content_block, store=store, key_prefix=prefix if single else f"{prefix}_{position}"
            )
            references.append(reference)
            descriptions.append(description)

        if len(descriptions) == 1:
            return references, f"Tool result offloaded to {descriptions[0]}"

        numbered = [f"{position}. {description}" for position, description in enumerate(descriptions, start=1)]
        return references, "\n".join([f"Tool result offloaded to {len(descriptions)} files:", *numbered])

    def _offload_content_block(
        self, content_block: TextContent | ImageContent | FileContent, store: ToolResultStore, key_prefix: str
    ) -> tuple[str, str]:
        """
        Write a single content block to the store and describe where it went.

        :param content_block: The content block to offload.
        :param store: The store to write to.
        :param key_prefix: The content block's store key without its extension.
        :returns: The store reference the content block was written to, and a one-line description for the pointer.
        """
        if isinstance(content_block, TextContent):
            text = content_block.text
            reference = store.write(key=f"{key_prefix}.txt", content=text)
            # An ellipsis marks a preview that was cut short, so the model can tell it is not the whole text.
            preview = f"{text[: self.preview_chars]}{'...' if len(text) > self.preview_chars else ''}"
            return reference, f"text ({len(text)} characters) at '{reference}'. Preview: {preview}"

        if isinstance(content_block, ImageContent):
            data = base64.b64decode(content_block.base64_image)
            label = content_block.mime_type or "image"
            filename = None
        else:
            data = base64.b64decode(content_block.base64_data)
            label = content_block.mime_type or "file"
            filename = content_block.filename

        # What the tool called the file wins over its MIME type. Only the suffix is taken.
        mime_extension = mimetypes.guess_extension(content_block.mime_type) if content_block.mime_type else None
        extension = (Path(filename).suffix if filename else "") or mime_extension or _FALLBACK_EXTENSION
        reference = store.write(key=f"{key_prefix}{extension}", content=data)

        named = f" named '{filename}'" if filename else ""
        return reference, f"{label}{named} ({len(data)} bytes) at '{reference}'"

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize the hook, including its store and per-tool offload strategies.

        :returns: A dictionary representation of the hook.
        """
        return default_to_dict(
            self,
            store=self.store.to_dict(),
            offload_strategies=_serialize_offload_strategies(strategies=self.offload_strategies),
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
            init_params["offload_strategies"] = _deserialize_offload_strategies(data=init_params["offload_strategies"])
        return default_from_dict(cls, data)
