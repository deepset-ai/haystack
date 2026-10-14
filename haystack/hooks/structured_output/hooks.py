# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import json
import re
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, best_match
from pydantic import BaseModel, ValidationError

from haystack import logging
from haystack.components.agents.state.state import State
from haystack.core.serialization import default_from_dict, default_to_dict
from haystack.dataclasses import ChatMessage, ChatRole
from haystack.utils.type_serialization import deserialize_type, serialize_type

logger = logging.getLogger(__name__)

# Meta key marking the correction messages the hook adds to the conversation.
_CORRECTION_META_KEY = "structured_output_correction"

# A markdown code fence wrapping the whole reply, with an optional language tag such as `json`.
_CODE_FENCE_PATTERN = re.compile(r"^```[\w-]*[ \t]*\n(.*?)\n?[ \t]*```$", re.DOTALL)


class StructuredOutputHook:
    """
    Parse the Agent's final text reply as JSON and store it in a `State` key.

    The hook runs at the `on_exit` hook point. When the Agent stops with a text reply (`exit_reason` is `"text"`),
    the hook parses the reply as JSON, validates it against `schema`, and stores the result under `state_key`.
    A markdown code fence around the whole reply (for example ```` ```json ````) is removed before parsing.
    With a Pydantic model as `schema`, the stored value is the validated model dumped to a dict.

    When the reply is not valid JSON or does not match the schema, and retries are left, the hook adds a user
    message describing the error and keeps the Agent running so the model can correct its reply. Retries count
    toward `max_agent_steps`. When no retries are left, or the Agent stops for any reason other than a text reply,
    `None` is stored under `state_key`.

    `state_key` must be declared in the Agent's `state_schema`, which also makes it an output of the Agent.

    The hook works with any chat generator. When the chat generator supports native structured output (for example
    `text_format` on `OpenAIResponsesChatGenerator`), pass the same schema there as well so the model is constrained
    to valid JSON and retries are rarely needed.

    <!-- test-ignore -->
    ```python
    from pydantic import BaseModel

    from haystack.components.agents import Agent
    from haystack.components.generators.chat import OpenAIResponsesChatGenerator
    from haystack.dataclasses import ChatMessage
    from haystack.hooks.structured_output import StructuredOutputHook


    class CityInfo(BaseModel):
        city: str
        country: str


    agent = Agent(
        chat_generator=OpenAIResponsesChatGenerator(generation_kwargs={"text_format": CityInfo}),
        state_schema={"structured_output": {"type": dict}},
        hooks={"on_exit": [StructuredOutputHook(schema=CityInfo)]},
    )

    result = agent.run(messages=[ChatMessage.from_user("Which city is the Eiffel Tower in?")])
    print(result["structured_output"])
    # >> {'city': 'Paris', 'country': 'France'}
    ```
    """

    allowed_hook_points = ("on_exit",)

    def __init__(
        self, *, schema: type[BaseModel] | dict[str, Any], state_key: str = "structured_output", max_retries: int = 2
    ) -> None:
        """
        Create a structured output hook.

        :param schema: A Pydantic model or a JSON schema the final reply is validated against.
        :param state_key: The `State` key the parsed reply is stored under. It must be declared in the Agent's
            `state_schema`. Defaults to `"structured_output"`.
        :param max_retries: How many times the model is asked to correct a reply that is not valid JSON or does not
            match the schema. Defaults to 2. Set it to 0 to store `None` on the first failure.
        :raises ValueError: If `schema` is not a valid JSON schema, or if `max_retries` is negative.
        :raises TypeError: If `schema` is neither a Pydantic model nor a dict.
        """
        if isinstance(schema, dict):
            try:
                Draft202012Validator.check_schema(schema)
            except SchemaError as e:
                raise ValueError(f"`schema` is not a valid JSON schema: {e.message}") from e
            self._validator: Draft202012Validator | None = Draft202012Validator(schema)
            self._model: type[BaseModel] | None = None
            json_schema = schema
        elif isinstance(schema, type) and issubclass(schema, BaseModel):
            self._validator = None
            self._model = schema
            json_schema = schema.model_json_schema()
        else:
            raise TypeError(f"`schema` must be a Pydantic model or a JSON schema dict, got {type(schema).__name__}.")
        if max_retries < 0:
            raise ValueError(f"`max_retries` must be 0 or greater, got {max_retries}.")

        self.schema = schema
        self.state_key = state_key
        self.max_retries = max_retries
        self._schema_text = json.dumps(json_schema, indent=2)

    def run(self, state: State) -> None:
        """
        Parse the final reply and store it, or ask the model to correct it.

        :param state: Agent state holding the conversation and the `exit_reason`.
        :raises ValueError: If `state_key` is not declared in the Agent's `state_schema`.
        """
        if self.state_key not in state.schema:
            raise ValueError(
                f"StructuredOutputHook stores its result under '{self.state_key}', which is not in the Agent's "
                f"`state_schema`. Add it, for example `state_schema={{'{self.state_key}': {{'type': dict}}}}`."
            )

        exit_reason = state.data.get("exit_reason")
        if exit_reason != "text":
            logger.warning(
                "The Agent stopped with exit reason '{exit_reason}' instead of a text reply, so there is no "
                "structured output to parse. Storing None under '{state_key}'.",
                exit_reason=exit_reason,
                state_key=self.state_key,
            )
            state.set(self.state_key, None)
            return

        messages = state.data["messages"]
        parsed, error = self._parse(text=messages[-1].text)
        if error is None:
            state.set(self.state_key, parsed)
            return

        state.set(self.state_key, None)
        retries_used = _count_corrections(messages=messages)
        if retries_used >= self.max_retries:
            logger.warning(
                "The Agent's final reply could not be parsed as structured output after {retries} retries: {error}",
                retries=retries_used,
                error=error,
            )
            return

        # Ask the model to correct its reply and keep the Agent running
        correction = ChatMessage.from_user(
            f"Your previous reply could not be used: {error}\n\n"
            f"Reply again with only JSON that matches this JSON schema:\n{self._schema_text}",
            meta={_CORRECTION_META_KEY: True},
        )
        state.set("messages", [correction])
        state.set("continue_run", True)

    def _parse(self, text: str | None) -> tuple[Any, str | None]:
        """
        Parse and validate a reply.

        :returns: The parsed value and `None` on success, or `None` and a description of the error on failure.
        """
        if not text:
            return None, "The reply has no text."
        try:
            data = json.loads(_strip_code_fence(text=text))
        except json.JSONDecodeError as e:
            return None, f"The reply is not valid JSON: {e}"

        # self._model is not None when the schema is a Pydantic model
        if self._model is not None:
            try:
                model = self._model.model_validate(data)
            except ValidationError as e:
                return None, f"The reply does not match the schema: {e}"
            return model.model_dump(mode="json"), None

        # self._validator is not None when the schema is a JSON schema dict
        if self._validator is not None and (validation_error := best_match(self._validator.iter_errors(data))):
            return None, f"The reply does not match the schema: {validation_error.message}"
        return data, None

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize this hook to a dictionary.

        :returns: Serialized representation of the hook.
        """
        schema = self.schema if isinstance(self.schema, dict) else serialize_type(self.schema)
        return default_to_dict(self, schema=schema, state_key=self.state_key, max_retries=self.max_retries)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StructuredOutputHook":
        """
        Create a hook from its serialized representation.

        :param data: Serialized hook data.
        :returns: The deserialized hook.
        """
        init_params = data.get("init_parameters", {})
        if isinstance(init_params.get("schema"), str):
            data = {**data, "init_parameters": {**init_params, "schema": deserialize_type(init_params["schema"])}}
        return default_from_dict(cls, data=data)


def _count_corrections(messages: list[ChatMessage]) -> int:
    """
    Count the correction messages the hook added since the last message from the user.

    Only the current turn is counted, so corrections from earlier turns of a multi-turn conversation don't use up
    the retries of the current one.
    """
    count = 0
    # Walk back from the newest message
    for message in reversed(messages):
        # Skip all non-user messages, including the assistant's reply and any tool results
        if not message.is_from(ChatRole.USER):
            continue
        # The first untagged user message is the start of the current turn, so stop counting
        if not message.meta.get(_CORRECTION_META_KEY):
            break
        count += 1
    return count


def _strip_code_fence(text: str) -> str:
    """
    Remove a markdown code fence that wraps the whole reply.

    Text with anything before or after the fence is returned unchanged.
    """
    match = _CODE_FENCE_PATTERN.match(text.strip())
    return match.group(1) if match else text
