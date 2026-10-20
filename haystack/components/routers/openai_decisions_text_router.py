# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
from typing import Any

from openai import AsyncOpenAI, OpenAI
from openai.types import Decision
from openai.types.decision_create_params import QuestionQuestionParamChoice, QuestionQuestionParamChoiceChoice

from haystack import component
from haystack.utils import Secret

_RESERVED_OUTPUT_NAMES = ("low_confidence", "refused")


@component
class OpenAIDecisionsTextRouter:
    """
    Routes a text to the connection of the label an OpenAI Decisions model picks for it.

    The model answers a single `choice` question over `labels`. The component has one output per label.
    When `min_confidence` is set, it also has a `low_confidence` output that receives the text whenever the
    answer's `confidence` is below the threshold, so uncertain texts can go to a fallback such as an LLM or a human.

    OpenAI can decline to answer a question. In that case, the text goes to the `refused` output,
    independently of `min_confidence`.

    ### Usage example

    ```python
    from haystack.components.routers import OpenAIDecisionsTextRouter

    # Set OPENAI_API_KEY in the environment.
    router = OpenAIDecisionsTextRouter(
        labels={
            "billing": "Payments, invoices, refunds and charges",
            "technical": "Bugs, crashes and errors in the product",
        },
        instructions="Which team should handle this support request?",
        min_confidence=0.6,
    )
    result = router.run(text="I was charged twice for my subscription.")
    # >> {"billing": "I was charged twice for my subscription."}
    ```
    """

    def __init__(
        self,
        labels: list[str] | dict[str, str | None],
        *,
        instructions: str = "Which category does this text belong to?",
        model: str = "gpt-6-luna",
        api_key: Secret = Secret.from_env_var("OPENAI_API_KEY"),
        api_base_url: str | None = None,
        min_confidence: float | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
    ) -> None:
        """
        Creates an OpenAIDecisionsTextRouter.

        :param labels:
            The labels to route between, each becoming an output connection. Pass a dict of label to description to
            tell the model what each label means. Requires between 2 and 255 unique, non-empty string labels.
            `low_confidence` and `refused` are reserved output names.
        :param instructions:
            The question the model answers by picking one of the labels.
            Defaults to "Which category does this text belong to?".
        :param model:
            Name of the Decisions model to use. Defaults to `gpt-6-luna`.
        :param api_key:
            OpenAI API key. Defaults to the `OPENAI_API_KEY` environment variable, resolved during warm-up.
        :param api_base_url:
            API base URL. If `None`, uses the SDK default or `OPENAI_BASE_URL` environment variable.
        :param min_confidence:
            Threshold between 0 and 1 for the answer's `confidence`. Texts below it are sent to the `low_confidence`
            output. If `None`, every non-refused text goes to a label output.
        :param timeout:
            Timeout for OpenAI client calls. If not set, it defaults to either the
            `OPENAI_TIMEOUT` environment variable, or 30 seconds.
        :param max_retries:
            Maximum number of retries to contact OpenAI after an internal error.
            If not set, it defaults to either the `OPENAI_MAX_RETRIES` environment variable, or set to 5.
        :raises ValueError:
            If labels are invalid, duplicated or reserved, or `min_confidence` is outside 0 to 1.
        """
        if not 2 <= len(labels) <= 255 or any(not isinstance(label, str) or not label for label in labels):
            raise ValueError("`labels` must contain between 2 and 255 non-empty string labels.")
        if len(set(labels)) != len(labels):
            raise ValueError("`labels` must be unique.")
        if set(labels).intersection(_RESERVED_OUTPUT_NAMES):
            raise ValueError("`low_confidence` and `refused` are reserved output names and cannot be labels.")
        if min_confidence is not None and not 0 <= min_confidence <= 1:
            raise ValueError("`min_confidence` must be between 0 and 1.")

        self.labels = labels
        self.instructions = instructions
        self.model = model
        self.api_key = api_key
        self.api_base_url = api_base_url
        self.min_confidence = min_confidence
        self.timeout = timeout
        self.max_retries = max_retries
        self._client: OpenAI | None = None
        self._async_client: AsyncOpenAI | None = None

        criteria = labels if isinstance(labels, dict) else dict.fromkeys(labels)
        choices: list[QuestionQuestionParamChoiceChoice] = []
        for label, description in criteria.items():
            choice: QuestionQuestionParamChoiceChoice = {"value": label}
            if description is not None:
                choice["description"] = description
            choices.append(choice)
        self._question: QuestionQuestionParamChoice = {
            "type": "choice",
            "name": "route",
            "instructions": instructions,
            "choices": choices,
        }
        outputs = [*labels, "refused"]
        if min_confidence is not None:
            outputs.append("low_confidence")
        component.set_output_types(self, **dict.fromkeys(outputs, str))

    def _client_kwargs(self) -> dict[str, Any]:
        timeout = self.timeout if self.timeout is not None else float(os.environ.get("OPENAI_TIMEOUT", "30.0"))
        max_retries = (
            self.max_retries if self.max_retries is not None else int(os.environ.get("OPENAI_MAX_RETRIES", "5"))
        )
        return {
            "api_key": self.api_key.resolve_value(),
            "base_url": self.api_base_url,
            "timeout": timeout,
            "max_retries": max_retries,
        }

    def warm_up(self) -> None:
        """Creates the synchronous OpenAI client if needed."""
        if self._client is None:
            self._client = OpenAI(**self._client_kwargs())

    async def warm_up_async(self) -> None:
        """Creates the asynchronous OpenAI client if needed."""
        if self._async_client is None:
            self._async_client = AsyncOpenAI(**self._client_kwargs())

    def close(self) -> None:
        """Releases the synchronous OpenAI client, if initialized."""
        if self._client is not None:
            self._client.close()
            self._client = None

    async def close_async(self) -> None:
        """Releases the asynchronous OpenAI client, if initialized."""
        if self._async_client is not None:
            await self._async_client.close()
            self._async_client = None

    def _route(self, *, text: str, response: Decision) -> dict[str, str]:
        """
        Picks the output for the text from the OpenAI Decisions response.

        :param text:
            The routed text.
        :param response:
            The OpenAI Decisions response for the text.
        :returns:
            A dictionary mapping `refused` to the text if the model declines to answer, or `low_confidence` if
            `min_confidence` is set and the answer's `confidence` is below it, otherwise mapping the picked label
            to the text.
        :raises ValueError:
            If the API returns an unexpected answer instead of a configured label or refusal.
        """
        answer = response.answers[0]
        if answer.type == "refusal":
            return {"refused": text}
        if answer.type != "choice" or not isinstance(answer.choice, str) or answer.choice not in self.labels:
            raise ValueError("The Decisions API returned an unexpected answer instead of a configured label.")
        if self.min_confidence is not None and answer.confidence < self.min_confidence:
            return {"low_confidence": text}
        return {answer.choice: text}

    def run(self, text: str) -> dict[str, str]:
        """
        Routes the text to the output of the label the model picks.

        :param text:
            The text to route.
        :returns:
            A dictionary with a single key, the picked label, `low_confidence`, or `refused`, mapped to the input text.
        :raises TypeError:
            If `text` is not a string.
        :raises ValueError:
            If the API returns an unexpected answer instead of a configured label or refusal.
        """
        if not isinstance(text, str):
            raise TypeError("OpenAIDecisionsTextRouter expects a str as input.")
        self.warm_up()
        # warm_up creates the client; the assertion narrows its optional type for mypy.
        assert self._client is not None
        response = self._client.decisions.create(model=self.model, input=text, questions=[self._question])
        return self._route(text=text, response=response)

    async def run_async(self, text: str) -> dict[str, str]:
        """
        Asynchronously routes the text to the output of the label the model picks.

        This is the asynchronous version of the `run` method with the same parameters and return values.

        :param text:
            The text to route.
        :returns:
            A dictionary with a single key, the picked label, `low_confidence`, or `refused`, mapped to the input text.
        :raises TypeError:
            If `text` is not a string.
        :raises ValueError:
            If the API returns an unexpected answer instead of a configured label or refusal.
        """
        if not isinstance(text, str):
            raise TypeError("OpenAIDecisionsTextRouter expects a str as input.")
        await self.warm_up_async()
        # warm_up_async creates the client; the assertion narrows its optional type for mypy.
        assert self._async_client is not None
        response = await self._async_client.decisions.create(model=self.model, input=text, questions=[self._question])
        return self._route(text=text, response=response)
