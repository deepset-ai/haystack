# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from typing import Any

from openai import APIError, AsyncOpenAI, OpenAI
from openai.types import Decision
from openai.types.decision_create_params import Question

from haystack import Document, component, logging
from haystack.utils import Secret

logger = logging.getLogger(__name__)

_QUESTION_TYPES = ("choice", "score", "predicate")
_ERROR_FIELD = "classification_error"


@component
class OpenAIDecisionsDocumentClassifier:
    """
    Answers typed questions about each document with an OpenAI Decisions model and stores the answers in its metadata.

    Each question has one of three types:
    - `choice`: picks one value from `choices`, a list of objects with a `value` and an optional `description`.
    - `score`: rates the text on the ordered `levels`, a list of objects with a `label` and an optional `description`.
    - `predicate`: returns the probability that the yes/no question in `instructions` is true.

    Answers are stored under `meta[metadata_field]`, keyed by question name. Questions are dicts in the OpenAI format.
    See the [OpenAI documentation](https://developers.openai.com/api/docs/guides/decisions) for the full question and
    answer format. A refusal is stored with `type="refusal"`, preserving the answers to the other questions.

    Documents without text to classify are returned in `failed_documents` with the reason in
    `meta["classification_error"]`. So are documents whose request failed, unless `raise_on_failure` is `True`.

    ### Usage example

    ```python
    from haystack import Document
    from haystack.components.classifiers import OpenAIDecisionsDocumentClassifier

    # Set OPENAI_API_KEY in the environment.
    classifier = OpenAIDecisionsDocumentClassifier(
        questions=[
            {
                "name": "department",
                "type": "choice",
                "instructions": "Which department should handle this ticket?",
                "choices": [{"value": "billing"}, {"value": "technical"}, {"value": "sales"}],
            },
            {
                "name": "urgency",
                "type": "score",
                "instructions": "How urgent is this ticket?",
                "levels": [{"label": "can wait"}, {"label": "this week"}, {"label": "today"}],
            },
            {"name": "refund", "type": "predicate", "instructions": "Is the customer asking for a refund?"},
        ]
    )

    result = classifier.run(documents=[Document(content="I was charged twice, please refund me today.")])
    print(result["documents"][0].meta["openai_decisions"])
    # Example response (values may vary):
    # >> {
    #     "department": {"type": "choice", "choice": "billing", "confidence": 0.95,
    #      "probabilities": [{"value": "billing", "probability": 0.97},
    #                        {"value": "technical", "probability": 0.02},
    #                        {"value": "sales", "probability": 0.01}]},
    #     "urgency": {"type": "score", "score": 1.95, "confidence": 0.94,
    #      "probabilities": [{"label": "can wait", "value": 0, "probability": 0.01},
    #                        {"label": "this week", "value": 1, "probability": 0.03},
    #                        {"label": "today", "value": 2, "probability": 0.96}]},
    #     "refund": {"type": "predicate", "probability": 0.95},
    # }
    print(result["failed_documents"])
    # >> []
    ```
    """

    def __init__(
        self,
        questions: list[Question],
        *,
        model: str = "gpt-6-luna",
        api_key: Secret = Secret.from_env_var("OPENAI_API_KEY"),
        api_base_url: str | None = None,
        classification_field: str | None = None,
        metadata_field: str = "openai_decisions",
        timeout: float | None = None,
        max_retries: int | None = None,
        max_workers: int = 3,
        raise_on_failure: bool = False,
    ) -> None:
        """
        Creates an OpenAIDecisionsDocumentClassifier.

        :param questions:
            Questions to answer for every document. Each item is a dict in the OpenAI format with a `type` (`choice`,
            `score` or `predicate`), `instructions`, and for `choice` and `score` questions, `choices` or `levels`,
            respectively. Each question must have a non-empty, unique string `name`, used as its answer's metadata key.
        :param model:
            The name of the Decisions model to use.
        :param api_key:
            The OpenAI API key. You can set it with an environment variable `OPENAI_API_KEY`, or pass with this
            parameter during initialization.
        :param api_base_url:
            Overrides the default base URL for all HTTP requests.
        :param classification_field:
            Name of the document's metadata field to classify. If `None`, `Document.content` is classified.
        :param metadata_field:
            Name of the metadata field the answers are written to.
        :param timeout:
            Timeout in seconds for each request attempt. If `None`, uses the OpenAI SDK default.
        :param max_retries:
            Maximum number of retries after a failed request attempt. If `None`, uses the OpenAI SDK default.
        :param max_workers:
            Maximum number of worker threads in `run` and of concurrent requests in `run_async`. Must be at least 1.
        :param raise_on_failure:
            If `True`, the first failed request raises its error. If `False`, documents whose request failed are
            returned in `failed_documents`.
        :raises ValueError:
            If `questions` is empty, a question has a missing, empty, non-string or duplicate `name`, an unknown
            `type`, a `choice` or `score` question has no `choices` or `levels`, respectively, or `max_workers` is
            smaller than 1.
        """
        if not questions:
            raise ValueError("`questions` must contain at least one question.")
        names: set[str] = set()
        for index, question in enumerate(questions):
            name = question.get("name")
            if not isinstance(name, str) or not name:
                msg = f"Question at index {index} must have a non-empty string `name`."
                raise ValueError(msg)
            if name in names:
                msg = f"Question name '{name}' is duplicated; names must be unique."
                raise ValueError(msg)
            names.add(name)
            if question.get("type") not in _QUESTION_TYPES:
                msg = f"Question '{name}' has type {question.get('type')!r}; use one of {_QUESTION_TYPES}."
                raise ValueError(msg)
            for question_type, field in (("choice", "choices"), ("score", "levels")):
                if question["type"] == question_type and not question.get(field):
                    msg = f"Question '{name}' of type '{question_type}' needs `{field}`."
                    raise ValueError(msg)
        if max_workers < 1:
            msg = f"`max_workers` must be at least 1, got {max_workers}."
            raise ValueError(msg)

        self.questions = questions
        self.model = model
        self.api_key = api_key
        self.api_base_url = api_base_url
        self.classification_field = classification_field
        self.metadata_field = metadata_field
        self.timeout = timeout
        self.max_retries = max_retries
        self.max_workers = max_workers
        self.raise_on_failure = raise_on_failure
        self._client: OpenAI | None = None
        self._async_client: AsyncOpenAI | None = None

    def _client_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"api_key": self.api_key.resolve_value(), "base_url": self.api_base_url}
        if self.timeout is not None:
            kwargs["timeout"] = self.timeout
        if self.max_retries is not None:
            kwargs["max_retries"] = self.max_retries
        return kwargs

    def warm_up(self) -> None:
        """
        Creates the synchronous OpenAI client.
        """
        if self._client is None:
            self._client = OpenAI(**self._client_kwargs())

    async def warm_up_async(self) -> None:
        """
        Creates the asynchronous OpenAI client.
        """
        if self._async_client is None:
            self._async_client = AsyncOpenAI(**self._client_kwargs())

    def close(self) -> None:
        """
        Releases the synchronous OpenAI client.
        """
        if self._client is not None:
            self._client.close()
            self._client = None

    async def close_async(self) -> None:
        """
        Releases the asynchronous OpenAI client.
        """
        if self._async_client is not None:
            await self._async_client.close()
            self._async_client = None

    def _succeed(self, document: Document, response: Decision) -> tuple[Document, bool]:
        """
        Adds the answers from an OpenAI response to a copy of the document.

        :param document:
            The classified document.
        :param response:
            The OpenAI response for the document.
        :returns:
            A copy of the document with the answers under `meta[metadata_field]` and no `classification_error`, and
            `True` to mark it as classified.
        """
        answers = {answer.name: answer.model_dump(mode="json", exclude={"name"}) for answer in response.answers}
        # Drop the error left by an earlier failed run
        meta = {key: value for key, value in document.meta.items() if key != _ERROR_FIELD}
        return replace(document, meta={**meta, self.metadata_field: answers}), True

    def _classify(self, client: OpenAI, document: Document) -> tuple[Document, bool]:
        """
        Sends one document to the OpenAI API and adds the answers to a copy of it.

        :param client:
            The synchronous OpenAI client.
        :param document:
            The document to classify.
        :returns:
            A copy of the document with the answers under `meta[metadata_field]` and `True`, or a copy with the reason
            under `meta["classification_error"]` and `False` if the document has no text or its request failed.
        :raises APIError:
            If the request fails and `raise_on_failure` is `True`.
        """
        text = document.content if self.classification_field is None else document.meta.get(self.classification_field)
        if not isinstance(text, str) or not text:
            return replace(document, meta={**document.meta, _ERROR_FIELD: "Document has no text to classify."}), False
        try:
            response = client.decisions.create(model=self.model, input=text, questions=self.questions)
        except APIError as error:
            if self.raise_on_failure:
                raise
            logger.warning("Classifying document {document_id} failed: {error}", document_id=document.id, error=error)
            return replace(document, meta={**document.meta, _ERROR_FIELD: str(error)}), False
        return self._succeed(document=document, response=response)

    async def _classify_async(
        self, client: AsyncOpenAI, document: Document, semaphore: asyncio.Semaphore
    ) -> tuple[Document, bool]:
        """
        Asynchronously sends one document to the OpenAI API and adds the answers to a copy of it.

        :param client:
            The asynchronous OpenAI client.
        :param document:
            The document to classify.
        :param semaphore:
            Limits how many requests run at once.
        :returns:
            A copy of the document with the answers under `meta[metadata_field]` and `True`, or a copy with the reason
            under `meta["classification_error"]` and `False` if the document has no text or its request failed.
        :raises APIError:
            If the request fails and `raise_on_failure` is `True`.
        """
        text = document.content if self.classification_field is None else document.meta.get(self.classification_field)
        if not isinstance(text, str) or not text:
            return replace(document, meta={**document.meta, _ERROR_FIELD: "Document has no text to classify."}), False
        try:
            async with semaphore:
                response = await client.decisions.create(model=self.model, input=text, questions=self.questions)
        except APIError as error:
            if self.raise_on_failure:
                raise
            logger.warning("Classifying document {document_id} failed: {error}", document_id=document.id, error=error)
            return replace(document, meta={**document.meta, _ERROR_FIELD: str(error)}), False
        return self._succeed(document=document, response=response)

    @component.output_types(documents=list[Document], failed_documents=list[Document])
    def run(self, documents: list[Document]) -> dict[str, list[Document]]:
        """
        Answers the questions for each document and adds the answers to its metadata.

        Each document is sent as its own request, with up to `max_workers` requests running at once.

        :param documents:
            Documents to classify.
        :returns:
            A dictionary with the following keys:
            - `documents`: The classified documents. Each has a `metadata_field` entry mapping each question name
              to its answer.
            - `failed_documents`: The documents that have no text to classify or whose request failed, with the
              reason in `meta["classification_error"]`.
        :raises APIError:
            If a request fails and `raise_on_failure` is `True`.
        """
        self.warm_up()
        assert self._client is not None  # noqa: S101  # mypy: the client is created by warm_up above

        client = self._client
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(executor.map(lambda document: self._classify(client=client, document=document), documents))
        return {
            "documents": [document for document, succeeded in results if succeeded],
            "failed_documents": [document for document, succeeded in results if not succeeded],
        }

    @component.output_types(documents=list[Document], failed_documents=list[Document])
    async def run_async(self, documents: list[Document]) -> dict[str, list[Document]]:
        """
        Asynchronously answers the questions for each document and adds the answers to its metadata.

        This is the asynchronous version of the `run` method with the same parameters and return values. Up to
        `max_workers` requests run concurrently.

        :param documents:
            Documents to classify.
        :returns:
            A dictionary with the following keys:
            - `documents`: The classified documents. Each has a `metadata_field` entry mapping each question name
              to its answer.
            - `failed_documents`: The documents that have no text to classify or whose request failed, with the
              reason in `meta["classification_error"]`.
        :raises APIError:
            If a request fails and `raise_on_failure` is `True`.
        """
        await self.warm_up_async()
        assert self._async_client is not None  # noqa: S101  # mypy: the client is created by warm_up_async above

        # Bound concurrency to max_workers, mirroring the ThreadPoolExecutor in the sync `run`
        semaphore = asyncio.Semaphore(self.max_workers)
        tasks = [
            asyncio.create_task(self._classify_async(client=self._async_client, document=document, semaphore=semaphore))
            for document in documents
        ]
        try:
            results = await asyncio.gather(*tasks)
        except Exception:
            # We only expect to get here when `raise_on_failure` is `True`, since `_classify_async` otherwise records a
            # failed request in `failed_documents` instead of raising. Cancel the remaining requests before re-raising.
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        return {
            "documents": [document for document, succeeded in results if succeeded],
            "failed_documents": [document for document, succeeded in results if not succeeded],
        }
