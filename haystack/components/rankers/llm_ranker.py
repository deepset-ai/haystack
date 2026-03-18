# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from haystack import Document, component, default_from_dict, default_to_dict, logging
from haystack.components.builders import PromptBuilder
from haystack.components.generators.chat.openai import OpenAIChatGenerator
from haystack.components.generators.chat.types import ChatGenerator
from haystack.core.serialization import component_to_dict
from haystack.dataclasses import ChatMessage
from haystack.utils import deserialize_chatgenerator_inplace
from haystack.utils.misc import _deduplicate_documents, _parse_dict_from_json

logger = logging.getLogger(__name__)


def _default_openai_chat_generator() -> ChatGenerator:
    return OpenAIChatGenerator(
        model="gpt-4.1-mini",
        generation_kwargs={
            "temperature": 0.0,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "document_ranking",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "documents": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {"index": {"type": "integer"}},
                                    "required": ["index"],
                                    "additionalProperties": False,
                                },
                            }
                        },
                        "required": ["documents"],
                        "additionalProperties": False,
                    },
                },
            },
        },
    )


DEFAULT_PROMPT_TEMPLATE = """
You are ranking retrieved documents for relevance to a query.

Return valid JSON only, with this structure:
{
  "documents": [
    {"index": 1}
  ]
}

Rules:
- Rank documents from most relevant to least relevant for answering the query.
- Only include documents that are relevant to the query.
- Do not return or rank documents that are not relevant.
- If none are relevant, return {"documents": []}.
- Use only document indices from the provided documents.
- Do not repeat document indices.
- Do not include explanations or any text outside the JSON object.

Query:
{{ query }}

Documents:
{% for document in documents %}
Document {{ loop.index }}:
content: {{ document.content or "" }}

{% endfor %}
""".strip()


@component
class LLMRanker:
    """
    Ranks documents for a query using a Large Language Model.

    The LLM is expected to return a JSON object containing ranked document indices.

    Usage example:

    ```python
    from haystack import Document
    from haystack.components.generators.chat import OpenAIChatGenerator
    from haystack.components.rankers import LLMRanker

    chat_generator = OpenAIChatGenerator(
        model="gpt-4.1-mini",
        generation_kwargs={
            "temperature": 0.0,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "document_ranking",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "documents": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {"index": {"type": "integer"}},
                                    "required": ["index"],
                                    "additionalProperties": False,
                                },
                            }
                        },
                        "required": ["documents"],
                        "additionalProperties": False,
                    },
                },
            },
        },
    )

    ranker = LLMRanker(chat_generator=chat_generator)

    documents = [
        Document(id="paris", content="Paris is the capital of France."),
        Document(id="berlin", content="Berlin is the capital of Germany."),
    ]

    result = ranker.run(query="capital of Germany", documents=documents)
    print(result["documents"][0].id)
    ```
    """

    def __init__(
        self,
        *,
        chat_generator: ChatGenerator | None = None,
        prompt: str = DEFAULT_PROMPT_TEMPLATE,
        top_k: int = 10,
        raise_on_failure: bool = False,
    ) -> None:
        """
        Initialize the LLMRanker component.

        :param chat_generator:
            The chat generator to use for reranking. If `None`, a default `OpenAIChatGenerator` configured for JSON
            output is used.
        :param prompt:
            Custom prompt template for reranking. The prompt must include exactly the variables `query` and
            `documents` and instruct the LLM to return ranked 1-based document indices as JSON.
        :param top_k:
            The maximum number of documents to return.
        :param raise_on_failure:
            If `True`, raise when generation or response parsing fails. If `False`, log the failure and return the
            input documents in fallback order.
        """
        if top_k <= 0:
            raise ValueError(f"top_k must be > 0, but got {top_k}")

        self.top_k = top_k
        self.raise_on_failure = raise_on_failure
        self.prompt = prompt
        self._prompt_builder = PromptBuilder(template=self.prompt, required_variables=["documents", "query"])
        if set(self._prompt_builder.variables) != {"documents", "query"}:
            raise ValueError("prompt must include exactly the variables 'documents' and 'query'.")

        if chat_generator is None:
            self._chat_generator = _default_openai_chat_generator()
        else:
            self._chat_generator = chat_generator
        self._is_warmed_up = False

    def warm_up(self) -> None:
        """
        Warm up the underlying chat generator.
        """
        if not self._is_warmed_up:
            if hasattr(self._chat_generator, "warm_up"):
                self._chat_generator.warm_up()
            self._is_warmed_up = True

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize this component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            chat_generator=component_to_dict(obj=self._chat_generator, name="chat_generator"),
            prompt=self.prompt,
            top_k=self.top_k,
            raise_on_failure=self.raise_on_failure,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LLMRanker":
        """
        Deserialize this component from a dictionary.

        :param data:
            The dictionary representation of the component.
        :returns:
            The deserialized component instance.
        """
        init_params = data.get("init_parameters", {})
        if init_params.get("chat_generator"):
            deserialize_chatgenerator_inplace(init_params, key="chat_generator")
        return default_from_dict(cls, data)

    @component.output_types(documents=list[Document])
    def run(self, query: str, documents: list[Document], top_k: int | None = None) -> dict[str, list[Document]]:
        """
        Rank documents for a query using an LLM.

        Before ranking, duplicate documents are removed.

        :param query:
            The query used for reranking.
        :param documents:
            Candidate documents to rerank.
        :param top_k:
            The maximum number of documents to return. Overrides the instance's `top_k` if provided.
        :returns:
            A dictionary with the ranked documents under the `documents` key.
        """
        if top_k is not None and top_k <= 0:
            raise ValueError(f"top_k must be > 0, but got {top_k}")

        if not documents:
            return {"documents": []}

        top_k = self.top_k if top_k is None else top_k
        deduplicated_documents = _deduplicate_documents(documents)
        fallback_documents = deduplicated_documents

        if not query.strip():
            logger.warning("Empty query provided to LLMRanker. Returning documents without reranking.")
            return {"documents": fallback_documents}

        if not self._is_warmed_up:
            self.warm_up()

        prompt = self._prompt_builder.run(query=query.strip(), documents=deduplicated_documents)

        try:
            result = self._chat_generator.run(messages=[ChatMessage.from_user(prompt["prompt"])])
        except Exception as exc:
            if self.raise_on_failure:
                raise
            logger.warning(
                "LLMRanker failed during chat generation. Returning fallback order. Error: {error}", error=exc
            )
            return {"documents": fallback_documents}

        try:
            reply_text = self._get_reply_text(result)
            ranked_documents = self._rank_documents_from_reply(reply_text=reply_text, documents=deduplicated_documents)
        except (TypeError, ValueError) as exc:
            if self.raise_on_failure:
                raise
            logger.warning(
                "LLMRanker failed while processing the chat response. Returning fallback order. Error: {error}",
                error=exc,
            )
            return {"documents": fallback_documents}

        return {"documents": ranked_documents[:top_k]}

    @staticmethod
    def _get_reply_text(result: dict[str, Any]) -> str:
        replies = result.get("replies") or []
        if not replies:
            raise ValueError("ChatGenerator returned no replies.")

        reply_text = replies[0].text
        if reply_text is None:
            raise ValueError("ChatGenerator returned a reply without text.")

        return reply_text

    @staticmethod
    def _rank_documents_from_reply(reply_text: str, documents: list[Document]) -> list[Document]:
        parsed_response = _parse_dict_from_json(reply_text, expected_keys=["documents"], raise_on_failure=True)
        ranked_entries = parsed_response["documents"]

        if not isinstance(ranked_entries, list):
            raise TypeError("Expected 'documents' in ranking response to be a list.")

        if not ranked_entries:
            return []

        ranked_documents: list[Document] = []

        for entry in ranked_entries:
            if not isinstance(entry, dict):
                raise TypeError("Expected each ranked document entry to be a JSON object.")

            document_index = entry.get("index")
            if document_index is None:
                continue

            try:
                # LLMs can return numeric indices as strings even when asked for integers.
                document_index = int(document_index)
            except (TypeError, ValueError):
                continue

            # Jinja's `loop.index` is 1-based:
            # https://jinja.palletsprojects.com/en/stable/templates/#for
            if document_index < 1 or document_index > len(documents):
                continue

            document = documents[document_index - 1]
            ranked_documents.append(document)

        if not ranked_documents:
            raise ValueError("Ranking response did not contain any valid document indices.")

        return ranked_documents
