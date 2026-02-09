# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import json
from typing import Any

from haystack import default_from_dict, default_to_dict, logging
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.components.generators.chat.openai import OpenAIChatGenerator
from haystack.components.generators.chat.types import ChatGenerator
from haystack.core.component import component
from haystack.core.serialization import component_to_dict
from haystack.dataclasses.chat_message import ChatMessage
from haystack.utils.deserialization import deserialize_chatgenerator_inplace

logger = logging.getLogger(__name__)


DEFAULT_PROMPT_TEMPLATE = """
You are part of an information system that processes user queries for retrieval.
You have to expand a given query into {{ n_expansions }} queries that are
semantically similar to improve retrieval recall.

Structure:
Follow the structure shown below in examples to generate expanded queries.

Examples:
1.  Query: "climate change effects"
    {"queries": ["impact of climate change", "consequences of global warming", "effects of environmental changes"]}

2.  Query: "machine learning algorithms"
    {"queries": ["neural networks", "clustering techniques", "supervised learning methods", "deep learning models"]}

3.  Query: "open source NLP frameworks"
    {"queries": ["natural language processing tools", "free nlp libraries", "open-source NLP platforms"]}

Guidelines:
- Generate queries that use different words and phrasings
- Include synonyms and related terms
- Maintain the same core meaning and intent
- Make queries that are likely to retrieve relevant information the original might miss
- Focus on variations that would work well with keyword-based search
- Respond in the same language as the input query

Your Task:
Query: "{{ query }}"

You *must* respond with a JSON object containing a "queries" array with the expanded queries.
Example: {"queries": ["query1", "query2", "query3"]}"""


@component
class QueryExpander:
    """
    A component that returns a list of semantically similar queries to improve retrieval recall in RAG systems.

    The component uses a chat generator to expand queries. The chat generator is expected to return a JSON response
    with the following structure:
    ```json
    {"queries": ["expanded query 1", "expanded query 2", "expanded query 3"]}
    ```

    ### Usage example

    ```python
    from haystack.components.generators.chat.openai import OpenAIChatGenerator
    from haystack.components.query import QueryExpander

    expander = QueryExpander(
        chat_generator=OpenAIChatGenerator(model="gpt-4.1-mini"),
        n_expansions=3
    )

    result = expander.run(query="green energy sources")
    print(result["queries"])
    # Output: ['alternative query 1', 'alternative query 2', 'alternative query 3', 'green energy sources']
    # Note: Up to 3 additional queries + 1 original query (if include_original_query=True)

    # To control total number of queries:
    expander = QueryExpander(n_expansions=2, include_original_query=True)  # Up to 3 total
    # or
    expander = QueryExpander(n_expansions=3, include_original_query=False)  # Exactly 3 total
    ```
    """

    def __init__(
        self,
        *,
        chat_generator: ChatGenerator | None = None,
        prompt_template: str | None = None,
        n_expansions: int = 4,
        include_original_query: bool = True,
    ) -> None:
        """
        Initialize the QueryExpander component.

        :param chat_generator: The chat generator component to use for query expansion.
            If None, a default OpenAIChatGenerator with gpt-4.1-mini model is used.
        :param prompt_template: Custom [PromptBuilder](https://docs.haystack.deepset.ai/docs/promptbuilder)
            template for query expansion. The template should instruct the LLM to return a JSON response with the
            structure: `{"queries": ["query1", "query2", "query3"]}`. The template should include 'query' and
            'n_expansions' variables.
        :param n_expansions: Number of alternative queries to generate (default: 4).
        :param include_original_query: Whether to include the original query in the output.
        """
        if n_expansions <= 0:
            raise ValueError("n_expansions must be positive")

        self.n_expansions = n_expansions
        self.include_original_query = include_original_query

        if chat_generator is None:
            self.chat_generator: ChatGenerator = OpenAIChatGenerator(
                model="gpt-4.1-mini",
                generation_kwargs={
                    "temperature": 0.7,
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "query_expansion",
                            "schema": {
                                "type": "object",
                                "properties": {"queries": {"type": "array", "items": {"type": "string"}}},
                                "required": ["queries"],
                                "additionalProperties": False,
                            },
                        },
                    },
                    "seed": 42,
                },
            )
        else:
            self.chat_generator = chat_generator

        self._is_warmed_up = False
        self.prompt_template = prompt_template or DEFAULT_PROMPT_TEMPLATE

        # Check if required variables are present in the template
        if "query" not in self.prompt_template:
            logger.warning(
                "The prompt template does not contain the 'query' variable. This may cause issues during execution."
            )
        if "n_expansions" not in self.prompt_template:
            logger.warning(
                "The prompt template does not contain the 'n_expansions' variable. "
                "This may cause issues during execution."
            )

        self._prompt_builder = PromptBuilder(
            template=self.prompt_template, required_variables=["n_expansions", "query"]
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :return: Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            chat_generator=component_to_dict(self.chat_generator, name="chat_generator"),
            prompt_template=self.prompt_template,
            n_expansions=self.n_expansions,
            include_original_query=self.include_original_query,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "QueryExpander":
        """
        Deserializes the component from a dictionary.

        :param data: Dictionary with serialized data.
        :return: Deserialized component.
        """
        init_params = data.get("init_parameters", {})

        deserialize_chatgenerator_inplace(init_params, key="chat_generator")

        return default_from_dict(cls, data)

    @component.output_types(queries=list[str])
    def run(self, query: str, n_expansions: int | None = None) -> dict[str, list[str]]:
        """
        Expand the input query into multiple semantically similar queries.

        The language of the original query is preserved in the expanded queries.

        :param query: The original query to expand.
        :param n_expansions: Number of additional queries to generate (not including the original).
            If None, uses the value from initialization. Can be 0 to generate no additional queries.
        :return: Dictionary with "queries" key containing the list of expanded queries.
            If include_original_query=True, the original query will be included in addition
            to the n_expansions alternative queries.
        :raises ValueError: If n_expansions is not positive (less than or equal to 0).
        """

        if not self._is_warmed_up:
            self.warm_up()

        response = {"queries": [query] if self.include_original_query else []}

        if not query.strip():
            logger.warning("Empty query provided to QueryExpander")
            return response

        expansion_count = n_expansions if n_expansions is not None else self.n_expansions
        if expansion_count <= 0:
            raise ValueError("n_expansions must be positive")

        try:
            prompt_result = self._prompt_builder.run(query=query.strip(), n_expansions=expansion_count)
            generator_result = self.chat_generator.run(messages=[ChatMessage.from_user(prompt_result["prompt"])])

            if not generator_result.get("replies") or len(generator_result["replies"]) == 0:
                logger.warning("ChatGenerator returned no replies for query: {query}", query=query)
                return response

            expanded_text = generator_result["replies"][0].text.strip()
            expanded_queries = self._parse_expanded_queries(expanded_text)

            # Limit the number of expanded queries to the requested amount
            if len(expanded_queries) > expansion_count:
                logger.warning(
                    "Generated {generated_count} queries but only {requested_count} were requested. "
                    "Truncating to the first {requested_count} queries. ",
                    generated_count=len(expanded_queries),
                    requested_count=expansion_count,
                )
                expanded_queries = expanded_queries[:expansion_count]

            # Add original query if requested and remove duplicates
            if self.include_original_query:
                expanded_queries_lower = [q.lower() for q in expanded_queries]
                if query.lower() not in expanded_queries_lower:
                    expanded_queries.append(query)

            response["queries"] = expanded_queries
            return response

        except Exception as e:
            # Fallback: return original query to maintain pipeline functionality
            logger.error("Failed to expand query {query}: {error}", query=query, error=str(e))
            return response

    def warm_up(self):
        """
        Warm up the LLM provider component.
        """
        if not self._is_warmed_up:
            if hasattr(self.chat_generator, "warm_up"):
                self.chat_generator.warm_up()
            self._is_warmed_up = True

    @staticmethod
    def _parse_expanded_queries(generator_response: str) -> list[str]:
        """
        Parse the generator response to extract individual expanded queries.

        :param generator_response: The raw text response from the generator.
        :return: List of parsed expanded queries.
        """
        if not generator_response.strip():
            return []

        try:
            parsed = json.loads(generator_response)
            if not isinstance(parsed, dict) or "queries" not in parsed:
                logger.warning(
                    "Generator response is not a JSON object containing a 'queries' array: {response}",
                    response=generator_response[:100],
                )
                return []

            queries = []
            for item in parsed["queries"]:
                if isinstance(item, str) and item.strip():
                    queries.append(item.strip())
                else:
                    logger.warning("Skipping non-string or empty query in response: {item}", item=item)

            return queries

        except json.JSONDecodeError as e:
            logger.warning(
                "Failed to parse JSON response: {error}. Response: {response}",
                error=str(e),
                response=generator_response[:100],
            )
            return []
