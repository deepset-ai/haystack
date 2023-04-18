import logging
import os
from typing import List, Optional, Tuple, Union

from haystack import Document
from haystack.environment import HAYSTACK_REMOTE_API_TIMEOUT_SEC
from haystack.nodes.answer_generator import BaseGenerator
from haystack.nodes.prompt import PromptTemplate
from haystack.utils.openai_utils import (
    load_openai_tokenizer,
    openai_request,
    count_openai_tokens,
    _openai_text_completion_tokenization_details,
    _check_openai_finish_reason,
)

logger = logging.getLogger(__name__)

OPENAI_TIMEOUT = float(os.environ.get(HAYSTACK_REMOTE_API_TIMEOUT_SEC, 30))


class OpenAIAnswerGenerator(BaseGenerator):
    """
    Uses the GPT-3 models from the OpenAI API to generate Answers based on the Documents it receives.
    The Documents can come from a Retriever or you can supply them manually.

    To use this Node, you need an API key from an active OpenAI account. You can sign-up for an account
    on the [OpenAI API website](https://openai.com/api/).
    """

    def __init__(
        self,
        api_key: str,
        azure_base_url: Optional[str] = None,
        azure_deployment_name: Optional[str] = None,
        model: str = "text-davinci-003",
        max_tokens: int = 50,
        api_version: str = "2022-12-01",
        top_k: int = 5,
        temperature: float = 0.2,
        presence_penalty: float = 0.1,
        frequency_penalty: float = 0.1,
        examples_context: Optional[str] = None,
        examples: Optional[List[List[str]]] = None,
        stop_words: Optional[List[str]] = None,
        progress_bar: bool = True,
        prompt_template: Optional[PromptTemplate] = None,
        context_join_str: str = " ",
    ):
        """
        :param api_key: Your API key from OpenAI. It is required for this node to work.
        :param azure_base_url: The base URL for the Azure OpenAI API. If not supplied, Azure OpenAI API will not be used.
                               This parameter is an OpenAI Azure endpoint, usually in the form `https://<your-endpoint>.openai.azure.com'

        :param azure_deployment_name: The name of the Azure OpenAI API deployment. If not supplied, Azure OpenAI API
                                     will not be used.
        :param model: ID of the engine to use for generating the answer. You can select one of `"text-ada-001"`,
                     `"text-babbage-001"`, `"text-curie-001"`, or `"text-davinci-003"`
                     (from worst to best and from cheapest to most expensive). For more information about the models,
                     refer to the [OpenAI Documentation](https://platform.openai.com/docs/models/gpt-3).
        :param max_tokens: The maximum number of tokens reserved for the generated Answer.
                           A higher number allows for longer answers without exceeding the max prompt length of the OpenAI model.
                           A lower number allows longer prompts with more documents passed as context, but the generated answer might be cut after max_tokens.
        :param api_version: The version of the Azure OpenAI API to use. The default is `2022-12-01` version.
        :param top_k: Number of generated Answers.
        :param temperature: What sampling temperature to use. Higher values mean the model will take more risks and
                            value 0 (argmax sampling) works better for scenarios with a well-defined Answer.
        :param presence_penalty: Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they have already appeared
                                 in the text. This increases the model's likelihood to talk about new topics. For more information about frequency and presence penalties, see
                                 [parameter details in OpenAI](https://platform.openai.com/docs/api-reference/parameter-details).
        :param frequency_penalty: Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing
                                  frequency in the text so far, decreasing the model's likelihood to repeat the same line
                                  verbatim.
                                  [See more information about frequency and presence penalties.](https://platform.openai.com/docs/api-reference/parameter-details)
        :param examples_context: A text snippet containing the contextual information used to generate the Answers for
                                 the examples you provide.
                                 If not supplied, the default from OpenAI API docs is used:
                                 `"In 2017, U.S. life expectancy was 78.6 years."`
        :param examples: List of (question, answer) pairs that helps steer the model towards the tone and answer
                         format you'd like. We recommend adding 2 to 3 examples.
                         If not supplied, the default from OpenAI API docs is used:
                         `[["Q: What is human life expectancy in the United States?", "A: 78 years."]]`
        :param stop_words: Up to four sequences where the API stops generating further tokens. The returned text does
                           not contain the stop sequence.
                           If you don't provide any stop words, the default value from OpenAI API docs is used: `["\n", "<|endoftext|>"]`.
        :param prompt_template: A PromptTemplate that tells the model how to generate answers given a
             `context` and `query` supplied at runtime. The `context` is automatically constructed at runtime from a
            list of provided documents. Use `example_context` and a list of `examples` to provide
            the model with examples to steer it towards the tone and answer format you would like.
            If not supplied, the default prompt template is:
            ```python
                PromptTemplate(
                    name="question-answering-with-examples",
                    prompt_text="Please answer the question according to the above context."
                                "\n===\nContext: {examples_context}\n===\n{examples}\n\n"
                                "===\nContext: {context}\n===\n{query}",
                )
            ```
            To learn how variables, such as'{context}', are substituted in the `prompt_text`, see
            [PromptTemplate](https://docs.haystack.deepset.ai/docs/prompt_node#template-structure).
        :param context_join_str: The separation string used to join the input documents to create the context
            used by the PromptTemplate.
        """
        super().__init__(progress_bar=progress_bar)
        if (examples is None and examples_context is not None) or (examples is not None and examples_context is None):
            logger.warning(
                "If providing examples or examples_context, we recommend providing both of them "
                "so the examples correctly refer to the examples_context."
            )
        if examples_context is None:
            examples_context = "In 2017, U.S. life expectancy was 78.6 years."
        if examples is None:
            examples = [["Q: What is human life expectancy in the United States?", "A: 78 years."]]
        if stop_words is None:
            stop_words = ["\n", "<|endoftext|>"]
        if prompt_template is None:
            prompt_template = PromptTemplate(
                name="question-answering-with-examples",
                prompt_text="Please answer the question according to the above context."
                "\n===\nContext: {examples_context}\n===\n{examples}\n\n"
                "===\nContext: {context}\n===\n{query}",
            )
        else:
            # Check for required prompts
            required_params = ["context", "query"]
            if not all(p in prompt_template.prompt_params for p in required_params):
                raise ValueError(
                    "The OpenAIAnswerGenerator requires a PromptTemplate that has `context` and "
                    "`query` in its `prompt_params`. Supply a different `prompt_template` or "
                    "use the default one."
                )

            # Check for unsupported prompt parameters
            optional_params = ["examples_context", "examples"]
            unknown_params = []
            for p in prompt_template.prompt_params:
                if p not in set(required_params + optional_params):
                    unknown_params.append(p)
            if len(unknown_params) > 1:
                raise ValueError(
                    f"The provided PromptTemplate has the prompt parameters, {unknown_params}, that are not supported "
                    f"by the OpenAIAnswerGenerator. The only prompt parameters that are supported are "
                    f"`examples_context`, `examples`, `context`, and `query`."
                )

        self.api_key = api_key
        self.azure_base_url = azure_base_url
        self.azure_deployment_name = azure_deployment_name
        self.api_version = api_version
        self.model = model
        self.max_tokens = max_tokens
        self.top_k = top_k
        self.temperature = temperature
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.examples_context = examples_context
        self.examples = examples
        self.stop_words = stop_words
        self.prompt_template = prompt_template
        self.context_join_str = context_join_str
        self.using_azure = self.azure_deployment_name is not None and self.azure_base_url is not None

        tokenizer_name, max_tokens_limit = _openai_text_completion_tokenization_details(model_name=self.model)

        self.MAX_TOKENS_LIMIT = max_tokens_limit
        self._tokenizer = load_openai_tokenizer(tokenizer_name=tokenizer_name)

    def predict(
        self,
        query: str,
        documents: List[Document],
        top_k: Optional[int] = None,
        max_tokens: Optional[int] = None,
        timeout: Union[float, Tuple[float, float]] = OPENAI_TIMEOUT,
    ):
        """
        Use the loaded QA model to generate Answers for a query based on the Documents it receives.

        Returns dictionaries containing Answers.
        Note that OpenAI doesn't return scores for those Answers.

        Example:
        ```python
        {
            'query': 'Who is the father of Arya Stark?',
            'answers':[Answer(
                         'answer': 'Eddard,',
                         'score': None,
                         ),...
                      ]
        }
        ```

        :param query: The query you want to provide. It's a string.
        :param documents: List of Documents in which to search for the Answer.
        :param top_k: The maximum number of Answers to return.
        :param max_tokens: The maximum number of tokens the generated Answer can have.
        :param timeout: How many seconds to wait for the server to send data before giving up,
            as a float, or a :ref:`(connect timeout, read timeout) <timeouts>` tuple.
            Defaults to 10 seconds.
        :return: Dictionary containing query and Answers.
        """
        if top_k is None:
            top_k = self.top_k

        # convert input to OpenAI format
        prompt, input_docs = self._build_prompt_within_max_length(query=query, documents=documents)
        logger.debug("Prompt being sent to OpenAI API with prompt %s.", prompt)

        payload = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": max_tokens or self.max_tokens,
            "stop": self.stop_words,
            "n": top_k,
            "temperature": self.temperature,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
        }
        if self.using_azure:
            url = f"{self.azure_base_url}/openai/deployments/{self.azure_deployment_name}/completions?api-version={self.api_version}"
        else:
            url = "https://api.openai.com/v1/completions"

        headers = {"Content-Type": "application/json"}
        if self.using_azure:
            headers["api-key"] = self.api_key
        else:
            headers["Authorization"] = f"Bearer {self.api_key}"

        res = openai_request(url=url, headers=headers, payload=payload, timeout=timeout)
        _check_openai_finish_reason(result=res, payload=payload)
        generated_answers = [ans["text"] for ans in res["choices"]]
        answers = self._create_answers(generated_answers, input_docs, prompt=prompt)
        result = {"query": query, "answers": answers}
        return result

    @staticmethod
    def _create_context(documents: List[Document], join_str: str = " ") -> str:
        """Join the documents to create a single context to be used in the PromptTemplate."""
        doc_contents = [doc.content for doc in documents]
        # We reverse the docs to put the most relevant documents at the bottom of the context
        context = join_str.join(reversed(doc_contents))
        return context

    def _fill_prompt(self, query: str, documents: List[Document]) -> str:
        """Fills in the `prompt_template` with its `prompt_params` and returns the full prompt."""
        example_prompts = "\n---\n".join([f"{query}\n{answer}" for query, answer in self.examples])
        qa_prompt = f"Q: {query}\nA:"

        kwargs = {"context": self._create_context(documents, join_str=self.context_join_str), "query": qa_prompt}
        if (
            "examples_context" in self.prompt_template.prompt_params
            and "examples" in self.prompt_template.prompt_params
        ):
            kwargs["examples_context"] = self.examples_context
            kwargs["examples"] = example_prompts
        full_prompt = next(self.prompt_template.fill(**kwargs))
        return full_prompt

    def _build_prompt_within_max_length(self, query: str, documents: List[Document]) -> Tuple[str, List[Document]]:
        """
        Builds the prompt for the GPT-3 model so that it can generate an Answer. If the prompt is too long based on the
        MAX_TOKENS_LIMIT of the OpenAI model and `max_tokens` you specify, then documents (used to
        construct the context) are thrown away until the prompt length fits within the MAX_TOKENS_LIMIT.
        """
        full_prompt = self._fill_prompt(query, documents)
        n_full_prompt_tokens = count_openai_tokens(text=full_prompt, tokenizer=self._tokenizer)

        # for length restrictions of prompt see: https://platform.openai.com/docs/api-reference/completions/create#completions/create-max_tokens
        leftover_token_len = self.MAX_TOKENS_LIMIT - n_full_prompt_tokens - self.max_tokens

        # Trim down the prompt (by removing documents) until it fits the models MAX_TOKENS_LIMIT
        input_docs = documents
        skipped_docs = 0
        # If leftover_token_len is negative we have gone past the MAX_TOKENS_LIMIT and the prompt must be trimmed
        if leftover_token_len < 0:
            n_skipped_tokens = 0
            # Reversing the order of documents b/c we want to throw away less relevant docs first
            for doc in reversed(documents):
                skipped_docs += 1
                n_skipped_tokens += count_openai_tokens(text=doc.content, tokenizer=self._tokenizer)
                # Only skip enough tokens to fit within the MAX_TOKENS_LIMIT
                if n_skipped_tokens >= abs(leftover_token_len):
                    break

            # Throw away least relevant docs
            input_docs = documents[:-skipped_docs]
            full_prompt = self._fill_prompt(query, input_docs)
            n_full_prompt_tokens = count_openai_tokens(text=full_prompt, tokenizer=self._tokenizer)

            if len(input_docs) == 0:
                logger.warning(
                    "Skipping all of the provided Documents, as none of them fits the maximum token limit of %s. "
                    "The generated answers will therefore not be conditioned on any context.",
                    self.MAX_TOKENS_LIMIT,
                )
            elif skipped_docs >= 1:
                logger.warning(
                    "Skipping %s of the provided Documents, as using them would exceed the maximum token limit of %s.",
                    skipped_docs,
                    self.MAX_TOKENS_LIMIT,
                )

        logger.debug("Number of tokens in full prompt: %s", n_full_prompt_tokens)
        logger.debug("Full prompt: %s", full_prompt)
        return full_prompt, input_docs
