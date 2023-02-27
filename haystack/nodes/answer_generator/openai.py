import json
import logging
import os
import platform
import sys
from typing import List, Optional, Tuple, Union

import requests

from haystack import Document
from haystack.environment import (
    HAYSTACK_REMOTE_API_BACKOFF_SEC,
    HAYSTACK_REMOTE_API_MAX_RETRIES,
    HAYSTACK_REMOTE_API_TIMEOUT_SEC,
)
from haystack.errors import OpenAIError, OpenAIRateLimitError
from haystack.nodes.answer_generator import BaseGenerator
from haystack.utils.reflection import retry_with_exponential_backoff
from haystack.nodes.prompt import PromptTemplate

logger = logging.getLogger(__name__)

machine = platform.machine().lower()
system = platform.system()

USE_TIKTOKEN = False
if sys.version_info >= (3, 8) and (machine in ["amd64", "x86_64"] or (machine == "arm64" and system == "Darwin")):
    USE_TIKTOKEN = True

if USE_TIKTOKEN:
    import tiktoken  # pylint: disable=import-error
else:
    logger.warning(
        "OpenAI tiktoken module is not available for Python < 3.8,Linux ARM64 and AARCH64. Falling back to GPT2TokenizerFast."
    )
    from transformers import GPT2TokenizerFast, PreTrainedTokenizerFast


OPENAI_TIMEOUT = float(os.environ.get(HAYSTACK_REMOTE_API_TIMEOUT_SEC, 30))
OPENAI_BACKOFF = float(os.environ.get(HAYSTACK_REMOTE_API_BACKOFF_SEC, 10))
OPENAI_MAX_RETRIES = int(os.environ.get(HAYSTACK_REMOTE_API_MAX_RETRIES, 5))


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
        model: str = "text-davinci-003",
        max_tokens: int = 50,
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
        :param model: ID of the engine to use for generating the answer. You can select one of `"text-ada-001"`,
                     `"text-babbage-001"`, `"text-curie-001"`, or `"text-davinci-003"`
                     (from worst to best and from cheapest to most expensive). For more information about the models,
                     refer to the [OpenAI Documentation](https://platform.openai.com/docs/models/gpt-3).
        :param max_tokens: The maximum number of tokens reserved for the generated Answer.
                           A higher number allows for longer answers without exceeding the max prompt length of the OpenAI model.
                           A lower number allows longer prompts with more documents passed as context, but the generated answer might be cut after max_tokens.
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
                                "\n===\nContext: $examples_context\n===\n$examples\n\n"
                                "===\nContext: $context\n===\n$query",
                    prompt_params=["examples_context", "examples", "context", "query"],
                )
            ```
            To learn how variables, such as'$context', are substituted in the `prompt_text`, see
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
                "\n===\nContext: $examples_context\n===\n$examples\n\n"
                "===\nContext: $context\n===\n$query",
                prompt_params=["examples_context", "examples", "context", "query"],
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

        tokenizer = "gpt2"
        if "davinci" in self.model:
            self.MAX_TOKENS_LIMIT = 4000
            if self.model.endswith("-003") and USE_TIKTOKEN:
                tokenizer = "cl100k_base"
        else:
            self.MAX_TOKENS_LIMIT = 2048

        if USE_TIKTOKEN:
            logger.debug("Using tiktoken %s tokenizer", tokenizer)
            self._tk_tokenizer: tiktoken.Encoding = tiktoken.get_encoding(tokenizer)
        else:
            logger.debug("Using GPT2TokenizerFast")
            self._hf_tokenizer: PreTrainedTokenizerFast = GPT2TokenizerFast.from_pretrained(tokenizer)

    @retry_with_exponential_backoff(
        backoff_in_seconds=OPENAI_BACKOFF, max_retries=OPENAI_MAX_RETRIES, errors=(OpenAIRateLimitError, OpenAIError)
    )
    def predict(
        self,
        query: str,
        documents: List[Document],
        top_k: Optional[int] = None,
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

        # get answers from OpenAI API
        url = "https://api.openai.com/v1/completions"

        payload = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": self.max_tokens,
            "stop": self.stop_words,
            "n": top_k,
            "temperature": self.temperature,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
        }

        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        response = requests.request("POST", url, headers=headers, data=json.dumps(payload), timeout=timeout)
        res = json.loads(response.text)

        if response.status_code != 200 or "choices" not in res:
            openai_error: OpenAIError
            if response.status_code == 429:
                openai_error = OpenAIRateLimitError(f"API rate limit exceeded: {response.text}")
            else:
                openai_error = OpenAIError(
                    f"OpenAI returned an error.\n"
                    f"Status code: {response.status_code}\n"
                    f"Response body: {response.text}",
                    status_code=response.status_code,
                )
            raise openai_error

        number_of_truncated_answers = sum(1 for ans in res["choices"] if ans["finish_reason"] == "length")
        if number_of_truncated_answers > 0:
            logger.warning(
                "%s out of the %s answers have been truncated before reaching a natural stopping point."
                "Consider increasing the max_tokens parameter to allow for longer answers.",
                number_of_truncated_answers,
                top_k,
            )

        generated_answers = [ans["text"] for ans in res["choices"]]
        answers = self._create_answers(generated_answers, input_docs)
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
        n_full_prompt_tokens = self._count_tokens(full_prompt)

        # for length restrictions of prompt see: https://platform.openai.com/docs/api-reference/completions/create#completions/create-max_tokens
        leftover_token_len = self.MAX_TOKENS_LIMIT - n_full_prompt_tokens - self.max_tokens

        # Trim down the prompt (by removing documents) until it fits the models MAX_TOKENS_LIMIT
        input_docs = documents
        skipped_docs = 0
        # If leftover_token_len is negative we have gone past the MAX_TOKENS_LIMIT and the prompt must be trimmed
        if leftover_token_len < 0:
            n_docs_tokens = [self._count_tokens(doc.content) for doc in documents]
            logger.debug("Number of tokens in documents: %s", n_docs_tokens)

            # Reversing the order of documents b/c we want to throw away less relevant docs first
            rev_n_docs_tokens = reversed(n_docs_tokens)
            n_skipped_tokens = 0
            for doc_token_len in rev_n_docs_tokens:
                n_skipped_tokens += doc_token_len
                skipped_docs += 1
                # Only skip enough tokens to fit within the MAX_TOKENS_LIMIT
                if n_skipped_tokens >= abs(leftover_token_len):
                    break

            # Throw away least relevant docs
            input_docs = documents[:-skipped_docs]
            full_prompt = self._fill_prompt(query, input_docs)
            n_full_prompt_tokens = self._count_tokens(full_prompt)

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

    def _count_tokens(self, text: str) -> int:
        if USE_TIKTOKEN:
            return len(self._tk_tokenizer.encode(text))
        else:
            return len(self._hf_tokenizer.tokenize(text))
