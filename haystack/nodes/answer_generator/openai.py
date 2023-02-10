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

logger = logging.getLogger(__name__)

machine = platform.machine()
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
        max_tokens: int = 40,
        top_k: int = 1,
        temperature: float = 0.2,
        presence_penalty: float = -2.0,
        frequency_penalty: float = 0,
        examples_context: Optional[str] = None,
        examples: Optional[List] = None,
        instructions: Optional[str] = None,
        add_runtime_instructions: bool = False,
        stop_words: Optional[List] = None,
        progress_bar: bool = True,
    ):
        """
        :param api_key: Your API key from OpenAI. It is required for this node to work.
        :param model: ID of the engine to use for generating the answer. You can select one of `"text-ada-001"`,
                     `"text-babbage-001"`, `"text-curie-001"`, or `"text-davinci-003"`
                     (from worst to best and from cheapest to most expensive). For more information about the models,
                     refer to the [OpenAI Documentation](https://platform.openai.com/docs/models/gpt-3).
        :param max_tokens: The maximum number of tokens allowed for the generated Answer.
        :param top_k: Number of generated Answers.
        :param temperature: What sampling temperature to use. Higher values mean the model will take more risks and
                            value 0 (argmax sampling) works better for scenarios with a well-defined Answer.
        :param presence_penalty: Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they have already appeared
                                 in the text. This increases the model's likelihood to talk about new topics.
        :param frequency_penalty: Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing
                                  frequency in the text so far, decreasing the model's likelihood to repeat the same line
                                  verbatim.
        :param examples_context: A text snippet containing the contextual information used to generate the Answers for
                                 the examples you provide.
        :param examples: List of (question, answer) pairs that helps steer the model towards the tone and answer
                         format you'd like.
        :param instructions: Here you can initialize custom instructions as prompt. Defaults to 'Create a concise and informative answer...'
        :param add_runtime_instructions: If you like to add the prompt instructions (the instructions around the question)
                                         during querying or not. Defaults to using predefined prompt instructions.
                                         If you do add instructions at runtime separate instructions and question like:
                                         "... <instructions> ... [SEPARATOR] <question>"
                                         Also make sure to mention "$documents" and "$query" in the <instructions>, such
                                         that those will be replaced in correctly.
        :param stop_words: Up to 4 sequences where the API stops generating further tokens. The returned text does
                           not contain the stop sequence.
                           If you don't provide it, the default from OpenAPI docs is used: ["\n", "<|endoftext|>"]
        """
        super().__init__(progress_bar=progress_bar)
        if not stop_words:
            stop_words = ["\n", "<|endoftext|>"]

        if not api_key:
            raise ValueError("OpenAIAnswerGenerator requires an API key.")

        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.top_k = top_k
        self.temperature = temperature
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.examples_context = examples_context
        self.examples = examples
        self.add_runtime_instructions = add_runtime_instructions
        if not instructions:
            self.instructions = (
                f"Create a concise and informative answer (no more than {max_tokens} words) for a given "
                f"question based solely on the given documents. You must only use information from the given "
                f"documents. Use an unbiased and journalistic tone. Do not repeat text. Cite the documents "
                f"using Document[$number] notation. If multiple documents contain the answer, cite "
                f"each document like Document[$number], Document[$number], Document[$number] ... If "
                f"the documents do not contain the answer to the question, say that "
                f"'answering is not possible given the available information.'\n "
                f"$documents; \n Question: $query; Answer: "
            )
        else:
            if "$documents" in instructions and "$query" in instructions:
                self.instructions = instructions
            else:
                logger.warning(
                    f"Instructions do not have the right format. You need to include '$documents' and '$query'. "
                    f"You supplied: {instructions}"
                )
                self.instructions = instructions
        self.stop_words = stop_words

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
        backoff_in_seconds=int(os.environ.get(HAYSTACK_REMOTE_API_BACKOFF_SEC, 1)),
        max_retries=int(os.environ.get(HAYSTACK_REMOTE_API_MAX_RETRIES, 5)),
        errors=(OpenAIRateLimitError, OpenAIError),
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
        prompt, input_docs = self._build_prompt(query=query, documents=documents)

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

    def _build_prompt(self, query: str, documents: List[Document]) -> Tuple[str, List[Document]]:
        """
        Builds the prompt for the GPT-3 model so that it can generate an Answer.
        """
        ## Clean documents
        for doc in documents:
            doc.content = self._clean_documents(doc.content)

        ## Add example prompts
        if self.examples and self.examples_context:
            example_context = f"===\nContext: {self.examples_context}\n===\n"
            example_QAs = "\n---\n".join([f"Q: {question}\nA: {answer}" for question, answer in self.examples])
            example_prompt = (
                "Please answer the question according to the above context.\n" + example_context + example_QAs
            )
            example_prompt = f"{example_prompt.strip()}\n\n"
        else:
            example_prompt = ""

        ## Compose prompt
        # Switch for adding the prompt instructions at runtime.
        if self.add_runtime_instructions:
            temp = query.split("[SEPARATOR]")
            if len(temp) != 2:
                logger.error(
                    f"Instructions given to the OpenAIAnswerGenerator were not correct, please follow the structure "
                    f"from the docstrings. You supplied: {query}"
                )
                current_prompt = ""
                query = "Say: incorrect prompt."
            else:
                current_prompt = temp[0].strip()
                query = temp[1].strip()
        else:
            current_prompt = self.instructions

        # Inserting the query into the prompt here.
        current_prompt = current_prompt.replace("$query", query)
        n_instruction_tokens = self._count_tokens(example_prompt + current_prompt)
        logger.debug("Number of tokens in instruction: %s", n_instruction_tokens)
        n_docs_tokens = [self._count_tokens(f"\nDocument[{i}]: " + doc.content) for i, doc in enumerate(documents)]
        logger.debug("Number of tokens in documents: %s", n_docs_tokens)

        # Add as many Documents as fit into the model.
        leftover_token_len = self.MAX_TOKENS_LIMIT - n_instruction_tokens - self.max_tokens
        input_docs = []
        input_docs_content = []
        skipped_docs = 0
        for i, (doc, doc_token_len) in enumerate(zip(documents, n_docs_tokens)):
            if doc_token_len <= leftover_token_len:
                input_docs.append(doc)
                input_docs_content.append(f"\nDocument[{i}]: " + doc.content)
                leftover_token_len -= doc_token_len
            else:
                skipped_docs += 1

        if len(input_docs) == 0:
            logger.warning(
                "Skipping all of the provided Documents, as none of them fits the maximum token limit of %s"
                "The generated answers will therefore not be conditioned on any context.",
                self.MAX_TOKENS_LIMIT,
            )
        elif skipped_docs >= 1:
            logger.warning(
                "Skipping %s of the provided Documents, as using them would exceed the maximum token limit of %s.",
                skipped_docs,
                self.MAX_TOKENS_LIMIT,
            )

        # Top ranked documents should go at the end.
        context_documents = " ".join(reversed(input_docs_content))

        current_prompt = current_prompt.replace("$documents", context_documents)
        logger.debug("Full prompt: %s", current_prompt)

        return current_prompt, input_docs

    def _count_tokens(self, text: str) -> int:
        if USE_TIKTOKEN:
            return len(self._tk_tokenizer.encode(text))
        else:
            return len(self._hf_tokenizer.tokenize(text))

    def _clean_documents(self, text: str) -> str:
        to_remove = {"$documents": "#documents", "$query": "#query", "\n": " "}
        for x in to_remove.keys():
            text = text.replace(x, to_remove[x])
        return text
