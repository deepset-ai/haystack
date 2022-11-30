# pylint: disable=missing-timeout

from typing import Optional, List, Tuple
import json
import logging
import requests

from transformers import GPT2TokenizerFast

from haystack.nodes.answer_generator import BaseGenerator
from haystack import Document
from haystack.errors import OpenAIError, OpenAIRateLimitError
from haystack.utils.reflection import retry_with_exponential_backoff

logger = logging.getLogger(__name__)


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
        model: str = "text-curie-001",
        max_tokens: int = 13,
        top_k: int = 5,
        temperature: float = 0.2,
        presence_penalty: float = -2.0,
        frequency_penalty: float = -2.0,
        examples_context: Optional[str] = None,
        examples: Optional[List] = None,
        stop_words: Optional[List] = None,
        progress_bar: bool = True,
    ):

        """
        :param api_key: Your API key from OpenAI. It is required for this node to work.
        :param model: ID of the engine to use for generating the answer. You can select one of `"text-ada-001"`,
                     `"text-babbage-001"`, `"text-curie-001"`, or `"text-davinci-002"`
                     (from worst to best and from cheapest to most expensive). For more information about the models,
                     refer to the [OpenAI Documentation](https://beta.openai.com/docs/models/gpt-3).
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
                                 If not supplied, the default from OpenAPI docs is used:
                                 "In 2017, U.S. life expectancy was 78.6 years."
        :param examples: List of (question, answer) pairs that helps steer the model towards the tone and answer
                         format you'd like. We recommend adding 2 to 3 examples.
                         If not supplied, the default from OpenAPI docs is used:
                         [["What is human life expectancy in the United States?", "78 years."]]
        :param stop_words: Up to 4 sequences where the API stops generating further tokens. The returned text does
                           not contain the stop sequence.
                           If you don't provide it, the default from OpenAPI docs is used: ["\n", "<|endoftext|>"]
        """
        super().__init__(progress_bar=progress_bar)
        if not examples_context:
            examples_context = "In 2017, U.S. life expectancy was 78.6 years."
        if not examples:
            examples = [["What is human life expectancy in the United States?", "78 years."]]
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
        self.stop_words = stop_words
        self._tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

        if "davinci" in self.model:
            self.MAX_TOKENS_LIMIT = 4000
        else:
            self.MAX_TOKENS_LIMIT = 2048

    @retry_with_exponential_backoff(backoff_in_seconds=10, max_retries=5)
    def predict(self, query: str, documents: List[Document], top_k: Optional[int] = None):
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
        response = requests.request("POST", url, headers=headers, data=json.dumps(payload))
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

        generated_answers = [ans["text"] for ans in res["choices"]]
        answers = self._create_answers(generated_answers, input_docs)
        result = {"query": query, "answers": answers}
        return result

    def _build_prompt(self, query: str, documents: List[Document]) -> Tuple[str, List[Document]]:
        """
        Builds the prompt for the GPT-3 model so that it can generate an Answer.
        """
        example_context = f"===\nContext: {self.examples_context}\n===\n"
        example_prompts = "\n---\n".join([f"Q: {question}\nA: {answer}" for question, answer in self.examples])
        instruction = "Please answer the question according to the above context.\n" + example_context + example_prompts
        instruction = f"{instruction.strip()}\n\n"

        qa_prompt = f"Q: {query}\nA:"

        n_instruction_tokens = len(self._tokenizer.encode(instruction + qa_prompt + "===\nContext: \n===\n"))
        n_docs_tokens = [len(self._tokenizer.encode(doc.content)) for doc in documents]
        # for length restrictions of prompt see: https://beta.openai.com/docs/api-reference/completions/create#completions/create-max_tokens
        leftover_token_len = self.MAX_TOKENS_LIMIT - n_instruction_tokens - self.max_tokens

        # Add as many Documents as context as fit into the model
        input_docs = []
        input_docs_content = []
        skipped_docs = 0
        for doc, doc_token_len in zip(documents, n_docs_tokens):
            if doc_token_len <= leftover_token_len:
                input_docs.append(doc)
                input_docs_content.append(doc.content)
                leftover_token_len -= doc_token_len
            else:
                skipped_docs += 1

        if len(input_docs) == 0:
            logger.warning(
                f"Skipping all of the provided Documents, as none of them fits the maximum token limit of "
                f"{self.MAX_TOKENS_LIMIT}. The generated answers will therefore not be conditioned on any context."
            )
        elif skipped_docs >= 1:
            logger.warning(
                f"Skipping {skipped_docs} of the provided Documents, as using them would exceed the maximum token "
                f"limit of {self.MAX_TOKENS_LIMIT}."
            )

        # Top ranked documents should go at the end
        context = " ".join(reversed(input_docs_content))
        context = f"===\nContext: {context}\n===\n"

        full_prompt = instruction + context + qa_prompt

        return full_prompt, input_docs
