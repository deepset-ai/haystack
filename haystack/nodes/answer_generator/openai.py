from typing import Optional, List, Union
import json
import requests

from haystack.nodes.answer_generator import BaseGenerator
from haystack import Answer, Document


class OpenAIAnswerGenerator(BaseGenerator):
    """
    Uses the GPT-3 models from the OpenAI API to generate answers based on supplied documents (e.g. from any retriever in Haystack).
    """

    def __init__(
        self,
        api_key: str,
        model: str = "curie",
        search_model: str = "ada",
        max_tokens: int = 7,
        top_k: int = 5,
        temperature: int = 0,
        examples_context: Optional[str] = None,
        examples: Optional[List] = None,
        stop_words: Optional[List] = None,
    ):

        """
        :param api_key: Your API key from OpenAI
        :param model: ID of the engine to use for generating the answer. You can select one of ada, babbage, curie, or davinci (from worst to best + cheapest to most expensive).
        :param search_model: ID of the engine to use for Search. You can select one of ada, babbage, curie, or davinci (from worst to best + cheapest to most expensive).
        :param max_tokens: The maximum number of tokens allowed for the generated answer.
        :param top_k: Number of generated answers
        :param temperature: What sampling temperature to use. Higher values mean the model will take more risks and value 0 (argmax sampling) works better for scenarios with a well-defined answer.
        :param examples_context: A text snippet containing the contextual information used to generate the answers for the examples you provide.
                                 If not supplied, the default from OpenAPI docs is used: "In 2017, U.S. life expectancy was 78.6 years.",
        :param examples: List of (question, answer) pairs that will help steer the model towards the tone and answer format you'd like. We recommend adding 2 to 3 examples.
                        If not supplied, the default from OpenAPI docs is used: [["What is human life expectancy in the United States?","78 years."]]
        :param stop_words: Up to 4 sequences where the API will stop generating further tokens. The returned text will not contain the stop sequence.
                        If not supplied, the default from OpenAPI docs is used: ["\n", "<|endoftext|>"]
        """
        super().__init__()
        if not examples_context:
            examples_context = "In 2017, U.S. life expectancy was 78.6 years."
        if not examples:
            examples = [["What is human life expectancy in the United States?", "78 years."]]
        if not stop_words:
            stop_words = ["\n", "<|endoftext|>"]

        self.model = model
        self.search_model = search_model
        self.max_tokens = max_tokens
        self.top_k = top_k
        self.examples_context = examples_context
        self.examples = examples
        self.api_key = api_key
        self.stop_words = stop_words

    def predict(self, query: str, documents: List[Document], top_k: Optional[int] = None):
        """
        Use loaded QA model to generate answers for a query based on the supplied list of Documents.

        Returns dictionaries containing answers.
        Be aware that OpenAI doesn't return scores for those answers.

        Example:
         ```python
            |{
            |    'query': 'Who is the father of Arya Stark?',
            |    'answers':[Answer(
            |                 'answer': 'Eddard,',
            |                 'score': None,
            |                 ),...
            |              ]
            |}
         ```

        :param query: Query string
        :param documents: List of Document in which to search for the answer
        :param top_k: The maximum number of answers to return
        :return: Dict containing query and answers
        """
        if top_k is None:
            top_k = self.top_k
        # convert input to OpenAI format
        inputs = [doc.content for doc in documents]

        # get answers from OpenAI API
        url = "https://api.openai.com/v1/answers"

        payload = {
            "documents": inputs,
            "question": query,
            "search_model": self.search_model,
            "model": self.model,
            "examples_context": self.examples_context,
            "examples": self.examples,
            "max_tokens": self.max_tokens,
            "stop": self.stop_words,
            "n": self.top_k,
        }

        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        response = requests.request("POST", url, headers=headers, data=json.dumps(payload))

        res = json.loads(response.text)
        answers: List[Answer] = [Answer(answer=a, type="generative") for a in res["answers"]]
        result = {"query": query, "answers": answers}

        return result

    def predict_batch(
        self,
        queries: List[str],
        documents: Union[List[Document], List[List[Document]]],
        top_k: Optional[int] = None,
        batch_size: Optional[int] = None,
    ):
        raise NotImplementedError("predict_batch() is not yet implemented for OpenAIAnswerGenerator")
