import logging
from typing import Any, Dict, List, Optional

from transformers import pipeline

from haystack import Document
from haystack.summarizer.base import BaseSummarizer

logger = logging.getLogger(__name__)


class TransformersSummarizer(BaseSummarizer):
    """
        Transformer based model to summarize the documents using the HuggingFace's transformers framework

        You can use any model that has been fine-tuned on a summarization task. For example:
        '`bart-large-cnn`', '`t5-small`', '`t5-base`', '`t5-large`', '`t5-3b`', '`t5-11b`'.
        See the up-to-date list of available models on
        `huggingface.co/models <https://huggingface.co/models?filter=summarization>`__

        **Example**

        ```python
        |     query = "Where is Eiffel Tower?"
        |
        |     # Retrieve related documents from retriever
        |     retrieved_docs = retriever.retrieve(query=query)
        |
        |     # Now summarize answer from query and retrieved documents
        |     summarizer.predict(
        |        query=query,
        |        documents=retrieved_docs,
        |        generate_one_summary=True
        |     )
        |
        |     # Answer
        |
        |     {'query': 'Where is Eiffel Tower?',
        |      'answers':
        |          [{'query': 'Where is Eiffel Tower?',
        |            'answer': 'The Eiffel Tower is a landmark in Paris, France.',
        |            'meta': {
        |                      'text': 'The tower is 324 metres ...'
        |      }}]}
        ```
    """

    def __init__(
            self,
            model_name_or_path: str = "google/pegasus-xsum",
            tokenizer: Optional[str] = None,
            max_length: int = 200,
            min_length: int = 5,
            use_gpu: int = 0,
            clean_up_tokenization_spaces: bool = True,
            separator_for_single_summary: str = " ",
    ):
        """
        Load a Summarization model from Transformers.
        See the up-to-date list of available models on
        `huggingface.co/models <https://huggingface.co/models?filter=summarization>`__

        :param model_name_or_path: Directory of a saved model or the name of a public model e.g.
                                   'facebook/rag-token-nq', 'facebook/rag-sequence-nq'.
                                   See https://huggingface.co/models?filter=summarization for full list of available models.
        :param tokenizer: Name of the tokenizer (usually the same as model)
        :param max_length: Maximum length of summarized text
        :param min_length: Minimum length of summarized text
        :param use_gpu: If < 0, then use cpu. If >= 0, this is the ordinal of the gpu to use
        :param clean_up_tokenization_spaces: Whether or not to clean up the potential extra spaces in the text output
        :param separator_for_single_summary: If `generate_single_summary=True` in `predict()`, we need to join all docs
                                             into a single text. This separator appears between those subsequent docs.
        """

        self.summarizer = pipeline("summarization", model=model_name_or_path, tokenizer=tokenizer, device=use_gpu)
        self.max_length = max_length
        self.min_length = min_length
        self.clean_up_tokenization_spaces = clean_up_tokenization_spaces
        self.separator_for_single_summary = separator_for_single_summary

    def predict(self, documents: List[Document], generate_single_summary: bool = False, query: str = None) -> Dict:
        """
        Produce the summarization from the supplied documents.
        These document can for example be retrieved via the Retriever.

        :param documents: Related documents (e.g. coming from a retriever) that the answer shall be conditioned on.
        :param generate_single_summary: Whether to generate a single summary for all documents or one summary per document.
                                        If set to "True", all docs will be joined to a single string that will then
                                        be summarized.
                                        Important: The summary will depend on the order of the supplied documents!
        :param query: Query
        :return: Generated answers plus additional infos in a dict like this:

        ```python
        |     {'query': 'Where is Eiffel Tower?',
        |      'answers':
        |          [{'query': 'Where is Eiffel Tower?',
        |            'answer': 'The Eiffel Tower is a landmark in Paris, France.',
        |            'meta': {
        |                      'text': 'The tower is 324 metres ...'
        |      }}]}
        ```
        """

        if self.min_length > self.max_length:
            raise AttributeError("min_length cannot be greater than max_length")

        if len(documents) == 0:
            raise AttributeError("Summarizer needs at least one document to produce a summary.")

        contexts: List[str] = [doc.text for doc in documents]

        if generate_single_summary:
            # Documents order is very important to produce summary.
            # Different order of same documents produce different summary.
            contexts = [self.separator_for_single_summary.join(contexts)]

        summarized_answers = self.summarizer(
            contexts,
            min_length=self.min_length,
            max_length=self.max_length,
            return_text=True,
            clean_up_tokenization_spaces=self.clean_up_tokenization_spaces,
        )

        answers: List[Any] = []

        for context, summarized_answer in zip(contexts, summarized_answers):
            cur_answer = {
                "query": query,
                "answer": summarized_answer['summary_text'],
                "meta": {
                    "text": context,
                }
            }
            answers.append(cur_answer)

        result = {"query": query, "answers": answers}

        return result
