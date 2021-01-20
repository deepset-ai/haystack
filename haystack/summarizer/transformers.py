import logging
from typing import Any, Dict, List, Optional

from transformers import pipeline
from transformers.models.auto.modeling_auto import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer

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
        |     docs = [Document(text="PG&E stated it scheduled the blackouts in response to forecasts for high winds amid dry conditions.
        |            The aim is to reduce the risk of wildfires. Nearly 800 thousand customers were scheduled to be affected by
        |            the shutoffs which were expected to last through at least midday tomorrow.")]
        |
        |     # Summarize
        |     summary = summarizer.predict(
        |        documents=docs,
        |        generate_single_summary=True
        |     )
        |
        |     # Show results (List of Documents, containing summary and original text)
        |     print(summary)
        |
        |    [
        |      {
        |        "text": "California's largest electricity provider has turned off power to hundreds of thousands of customers.",
        |        ...
        |        "meta": {
        |          "context": "PGE stated it scheduled the blackouts in response to forecasts for high winds amid dry conditions. ....
        |              },
        |        ...
        |      },
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

        # TODO AutoModelForSeq2SeqLM is only necessary with transformers==4.1.1, with newer versions use the pipeline directly
        if tokenizer is None:
            tokenizer = model_name_or_path
        model = AutoModelForSeq2SeqLM.from_pretrained(pretrained_model_name_or_path=model_name_or_path)
        self.summarizer = pipeline("summarization", model=model, tokenizer=tokenizer, device=use_gpu)
        self.max_length = max_length
        self.min_length = min_length
        self.clean_up_tokenization_spaces = clean_up_tokenization_spaces
        self.separator_for_single_summary = separator_for_single_summary

    def predict(self, documents: List[Document], generate_single_summary: bool = False) -> List[Document]:
        """
        Produce the summarization from the supplied documents.
        These document can for example be retrieved via the Retriever.

        :param documents: Related documents (e.g. coming from a retriever) that the answer shall be conditioned on.
        :param generate_single_summary: Whether to generate a single summary for all documents or one summary per document.
                                        If set to "True", all docs will be joined to a single string that will then
                                        be summarized.
                                        Important: The summary will depend on the order of the supplied documents!
        :return: List of Documents, where Document.text contains the summarization and Document.meta["context"]
                 the original, not summarized text
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

        summaries = self.summarizer(
            contexts,
            min_length=self.min_length,
            max_length=self.max_length,
            return_text=True,
            clean_up_tokenization_spaces=self.clean_up_tokenization_spaces,
        )

        result: List[Document] = []

        for context, summarized_answer in zip(contexts, summaries):
            cur_doc = Document(text=summarized_answer['summary_text'], meta={"context": context})
            result.append(cur_doc)

        return result
