import logging
from abc import abstractmethod
from typing import Dict, List, Optional

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from haystack import Document
from haystack.generator.base import BaseGenerator
from haystack.retriever.base import BaseRetriever
from haystack.retriever.dense import EmbeddingRetriever

logger = logging.getLogger(__name__)


class Seq2SeqGenerator(BaseGenerator):
    """
        A generic sequence-to-sequence generator based on HuggingFace. Subclasses should
        implement prepare_model_input abstract method

        Text generation is supported by so called auto-regressive language models like GPT2,
        XLNet, XLM, Bart, T5 and others. In fact, any HuggingFace language model that extends
        GenerationMixin can be used by Seq2SeqGenerator

        See https://huggingface.co/transformers/main_classes/model.html?transformers.generation_utils.GenerationMixin#transformers.generation_utils.GenerationMixin
        as well as https://huggingface.co/blog/how-to-generate

        For a list of all text-generation models see https://huggingface.co/models?pipeline_tag=text-generation

        **Example**

        ```python
        |     query = "Why is Dothraki language important?"
        |
        |     # Retrieve related documents from retriever
        |     retrieved_docs = retriever.retrieve(query=query)
        |
        |     # Now generate answer from query and retrieved documents
        |     generator.predict(
        |        query=query,
        |        documents=retrieved_docs,
        |        top_k=1
        |     )
        |
        |     # Answer
        |
        |     {'answers': [" The Dothraki language is a constructed fictional language. It's important because George R.R. Martin wrote it."],
        |      'query': 'Why is Dothraki language important?'}
        |
        ```
    """

    def __init__(
            self,
            model_name_or_path: str,
            retriever: Optional[BaseRetriever] = None,
            top_k: int = 1,
            max_length: int = 200,
            min_length: int = 2,
            num_beams: int = 8,
            use_gpu: bool = True,
    ):
        """


        :param model_name_or_path: a HF model name for auto-regressive language model like GPT2, XLNet, XLM, Bart, T5
        and others (e.g. `t5-base`)
        :param retriever: `BaseRetriever` used to retrieve passage
        :param top_k: Number of independently generated text to return
        :param max_length: Maximum length of generated text
        :param min_length: Minimum length of generated text
        :param num_beams: Number of beams for beam search. 1 means no beam search.
        :param use_gpu: Whether to use GPU (if available)
        """

        self.model_name_or_path = model_name_or_path
        self.max_length = max_length
        self.min_length = min_length
        self.num_beams = num_beams
        self.retriever = retriever

        if top_k > self.num_beams:
            top_k = self.num_beams
            logger.warning(f'top_k value should not be greater than num_beams, hence setting it to {num_beams}')

        self.top_k = top_k

        if use_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path).to(self.device)
        self.model.eval()

    @abstractmethod
    def prepare_model_input(self, query: str, documents: List[Document], top_k: Optional[int] = None) -> Dict:
        pass

    def predict(self, query: str, documents: List[Document], top_k: Optional[int] = None) -> Dict:
        """
        Generate the answer to the input query. The generation will be conditioned on the supplied documents.
        These document can be retrieved via the Retriever or supplied directly via predict method.

        :param query: Query
        :param documents: Related documents (e.g. coming from a retriever) that the answer shall be conditioned on.
        :param top_k: Number of returned answers
        :return: Generated answers

        """
        torch.set_grad_enabled(False)
        if len(documents) == 0:
            raise AttributeError("generator needs documents to predict the answer")

        top_k = top_k if top_k is not None else self.top_k

        if top_k > self.num_beams:
            top_k = self.num_beams
            logger.warning(f'top_k value should not be greater than num_beams, hence setting it to {top_k}')

        query_and_docs_encoded = self.prepare_model_input(query, documents, top_k)

        generated_answers_encoded = self.model.generate(
            input_ids=query_and_docs_encoded["input_ids"],
            attention_mask=query_and_docs_encoded["attention_mask"],
            min_length=self.min_length,
            max_length=self.max_length,
            do_sample=True if self.num_beams == 1 else False,
            early_stopping=True,
            num_beams=self.num_beams,
            temperature=1.0,
            top_k=None,
            top_p=None,
            eos_token_id=self.tokenizer.eos_token_id,
            no_repeat_ngram_size=3,
            num_return_sequences=top_k,
            decoder_start_token_id=self.tokenizer.bos_token_id
        )
        generated_answers = self.tokenizer.batch_decode(generated_answers_encoded, skip_special_tokens=True)
        return {"query": query, "answers": generated_answers}


class BartEli5Generator(Seq2SeqGenerator):
    """
       A sequence-to-sequence model (https://huggingface.co/yjernite/bart_eli5) based on the BART architecture
       fine-tuned on ELI5 dataset (https://arxiv.org/abs/1907.09190)

       For more details refer to Yacine Jernite's excellent LFQA contributions at https://yjernite.github.io/lfqa.html

       **Example**

       ```python
       |     query = "Why is Dothraki language important?"
       |
       |     # Retrieve related documents from retriever
       |     retrieved_docs = retriever.retrieve(query=query)
       |
       |     # Now generate answer from query and retrieved documents
       |     generator.predict(
       |        query=query,
       |        documents=retrieved_docs,
       |        top_k=1
       |     )
       |
       |     # Answer
       |
       |     {'answers': [" The Dothraki language is a constructed fictional language. It's important because George R.R. Martin wrote it."],
       |      'query': 'Why is Dothraki language important?'}
       |
       ```
    """

    def __init__(
            self,
            model_name_or_path: str = "yjernite/bart_eli5",
            retriever: Optional[EmbeddingRetriever] = None,
            top_k: int = 1,
            max_length: int = 200,
            min_length: int = 2,
            num_beams: int = 8,
            use_gpu: bool = True,
    ):
        """

        :param model_name_or_path: ELI5 BART model from HF hub 'yjernite/bart_eli5'
        :param retriever: `EmbeddingRetriever` used to embedded passage
        :param top_k: Number of independently generated text to return
        :param max_length: Maximum length of generated text
        :param min_length: Minimum length of generated text
        :param num_beams: Number of beams for beam search. 1 means no beam search.
        :param use_gpu: Whether to use GPU (if available)
        """
        super(BartEli5Generator, self).__init__(model_name_or_path, retriever, top_k, max_length,
                                                min_length, num_beams, use_gpu)

    def prepare_model_input(self, query: str, documents: List[Document], top_k: Optional[int] = None) -> Dict:
        conditioned_doc = "<P> " + " <P> ".join([d.text for d in documents])

        # concatenate question and support document into BART input
        query_and_docs = "question: {} context: {}".format(query, conditioned_doc)

        return self.tokenizer([(query_and_docs, "A")], truncation=True,
                              padding=True, return_tensors="pt").to(self.device)
