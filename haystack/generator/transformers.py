import logging
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy
import torch
from transformers import RagTokenizer, RagTokenForGeneration

from haystack import Document
from haystack.generator.base import BaseGenerator
from haystack.retriever.dense import DensePassageRetriever

logger = logging.getLogger(__name__)


class RAGeneratorType(Enum):
    TOKEN = 1,
    SEQUENCE = 2


class RAGenerator(BaseGenerator):
    """
        Implementation of Facebook's Retrieval-Augmented Generator (https://arxiv.org/abs/2005.11401) based on
        HuggingFace's transformers (https://huggingface.co/transformers/model_doc/rag.html).

        Instead of "finding" the answer within a document, these models **generate** the answer.
        In that sense, RAG follows a similar approach as GPT-3 but it comes with two huge advantages
        for real-world applications:
        a) it has a manageable model size
        b) the answer generation is conditioned on retrieved documents,
        i.e. the model can easily adjust to domain documents even after training has finished
        (in contrast: GPT-3 relies on the web data seen during training)

        **Example**

        ```python
        |     query = "who got the first nobel prize in physics?"
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
        |     {'query': 'who got the first nobel prize in physics',
        |      'answers':
        |          [{'query': 'who got the first nobel prize in physics',
        |            'answer': ' albert einstein',
        |            'meta': { 'doc_ids': [...],
        |                      'doc_scores': [80.42758 ...],
        |                      'doc_probabilities': [40.71379089355469, ...
        |                      'texts': ['Albert Einstein was a ...]
        |                      'titles': ['"Albert Einstein"', ...]
        |      }}]}
        ```
    """

    def __init__(
            self,
            model_name_or_path: str = "facebook/rag-token-nq",
            retriever: Optional[DensePassageRetriever] = None,
            generator_type: RAGeneratorType = RAGeneratorType.TOKEN,
            top_k_answers: int = 2,
            max_length: int = 200,
            min_length: int = 2,
            num_beams: int = 2,
            embed_title: bool = True,
            prefix: Optional[str] = None,
            use_gpu: bool = True,
    ):
        """
        Load a RAG model from Transformers along with passage_embedding_model.
        See https://huggingface.co/transformers/model_doc/rag.html for more details

        :param model_name_or_path: Directory of a saved model or the name of a public model e.g.
                                   'facebook/rag-token-nq', 'facebook/rag-sequence-nq'.
                                   See https://huggingface.co/models for full list of available models.
        :param retriever: `DensePassageRetriever` used to embedded passage
        :param generator_type: Which RAG generator implementation to use? RAG-TOKEN or RAG-SEQUENCE
        :param top_k_answers: Number of independently generated text to return
        :param max_length: Maximum length of generated text
        :param min_length: Minimum length of generated text
        :param num_beams: Number of beams for beam search. 1 means no beam search.
        :param embed_title: Embedded the title of passage while generating embedding
        :param prefix: The prefix used by the generator's tokenizer.
        :param use_gpu: Whether to use GPU (if available)
        """

        self.model_name_or_path = model_name_or_path
        self.max_length = max_length
        self.min_length = min_length
        self.generator_type = generator_type
        self.num_beams = num_beams
        self.embed_title = embed_title
        self.prefix = prefix
        self.retriever = retriever

        if top_k_answers > self.num_beams:
            top_k_answers = self.num_beams
            logger.warning(f'top_k_answers value should not be greater than num_beams, hence setting it to {num_beams}')

        self.top_k_answers = top_k_answers

        if use_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.tokenizer = RagTokenizer.from_pretrained(model_name_or_path)

        if self.generator_type == RAGeneratorType.SEQUENCE:
            raise NotImplementedError("RagSequenceForGeneration is not implemented yet")
            # TODO: Enable when transformers have it. Refer https://github.com/huggingface/transformers/issues/7905
            # Also refer refer https://github.com/huggingface/transformers/issues/7829
            # self.model = RagSequenceForGeneration.from_pretrained(model_name_or_path)
        else:
            self.model = RagTokenForGeneration.from_pretrained(model_name_or_path).to(self.device)

    # Copied cat_input_and_doc method from transformers.RagRetriever
    # Refer section 2.3 of https://arxiv.org/abs/2005.11401
    def _cat_input_and_doc(self, doc_title: str, doc_text: str, input_string: str, prefix: Optional[str]):
        if doc_title.startswith('"'):
            doc_title = doc_title[1:]
        if doc_title.endswith('"'):
            doc_title = doc_title[:-1]
        if prefix is None:
            prefix = ""
        out = (prefix + doc_title + self.model.config.title_sep + doc_text + self.model.config.doc_sep +
               input_string).replace("  ", " ")

        return out

    # Copied postprocess_docs method from transformers.RagRetriever and modified
    def _get_contextualized_inputs(self, texts: List[str], query: str, titles: Optional[List[str]] = None,
                                   return_tensors: str = "pt"):

        titles_list = titles if self.embed_title and titles is not None else [""] * len(texts)
        prefix = self.prefix if self.prefix is not None else self.model.config.generator.prefix

        rag_input_strings = [
            self._cat_input_and_doc(
                doc_title=titles_list[i],
                doc_text=texts[i],
                input_string=query,
                prefix=prefix
            )
            for i in range(len(texts))
        ]

        contextualized_inputs = self.tokenizer.generator.batch_encode_plus(
            rag_input_strings,
            max_length=self.model.config.max_combined_length,
            return_tensors=return_tensors,
            padding="max_length",
            truncation=True,
        )

        return contextualized_inputs["input_ids"].to(self.device), \
               contextualized_inputs["attention_mask"].to(self.device)

    def _prepare_passage_embeddings(self, docs: List[Document], embeddings: List[Optional[numpy.ndarray]]) -> torch.Tensor:

        # If document missing embedding, then need embedding for all the documents
        is_embedding_required = embeddings is None or any(embedding is None for embedding in embeddings)

        if is_embedding_required:
            if self.retriever is None:
                raise AttributeError("_prepare_passage_embeddings need a DPR instance as self.retriever to embed document")

            embeddings = self.retriever.embed_passages(docs)

        embeddings_in_tensor = torch.cat(
            [torch.from_numpy(embedding).unsqueeze(0) for embedding in embeddings],
            dim=0
        )

        return embeddings_in_tensor.to(self.device)

    def predict(self, query: str, documents: List[Document], top_k: Optional[int] = None) -> Dict:
        """
        Generate the answer to the input query. The generation will be conditioned on the supplied documents.
        These document can for example be retrieved via the Retriever.

        :param query: Query
        :param documents: Related documents (e.g. coming from a retriever) that the answer shall be conditioned on.
        :param top_k: Number of returned answers
        :return: Generated answers plus additional infos in a dict like this:

        ```python
        |     {'query': 'who got the first nobel prize in physics',
        |      'answers':
        |          [{'query': 'who got the first nobel prize in physics',
        |            'answer': ' albert einstein',
        |            'meta': { 'doc_ids': [...],
        |                      'doc_scores': [80.42758 ...],
        |                      'doc_probabilities': [40.71379089355469, ...
        |                      'texts': ['Albert Einstein was a ...]
        |                      'titles': ['"Albert Einstein"', ...]
        |      }}]}
        ```
        """
        torch.set_grad_enabled(False)
        if len(documents) == 0:
            raise AttributeError("generator need documents to predict the answer")

        top_k_answers = top_k if top_k is not None else self.top_k_answers

        if top_k_answers > self.num_beams:
            top_k_answers = self.num_beams
            logger.warning(f'top_k value should not be greater than num_beams, '
                           f'hence setting it to {top_k_answers}')

        # Flatten the documents so easy to reference
        flat_docs_dict: Dict[str, Any] = {}
        for document in documents:
            for k, v in document.__dict__.items():
                if k not in flat_docs_dict:
                    flat_docs_dict[k] = []
                flat_docs_dict[k].append(v)

        # Extract title
        titles = [d.meta["name"] if d.meta and "name" in d.meta else "" for d in documents]

        # Raw document embedding and set device of query_embedding
        passage_embeddings = self._prepare_passage_embeddings(docs=documents, embeddings=flat_docs_dict["embedding"])

        # Query tokenization
        input_dict = self.tokenizer.prepare_seq2seq_batch(
            src_texts=[query],
            return_tensors="pt"
        )
        input_ids = input_dict['input_ids'].to(self.device)
        # Query embedding
        query_embedding = self.model.question_encoder(input_ids)[0]

        # Prepare contextualized input_ids of documents
        # (will be transformed into contextualized inputs inside generator)
        context_input_ids, context_attention_mask = self._get_contextualized_inputs(
            texts=flat_docs_dict["text"],
            titles=titles,
            query=query
        )

        # Compute doc scores from docs_embedding
        doc_scores = torch.bmm(query_embedding.unsqueeze(1),
                               passage_embeddings.unsqueeze(0).transpose(1, 2)).squeeze(1)

        # TODO Need transformers 3.4.0
        # Refer https://github.com/huggingface/transformers/issues/7874
        # Pass it as parameter to generate function as follows -
        # n_docs=len(flat_docs_dict["text"])
        self.model.config.n_docs = len(flat_docs_dict["text"])

        # Get generated ids from generator
        generator_ids = self.model.generate(
            # TODO: Need transformers 3.4.0
            # Refer https://github.com/huggingface/transformers/issues/7871
            # Remove input_ids parameter once upgraded to 3.4.0
            input_ids=input_ids,
            context_input_ids=context_input_ids,
            context_attention_mask=context_attention_mask,
            doc_scores=doc_scores,
            num_return_sequences=top_k_answers,
            num_beams=self.num_beams,
            max_length=self.max_length,
            min_length=self.min_length,
        )

        generated_answers = self.tokenizer.batch_decode(generator_ids, skip_special_tokens=True)
        answers: List[Any] = []

        for generated_answer in generated_answers:
            cur_answer = {
                "query": query,
                "answer": generated_answer,
                "meta": {
                    "doc_ids": flat_docs_dict["id"],
                    "doc_scores": flat_docs_dict["score"],
                    "doc_probabilities": flat_docs_dict["probability"],
                    "texts": flat_docs_dict["text"],
                    "titles": titles,
                }
            }
            answers.append(cur_answer)

        result = {"query": query, "answers": answers}

        return result
