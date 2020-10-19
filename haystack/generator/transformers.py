import logging
from enum import Enum
from typing import List, Optional, Union

import numpy
import torch
from transformers import DPRContextEncoder, DPRContextEncoderTokenizerFast, DPRContextEncoderTokenizer, \
    RagSequenceForGeneration, RagTokenizer, RagTokenForGeneration

from haystack import Document
from haystack.generator.base import BaseGenerator

logger = logging.getLogger(__name__)


class RAGeneratorType(Enum):
    TOKEN = 1,
    SEQUENCE = 2


class RAGenerator(BaseGenerator):
    """
        Implementation of Facebook's Retrieval-Augmented Generator (https://arxiv.org/abs/2005.11401) based on
        HuggingFace's transformers (https://huggingface.co/transformers/model_doc/rag.html).

        |  With the generator, you can:

            - directly get generate predictions via predict()
    """

    def __init__(
            self,
            model_name_or_path: str = "facebook/rag-token-nq",
            passage_embedding_model: Optional[str] = None,
            generator_type: RAGeneratorType = RAGeneratorType.TOKEN,
            top_k_answers: int = 2,
            max_length: int = 200,
            min_length: int = 2,
            num_beams: int = 2,
            embed_title: bool = True,
            prefix: Optional[str] = None,
            use_gpu: bool = True
    ):
        """
        Load a RAG model from Transformers along with passage_embedding_model.
        See https://huggingface.co/transformers/model_doc/rag.html for more details

        :param model_name_or_path: Directory of a saved model or the name of a public model e.g.
                                   'facebook/rag-token-nq', 'facebook/rag-sequence-nq'.
                                   See https://huggingface.co/models for full list of available models.
        :param passage_embedding_model: Local path or remote name of passage encoder checkpoint. The format equals the
                                        one used by hugging-face transformers' model hub models.
                                        Currently available remote names: ``"facebook/dpr-ctx_encoder-single-nq-base"``
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
        self.top_k_answers = top_k_answers
        self.max_length = max_length
        self.min_length = min_length
        self.generator_type = generator_type
        self.num_beams = num_beams
        self.embed_title = embed_title
        self.prefix = prefix

        if use_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.tokenizer = RagTokenizer.from_pretrained(model_name_or_path)

        if self.generator_type == RAGeneratorType.SEQUENCE:
            self.model = RagSequenceForGeneration.from_pretrained(model_name_or_path)
        else:
            self.model = RagTokenForGeneration.from_pretrained(model_name_or_path)

        if passage_embedding_model is not None:
            self.passage_tokenizer = DPRContextEncoderTokenizerFast.from_pretrained(passage_embedding_model)
            self.passage_encoder = DPRContextEncoder.from_pretrained(passage_embedding_model).to(self.device)

    def set_passage_tokenizer_and_encoder(
            self,
            tokenizer: Union[DPRContextEncoderTokenizer, DPRContextEncoderTokenizerFast],
            encoder: DPRContextEncoder
    ):
        self.passage_tokenizer = tokenizer
        self.passage_encoder = encoder

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
    def _get_contextualized_inputs(self, texts: List[str], question: str, titles: Optional[List[str]] = None,
                                   return_tensors: str = "pt"):

        titles_list = titles if self.embed_title and titles is not None else [""] * len(texts)
        prefix = self.prefix if self.prefix is not None else self.model.config.generator.prefix

        rag_input_strings = [
            self._cat_input_and_doc(
                doc_title=titles_list[i],
                doc_text=texts[i],
                input_string=question,
                prefix=prefix
            )
            for i in range(len(texts))
        ]

        contextualized_inputs = self.tokenizer.question_encoder.batch_encode_plus(
            rag_input_strings,
            max_length=self.model.config.max_length,
            return_tensors=return_tensors,
            padding="max_length",
            truncation=True,
        )

        return contextualized_inputs["input_ids"].to(self.device), \
               contextualized_inputs["attention_mask"].to(self.device)

    def _embed_passage(self, title: str, text: str) -> torch.Tensor:
        if self.passage_tokenizer is None or self.passage_encoder is None:
            raise AttributeError("_embed_passage need self.passage_tokenizer and self.passage_encoder")

        """Compute the DPR embeddings of document passages"""
        input_ids = self.passage_tokenizer(
            title,
            text,
            truncation=True,
            padding="longest",
            return_tensors="pt"
        )["input_ids"]

        embeddings = self.passage_encoder(
            input_ids.to(device=self.device),
            return_dict=True
        ).pooler_output

        return embeddings.detach().to(self.device)

    def embed_passage_in_tensor(
        self,
        titles: List[str],
        texts: List[str],
        embeddings: List[Optional[numpy.ndarray]],
        batch_size: Optional[int] = 16
    ) -> torch.Tensor:

        embeddings_in_tensor: List[torch.Tensor] = []

        # If of document missing embedding, then need embedding for all the documents
        is_embedding_required = embeddings.__contains__(None)

        if is_embedding_required:
            for title, text in zip(titles, texts):
                embeddings_in_tensor.append(
                    self._embed_passage(
                        title=title,
                        text=text
                    )
                )
        else:
            embeddings_in_tensor = [torch.from_numpy(embedding) for embedding in embeddings]

        return torch.cat(embeddings_in_tensor, dim=0).to(self.device)

    def predict(self, question: str, documents: List[Document], top_k: Optional[int] = None):
        if len(documents) == 0:
            raise AttributeError("generator need documents to predict the answer")

        top_k_answers = top_k if top_k is None else self.top_k_answers

        # Flatten the documents so easy to reference
        flat_docs_dict = {}
        for document in documents:
            for k, v in document.__dict__.items():
                if k not in flat_docs_dict:
                    flat_docs_dict[k] = []
                flat_docs_dict[k].append(v)

        # Extract title
        titles = [d.meta["name"] if d.meta and "name" in d.meta else "" for d in documents]

        # Raw document embedding and set device of question_embedding
        passage_embeddings = self.embed_passage_in_tensor(
            titles=titles,
            texts=flat_docs_dict["text"],
            embeddings=flat_docs_dict["embedding"]
        )

        # Question tokenization
        input_dict = self.tokenizer.prepare_seq2seq_batch(
            src_texts=[question],
            return_tensors="pt"
        )

        # Question embedding
        question_embedding = self.model.question_encoder(input_dict["input_ids"])[0]

        # Prepare contextualized input_ids of documents
        # (will be transformed into contextualized inputs inside generator)
        context_input_ids, context_attention_mask = self._get_contextualized_inputs(
            texts=flat_docs_dict["text"],
            titles=titles,
            question=question
        )

        # Compute doc scores from docs_embedding
        doc_scores = torch.bmm(question_embedding.unsqueeze(1),
                               passage_embeddings.unsqueeze(0).transpose(1, 2)).squeeze(1)

        # TODO Bug in extend_enc_output function of generator
        # Refer https://github.com/huggingface/transformers/issues/7874
        self.model.config.n_docs = len(flat_docs_dict["text"])

        # Get generated ids from generator
        # TODO: Handle RagSequenceForGeneration case refer https://github.com/huggingface/transformers/issues/7829
        generator_ids = self.model.generate(
            context_input_ids=context_input_ids,
            context_attention_mask=context_attention_mask,
            doc_scores=doc_scores,
            num_return_sequences=top_k_answers,
            num_beams=self.num_beams,
            max_length=self.max_length,
            min_length=self.min_length
        )

        result = {"question": question, "answers": []}
        answers = self.tokenizer.batch_decode(generator_ids, skip_special_tokens=True)

        for answer in answers:
            cur_answer = {
                "question": question,
                "answer": answer,
                "meta": {
                    "doc_ids": flat_docs_dict["id"],
                    "doc_scores": flat_docs_dict["score"],
                    "doc_probabilities": flat_docs_dict["probability"],
                    "texts": flat_docs_dict["text"],
                    "titles": titles
                    # TODO: Meta as well?
                }
            }
            result["answers"].append(cur_answer)

        return result
