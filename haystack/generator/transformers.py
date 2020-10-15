import logging
from enum import Enum
from typing import List, Optional

import numpy
import torch
from transformers import RagSequenceForGeneration, RagTokenForGeneration, RagTokenizer

from haystack import Document
from haystack.generator.base import BaseGenerator

logger = logging.getLogger(__name__)


class RAGGeneratorType(Enum):
    TOKEN = 1,
    SEQUENCE = 2


class RAGGenerator(BaseGenerator):

    def __init__(
            self,
            model_name_or_path: str = "facebook/rag-token-nq",
            generator_type: RAGGeneratorType = RAGGeneratorType.TOKEN,
            top_k_answers: int = 5,
            max_length: int = 200,
            min_length: int = 2,
    ):
        self.model_name_or_path = model_name_or_path
        self.top_k_answers = top_k_answers
        self.max_length = max_length
        self.min_length = min_length
        self.tokenizer = RagTokenizer.from_pretrained(model_name_or_path)
        self.generator_type = generator_type

        if self.generator_type == RAGGeneratorType.SEQUENCE:
            self.model = RagSequenceForGeneration.from_pretrained(model_name_or_path)
        else:
            self.model = RagTokenForGeneration.from_pretrained(model_name_or_path)


    # TODO: Copy from RagRetriever and modified
    # Need to understand the logic
    def _postprocess_docs(self, retrieved_texts: List[str], question: str, return_tensors: str = "pt"):
        def cat_input_and_doc(doc_text, input_string):
            out = (" / " + doc_text + " // " + input_string).replace("  ", " ")
            return out

        rag_input_strings = [
            cat_input_and_doc(
                retrieved_texts[i],
                question,
            )
            for i in range(len(retrieved_texts))
        ]

        contextualized_inputs = self.tokenizer.question_encoder.batch_encode_plus(
            rag_input_strings,
            max_length=300,
            return_tensors=return_tensors,
            padding="max_length",
            truncation=True,
        )

        return contextualized_inputs["input_ids"], contextualized_inputs["attention_mask"]

    def predict(self, question: str, documents: List[Document], top_k: Optional[int] = None):
        if len(documents) == 0 or documents[0].embedding is None:
            raise AttributeError("generator need documents with embedding")

        top_k_answers = top_k or self.top_k_answers

        docs_embedding = []
        retrieved_texts = []
        for doc in documents:
            docs_embedding.append(doc.embedding)
            retrieved_texts.append(doc.text)

        # Question tokenization and encoding
        input_dict = self.tokenizer.prepare_seq2seq_batch(src_texts=[question], return_tensors="pt")
        question_hidden_states = self.model.question_encoder(input_dict["input_ids"])[0]
        embedding_in_tensor = torch.from_numpy(numpy.array(docs_embedding)).to(question_hidden_states).float()

        # Compute doc scores
        doc_scores = torch.bmm(question_hidden_states.unsqueeze(1),
                               embedding_in_tensor.unsqueeze(0).transpose(1, 2)).squeeze(1)

        # Get context_input_ids and context_attention_mask
        context_input_ids, context_attention_mask = self._postprocess_docs(retrieved_texts=retrieved_texts,
                                                                           question=question)

        # Get generated ids from generator
        generator_ids = self.model.generate(
            input_ids=input_dict["input_ids"],
            context_input_ids=context_input_ids,
            context_attention_mask=context_attention_mask,
            doc_scores=doc_scores,
            num_beams=top_k_answers,
            max_length=self.max_length,
            min_length=self.min_length
        )

        answers = self.tokenizer.batch_decode(generator_ids, skip_special_tokens=True)

        result = {"question": question, "answers": answers}

        return result
