import logging
from abc import abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from haystack.document_stores.base import BaseDocumentStore
from haystack.schema import Document

logger = logging.getLogger(__name__)


class _BaseEmbeddingEncoder:
    @abstractmethod
    def embed_queries(self, queries: List[str]) -> np.ndarray:
        """
        Create embeddings for a list of queries.

        :param queries: List of queries to embed.
        :return: Embeddings, one per input query, shape: (queries, embedding_dim)
        """
        pass

    @abstractmethod
    def embed_documents(self, docs: List[Document]) -> np.ndarray:
        """
        Create embeddings for a list of documents.

        :param docs: List of documents to embed.
        :return: Embeddings, one per input document, shape: (documents, embedding_dim)
        """
        pass

    def train(
        self,
        training_data: List[Dict[str, Any]],
        learning_rate: float = 2e-5,
        n_epochs: int = 1,
        num_warmup_steps: Optional[int] = None,
        batch_size: int = 16,
    ):
        """
        Trains or adapts the underlying embedding model.

        Each training data example is a dictionary with the following keys:

        * question: The question string.
        * pos_doc: Positive document string (the document containing the answer).
        * neg_doc: Negative document string (the document that doesn't contain the answer).
        * score: The score margin the answer must fall within.


        :param training_data: The training data in a dictionary format. Required.
        :type training_data: List[Dict[str, Any]]
        :param learning_rate: The speed at which the model learns. Required. We recommend that you leave the default `2e-5` value.
        :type learning_rate: float
        :param n_epochs: The number of epochs (complete passes of the training data through the algorithm) that you want the model to go through. Required.
        :type n_epochs: int
        :param num_warmup_steps: The number of warmup steps for the model. Warmup steps are epochs when the learning rate is very low. You can use them at the beginning of the training to prevent early overfitting of your model. Required.
        :type num_warmup_steps: int
        :param batch_size: The batch size to use for the training. Optional. The default values is 16.
        :type batch_size: int (optional)
        """
        pass

    def save(self, save_dir: Union[Path, str]):
        """
        Save the model to the directory you specify.

        :param save_dir: The directory where the model is saved. Required.
        :type save_dir: Union[Path, str]
        """
        pass

    def _check_docstore_similarity_function(self, document_store: BaseDocumentStore, model_name: str):
        """
        Check that document_store uses a similarity function
        compatible with the embedding model
        """
        if "sentence-transformers" in model_name.lower():
            model_similarity = None
            if "-cos-" in model_name.lower():
                model_similarity = "cosine"
            elif "-dot-" in model_name.lower():
                model_similarity = "dot_product"

            if model_similarity is not None and document_store.similarity != model_similarity:
                logger.warning(
                    "You seem to be using %s model with the %s function instead of the recommended %s. "
                    "This can be set when initializing the DocumentStore",
                    model_name,
                    document_store.similarity,
                    model_similarity,
                )
        elif "dpr" in model_name.lower() and document_store.similarity != "dot_product":
            logger.warning(
                "You seem to be using a DPR model with the %s function. "
                "We recommend using dot_product instead. "
                "This can be set when initializing the DocumentStore",
                document_store.similarity,
            )
