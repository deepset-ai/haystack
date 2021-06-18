import logging
from pathlib import Path
from typing import List, Optional, Union

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from haystack import Document
from haystack.ranker.base import BaseRanker

logger = logging.getLogger(__name__)


class SentenceTransformersRanker(BaseRanker):
    """
    Sentence Transformer based pre-trained Cross-Encoder model for Document Re-ranking (https://huggingface.co/cross-encoder).

    |  With a SentenceTransformersRanker, you can:

     - directly get predictions via predict()
    """

    def __init__(
            self,
            model_name_or_path: Union[str, Path],
            model_version: Optional[str] = None,
            top_k: int = 10
    ):

        """
        :param model_name_or_path: Directory of a saved model or the name of a public model e.g. 'bert-base-cased',
        'cross-encoder/ms-marco-MiniLM-L-12-v2'.
        See https://huggingface.co/cross-encoder for full list of available models
        :param model_version: The version of model to use from the HuggingFace model hub. Can be tag name, branch name, or commit hash.
        :param batch_size: Number of samples the model receives in one batch for inference.
                           Memory consumption is much lower in inference mode. Recommendation: Increase the batch size
                           to a value so only a single batch is used.
        :param use_gpu: Whether to use GPU (if available)
        :param top_k: The maximum number of documents to return
        :param num_processes: The number of processes for `multiprocessing.Pool`. Set to value of 0 to disable
                              multiprocessing. Set to None to let Inferencer determine optimum number. If you
                              want to debug the Language Model, you might need to disable multiprocessing!
        :param max_seq_len: Max sequence length of one input text for the model
        """

        # save init parameters to enable export of component config as YAML
        self.set_config(
            model_name_or_path=model_name_or_path, model_version=model_version,
            top_k=top_k,
        )

        self.top_k = top_k

        self.transformer_model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=model_name_or_path, revision=model_version)
        self.transformer_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name_or_path, revision=model_version)
        self.transformer_model.eval()

    def predict_batch(self, query_doc_list: List[dict], top_k: int = None, batch_size: int = None):
        """
        Use loaded Ranker model to, for a list of queries, rank each query's supplied list of Document.

        Returns list of dictionary of query and list of document sorted by (desc.) similarity with query

        :param query_doc_list: List of dictionaries containing queries with their retrieved documents
        :param top_k: The maximum number of answers to return for each query
        :param batch_size: Number of samples the model receives in one batch for inference
        :return: List of dictionaries containing query and ranked list of Document
        """
        raise NotImplementedError

    def predict(self, query: str, documents: List[Document], top_k: Optional[int] = None):
        """
        Use loaded ranker model to re-rank the supplied list of Document.

        Returns list of Document sorted by (desc.) TextPairClassification similarity with the query.

        :param query: Query string
        :param documents: List of Document to be re-ranked
        :param top_k: The maximum number of documents to return
        :return: List of Document
        """
        if top_k is None:
            top_k = self.top_k

        features = self.transformer_tokenizer([query for doc in documents], [doc.text for doc in documents],
                                              padding=True, truncation=True, return_tensors="pt")

        # In contrast to FARMRanker, SentenceTransformerRanker uses the logit as similarity score and not the classifier's probability of label "1"
        # https://www.sbert.net/docs/pretrained-models/ce-msmarco.html#usage-with-transformers
        with torch.no_grad():
            similarity_scores = self.transformer_model(**features).logits

        # rank documents according to scores
        sorted_scores_and_documents = sorted(zip(similarity_scores, documents),
                                             key=lambda similarity_document_tuple: similarity_document_tuple[0],
                                             reverse=True)
        sorted_documents = [doc for _, doc in sorted_scores_and_documents]
        return sorted_documents[:top_k]
