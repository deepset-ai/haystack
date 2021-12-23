from typing import List, Optional, Union
import logging
from pathlib import Path

import torch
from torch.nn import DataParallel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from haystack.schema import Document
from haystack.nodes.ranker import BaseRanker
from haystack.modeling.utils import initialize_device_settings

logger = logging.getLogger(__name__)


class SentenceTransformersRanker(BaseRanker):
    """
    Sentence Transformer based pre-trained Cross-Encoder model for Document Re-ranking (https://huggingface.co/cross-encoder).
    Re-Ranking can be used on top of a retriever to boost the performance for document search. This is particularly useful if the retriever has a high recall but is bad in sorting the documents by relevance.

    SentenceTransformerRanker handles Cross-Encoder models
        - use a single logit as similarity score e.g.  cross-encoder/ms-marco-MiniLM-L-12-v2
        - use two output logits (no_answer, has_answer) e.g. deepset/gbert-base-germandpr-reranking
    https://www.sbert.net/docs/pretrained-models/ce-msmarco.html#usage-with-transformers

    |  With a SentenceTransformersRanker, you can:
     - directly get predictions via predict()

    Usage example:
    ...
    retriever = ElasticsearchRetriever(document_store=document_store)
    ranker = SentenceTransformersRanker(model_name_or_path="cross-encoder/ms-marco-MiniLM-L-12-v2")
    p = Pipeline()
    p.add_node(component=retriever, name="ESRetriever", inputs=["Query"])
    p.add_node(component=ranker, name="Ranker", inputs=["ESRetriever"])
    """

    def __init__(
            self,
            model_name_or_path: Union[str, Path],
            model_version: Optional[str] = None,
            top_k: int = 10,
            use_gpu: bool = True,
            devices: Optional[List[Union[int, str, torch.device]]] = None
    ):
        """
        :param model_name_or_path: Directory of a saved model or the name of a public model e.g.
        'cross-encoder/ms-marco-MiniLM-L-12-v2'.
        See https://huggingface.co/cross-encoder for full list of available models
        :param model_version: The version of model to use from the HuggingFace model hub. Can be tag name, branch name, or commit hash.
        :param top_k: The maximum number of documents to return
        :param use_gpu: Whether to use all available GPUs or the CPU. Falls back on CPU if no GPU is available.
        :param devices: List of GPU devices to limit inference to certain GPUs and not use all available ones (e.g. ["cuda:0"]).
        """

        # save init parameters to enable export of component config as YAML
        self.set_config(
            model_name_or_path=model_name_or_path, model_version=model_version,
            top_k=top_k,
        )

        self.top_k = top_k

        if devices is not None:
            self.devices = devices
        else:
            self.devices, _ = initialize_device_settings(use_cuda=use_gpu, multi_gpu=True)
        self.transformer_model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=model_name_or_path, revision=model_version)
        self.transformer_model.to(str(self.devices[0]))
        self.transformer_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name_or_path,
                                                                   revision=model_version)
        self.transformer_model.eval()

        if len(self.devices) > 1:
            self.model = DataParallel(self.transformer_model, device_ids=self.devices)

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

    def predict(self, query: str, documents: List[Document], top_k: Optional[int] = None) -> List[Document]:
        """
        Use loaded ranker model to re-rank the supplied list of Document.

        Returns list of Document sorted by (desc.) similarity with the query.

        :param query: Query string
        :param documents: List of Document to be re-ranked
        :param top_k: The maximum number of documents to return
        :return: List of Document
        """
        if top_k is None:
            top_k = self.top_k

        features = self.transformer_tokenizer([query for doc in documents], [doc.content for doc in documents],
                                              padding=True, truncation=True, return_tensors="pt").to(self.devices[0])

        # SentenceTransformerRanker uses:
        # 1. the logit as similarity score/answerable classification
        # 2. the logits as answerable classification  (no_answer / has_answer)
        # https://www.sbert.net/docs/pretrained-models/ce-msmarco.html#usage-with-transformers
        with torch.no_grad():
            similarity_scores = self.transformer_model(**features).logits

        logits_dim = similarity_scores.shape[1]  # [batch_size, logits_dim]
        sorted_scores_and_documents = sorted(
            zip(similarity_scores, documents), key=lambda similarity_document_tuple:
            # assume the last element in logits represents the `has_answer` label
            similarity_document_tuple[0][-1] if logits_dim >= 2 else similarity_document_tuple[0],
            reverse=True)

        # rank documents according to scores
        sorted_documents = [doc for _, doc in sorted_scores_and_documents]
        return sorted_documents[:top_k]
