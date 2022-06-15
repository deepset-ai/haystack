from typing import List, Optional, Union, Tuple, Iterator
import logging
from pathlib import Path

import torch
from torch.nn import DataParallel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from haystack.errors import HaystackError
from haystack.schema import Document
from haystack.nodes.ranker.base import BaseRanker
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
    retriever = BM25Retriever(document_store=document_store)
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
        devices: Optional[List[Union[str, torch.device]]] = None,
        batch_size: Optional[int] = None,
    ):
        """
        :param model_name_or_path: Directory of a saved model or the name of a public model e.g.
        'cross-encoder/ms-marco-MiniLM-L-12-v2'.
        See https://huggingface.co/cross-encoder for full list of available models
        :param model_version: The version of model to use from the HuggingFace model hub. Can be tag name, branch name, or commit hash.
        :param top_k: The maximum number of documents to return
        :param use_gpu: Whether to use all available GPUs or the CPU. Falls back on CPU if no GPU is available.
        :param devices: List of GPU (or CPU) devices, to limit inference to certain GPUs and not use all available ones
                        The strings will be converted into pytorch devices, so use the string notation described here:
                        https://pytorch.org/docs/stable/tensor_attributes.html?highlight=torch%20device#torch.torch.device
                        (e.g. ["cuda:0"]).
        :param batch_size: Number of documents to process at a time.
        """
        super().__init__()

        self.top_k = top_k

        if devices is not None:
            self.devices = [torch.device(device) for device in devices]
        else:
            self.devices, _ = initialize_device_settings(use_cuda=use_gpu, multi_gpu=True)

        self.transformer_model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=model_name_or_path, revision=model_version
        )
        self.transformer_model.to(str(self.devices[0]))
        self.transformer_tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=model_name_or_path, revision=model_version
        )
        self.transformer_model.eval()

        if len(self.devices) > 1:
            self.model = DataParallel(self.transformer_model, device_ids=self.devices)

        self.batch_size = batch_size

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

        features = self.transformer_tokenizer(
            [query for doc in documents],
            [doc.content for doc in documents],
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.devices[0])

        # SentenceTransformerRanker uses:
        # 1. the logit as similarity score/answerable classification
        # 2. the logits as answerable classification  (no_answer / has_answer)
        # https://www.sbert.net/docs/pretrained-models/ce-msmarco.html#usage-with-transformers
        with torch.no_grad():
            similarity_scores = self.transformer_model(**features).logits

        logits_dim = similarity_scores.shape[1]  # [batch_size, logits_dim]
        sorted_scores_and_documents = sorted(
            zip(similarity_scores, documents),
            key=lambda similarity_document_tuple:
            # assume the last element in logits represents the `has_answer` label
            similarity_document_tuple[0][-1] if logits_dim >= 2 else similarity_document_tuple[0],
            reverse=True,
        )

        # rank documents according to scores
        sorted_documents = [doc for _, doc in sorted_scores_and_documents]
        return sorted_documents[:top_k]

    def predict_batch(
        self,
        queries: List[str],
        documents: Union[List[Document], List[List[Document]]],
        top_k: Optional[int] = None,
        batch_size: Optional[int] = None,
    ) -> Union[List[Document], List[List[Document]]]:
        """
        Use loaded ranker model to re-rank the supplied lists of Documents.

        Returns lists of Documents sorted by (desc.) similarity with the corresponding queries.


        - If you provide a list containing a single query...

            - ... and a single list of Documents, the single list of Documents will be re-ranked based on the
              supplied query.
            - ... and a list of lists of Documents, each list of Documents will be re-ranked individually based on the
              supplied query.


        - If you provide a list of multiple queries...

            - ... you need to provide a list of lists of Documents. Each list of Documents will be re-ranked based on
              its corresponding query.

        :param queries: Single query string or list of queries
        :param documents: Single list of Documents or list of lists of Documents to be reranked.
        :param top_k: The maximum number of documents to return per Document list.
        :param batch_size: Number of Documents to process at a time.
        """
        if top_k is None:
            top_k = self.top_k

        if batch_size is None:
            batch_size = self.batch_size

        number_of_docs, all_queries, all_docs, single_list_of_docs = self._preprocess_batch_queries_and_docs(
            queries=queries, documents=documents
        )

        batches = self._get_batches(all_queries=all_queries, all_docs=all_docs, batch_size=batch_size)
        preds = []
        for cur_queries, cur_docs in batches:
            features = self.transformer_tokenizer(
                cur_queries, [doc.content for doc in cur_docs], padding=True, truncation=True, return_tensors="pt"
            ).to(self.devices[0])

            with torch.no_grad():
                similarity_scores = self.transformer_model(**features).logits
                preds.extend(similarity_scores)

        logits_dim = similarity_scores.shape[1]  # [batch_size, logits_dim]
        if single_list_of_docs:
            sorted_scores_and_documents = sorted(
                zip(similarity_scores, documents),
                key=lambda similarity_document_tuple:
                # assume the last element in logits represents the `has_answer` label
                similarity_document_tuple[0][-1] if logits_dim >= 2 else similarity_document_tuple[0],
                reverse=True,
            )

            # rank documents according to scores
            sorted_documents = [doc for _, doc in sorted_scores_and_documents if isinstance(doc, Document)]
            return sorted_documents[:top_k]
        else:
            # Group predictions together
            grouped_predictions = []
            left_idx = 0
            right_idx = 0
            for number in number_of_docs:
                right_idx = left_idx + number
                grouped_predictions.append(similarity_scores[left_idx:right_idx])
                left_idx = right_idx

            result = []
            for pred_group, doc_group in zip(grouped_predictions, documents):
                sorted_scores_and_documents = sorted(
                    zip(pred_group, doc_group),  # type: ignore
                    key=lambda similarity_document_tuple:
                    # assume the last element in logits represents the `has_answer` label
                    similarity_document_tuple[0][-1] if logits_dim >= 2 else similarity_document_tuple[0],
                    reverse=True,
                )

                # rank documents according to scores
                sorted_documents = [doc for _, doc in sorted_scores_and_documents if isinstance(doc, Document)][:top_k]
                result.append(sorted_documents)

            return result

    def _preprocess_batch_queries_and_docs(
        self, queries: List[str], documents: Union[List[Document], List[List[Document]]]
    ) -> Tuple[List[int], List[str], List[Document], bool]:
        number_of_docs = []
        all_queries = []
        all_docs: List[Document] = []
        single_list_of_docs = False

        # Docs case 1: single list of Documents -> rerank single list of Documents based on single query
        if len(documents) > 0 and isinstance(documents[0], Document):
            if len(queries) != 1:
                raise HaystackError("Number of queries must be 1 if a single list of Documents is provided.")
            query = queries[0]
            number_of_docs = [len(documents)]
            all_queries = [query] * len(documents)
            all_docs = documents  # type: ignore
            single_list_of_docs = True

        # Docs case 2: list of lists of Documents -> rerank each list of Documents based on corresponding query
        # If queries contains a single query, apply it to each list of Documents
        if len(documents) > 0 and isinstance(documents[0], list):
            if len(queries) == 1:
                queries = queries * len(documents)
            if len(queries) != len(documents):
                raise HaystackError("Number of queries must be equal to number of provided Document lists.")
            for query, cur_docs in zip(queries, documents):
                if not isinstance(cur_docs, list):
                    raise HaystackError(f"cur_docs was of type {type(cur_docs)}, but expected a list of Documents.")
                number_of_docs.append(len(cur_docs))
                all_queries.extend([query] * len(cur_docs))
                all_docs.extend(cur_docs)

        return number_of_docs, all_queries, all_docs, single_list_of_docs

    @staticmethod
    def _get_batches(
        all_queries: List[str], all_docs: List[Document], batch_size: Optional[int]
    ) -> Iterator[Tuple[List[str], List[Document]]]:
        if batch_size is None:
            yield all_queries, all_docs
            return
        else:
            for index in range(0, len(all_queries), batch_size):
                yield all_queries[index : index + batch_size], all_docs[index : index + batch_size]
