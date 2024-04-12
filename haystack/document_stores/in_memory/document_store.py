import heapq
import math
import re
from collections import Counter
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple

import numpy as np

from haystack import default_from_dict, default_to_dict, logging
from haystack.dataclasses import Document
from haystack.document_stores.errors import DocumentStoreError, DuplicateDocumentError
from haystack.document_stores.types import DuplicatePolicy
from haystack.utils import expit
from haystack.utils.filters import convert, document_matches_filter

logger = logging.getLogger(__name__)

# document scores are essentially unbounded and will be scaled to values between 0 and 1 if scale_score is set to
# True (default). Scaling uses the expit function (inverse of the logit function) after applying a scaling factor
# (e.g., BM25_SCALING_FACTOR for the bm25_retrieval method).
# Larger scaling factor decreases scaled scores. For example, an input of 10 is scaled to 0.99 with BM25_SCALING_FACTOR=2
# but to 0.78 with BM25_SCALING_FACTOR=8 (default). The defaults were chosen empirically. Increase the default if most
# unscaled scores are larger than expected (>30) and otherwise would incorrectly all be mapped to scores ~1.
BM25_SCALING_FACTOR = 8
DOT_PRODUCT_SCALING_FACTOR = 100


class InMemoryDocumentStore:
    """
    Stores data in-memory. It's ephemeral and cannot be saved to disk.
    """

    def __init__(
        self,
        bm25_tokenization_regex: str = r"(?u)\b\w\w+\b",
        bm25_algorithm: Literal["BM25Okapi", "BM25L", "BM25Plus"] = "BM25L",
        bm25_parameters: Optional[Dict] = None,
        embedding_similarity_function: Literal["dot_product", "cosine"] = "dot_product",
    ):
        """
        Initializes the DocumentStore.

        :param bm25_tokenization_regex: The regular expression used to tokenize the text for BM25 retrieval.
        :param bm25_algorithm: The BM25 algorithm to use. One of "BM25Okapi", "BM25L", or "BM25Plus".
        :param bm25_parameters: Parameters for BM25 implementation in a dictionary format.
                                For example: {'k1':1.5, 'b':0.75, 'epsilon':0.25}
                                You can learn more about these parameters by visiting https://github.com/dorianbrown/rank_bm25.
                                By default, no parameters are set.
        :param embedding_similarity_function: The similarity function used to compare Documents embeddings.
                                              One of "dot_product" (default) or "cosine".
                                              To choose the most appropriate function, look for information about your embedding model.
        """
        self.storage: Dict[str, Document] = {}
        self._bm25_tokenization_regex = bm25_tokenization_regex
        self.tokenizer = re.compile(bm25_tokenization_regex).findall

        self.bm25_algorithm_name = bm25_algorithm
        self.bm25_algorithm = self._dispatch_bm25()
        self.bm25_parameters = bm25_parameters or {}
        self.embedding_similarity_function = embedding_similarity_function

        # Global BM25 statistics
        self._avg_doc_len: float = 0.0
        self._freq_doc: Counter = Counter()

        # Per-document statistics
        self._bm25_attr: Dict[str, Tuple[Dict[str, int], int]] = {}

    def _dispatch_bm25(self):
        """
        Select the correct BM25 algorithm based on user specification.

        :returns:
            The BM25 algorithm method.
        """
        # TODO other implementations are not available yet, always return BM25+

        if self.bm25_algorithm_name not in {"BM25Okapi", "BM25L", "BM25Plus"}:
            raise ValueError(f"BM25 algorithm '{self.bm25_algorithm_name}' is not supported.")
        return self._score_bm25plus

    def _tokenize_bm25(self, text: str) -> List[str]:
        """
        Tokenize text using the BM25 tokenization regex.

        Here we explicitly create a tokenization method to encapsulate
        all pre-processing logic used to create BM25 tokens, such as
        lowercasing. This helps track the exact tokenization process
        used for BM25 scoring at any given time.

        :param text:
            The text to tokenize.
        :returns:
            A list of tokens.
        """
        text = text.lower()
        return self.tokenizer(text)

    def _score_bm25plus(
        self, query: str, documents: List[Document], scale_score: bool = False
    ) -> List[Tuple[Document, float]]:
        """
        Calculate BM25+ scores for the given query and filtered documents.

        :param query:
            The query string.
        :param documents:
            The list of documents to score, should be produced by
            the filter_documents method; may be an empty list.
        :param scale_score:
            Whether to scale the scores of the retrieved documents.
        :returns:
            A list of tuples, each containing a Document and its BM25+ score.
        """

        def _compute_idf(tokens: List[str]) -> Dict[str, float]:
            """Per-token IDF computation."""
            n = lambda t: self._freq_doc.get(t, 0)
            idf = {t: math.log(1 + (len(self._index) - n(t) + 0.5) / (n(t) + 0.5)) for t in tokens}
            return idf

        def _compute_damping(doc_len: int) -> float:
            """Shrink effect of token frequency on the final score.
            as a function of document length."""
            return k * (1 - b + b * doc_len / self._avg_doc_len)

        def _compute_bm25(token: str, freq: Dict[str, int], damp: float) -> float:
            """Per-token BM25+ computation."""
            freq_term = freq.get(token, 0.0)
            freq_norm = freq_term / (freq_term + damp) + delta
            return idf[token] * freq_norm

        # Retrieve args
        k = self.bm25_parameters.get("k1", 1.5)
        b = self.bm25_parameters.get("b", 0.75)

        # Adjust the delta value so that we can bring the `(k1 + 1)`
        # term out of the 'term frequency' term in BM25+ formula and
        # delete it; this will not affect the ranking
        delta = self.bm25_parameters.get("delta", 1.0) / (k + 1.0)

        idf = _compute_idf(self._tokenize_bm25(query))
        bm25_attr = {doc.id: self._bm25_attr[doc.id] for doc in documents}

        ret = []
        for doc in documents:
            freq, doc_len = bm25_attr[doc.id]
            damp = _compute_damping(doc_len)

            score = sum(_compute_bm25(tok, freq, damp) for tok in idf.keys())
            if scale_score:
                score = expit(score / BM25_SCALING_FACTOR)
            ret.append((doc, score))

        return ret

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            bm25_tokenization_regex=self._bm25_tokenization_regex,
            bm25_algorithm=self.bm25_algorithm.__name__,
            bm25_parameters=self.bm25_parameters,
            embedding_similarity_function=self.embedding_similarity_function,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InMemoryDocumentStore":
        """
        Deserializes the component from a dictionary.

        :param data:
            The dictionary to deserialize from.
        :returns:
            The deserialized component.
        """
        return default_from_dict(cls, data)

    def count_documents(self) -> int:
        """
        Returns the number of how many documents are present in the DocumentStore.
        """
        return len(self.storage.keys())

    def filter_documents(self, filters: Optional[Dict[str, Any]] = None) -> List[Document]:
        """
        Returns the documents that match the filters provided.

        For a detailed specification of the filters, refer to the DocumentStore.filter_documents() protocol documentation.

        :param filters: The filters to apply to the document list.
        :returns: A list of Documents that match the given filters.
        """
        if filters:
            if "operator" not in filters and "conditions" not in filters:
                filters = convert(filters)
            return [doc for doc in self.storage.values() if document_matches_filter(filters=filters, document=doc)]
        return list(self.storage.values())

    def write_documents(self, documents: List[Document], policy: DuplicatePolicy = DuplicatePolicy.NONE) -> int:
        """
        Refer to the DocumentStore.write_documents() protocol documentation.

        If `policy` is set to `DuplicatePolicy.NONE` defaults to `DuplicatePolicy.FAIL`.
        """
        if (
            not isinstance(documents, Iterable)
            or isinstance(documents, str)
            or any(not isinstance(doc, Document) for doc in documents)
        ):
            raise ValueError("Please provide a list of Documents.")

        if policy == DuplicatePolicy.NONE:
            policy = DuplicatePolicy.FAIL

        written_documents = len(documents)
        for document in documents:
            if policy != DuplicatePolicy.OVERWRITE and document.id in self.storage.keys():
                if policy == DuplicatePolicy.FAIL:
                    raise DuplicateDocumentError(f"ID '{document.id}' already exists.")
                if policy == DuplicatePolicy.SKIP:
                    logger.warning("ID '{document_id}' already exists", document_id=document.id)
                    written_documents -= 1
                    continue

            # Since the statistics are updated in an incremental manner,
            # we need to explicitly remove the existing document to revert
            # the statistics before updating them with the new document.
            if document.id in self.storage.keys():
                self.delete_documents([document.id])

            # This processing logic is extracted from the original bm25_retrieval method.
            # Since we are creating index incrementally before the first retrieval,
            # we need to determine what content to use for indexing here, not at query time.
            if document.content is not None:
                if document.dataframe is not None:
                    logger.warning(
                        "Document '{document_id}' has both text and dataframe content. "
                        "Using text content for retrieval and skipping dataframe content.",
                        document_id=document.id,
                    )
                tokens = self._tokenize_bm25(document.content)
            elif document.dataframe is not None:
                str_content = document.dataframe.astype(str)
                csv_content = str_content.to_csv(index=False)
                tokens = self._tokenize_bm25(csv_content)
            else:
                tokens = []

            self.storage[document.id] = document

            self._bm25_attr[document.id] = (Counter(tokens), len(tokens))
            self._freq_doc.update(set(tokens))
            self._avg_doc_len = (len(tokens) + self._avg_doc_len * len(self._bm25_attr)) / (len(self._bm25_attr) + 1)
        return written_documents

    def delete_documents(self, document_ids: List[str]) -> None:
        """
        Deletes all documents with matching document_ids from the DocumentStore.

        :param document_ids: The object_ids to delete.
        """
        for doc_id in document_ids:
            if doc_id not in self.storage.keys():
                continue
            del self.storage[doc_id]

            # Update statistics accordingly
            freq, doc_len = self._bm25_attr.pop(doc_id)
            self._freq_doc.subtract(Counter(freq.keys()))
            try:
                self._avg_doc_len = (self._avg_doc_len * (len(self._bm25_attr) + 1) - doc_len) / len(self._bm25_attr)
            except ZeroDivisionError:
                self._avg_doc_len = 0

    def bm25_retrieval(
        self, query: str, filters: Optional[Dict[str, Any]] = None, top_k: int = 10, scale_score: bool = False
    ) -> List[Document]:
        """
        Retrieves documents that are most relevant to the query using BM25 algorithm.

        :param query: The query string.
        :param filters: A dictionary with filters to narrow down the search space.
        :param top_k: The number of top documents to retrieve. Default is 10.
        :param scale_score: Whether to scale the scores of the retrieved documents. Default is False.
        :returns: A list of the top_k documents most relevant to the query.
        """
        if not query:
            raise ValueError("Query should be a non-empty string")

        content_type_filter = {
            "operator": "OR",
            "conditions": [
                {"field": "content", "operator": "!=", "value": None},
                {"field": "dataframe", "operator": "!=", "value": None},
            ],
        }
        if filters:
            if "operator" not in filters:
                filters = convert(filters)
            filters = {"operator": "AND", "conditions": [content_type_filter, filters]}
        else:
            filters = content_type_filter

        all_documents = self.filter_documents(filters=filters)
        if len(all_documents) == 0:
            logger.info("No documents found for BM25 retrieval. Returning empty list.")
            return []

        # initialize BM25
        results = self.bm25_algorithm(query, all_documents, scale_score)

        # get the last top_k indexes and reverse them
        top_results = heapq.nlargest(top_k, results, key=lambda x: x[1])

        # BM25Okapi can return meaningful negative values, so they should not be filtered out when scale_score is False.
        # It's the only algorithm supported by rank_bm25 at the time of writing (2024) that can return negative scores.
        # see https://github.com/deepset-ai/haystack/pull/6889 for more context.
        negatives_are_valid = self.bm25_algorithm_name == "BM25Okapi" and not scale_score

        # Create documents with the BM25 score to return them
        return_documents = []
        for doc, score in top_results:
            if not negatives_are_valid and score <= 0.0:
                continue

            doc_fields = doc.to_dict()
            doc_fields["score"] = score
            return_document = Document.from_dict(doc_fields)
            return_documents.append(return_document)

        return return_documents

    def embedding_retrieval(
        self,
        query_embedding: List[float],
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 10,
        scale_score: bool = False,
        return_embedding: bool = False,
    ) -> List[Document]:
        """
        Retrieves documents that are most similar to the query embedding using a vector similarity metric.

        :param query_embedding: Embedding of the query.
        :param filters: A dictionary with filters to narrow down the search space.
        :param top_k: The number of top documents to retrieve. Default is 10.
        :param scale_score: Whether to scale the scores of the retrieved Documents. Default is False.
        :param return_embedding: Whether to return the embedding of the retrieved Documents. Default is False.
        :returns: A list of the top_k documents most relevant to the query.
        """
        if len(query_embedding) == 0 or not isinstance(query_embedding[0], float):
            raise ValueError("query_embedding should be a non-empty list of floats.")

        filters = filters or {}
        all_documents = self.filter_documents(filters=filters)

        documents_with_embeddings = [doc for doc in all_documents if doc.embedding is not None]
        if len(documents_with_embeddings) == 0:
            logger.warning(
                "No Documents found with embeddings. Returning empty list. "
                "To generate embeddings, use a DocumentEmbedder."
            )
            return []
        elif len(documents_with_embeddings) < len(all_documents):
            logger.info(
                "Skipping some Documents that don't have an embedding. "
                "To generate embeddings, use a DocumentEmbedder."
            )

        scores = self._compute_query_embedding_similarity_scores(
            embedding=query_embedding, documents=documents_with_embeddings, scale_score=scale_score
        )

        # create Documents with the similarity score for the top k results
        top_documents = []
        for doc, score in sorted(zip(documents_with_embeddings, scores), key=lambda x: x[1], reverse=True)[:top_k]:
            doc_fields = doc.to_dict()
            doc_fields["score"] = score
            if return_embedding is False:
                doc_fields["embedding"] = None
            top_documents.append(Document.from_dict(doc_fields))

        return top_documents

    def _compute_query_embedding_similarity_scores(
        self, embedding: List[float], documents: List[Document], scale_score: bool = False
    ) -> List[float]:
        """
        Computes the similarity scores between the query embedding and the embeddings of the documents.

        :param embedding: Embedding of the query.
        :param documents: A list of Documents.
        :param scale_score: Whether to scale the scores of the Documents. Default is False.
        :returns: A list of scores.
        """

        query_embedding = np.array(embedding)
        if query_embedding.ndim == 1:
            query_embedding = np.expand_dims(a=query_embedding, axis=0)

        try:
            document_embeddings = np.array([doc.embedding for doc in documents])
        except ValueError as e:
            if "inhomogeneous shape" in str(e):
                raise DocumentStoreError(
                    "The embedding size of all Documents should be the same. "
                    "Please make sure that the Documents have been embedded with the same model."
                ) from e
            raise e
        if document_embeddings.ndim == 1:
            document_embeddings = np.expand_dims(a=document_embeddings, axis=0)

        if self.embedding_similarity_function == "cosine":
            # cosine similarity is a normed dot product
            query_embedding /= np.linalg.norm(x=query_embedding, axis=1, keepdims=True)
            document_embeddings /= np.linalg.norm(x=document_embeddings, axis=1, keepdims=True)

        try:
            scores = np.dot(a=query_embedding, b=document_embeddings.T)[0].tolist()
        except ValueError as e:
            if "shapes" in str(e) and "not aligned" in str(e):
                raise DocumentStoreError(
                    "The embedding size of the query should be the same as the embedding size of the Documents. "
                    "Please make sure that the query has been embedded with the same model as the Documents."
                ) from e
            raise e

        if scale_score:
            if self.embedding_similarity_function == "dot_product":
                scores = [expit(float(score / DOT_PRODUCT_SCALING_FACTOR)) for score in scores]
            elif self.embedding_similarity_function == "cosine":
                scores = [(score + 1) / 2 for score in scores]

        return scores
