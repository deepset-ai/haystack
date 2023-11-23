import re
from typing import Literal, Any, Dict, List, Optional, Iterable

import logging

import numpy as np
import rank_bm25
from tqdm.auto import tqdm

from haystack.preview import default_from_dict, default_to_dict
from haystack.preview.document_stores.decorator import document_store
from haystack.preview.dataclasses import Document
from haystack.preview.document_stores.protocols import DuplicatePolicy
from haystack.preview.utils.filters import document_matches_filter
from haystack.preview.document_stores.errors import DuplicateDocumentError, DocumentStoreError
from haystack.preview.utils import expit

logger = logging.getLogger(__name__)

# document scores are essentially unbounded and will be scaled to values between 0 and 1 if scale_score is set to
# True (default). Scaling uses the expit function (inverse of the logit function) after applying a scaling factor
# (e.g., BM25_SCALING_FACTOR for the bm25_retrieval method).
# Larger scaling factor decreases scaled scores. For example, an input of 10 is scaled to 0.99 with BM25_SCALING_FACTOR=2
# but to 0.78 with BM25_SCALING_FACTOR=8 (default). The defaults were chosen empirically. Increase the default if most
# unscaled scores are larger than expected (>30) and otherwise would incorrectly all be mapped to scores ~1.
BM25_SCALING_FACTOR = 8
DOT_PRODUCT_SCALING_FACTOR = 100


@document_store
class InMemoryDocumentStore:
    """
    Stores data in-memory. It's ephemeral and cannot be saved to disk.
    """

    def __init__(
        self,
        bm25_tokenization_regex: str = r"(?u)\b\w\w+\b",
        bm25_algorithm: Literal["BM25Okapi", "BM25L", "BM25Plus"] = "BM25Okapi",
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
        algorithm_class = getattr(rank_bm25, bm25_algorithm)
        if algorithm_class is None:
            raise ValueError(f"BM25 algorithm '{bm25_algorithm}' not found.")
        self.bm25_algorithm = algorithm_class
        self.bm25_parameters = bm25_parameters or {}
        self.embedding_similarity_function = embedding_similarity_function

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes this store to a dictionary.
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
        Deserializes the store from a dictionary.
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

        Filters are defined as nested dictionaries. The keys of the dictionaries can be a logical operator (`"$and"`,
        `"$or"`, `"$not"`), a comparison operator (`"$eq"`, `$ne`, `"$in"`, `$nin`, `"$gt"`, `"$gte"`, `"$lt"`,
        `"$lte"`) or a metadata field name.

        Logical operator keys take a dictionary of metadata field names and/or logical operators as value. Metadata
        field names take a dictionary of comparison operators as value. Comparison operator keys take a single value or
        (in case of `"$in"`) a list of values as value. If no logical operator is provided, `"$and"` is used as default
        operation. If no comparison operator is provided, `"$eq"` (or `"$in"` if the comparison value is a list) is used
        as default operation.

        Example:

        ```python
        filters = {
            "$and": {
                "type": {"$eq": "article"},
                "date": {"$gte": "2015-01-01", "$lt": "2021-01-01"},
                "rating": {"$gte": 3},
                "$or": {
                    "genre": {"$in": ["economy", "politics"]},
                    "publisher": {"$eq": "nytimes"}
                }
            }
        }
        # or simpler using default operators
        filters = {
            "type": "article",
            "date": {"$gte": "2015-01-01", "$lt": "2021-01-01"},
            "rating": {"$gte": 3},
            "$or": {
                "genre": ["economy", "politics"],
                "publisher": "nytimes"
            }
        }
        ```

        To use the same logical operator multiple times on the same level, logical operators can take a list of
        dictionaries as value.

        Example:

        ```python
        filters = {
            "$or": [
                {
                    "$and": {
                        "Type": "News Paper",
                        "Date": {
                            "$lt": "2019-01-01"
                        }
                    }
                },
                {
                    "$and": {
                        "Type": "Blog Post",
                        "Date": {
                            "$gte": "2019-01-01"
                        }
                    }
                }
            ]
        }
        ```

        :param filters: The filters to apply to the document list.
        :return: A list of Documents that match the given filters.
        """
        if filters:
            return [doc for doc in self.storage.values() if document_matches_filter(conditions=filters, document=doc)]
        return list(self.storage.values())

    def write_documents(self, documents: List[Document], policy: DuplicatePolicy = DuplicatePolicy.FAIL) -> int:
        """
        Writes (or overwrites) documents into the DocumentStore.

        :param documents: A list of documents.
        :param policy: Documents with the same ID count as duplicates. When duplicates are met,
            the DocumentStore can:
             - skip: keep the existing document and ignore the new one.
             - overwrite: remove the old document and write the new one.
             - fail: an error is raised.
        :raises DuplicateError: Exception trigger on duplicate document if `policy=DuplicatePolicy.FAIL`
        :return: None
        """
        if (
            not isinstance(documents, Iterable)
            or isinstance(documents, str)
            or any(not isinstance(doc, Document) for doc in documents)
        ):
            raise ValueError("Please provide a list of Documents.")

        written_documents = len(documents)
        for document in documents:
            if policy != DuplicatePolicy.OVERWRITE and document.id in self.storage.keys():
                if policy == DuplicatePolicy.FAIL:
                    raise DuplicateDocumentError(f"ID '{document.id}' already exists.")
                if policy == DuplicatePolicy.SKIP:
                    logger.warning("ID '%s' already exists", document.id)
                    written_documents -= 1
                    continue
            self.storage[document.id] = document
        return written_documents

    def delete_documents(self, document_ids: List[str]) -> None:
        """
        Deletes all documents with matching document_ids from the DocumentStore.
        :param object_ids: The object_ids to delete.
        """
        for doc_id in document_ids:
            if doc_id not in self.storage.keys():
                continue
            del self.storage[doc_id]

    def bm25_retrieval(
        self, query: str, filters: Optional[Dict[str, Any]] = None, top_k: int = 10, scale_score: bool = False
    ) -> List[Document]:
        """
        Retrieves documents that are most relevant to the query using BM25 algorithm.

        :param query: The query string.
        :param filters: A dictionary with filters to narrow down the search space.
        :param top_k: The number of top documents to retrieve. Default is 10.
        :param scale_score: Whether to scale the scores of the retrieved documents. Default is False.
        :return: A list of the top_k documents most relevant to the query.
        """
        if not query:
            raise ValueError("Query should be a non-empty string")

        content_type_filter = {"$or": {"content": {"$not": None}, "dataframe": {"$not": None}}}
        if filters:
            filters = {"$and": [content_type_filter, filters]}
        else:
            filters = content_type_filter
        all_documents = self.filter_documents(filters=filters)

        # Lowercase all documents
        lower_case_documents = []
        for doc in all_documents:
            if doc.content is None and doc.dataframe is None:
                logger.info("Document '%s' has no text or dataframe content. Skipping it.", doc.id)
            else:
                if doc.content is not None:
                    lower_case_documents.append(doc.content.lower())
                    if doc.dataframe is not None:
                        logger.warning(
                            "Document '%s' has both text and dataframe content. "
                            "Using text content and skipping dataframe content.",
                            doc.id,
                        )
                        continue
                if doc.dataframe is not None:
                    str_content = doc.dataframe.astype(str)
                    csv_content = str_content.to_csv(index=False)
                    lower_case_documents.append(csv_content.lower())

        # Tokenize the entire content of the DocumentStore
        tokenized_corpus = [
            self.tokenizer(doc) for doc in tqdm(lower_case_documents, unit=" docs", desc="Ranking by BM25...")
        ]
        if len(tokenized_corpus) == 0:
            logger.info("No documents found for BM25 retrieval. Returning empty list.")
            return []

        # initialize BM25
        bm25_scorer = self.bm25_algorithm(tokenized_corpus, **self.bm25_parameters)
        # tokenize query
        tokenized_query = self.tokenizer(query.lower())
        # get scores for the query against the corpus
        docs_scores = bm25_scorer.get_scores(tokenized_query)
        if scale_score:
            docs_scores = [expit(float(score / BM25_SCALING_FACTOR)) for score in docs_scores]
        # get the last top_k indexes and reverse them
        top_docs_positions = np.argsort(docs_scores)[-top_k:][::-1]

        # Create documents with the BM25 score to return them
        return_documents = []
        for i in top_docs_positions:
            doc = all_documents[i]
            doc_fields = doc.to_dict()
            doc_fields["score"] = docs_scores[i]
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
        :return: A list of the top_k documents most relevant to the query.
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
        :return: A list of scores.
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
