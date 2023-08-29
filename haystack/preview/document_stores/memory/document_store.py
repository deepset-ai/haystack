import re
from typing import Literal, Any, Dict, List, Optional, Iterable

import logging

import numpy as np
import rank_bm25
from tqdm.auto import tqdm

from haystack.preview import default_from_dict, default_to_dict
from haystack.preview.document_stores.decorator import document_store
from haystack.preview.dataclasses import Document
from haystack.preview.document_stores.protocols import DuplicatePolicy, DocumentStore
from haystack.preview.document_stores.memory._filters import match
from haystack.preview.document_stores.errors import DuplicateDocumentError, MissingDocumentError
from haystack.preview.utils import expit

logger = logging.getLogger(__name__)

# document scores are essentially unbounded and will be scaled to values between 0 and 1 if scale_score is set to
# True (default). Scaling uses the expit function (inverse of the logit function) after applying a SCALING_FACTOR. A
# larger SCALING_FACTOR decreases scaled scores. For example, an input of 10 is scaled to 0.99 with SCALING_FACTOR=2
# but to 0.78 with SCALING_FACTOR=8 (default). The default was chosen empirically. Increase the default if most
# unscaled scores are larger than expected (>30) and otherwise would incorrectly all be mapped to scores ~1.
SCALING_FACTOR = 8


@document_store
class MemoryDocumentStore:
    """
    Stores data in-memory. It's ephemeral and cannot be saved to disk.
    """

    def __init__(
        self,
        bm25_tokenization_regex: str = r"(?u)\b\w\w+\b",
        bm25_algorithm: Literal["BM25Okapi", "BM25L", "BM25Plus"] = "BM25Okapi",
        bm25_parameters: Optional[Dict] = None,
    ):
        """
        Initializes the DocumentStore.
        """
        self.storage: Dict[str, Document] = {}
        self._bm25_tokenization_regex = bm25_tokenization_regex
        self.tokenizer = re.compile(bm25_tokenization_regex).findall
        algorithm_class = getattr(rank_bm25, bm25_algorithm)
        if algorithm_class is None:
            raise ValueError(f"BM25 algorithm '{bm25_algorithm}' not found.")
        self.bm25_algorithm = algorithm_class
        self.bm25_parameters = bm25_parameters or {}

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes this store to a dictionary.
        """
        return default_to_dict(
            self,
            bm25_tokenization_regex=self._bm25_tokenization_regex,
            bm25_algorithm=self.bm25_algorithm.__name__,
            bm25_parameters=self.bm25_parameters,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentStore":
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

        :param filters: the filters to apply to the document list.
        :return: a list of Documents that match the given filters.
        """
        if filters:
            return [doc for doc in self.storage.values() if match(conditions=filters, document=doc)]
        return list(self.storage.values())

    def write_documents(self, documents: List[Document], policy: DuplicatePolicy = DuplicatePolicy.FAIL) -> None:
        """
        Writes (or overwrites) documents into the DocumentStore.

        :param documents: a list of documents.
        :param policy: documents with the same ID count as duplicates. When duplicates are met,
            the DocumentStore can:
             - skip: keep the existing document and ignore the new one.
             - overwrite: remove the old document and write the new one.
             - fail: an error is raised
        :raises DuplicateError: Exception trigger on duplicate document if `policy=DuplicatePolicy.FAIL`
        :return: None
        """
        if (
            not isinstance(documents, Iterable)
            or isinstance(documents, str)
            or any(not isinstance(doc, Document) for doc in documents)
        ):
            raise ValueError("Please provide a list of Documents.")

        for document in documents:
            if policy != DuplicatePolicy.OVERWRITE and document.id in self.storage.keys():
                if policy == DuplicatePolicy.FAIL:
                    raise DuplicateDocumentError(f"ID '{document.id}' already exists.")
                if policy == DuplicatePolicy.SKIP:
                    logger.warning("ID '%s' already exists", document.id)
            self.storage[document.id] = document

    def delete_documents(self, document_ids: List[str]) -> None:
        """
        Deletes all documents with a matching document_ids from the DocumentStore.
        Fails with `MissingDocumentError` if no document with this id is present in the DocumentStore.

        :param object_ids: the object_ids to delete
        """
        for doc_id in document_ids:
            if not doc_id in self.storage.keys():
                raise MissingDocumentError(f"ID '{doc_id}' not found, cannot delete it.")
            del self.storage[doc_id]

    def bm25_retrieval(
        self, query: str, filters: Optional[Dict[str, Any]] = None, top_k: int = 10, scale_score: bool = True
    ) -> List[Document]:
        """
        Retrieves documents that are most relevant to the query using BM25 algorithm.

        :param query: The query string.
        :param filters: A dictionary with filters to narrow down the search space.
        :param top_k: The number of top documents to retrieve. Default is 10.
        :param scale_score: Whether to scale the scores of the retrieved documents. Default is True.
        :return: A list of the top 'k' documents most relevant to the query.
        """
        if not query:
            raise ValueError("Query should be a non-empty string")

        # Get all documents that match the user's filters AND are either 'table' or 'text'.
        # Raises an exception if the user was trying to include other content types.
        if filters and "content_type" in filters:
            content_types = filters["content_type"]
            if isinstance(content_types, str):
                content_types = [content_types]
            if any(type_ not in ["text", "table"] for type_ in content_types):
                raise ValueError(
                    "MemoryDocumentStore can do BM25 retrieval on no other document type than text or table."
                )
        else:
            filters = filters or {}
            filters = {**filters, "content_type": ["text", "table"]}
        all_documents = self.filter_documents(filters=filters)

        # FIXME: remove this guard after resolving https://github.com/deepset-ai/canals/issues/33
        top_k = top_k if top_k is not None else 10

        # Lowercase all documents
        lower_case_documents = []
        for doc in all_documents:
            if doc.content_type == "text":
                lower_case_documents.append(doc.content.lower())
            elif doc.content_type == "table":
                str_content = doc.content.astype(str)
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
            docs_scores = [expit(float(score / SCALING_FACTOR)) for score in docs_scores]
        # get the last top_k indexes and reverse them
        top_docs_positions = np.argsort(docs_scores)[-top_k:][::-1]

        # Create documents with the BM25 score to return them
        return_documents = []
        for i in top_docs_positions:
            doc = all_documents[i]
            doc_fields = doc.to_dict()
            doc_fields["score"] = docs_scores[i]
            return_document = Document(**doc_fields)
            return_documents.append(return_document)
        return return_documents
