from typing import Protocol, Optional, Dict, Any, List
import logging
from enum import Enum

from haystack.preview.dataclasses import Document


# Ellipsis are needed for the type checker, it's safe to disable module-wide
# pylint: disable=unnecessary-ellipsis

logger = logging.getLogger(__name__)


class DuplicatePolicy(Enum):
    SKIP = "skip"
    OVERWRITE = "overwrite"
    FAIL = "fail"


class DocumentStore(Protocol):
    """
    Stores Documents to be used by the components of a Pipeline.

    Classes implementing this protocol often store the documents permanently and allow specialized components to
    perform retrieval on them, either by embedding, by keyword, hybrid, and so on, depending on the backend used.

    In order to retrieve documents, consider using a Retriever that supports the DocumentStore implementation that
    you're using.
    """

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes this store to a dictionary.
        """
        ...

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentStore":
        """
        Deserializes the store from a dictionary.
        """
        ...

    def count_documents(self) -> int:
        """
        Returns the number of documents stored.
        """
        ...

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
        ...

    def write_documents(self, documents: List[Document], policy: DuplicatePolicy = DuplicatePolicy.FAIL) -> int:
        """
        Writes (or overwrites) documents into the DocumentStore.

        :param documents: a list of documents.
        :param policy: documents with the same ID count as duplicates. When duplicates are met,
            the DocumentStore can:
             - skip: keep the existing document and ignore the new one.
             - overwrite: remove the old document and write the new one.
             - fail: an error is raised
        :raises DuplicateError: Exception trigger on duplicate document if `policy=DuplicatePolicy.FAIL`
        :return: The number of documents that was written.
            If DuplicatePolicy.OVERWRITE is used, this number is always equal to the number of documents in input.
            If DuplicatePolicy.SKIP is used, this number can be lower than the number of documents in the input list.
        """
        ...

    def delete_documents(self, document_ids: List[str]) -> None:
        """
        Deletes all documents with a matching document_ids from the DocumentStore.
        Fails with `MissingDocumentError` if no document with this id is present in the DocumentStore.

        :param object_ids: the object_ids to delete
        """
        ...
