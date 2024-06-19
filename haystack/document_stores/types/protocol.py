# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, List, Optional, Protocol

from haystack import logging
from haystack.dataclasses import Document
from haystack.document_stores.types.policy import DuplicatePolicy

# Ellipsis are needed for the type checker, it's safe to disable module-wide
# pylint: disable=unnecessary-ellipsis

logger = logging.getLogger(__name__)


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

        Filters are defined as nested dictionaries that can be of two types:
        - Comparison
        - Logic

        Comparison dictionaries must contain the keys:

        - `field`
        - `operator`
        - `value`

        Logic dictionaries must contain the keys:

        - `operator`
        - `conditions`

        The `conditions` key must be a list of dictionaries, either of type Comparison or Logic.

        The `operator` value in Comparison dictionaries must be one of:

        - `==`
        - `!=`
        - `>`
        - `>=`
        - `<`
        - `<=`
        - `in`
        - `not in`

        The `operator` values in Logic dictionaries must be one of:

        - `NOT`
        - `OR`
        - `AND`


        A simple filter:
        ```python
        filters = {"field": "meta.type", "operator": "==", "value": "article"}
        ```

        A more complex filter:
        ```python
        filters = {
            "operator": "AND",
            "conditions": [
                {"field": "meta.type", "operator": "==", "value": "article"},
                {"field": "meta.date", "operator": ">=", "value": 1420066800},
                {"field": "meta.date", "operator": "<", "value": 1609455600},
                {"field": "meta.rating", "operator": ">=", "value": 3},
                {
                    "operator": "OR",
                    "conditions": [
                        {"field": "meta.genre", "operator": "in", "value": ["economy", "politics"]},
                        {"field": "meta.publisher", "operator": "==", "value": "nytimes"},
                    ],
                },
            ],
        }

        :param filters: the filters to apply to the document list.
        :returns: a list of Documents that match the given filters.
        """
        ...

    def write_documents(self, documents: List[Document], policy: DuplicatePolicy = DuplicatePolicy.NONE) -> int:
        """
        Writes Documents into the DocumentStore.

        :param documents: a list of Document objects.
        :param policy: the policy to apply when a Document with the same id already exists in the DocumentStore.
            - `DuplicatePolicy.NONE`: Default policy, behaviour depends on the Document Store.
            - `DuplicatePolicy.SKIP`: If a Document with the same id already exists, it is skipped and not written.
            - `DuplicatePolicy.OVERWRITE`: If a Document with the same id already exists, it is overwritten.
            - `DuplicatePolicy.FAIL`: If a Document with the same id already exists, an error is raised.
        :raises DuplicateError: If `policy` is set to `DuplicatePolicy.FAIL` and a Document with the same id already
            exists.
        :returns: The number of Documents written.
            If `DuplicatePolicy.OVERWRITE` is used, this number is always equal to the number of documents in input.
            If `DuplicatePolicy.SKIP` is used, this number can be lower than the number of documents in the input list.
        """
        ...

    def delete_documents(self, document_ids: List[str]) -> None:
        """
        Deletes all documents with a matching document_ids from the DocumentStore.

        Fails with `MissingDocumentError` if no document with this id is present in the DocumentStore.

        :param document_ids: the object_ids to delete
        """
        ...
