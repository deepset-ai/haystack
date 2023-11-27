from haystack.document_stores.protocols import DocumentStore, DuplicatePolicy
from haystack.document_stores.in_memory.document_store import InMemoryDocumentStore
from haystack.document_stores.errors import DocumentStoreError, DuplicateDocumentError, MissingDocumentError
from haystack.document_stores.decorator import document_store

__all__ = [
    "DocumentStore",
    "DuplicatePolicy",
    "InMemoryDocumentStore",
    "DocumentStoreError",
    "DuplicateDocumentError",
    "MissingDocumentError",
    "document_store",
]
