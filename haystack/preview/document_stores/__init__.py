from haystack.preview.document_stores.protocols import DocumentStore, DuplicatePolicy
from haystack.preview.document_stores.memory.document_store import MemoryDocumentStore
from haystack.preview.document_stores.errors import DocumentStoreError, DuplicateDocumentError, MissingDocumentError
from haystack.preview.document_stores.decorator import document_store

__all__ = [
    "DocumentStore",
    "DuplicatePolicy",
    "MemoryDocumentStore",
    "DocumentStoreError",
    "DuplicateDocumentError",
    "MissingDocumentError",
    "document_store",
]
