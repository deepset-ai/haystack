from haystack.preview.document_stores.protocols import Store, DuplicatePolicy
from haystack.preview.document_stores.mixins import StoreAwareMixin
from haystack.preview.document_stores.memory.document_store import MemoryDocumentStore
from haystack.preview.document_stores.errors import StoreError, DuplicateDocumentError, MissingDocumentError
