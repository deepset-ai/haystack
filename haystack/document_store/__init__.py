from ..utils import (is_faiss_available, is_elasticsearch_available)

if is_faiss_available():
    from .faiss import FAISSDocumentStore

if is_elasticsearch_available():
    from .elasticsearch import ElasticsearchDocumentStore, OpenDistroElasticsearchDocumentStore, OpenSearchDocumentStore

from haystack.document_store.memory import InMemoryDocumentStore
from haystack.document_store.milvus import MilvusDocumentStore
from haystack.document_store.sql import SQLDocumentStore
