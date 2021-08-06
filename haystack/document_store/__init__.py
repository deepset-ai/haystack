from ..utils import (
    FAISS_AVAILABLE,
    ELASTICSEARCH_AVAILABLE,
    PYMILVUS_AVAILABLE,
    SQLALCHEMY_AVAILABLE,
)

if FAISS_AVAILABLE:
    from .faiss import FAISSDocumentStore

if ELASTICSEARCH_AVAILABLE:
    from .elasticsearch import (
        ElasticsearchDocumentStore,
        OpenDistroElasticsearchDocumentStore,
        OpenSearchDocumentStore,
    )

if SQLALCHEMY_AVAILABLE:
    from .sql import SQLDocumentStore

if PYMILVUS_AVAILABLE:
    from .milvus import MilvusDocumentStore

from haystack.document_store.memory import InMemoryDocumentStore
