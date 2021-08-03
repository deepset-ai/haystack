from ..utils import (
    is_faiss_available,
    is_elasticsearch_available,
    is_pymilvus_available,
    is_sqlalchemy_available,
)

if is_faiss_available():
    from .faiss import FAISSDocumentStore

if is_elasticsearch_available():
    from .elasticsearch import (
        ElasticsearchDocumentStore,
        OpenDistroElasticsearchDocumentStore,
        OpenSearchDocumentStore,
    )

if is_sqlalchemy_available():
    from .sql import SQLDocumentStore

if is_pymilvus_available():
    from .milvus import MilvusDocumentStore

from haystack.document_store.memory import InMemoryDocumentStore
