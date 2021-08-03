from ..utils import is_faiss_available

if is_faiss_available():
    from .faiss import FAISSDocumentStore

from haystack.document_store.elasticsearch import ElasticsearchDocumentStore, OpenDistroElasticsearchDocumentStore
from haystack.document_store.memory import InMemoryDocumentStore
from haystack.document_store.milvus import MilvusDocumentStore
from haystack.document_store.sql import SQLDocumentStore
