import os

from haystack.document_store.elasticsearch import ElasticsearchDocumentStore, OpenDistroElasticsearchDocumentStore, OpenSearchDocumentStore
from haystack.document_store.faiss import FAISSDocumentStore
from haystack.document_store.memory import InMemoryDocumentStore

if os.getenv("MILVUS2_ENABLED"):
    print("Using experimental Milvus2DocumentStore")
    from haystack.document_store.milvus2x import Milvus2DocumentStore as MilvusDocumentStore
else:
    from haystack.document_store.milvus import MilvusDocumentStore #type: ignore    

from haystack.document_store.sql import SQLDocumentStore
from haystack.document_store.weaviate import WeaviateDocumentStore
