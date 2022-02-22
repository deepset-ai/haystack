import os
import importlib
from haystack.utils.import_utils import safe_import
from haystack.document_stores.base import BaseDocumentStore, BaseKnowledgeGraph, KeywordDocumentStore

ElasticsearchDocumentStore = safe_import(
    "haystack.document_stores.elasticsearch", "ElasticsearchDocumentStore", "elasticsearch"
)
OpenDistroElasticsearchDocumentStore = safe_import(
    "haystack.document_stores.elasticsearch", "OpenDistroElasticsearchDocumentStore", "elasticsearch"
)
OpenSearchDocumentStore = safe_import(
    "haystack.document_stores.elasticsearch", "OpenSearchDocumentStore", "elasticsearch"
)

SQLDocumentStore = safe_import("haystack.document_stores.sql", "SQLDocumentStore", "sql")
FAISSDocumentStore = safe_import("haystack.document_stores.faiss", "FAISSDocumentStore", "faiss")
if os.getenv("MILVUS2_ENABLED"):
    MilvusDocumentStore = safe_import("haystack.document_stores.milvus2x", "MilvusDocumentStore", "milvus2")
else:
    MilvusDocumentStore = safe_import("haystack.document_stores.milvus", "MilvusDocumentStore", "milvus")
WeaviateDocumentStore = safe_import("haystack.document_stores.weaviate", "WeaviateDocumentStore", "weaviate")
GraphDBKnowledgeGraph = safe_import("haystack.document_stores.graphdb", "GraphDBKnowledgeGraph", "graphdb")

from haystack.document_stores.memory import InMemoryDocumentStore
from haystack.document_stores.deepsetcloud import DeepsetCloudDocumentStore
from haystack.document_stores.utils import (
    eval_data_from_json,
    eval_data_from_jsonl,
    squad_json_to_jsonl,
    es_index_to_document_store,
)
