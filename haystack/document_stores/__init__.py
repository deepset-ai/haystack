import os
import importlib
from haystack.utils.import_utils import safe_import
from haystack.document_stores.base import BaseDocumentStore, BaseKnowledgeGraph, KeywordDocumentStore

from haystack.document_stores.memory import InMemoryDocumentStore
from haystack.document_stores.deepsetcloud import DeepsetCloudDocumentStore
from haystack.document_stores.utils import eval_data_from_json, eval_data_from_jsonl, squad_json_to_jsonl

from haystack.document_stores.elasticsearch import ElasticsearchDocumentStore
from haystack.document_stores.es_converter import (
    elasticsearch_index_to_document_store,
    open_search_index_to_document_store,
)

OpenSearchDocumentStore = safe_import("haystack.document_stores.opensearch", "OpenSearchDocumentStore", "opensearch")
OpenDistroElasticsearchDocumentStore = safe_import(
    "haystack.document_stores.opensearch", "OpenDistroElasticsearchDocumentStore", "opensearch"
)
SQLDocumentStore = safe_import("haystack.document_stores.sql", "SQLDocumentStore", "sql")
FAISSDocumentStore = safe_import("haystack.document_stores.faiss", "FAISSDocumentStore", "faiss")
PineconeDocumentStore = safe_import("haystack.document_stores.pinecone", "PineconeDocumentStore", "pinecone")
MilvusDocumentStore = safe_import("haystack.document_stores.milvus", "MilvusDocumentStore", "milvus")
WeaviateDocumentStore = safe_import("haystack.document_stores.weaviate", "WeaviateDocumentStore", "weaviate")
GraphDBKnowledgeGraph = safe_import("haystack.document_stores.graphdb", "GraphDBKnowledgeGraph", "graphdb")
InMemoryKnowledgeGraph = safe_import(
    "haystack.document_stores.memory_knowledgegraph", "InMemoryKnowledgeGraph", "inmemorygraph"
)
