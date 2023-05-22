from haystack.utils.import_utils import safe_import
from haystack.document_stores.base import BaseDocumentStore, KeywordDocumentStore

from haystack.document_stores.memory import InMemoryDocumentStore
from haystack.document_stores.deepsetcloud import DeepsetCloudDocumentStore
from haystack.document_stores.utils import eval_data_from_json, eval_data_from_jsonl, squad_json_to_jsonl

elasticsearch_index_to_document_store = safe_import(
    "haystack.document_stores.es_converter", "elasticsearch_index_to_document_store", "elasticsearch"
)
open_search_index_to_document_store = safe_import(
    "haystack.document_stores.es_converter", "open_search_index_to_document_store", "elasticsearch"
)
ElasticsearchDocumentStore = safe_import(
    "haystack.document_stores.elasticsearch", "ElasticsearchDocumentStore", "elasticsearch"
)
OpenSearchDocumentStore = safe_import("haystack.document_stores.opensearch", "OpenSearchDocumentStore", "opensearch")
SQLDocumentStore = safe_import("haystack.document_stores.sql", "SQLDocumentStore", "sql")
FAISSDocumentStore = safe_import("haystack.document_stores.faiss", "FAISSDocumentStore", "faiss")
PineconeDocumentStore = safe_import("haystack.document_stores.pinecone", "PineconeDocumentStore", "pinecone")
WeaviateDocumentStore = safe_import("haystack.document_stores.weaviate", "WeaviateDocumentStore", "weaviate")
