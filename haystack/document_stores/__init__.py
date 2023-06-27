import typing

from haystack.lazy_imports import LazyImport
from haystack.document_stores.base import BaseDocumentStore, KeywordDocumentStore

from haystack.document_stores.memory import InMemoryDocumentStore
from haystack.document_stores.deepsetcloud import DeepsetCloudDocumentStore
from haystack.document_stores.utils import eval_data_from_json, eval_data_from_jsonl, squad_json_to_jsonl

from haystack.document_stores.es_converter import elasticsearch_index_to_document_store
from haystack.document_stores.es_converter import open_search_index_to_document_store

try:
    # Use appropriate ElasticsearchDocumentStore depending on ES client version
    with LazyImport() as elasticsearch_import:
        from elasticsearch import __version__ as ES_VERSION

        if ES_VERSION[0] == 7:
            from haystack.document_stores.elasticsearch7 import ElasticsearchDocumentStore  # type: ignore  # pylint: disable=reimported,ungrouped-imports
        elif ES_VERSION[0] == 8:
            from haystack.document_stores.elasticsearch8 import ElasticsearchDocumentStore  # type: ignore  # pylint: disable=reimported,ungrouped-imports
        else:
            # Use Elasticsearch 7 as default
            from haystack.document_stores.elasticsearch7 import ElasticsearchDocumentStore
    elasticsearch_import.check()
except (ModuleNotFoundError, ImportError):
    # No need to import anything if ES could not be imported
    from haystack.document_stores.elasticsearch7 import ElasticsearchDocumentStore

from haystack.document_stores.opensearch import OpenSearchDocumentStore
from haystack.document_stores.sql import SQLDocumentStore
from haystack.document_stores.faiss import FAISSDocumentStore
from haystack.document_stores.pinecone import PineconeDocumentStore
from haystack.document_stores.weaviate import WeaviateDocumentStore
