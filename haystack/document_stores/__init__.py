from haystack.document_stores.base import BaseDocumentStore, BaseKnowledgeGraph
from haystack.document_stores.elasticsearch import ElasticsearchDocumentStore, OpenDistroElasticsearchDocumentStore, OpenSearchDocumentStore
from haystack.document_stores.faiss import FAISSDocumentStore
from haystack.document_stores.memory import InMemoryDocumentStore

import os
if os.getenv("MILVUS2_ENABLED"):
    print("Using experimental Milvus2DocumentStore")
    from haystack.document_stores.milvus2x import Milvus2DocumentStore as MilvusDocumentStore
else:
    from haystack.document_stores.milvus import MilvusDocumentStore  # type: ignore    

from haystack.document_stores.sql import SQLDocumentStore
from haystack.document_stores.weaviate import WeaviateDocumentStore
from haystack.document_stores.graphdb import GraphDBKnowledgeGraph
from haystack.document_stores.utils import eval_data_from_json, eval_data_from_jsonl, squad_json_to_jsonl
