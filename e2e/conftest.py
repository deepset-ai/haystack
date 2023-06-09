import os
import uuid
from contextlib import contextmanager
from pathlib import Path

import pytest

from haystack.schema import Document
from haystack.modeling.utils import set_all_seeds
from haystack.document_stores import (
    InMemoryDocumentStore,
    ElasticsearchDocumentStore,
    WeaviateDocumentStore,
    PineconeDocumentStore,
    OpenSearchDocumentStore,
    FAISSDocumentStore,
)


set_all_seeds(0)


@pytest.fixture
def samples_path():
    return Path(__file__).parent / "samples"


@pytest.fixture
def preview_samples_path():
    return Path(__file__).parent / "preview" / "test_files"


@pytest.fixture
def docs_all_formats():
    return [
        # metafield at the top level for backward compatibility
        {
            "content": "My name is Paul and I live in New York",
            "meta_field": "test2",
            "name": "filename2",
            "date_field": "2019-10-01",
            "numeric_field": 5.0,
        },
        # "dict" format
        {
            "content": "My name is Carla and I live in Berlin",
            "meta": {"meta_field": "test1", "name": "filename1", "date_field": "2020-03-01", "numeric_field": 5.5},
        },
        # Document object
        Document(
            content="My name is Christelle and I live in Paris",
            meta={"meta_field": "test3", "name": "filename3", "date_field": "2018-10-01", "numeric_field": 4.5},
        ),
        Document(
            content="My name is Camila and I live in Madrid",
            meta={"meta_field": "test4", "name": "filename4", "date_field": "2021-02-01", "numeric_field": 3.0},
        ),
        Document(
            content="My name is Matteo and I live in Rome",
            meta={"meta_field": "test5", "name": "filename5", "date_field": "2019-01-01", "numeric_field": 0.0},
        ),
    ]


@pytest.fixture
def docs(docs_all_formats):
    return [Document.from_dict(doc) if isinstance(doc, dict) else doc for doc in docs_all_formats]


@contextmanager
def document_store(
    name,
    docs,
    tmp_path,
    embedding_dim=768,
    embedding_field="embedding",
    index="haystack_test",
    similarity="cosine",  # cosine is default similarity as dot product is not supported by Weaviate
    recreate_index=True,
):
    if name == "memory":
        document_store = InMemoryDocumentStore(
            return_embedding=True,
            embedding_dim=embedding_dim,
            embedding_field=embedding_field,
            index=index,
            similarity=similarity,
            use_bm25=True,
        )

    elif name == "elasticsearch":
        # make sure we start from a fresh index
        document_store = ElasticsearchDocumentStore(
            index=index,
            return_embedding=True,
            embedding_dim=embedding_dim,
            embedding_field=embedding_field,
            similarity=similarity,
            recreate_index=recreate_index,
        )

    elif name == "faiss":
        document_store = FAISSDocumentStore(
            embedding_dim=embedding_dim,
            sql_url=f"sqlite:///{tmp_path}/haystack_test.db",
            return_embedding=True,
            embedding_field=embedding_field,
            index=index,
            similarity=similarity,
            isolation_level="AUTOCOMMIT",
        )

    elif name == "weaviate":
        document_store = WeaviateDocumentStore(
            index=index, similarity=similarity, embedding_dim=embedding_dim, recreate_index=recreate_index
        )
        for d in docs:
            d.id = str(uuid.uuid4())

    elif name == "pinecone":
        document_store = PineconeDocumentStore(
            api_key=os.environ.get("PINECONE_API_KEY") or "fake-haystack-test-key",
            embedding_dim=embedding_dim,
            embedding_field=embedding_field,
            index=index,
            similarity=similarity,
            recreate_index=recreate_index,
            metadata_config={
                "indexed": [
                    "meta_field",
                    "name",
                    "date_field",
                    "numeric_field",
                    "f1",
                    "f3",
                    "meta_id",
                    "meta_field_for_count",
                    "meta_key_1",
                    "meta_key_2",
                ]
            },
        )

    elif name == "opensearch_faiss":
        document_store = OpenSearchDocumentStore(
            index=index,
            return_embedding=True,
            embedding_dim=embedding_dim,
            embedding_field=embedding_field,
            similarity=similarity,
            recreate_index=recreate_index,
            port=9201,
            knn_engine="faiss",
        )

    else:
        raise Exception(f"No document store fixture for '{name}'")

    document_store.write_documents(docs)
    yield document_store
    document_store.delete_index(document_store.index)
