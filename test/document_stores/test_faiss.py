import sys

import yaml
import faiss
import pytest
import numpy as np

from haystack.schema import Document
from haystack.document_stores.faiss import FAISSDocumentStore

from haystack.pipelines import Pipeline
from haystack.nodes.retriever.dense import EmbeddingRetriever

from ..conftest import MockDenseRetriever


DOCUMENTS = [
    {
        "meta": {"name": "name_1", "year": "2020", "month": "01"},
        "content": "text_1",
        "embedding": np.random.rand(768).astype(np.float32),
    },
    {
        "meta": {"name": "name_2", "year": "2020", "month": "02"},
        "content": "text_2",
        "embedding": np.random.rand(768).astype(np.float32),
    },
    {
        "meta": {"name": "name_3", "year": "2020", "month": "03"},
        "content": "text_3",
        "embedding": np.random.rand(768).astype(np.float64),
    },
    {
        "meta": {"name": "name_4", "year": "2021", "month": "01"},
        "content": "text_4",
        "embedding": np.random.rand(768).astype(np.float32),
    },
    {
        "meta": {"name": "name_5", "year": "2021", "month": "02"},
        "content": "text_5",
        "embedding": np.random.rand(768).astype(np.float32),
    },
    {
        "meta": {"name": "name_6", "year": "2021", "month": "03"},
        "content": "text_6",
        "embedding": np.random.rand(768).astype(np.float64),
    },
]


@pytest.mark.skipif(sys.platform in ["win32", "cygwin"], reason="Test with tmp_path not working on windows runner")
def test_faiss_index_save_and_load(tmp_path, sql_url):
    document_store = FAISSDocumentStore(
        sql_url=sql_url,
        index="haystack_test",
        progress_bar=False,  # Just to check if the init parameters are kept
        isolation_level="AUTOCOMMIT",
    )
    document_store.write_documents(DOCUMENTS)

    # test saving the index
    document_store.save(tmp_path / "haystack_test_faiss")

    # clear existing faiss_index
    document_store.faiss_indexes[document_store.index].reset()

    # test faiss index is cleared
    assert document_store.faiss_indexes[document_store.index].ntotal == 0

    # test loading the index
    new_document_store = FAISSDocumentStore.load(tmp_path / "haystack_test_faiss")

    # check faiss index is restored
    assert new_document_store.faiss_indexes[document_store.index].ntotal == len(DOCUMENTS)
    # check if documents are restored
    assert len(new_document_store.get_all_documents()) == len(DOCUMENTS)
    # Check if the init parameters are kept
    assert not new_document_store.progress_bar

    # test saving and loading the loaded faiss index
    new_document_store.save(tmp_path / "haystack_test_faiss")
    reloaded_document_store = FAISSDocumentStore.load(tmp_path / "haystack_test_faiss")

    # check faiss index is restored
    assert reloaded_document_store.faiss_indexes[document_store.index].ntotal == len(DOCUMENTS)
    # check if documents are restored
    assert len(reloaded_document_store.get_all_documents()) == len(DOCUMENTS)
    # Check if the init parameters are kept
    assert not reloaded_document_store.progress_bar

    # test loading the index via init
    new_document_store = FAISSDocumentStore(faiss_index_path=tmp_path / "haystack_test_faiss")

    # check faiss index is restored
    assert new_document_store.faiss_indexes[document_store.index].ntotal == len(DOCUMENTS)
    # check if documents are restored
    assert len(new_document_store.get_all_documents()) == len(DOCUMENTS)
    # Check if the init parameters are kept
    assert not new_document_store.progress_bar


@pytest.mark.skipif(sys.platform in ["win32", "cygwin"], reason="Test with tmp_path not working on windows runner")
def test_faiss_index_save_and_load_custom_path(tmp_path, sql_url):
    document_store = FAISSDocumentStore(
        sql_url=sql_url,
        index="haystack_test",
        progress_bar=False,  # Just to check if the init parameters are kept
        isolation_level="AUTOCOMMIT",
    )
    document_store.write_documents(DOCUMENTS)

    # test saving the index
    document_store.save(index_path=tmp_path / "haystack_test_faiss", config_path=tmp_path / "custom_path.json")

    # clear existing faiss_index
    document_store.faiss_indexes[document_store.index].reset()

    # test faiss index is cleared
    assert document_store.faiss_indexes[document_store.index].ntotal == 0

    # test loading the index
    new_document_store = FAISSDocumentStore.load(
        index_path=tmp_path / "haystack_test_faiss", config_path=tmp_path / "custom_path.json"
    )

    # check faiss index is restored
    assert new_document_store.faiss_indexes[document_store.index].ntotal == len(DOCUMENTS)
    # check if documents are restored
    assert len(new_document_store.get_all_documents()) == len(DOCUMENTS)
    # Check if the init parameters are kept
    assert not new_document_store.progress_bar

    # test saving and loading the loaded faiss index
    new_document_store.save(tmp_path / "haystack_test_faiss", config_path=tmp_path / "custom_path.json")
    reloaded_document_store = FAISSDocumentStore.load(
        tmp_path / "haystack_test_faiss", config_path=tmp_path / "custom_path.json"
    )

    # check faiss index is restored
    assert reloaded_document_store.faiss_indexes[document_store.index].ntotal == len(DOCUMENTS)
    # check if documents are restored
    assert len(reloaded_document_store.get_all_documents()) == len(DOCUMENTS)
    # Check if the init parameters are kept
    assert not reloaded_document_store.progress_bar

    # test loading the index via init
    new_document_store = FAISSDocumentStore(
        faiss_index_path=tmp_path / "haystack_test_faiss", faiss_config_path=tmp_path / "custom_path.json"
    )

    # check faiss index is restored
    assert new_document_store.faiss_indexes[document_store.index].ntotal == len(DOCUMENTS)
    # check if documents are restored
    assert len(new_document_store.get_all_documents()) == len(DOCUMENTS)
    # Check if the init parameters are kept
    assert not new_document_store.progress_bar


@pytest.mark.skipif(sys.platform in ["win32", "cygwin"], reason="Test with tmp_path not working on windows runner")
def test_faiss_index_mutual_exclusive_args(tmp_path):
    with pytest.raises(ValueError):
        FAISSDocumentStore(
            sql_url=f"sqlite:////{tmp_path/'haystack_test.db'}",
            faiss_index_path=f"{tmp_path/'haystack_test'}",
            isolation_level="AUTOCOMMIT",
        )

    with pytest.raises(ValueError):
        FAISSDocumentStore(
            f"sqlite:////{tmp_path/'haystack_test.db'}",
            faiss_index_path=f"{tmp_path/'haystack_test'}",
            isolation_level="AUTOCOMMIT",
        )


@pytest.mark.parametrize("document_store", ["faiss"], indirect=True)
@pytest.mark.parametrize("index_buffer_size", [10_000, 2])
@pytest.mark.parametrize("batch_size", [2])
def test_faiss_write_docs(document_store, index_buffer_size, batch_size):
    document_store.index_buffer_size = index_buffer_size

    # Write in small batches
    for i in range(0, len(DOCUMENTS), batch_size):
        document_store.write_documents(DOCUMENTS[i : i + batch_size])

    documents_indexed = document_store.get_all_documents()
    assert len(documents_indexed) == len(DOCUMENTS)

    # test if correct vectors are associated with docs
    for i, doc in enumerate(documents_indexed):
        # we currently don't get the embeddings back when we call document_store.get_all_documents()
        original_doc = [d for d in DOCUMENTS if d["content"] == doc.content][0]
        stored_emb = document_store.faiss_indexes[document_store.index].reconstruct(int(doc.meta["vector_id"]))
        # compare original input vec with stored one (ignore extra dim added by hnsw)
        # original input vec is normalized as faiss only stores normalized vectors
        assert np.allclose(original_doc["embedding"] / np.linalg.norm(original_doc["embedding"]), stored_emb, rtol=0.01)


@pytest.mark.parametrize("document_store", ["faiss"], indirect=True)
def test_faiss_write_docs_different_indexes(document_store):
    document_store.write_documents(DOCUMENTS, index="index1")
    document_store.write_documents(DOCUMENTS, index="index2")

    docs_from_index1 = document_store.get_all_documents(index="index1", return_embedding=False)
    assert len(docs_from_index1) == len(DOCUMENTS)
    assert {int(doc.meta["vector_id"]) for doc in docs_from_index1} == set(range(0, 6))

    docs_from_index2 = document_store.get_all_documents(index="index2", return_embedding=False)
    assert len(docs_from_index2) == len(DOCUMENTS)
    assert {int(doc.meta["vector_id"]) for doc in docs_from_index2} == set(range(0, 6))


@pytest.mark.parametrize("document_store", ["faiss"], indirect=True)
def test_faiss_update_docs_different_indexes(document_store):
    retriever = MockDenseRetriever(document_store=document_store)

    document_store.write_documents(DOCUMENTS, index="index1")
    document_store.write_documents(DOCUMENTS, index="index2")

    document_store.update_embeddings(retriever=retriever, update_existing_embeddings=True, index="index1")
    document_store.update_embeddings(retriever=retriever, update_existing_embeddings=True, index="index2")

    docs_from_index1 = document_store.get_all_documents(index="index1", return_embedding=False)
    assert len(docs_from_index1) == len(DOCUMENTS)
    assert {int(doc.meta["vector_id"]) for doc in docs_from_index1} == set(range(0, 6))

    docs_from_index2 = document_store.get_all_documents(index="index2", return_embedding=False)
    assert len(docs_from_index2) == len(DOCUMENTS)
    assert {int(doc.meta["vector_id"]) for doc in docs_from_index2} == set(range(0, 6))


@pytest.mark.skipif(sys.platform in ["win32", "cygwin"], reason="Test with tmp_path not working on windows runner")
@pytest.mark.parametrize("index_factory", ["Flat", "HNSW", "IVF1,Flat"])
def test_faiss_retrieving(index_factory, tmp_path):
    document_store = FAISSDocumentStore(
        sql_url=f"sqlite:////{tmp_path/'test_faiss_retrieving.db'}",
        faiss_index_factory_str=index_factory,
        isolation_level="AUTOCOMMIT",
    )

    document_store.delete_all_documents(index="document")
    if "ivf" in index_factory.lower():
        document_store.train_index(DOCUMENTS)
    document_store.write_documents(DOCUMENTS)

    retriever = EmbeddingRetriever(
        document_store=document_store, embedding_model="deepset/sentence_bert", use_gpu=False
    )
    result = retriever.retrieve(query="How to test this?")

    assert len(result) == len(DOCUMENTS)
    assert type(result[0]) == Document

    # Cleanup
    document_store.faiss_indexes[document_store.index].reset()


@pytest.mark.skipif(sys.platform in ["win32", "cygwin"], reason="Test with tmp_path not working on windows runner")
def test_faiss_passing_index_from_outside(tmp_path):
    d = 768
    nlist = 2
    quantizer = faiss.IndexFlatIP(d)
    index = "haystack_test_1"
    faiss_index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
    faiss_index.set_direct_map_type(faiss.DirectMap.Hashtable)
    faiss_index.nprobe = 2
    document_store = FAISSDocumentStore(
        sql_url=f"sqlite:////{tmp_path/'haystack_test_faiss.db'}",
        faiss_index=faiss_index,
        index=index,
        isolation_level="AUTOCOMMIT",
    )

    document_store.delete_documents()
    # as it is a IVF index we need to train it before adding docs
    document_store.train_index(DOCUMENTS)

    document_store.write_documents(documents=DOCUMENTS)
    documents_indexed = document_store.get_all_documents()

    # test if vectors ids are associated with docs
    for doc in documents_indexed:
        assert 0 <= int(doc.meta["vector_id"]) <= 7


@pytest.mark.integration
def test_pipeline_with_existing_faiss_docstore(tmp_path):

    document_store: FAISSDocumentStore = FAISSDocumentStore(
        sql_url=f'sqlite:///{(tmp_path / "faiss_document_store.db").absolute()}'
    )
    retriever = MockDenseRetriever(document_store=document_store)
    document_store.write_documents(DOCUMENTS)
    document_store.update_embeddings(retriever=retriever, update_existing_embeddings=True)

    document_store.save(tmp_path / "existing_faiss_document_store")

    query_config = f"""
version: ignore
components:
  - name: DPRRetriever
    type: MockDenseRetriever
    params:
      document_store: ExistingFAISSDocumentStore
  - name: ExistingFAISSDocumentStore
    type: FAISSDocumentStore
    params:
      faiss_index_path: '{tmp_path / "existing_faiss_document_store"}'
pipelines:
  - name: query_pipeline
    nodes:
      - name: DPRRetriever
        inputs: [Query]
    """
    pipeline = Pipeline.load_from_config(yaml.safe_load(query_config))
    existing_document_store = pipeline.get_document_store()
    faiss_index = existing_document_store.faiss_indexes["document"]
    assert faiss_index.ntotal == len(DOCUMENTS)
