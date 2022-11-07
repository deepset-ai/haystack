import sys
import os

import yaml
import faiss
import pytest
import numpy as np

from haystack.schema import Document
from haystack.document_stores.faiss import FAISSDocumentStore

from .test_base import DocumentStoreBaseTestAbstract

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


class TestFAISSDocumentStore(DocumentStoreBaseTestAbstract):
    index_name = __name__

    @pytest.fixture
    def ds(self, tmp_path):
        return FAISSDocumentStore(
            sql_url=f"sqlite:///{tmp_path}/haystack_test.db",
            return_embedding=True,
            index=self.index_name,
            isolation_level="AUTOCOMMIT",
            progress_bar=False,
            similarity="cosine",
        )

    @pytest.fixture(scope="class")
    def documents_embedding_only(self, documents):
        # drop documents without embeddings from the original fixture
        return [d for d in documents if d.embedding is not None]

    @pytest.mark.unit
    def test_index_mutual_exclusive_args(self, tmp_path):
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

    @pytest.mark.integration
    def test_delete_index(self, ds, documents):
        """Contrary to other Document Stores, FAISSDocumentStore doesn't raise if the index is empty"""
        ds.write_documents(documents, index="custom_index")
        assert ds.get_document_count(index="custom_index") == len(documents)
        ds.delete_index(index="custom_index")
        assert ds.get_document_count(index="custom_index") == 0

    @pytest.mark.integration
    def test_index_save_and_load(self, ds, documents_embedding_only, tmp_path):
        ds.write_documents(documents_embedding_only)

        # test saving the index
        ds.save(tmp_path / "haystack_test_faiss")

        # clear existing faiss_index
        ds.faiss_indexes[ds.index].reset()

        # test faiss index is cleared
        assert ds.faiss_indexes[ds.index].ntotal == 0

        # test loading the index
        new_document_store = FAISSDocumentStore.load(tmp_path / "haystack_test_faiss")

        # check faiss index is restored
        assert new_document_store.faiss_indexes[ds.index].ntotal == len(documents_embedding_only)
        # check if documents are restored
        assert len(new_document_store.get_all_documents()) == len(documents_embedding_only)
        # Check if the init parameters are kept
        assert not new_document_store.progress_bar

        # test saving and loading the loaded faiss index
        new_document_store.save(tmp_path / "haystack_test_faiss")
        reloaded_document_store = FAISSDocumentStore.load(tmp_path / "haystack_test_faiss")

        # check faiss index is restored
        assert reloaded_document_store.faiss_indexes[ds.index].ntotal == len(documents_embedding_only)
        # check if documents are restored
        assert len(reloaded_document_store.get_all_documents()) == len(documents_embedding_only)
        # Check if the init parameters are kept
        assert not reloaded_document_store.progress_bar

        # test loading the index via init
        new_document_store = FAISSDocumentStore(faiss_index_path=tmp_path / "haystack_test_faiss")

        # check faiss index is restored
        assert new_document_store.faiss_indexes[ds.index].ntotal == len(documents_embedding_only)
        # check if documents are restored
        assert len(new_document_store.get_all_documents()) == len(documents_embedding_only)
        # Check if the init parameters are kept
        assert not new_document_store.progress_bar

    @pytest.mark.integration
    def test_index_save_and_load_custom_path(self, ds, documents_embedding_only, tmp_path):
        ds.write_documents(documents_embedding_only)

        # test saving the index
        ds.save(index_path=tmp_path / "haystack_test_faiss", config_path=tmp_path / "custom_path.json")

        # clear existing faiss_index
        ds.faiss_indexes[ds.index].reset()

        # test faiss index is cleared
        assert ds.faiss_indexes[ds.index].ntotal == 0

        # test loading the index
        new_document_store = FAISSDocumentStore.load(
            index_path=tmp_path / "haystack_test_faiss", config_path=tmp_path / "custom_path.json"
        )

        # check faiss index is restored
        assert new_document_store.faiss_indexes[ds.index].ntotal == len(documents_embedding_only)
        # check if documents are restored
        assert len(new_document_store.get_all_documents()) == len(documents_embedding_only)
        # Check if the init parameters are kept
        assert not new_document_store.progress_bar

        # test saving and loading the loaded faiss index
        new_document_store.save(tmp_path / "haystack_test_faiss", config_path=tmp_path / "custom_path.json")
        reloaded_document_store = FAISSDocumentStore.load(
            tmp_path / "haystack_test_faiss", config_path=tmp_path / "custom_path.json"
        )

        # check faiss index is restored
        assert reloaded_document_store.faiss_indexes[ds.index].ntotal == len(documents_embedding_only)
        # check if documents are restored
        assert len(reloaded_document_store.get_all_documents()) == len(documents_embedding_only)
        # Check if the init parameters are kept
        assert not reloaded_document_store.progress_bar

        # test loading the index via init
        new_document_store = FAISSDocumentStore(
            faiss_index_path=tmp_path / "haystack_test_faiss", faiss_config_path=tmp_path / "custom_path.json"
        )

        # check faiss index is restored
        assert new_document_store.faiss_indexes[ds.index].ntotal == len(documents_embedding_only)
        # check if documents are restored
        assert len(new_document_store.get_all_documents()) == len(documents_embedding_only)
        # Check if the init parameters are kept
        assert not new_document_store.progress_bar

    @pytest.mark.integration
    @pytest.mark.parametrize("index_buffer_size", [10_000, 2])
    def test_write_index_docs(self, ds, documents_embedding_only, index_buffer_size):
        batch_size = 2
        ds.index_buffer_size = index_buffer_size

        # Write in small batches
        for i in range(0, len(documents_embedding_only), batch_size):
            ds.write_documents(documents_embedding_only[i : i + batch_size])

        documents_indexed = ds.get_all_documents()
        assert len(documents_indexed) == len(documents_embedding_only)

        # test if correct vectors are associated with docs
        for i, doc in enumerate(documents_indexed):
            # we currently don't get the embeddings back when we call document_store.get_all_documents()
            original_doc = [d for d in documents_embedding_only if d.content == doc.content][0]
            stored_emb = ds.faiss_indexes[ds.index].reconstruct(int(doc.meta["vector_id"]))
            # compare original input vec with stored one (ignore extra dim added by hnsw)
            # original input vec is normalized as faiss only stores normalized vectors
            assert np.allclose(original_doc.embedding / np.linalg.norm(original_doc.embedding), stored_emb, rtol=0.01)

    @pytest.mark.integration
    def test_write_docs_different_indexes(self, ds, documents_embedding_only):
        ds.write_documents(documents_embedding_only, index="index1")
        ds.write_documents(documents_embedding_only, index="index2")

        docs_from_index1 = ds.get_all_documents(index="index1", return_embedding=False)
        assert len(docs_from_index1) == len(documents_embedding_only)
        assert {int(doc.meta["vector_id"]) for doc in docs_from_index1} == set(range(0, 6))

        docs_from_index2 = ds.get_all_documents(index="index2", return_embedding=False)
        assert len(docs_from_index2) == len(documents_embedding_only)
        assert {int(doc.meta["vector_id"]) for doc in docs_from_index2} == set(range(0, 6))

    @pytest.mark.integration
    def test_update_docs_different_indexes(self, ds, documents_embedding_only):
        retriever = MockDenseRetriever(document_store=ds)

        ds.write_documents(documents_embedding_only, index="index1")
        ds.write_documents(documents_embedding_only, index="index2")

        ds.update_embeddings(retriever=retriever, update_existing_embeddings=True, index="index1")
        ds.update_embeddings(retriever=retriever, update_existing_embeddings=True, index="index2")

        docs_from_index1 = ds.get_all_documents(index="index1", return_embedding=False)
        assert len(docs_from_index1) == len(documents_embedding_only)
        assert {int(doc.meta["vector_id"]) for doc in docs_from_index1} == set(range(0, 6))

        docs_from_index2 = ds.get_all_documents(index="index2", return_embedding=False)
        assert len(docs_from_index2) == len(documents_embedding_only)
        assert {int(doc.meta["vector_id"]) for doc in docs_from_index2} == set(range(0, 6))

    @pytest.mark.integration
    @pytest.mark.parametrize("index_factory", ["Flat", "HNSW", "IVF1,Flat"])
    def test_retrieving(self, documents_embedding_only, index_factory, tmp_path):
        document_store = FAISSDocumentStore(
            sql_url=f"sqlite:////{tmp_path/'test_faiss_retrieving.db'}",
            faiss_index_factory_str=index_factory,
            isolation_level="AUTOCOMMIT",
        )

        document_store.delete_all_documents(index="document")
        if "ivf" in index_factory.lower():
            document_store.train_index(documents_embedding_only)
        document_store.write_documents(documents_embedding_only)

        retriever = EmbeddingRetriever(
            document_store=document_store, embedding_model="nreimers/albert-small-v2", use_gpu=False
        )
        result = retriever.retrieve(query="How to test this?")

        assert len(result) == len(documents_embedding_only)
        assert type(result[0]) == Document

        # Cleanup
        document_store.faiss_indexes[document_store.index].reset()

    @pytest.mark.integration
    def test_passing_index_from_outside(self, documents_embedding_only, tmp_path):
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
        document_store.train_index(documents_embedding_only)

        document_store.write_documents(documents=documents_embedding_only)
        documents_indexed = document_store.get_all_documents()

        # test if vectors ids are associated with docs
        for doc in documents_indexed:
            assert 0 <= int(doc.meta["vector_id"]) <= 7

    @pytest.mark.integration
    def test_pipeline_with_existing_faiss_docstore(self, ds, documents_embedding_only, tmp_path):

        retriever = MockDenseRetriever(document_store=ds)
        ds.write_documents(documents_embedding_only)
        ds.update_embeddings(retriever=retriever, update_existing_embeddings=True)

        ds.save(tmp_path / "existing_faiss_document_store")

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
        faiss_index = existing_document_store.faiss_indexes[ds.index]
        assert faiss_index.ntotal == len(documents_embedding_only)

    # See TestSQLDocumentStore about why we have to skip these tests

    @pytest.mark.skip
    @pytest.mark.integration
    def test_ne_filters(self, ds, documents):
        pass

    @pytest.mark.skip
    @pytest.mark.integration
    def test_nin_filters(self, ds, documents):
        pass

    @pytest.mark.skip
    @pytest.mark.integration
    def test_comparison_filters(self, ds, documents):
        pass

    @pytest.mark.skip
    @pytest.mark.integration
    def test_nested_condition_filters(self, ds, documents):
        pass

    @pytest.mark.skip
    @pytest.mark.integration
    def test_nested_condition_not_filters(self, ds, documents):
        pass

    @pytest.mark.skip
    @pytest.mark.integration
    def test_delete_labels_by_filter(self, ds, labels):
        pass

    @pytest.mark.skip
    @pytest.mark.integration
    def test_delete_labels_by_filter_id(self, ds, labels):
        pass
