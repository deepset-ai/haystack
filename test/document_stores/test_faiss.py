import faiss
import pytest
import numpy as np

from haystack.document_stores.faiss import FAISSDocumentStore
from haystack.testing import DocumentStoreBaseTestAbstract

from haystack.pipelines import Pipeline

from ..conftest import MockDenseRetriever


class TestFAISSDocumentStore(DocumentStoreBaseTestAbstract):
    @pytest.fixture
    def ds(self, tmp_path):
        return FAISSDocumentStore(
            sql_url=f"sqlite:///{tmp_path}/haystack_test.db",
            return_embedding=True,
            isolation_level="AUTOCOMMIT",
            progress_bar=False,
            similarity="cosine",
        )

    @pytest.fixture
    def documents_with_embeddings(self, documents):
        # drop documents without embeddings from the original fixture
        return [d for d in documents if d.embedding is not None]

    @pytest.mark.unit
    def test_index_mutual_exclusive_args(self, tmp_path):
        with pytest.raises(ValueError, match="faiss_index_path"):
            FAISSDocumentStore(
                sql_url=f"sqlite:////{tmp_path/'haystack_test.db'}",
                faiss_index_path=f"{tmp_path/'haystack_test'}",
                isolation_level="AUTOCOMMIT",
            )

        with pytest.raises(ValueError, match="faiss_index_path"):
            FAISSDocumentStore(
                f"sqlite:////{tmp_path/'haystack_test.db'}",
                faiss_index_path=f"{tmp_path/'haystack_test'}",
                isolation_level="AUTOCOMMIT",
            )

    @pytest.mark.integration
    def test_delete_index(self, ds, documents):
        """Contrary to other Document Stores, FAISSDocumentStore doesn't raise if the index is empty"""
        ds.write_documents(documents)
        assert ds.get_document_count() == len(documents)
        ds.delete_index(ds.index)
        assert ds.get_document_count() == 0

    @pytest.mark.integration
    @pytest.mark.parametrize("config_path", [None, "custom_path.json"])
    def test_index_save_and_load(self, ds, documents_with_embeddings, tmp_path, config_path):
        if config_path:
            config_path = tmp_path / config_path

        ds.write_documents(documents_with_embeddings)

        # test saving the index
        ds.save(index_path=tmp_path / "haystack_test_faiss", config_path=config_path)

        # clear existing faiss_index
        ds.faiss_indexes[ds.index].reset()

        # test faiss index is cleared
        assert ds.faiss_indexes[ds.index].ntotal == 0

        # test loading the index
        new_document_store = FAISSDocumentStore.load(
            index_path=tmp_path / "haystack_test_faiss", config_path=config_path
        )

        # check faiss index is restored
        assert new_document_store.faiss_indexes[ds.index].ntotal == len(documents_with_embeddings)
        # check if documents are restored
        assert len(new_document_store.get_all_documents()) == len(documents_with_embeddings)
        # Check if the init parameters are kept
        assert not new_document_store.progress_bar

        # test saving and loading the loaded faiss index
        new_document_store.save(tmp_path / "haystack_test_faiss", config_path=config_path)
        reloaded_document_store = FAISSDocumentStore.load(tmp_path / "haystack_test_faiss", config_path=config_path)

        # check faiss index is restored
        assert reloaded_document_store.faiss_indexes[ds.index].ntotal == len(documents_with_embeddings)
        # check if documents are restored
        assert len(reloaded_document_store.get_all_documents()) == len(documents_with_embeddings)
        # Check if the init parameters are kept
        assert not reloaded_document_store.progress_bar

        # test loading the index via init
        new_document_store = FAISSDocumentStore(
            faiss_index_path=tmp_path / "haystack_test_faiss", faiss_config_path=config_path
        )

        # check faiss index is restored
        assert new_document_store.faiss_indexes[ds.index].ntotal == len(documents_with_embeddings)
        # check if documents are restored
        assert len(new_document_store.get_all_documents()) == len(documents_with_embeddings)
        # Check if the init parameters are kept
        assert not new_document_store.progress_bar

    @pytest.mark.integration
    @pytest.mark.parametrize("index_buffer_size", [10_000, 2])
    @pytest.mark.parametrize("index_factory", ["Flat", "HNSW", "IVF1,Flat"])
    def test_write_index_docs(self, documents_with_embeddings, tmp_path, index_buffer_size, index_factory):
        document_store = FAISSDocumentStore(
            sql_url=f"sqlite:///{tmp_path}/test_faiss_retrieving_{index_factory}.db",
            faiss_index_factory_str=index_factory,
            isolation_level="AUTOCOMMIT",
            return_embedding=True,
        )
        batch_size = 2
        document_store.index_buffer_size = index_buffer_size
        document_store.delete_all_documents(index=document_store.index)
        if "ivf" in index_factory.lower():
            document_store.train_index(documents_with_embeddings)
            document_store.faiss_indexes[document_store.index].make_direct_map()

        # Write in batches
        for i in range(0, len(documents_with_embeddings), batch_size):
            document_store.write_documents(documents_with_embeddings[i : i + batch_size])

        documents_indexed = document_store.get_all_documents()
        assert len(documents_indexed) == len(documents_with_embeddings)
        assert all(doc.embedding is not None for doc in documents_indexed)
        # Check that get_embedding_count works as expected
        assert document_store.get_embedding_count() == len(documents_with_embeddings)

    @pytest.mark.integration
    def test_write_docs_no_training(self, documents_with_embeddings, tmp_path, caplog):
        document_store = FAISSDocumentStore(
            sql_url=f"sqlite:///{tmp_path}/test_write_docs_no_training.db",
            faiss_index_factory_str="IVF1,Flat",
            isolation_level="AUTOCOMMIT",
            return_embedding=True,
        )
        with pytest.raises(ValueError, match="must be trained before adding vectors"):
            document_store.write_documents(documents_with_embeddings)

    @pytest.mark.integration
    def test_train_index_from_docs(self, documents_with_embeddings, tmp_path):
        document_store = FAISSDocumentStore(
            sql_url=f"sqlite:///{tmp_path}/test_faiss_retrieving.db",
            faiss_index_factory_str="IVF1,Flat",
            isolation_level="AUTOCOMMIT",
            return_embedding=True,
        )
        document_store.delete_all_documents(index=document_store.index)

        assert not document_store.faiss_indexes[document_store.index].is_trained
        document_store.train_index(documents_with_embeddings)
        assert document_store.faiss_indexes[document_store.index].is_trained

    @pytest.mark.integration
    def test_train_index_from_embeddings(self, documents_with_embeddings, tmp_path):
        document_store = FAISSDocumentStore(
            sql_url=f"sqlite:///{tmp_path}/test_faiss_retrieving.db",
            faiss_index_factory_str="IVF1,Flat",
            isolation_level="AUTOCOMMIT",
            return_embedding=True,
        )
        document_store.delete_all_documents(index=document_store.index)

        embeddings = np.array([doc.embedding for doc in documents_with_embeddings])
        assert not document_store.faiss_indexes[document_store.index].is_trained
        document_store.train_index(embeddings=embeddings)
        assert document_store.faiss_indexes[document_store.index].is_trained

    @pytest.mark.integration
    def test_write_docs_different_indexes(self, ds, documents_with_embeddings):
        docs_a = documents_with_embeddings[:2]
        docs_b = documents_with_embeddings[2:]
        ds.write_documents(docs_a, index="index_a")
        ds.write_documents(docs_b, index="index_b")

        docs_from_index_a = ds.get_all_documents(index="index_a", return_embedding=False)
        assert len(docs_from_index_a) == len(docs_a)
        assert {int(doc.meta["vector_id"]) for doc in docs_from_index_a} == {0, 1}

        docs_from_index_b = ds.get_all_documents(index="index_b", return_embedding=False)
        assert len(docs_from_index_b) == len(docs_b)
        assert {int(doc.meta["vector_id"]) for doc in docs_from_index_b} == {0, 1, 2, 3}

    @pytest.mark.integration
    def test_update_docs_different_indexes(self, ds, documents_with_embeddings):
        retriever = MockDenseRetriever(document_store=ds)

        docs_a = documents_with_embeddings[:2]
        docs_b = documents_with_embeddings[2:]
        ds.write_documents(docs_a, index="index_a")
        ds.write_documents(docs_b, index="index_b")

        ds.update_embeddings(retriever=retriever, update_existing_embeddings=True, index="index_a")
        ds.update_embeddings(retriever=retriever, update_existing_embeddings=True, index="index_b")

        docs_from_index_a = ds.get_all_documents(index="index_a", return_embedding=False)
        assert len(docs_from_index_a) == len(docs_a)
        assert {int(doc.meta["vector_id"]) for doc in docs_from_index_a} == {0, 1}

        docs_from_index_b = ds.get_all_documents(index="index_b", return_embedding=False)
        assert len(docs_from_index_b) == len(docs_b)
        assert {int(doc.meta["vector_id"]) for doc in docs_from_index_b} == {0, 1, 2, 3}

    @pytest.mark.integration
    def test_dont_update_existing_embeddings(self, ds, docs):
        retriever = MockDenseRetriever(document_store=ds)
        first_doc_id = docs[0].id

        for i in range(1, 4):
            ds.write_documents(docs[:i])
            ds.update_embeddings(retriever=retriever, update_existing_embeddings=False)

            assert ds.get_document_count() == i
            assert ds.get_embedding_count() == i
            assert ds.get_document_by_id(id=first_doc_id).meta["vector_id"] == "0"

            # Check if the embeddings of the first document remain unchanged after multiple updates
            if i == 1:
                first_doc_embedding = ds.get_document_by_id(id=first_doc_id).embedding
            else:
                assert np.array_equal(ds.get_document_by_id(id=first_doc_id).embedding, first_doc_embedding)

    @pytest.mark.integration
    def test_passing_index_from_outside(self, documents_with_embeddings, tmp_path):
        d = 768
        nlist = 2
        quantizer = faiss.IndexFlatIP(d)
        index = "haystack_test_1"
        faiss_index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
        faiss_index.set_direct_map_type(faiss.DirectMap.Hashtable)
        faiss_index.nprobe = 2
        document_store = FAISSDocumentStore(
            sql_url=f"sqlite:///", faiss_index=faiss_index, index=index, isolation_level="AUTOCOMMIT"
        )

        document_store.delete_documents()
        # as it is a IVF index we need to train it before adding docs
        document_store.train_index(documents_with_embeddings)

        document_store.write_documents(documents=documents_with_embeddings)
        documents_indexed = document_store.get_all_documents()

        # test if vectors ids are associated with docs
        for doc in documents_indexed:
            assert 0 <= int(doc.meta["vector_id"]) <= 7

    @pytest.mark.integration
    def test_pipeline_with_existing_faiss_docstore(self, ds, documents_with_embeddings, tmp_path):
        ds.write_documents(documents_with_embeddings)
        ds.save(tmp_path / "existing_faiss_document_store")
        pipeline_config = {
            "version": "ignore",
            "components": [
                {
                    "name": "DPRRetriever",
                    "type": "MockDenseRetriever",
                    "params": {"document_store": "ExistingFAISSDocumentStore"},
                },
                {
                    "name": "ExistingFAISSDocumentStore",
                    "type": "FAISSDocumentStore",
                    "params": {"faiss_index_path": f"{tmp_path / 'existing_faiss_document_store'}"},
                },
            ],
            "pipelines": [{"name": "query_pipeline", "nodes": [{"name": "DPRRetriever", "inputs": ["Query"]}]}],
        }
        pipeline = Pipeline.load_from_config(pipeline_config)
        existing_document_store = pipeline.get_document_store()
        faiss_index = existing_document_store.faiss_indexes[ds.index]
        assert faiss_index.ntotal == len(documents_with_embeddings)

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

    @pytest.mark.skip(reason="labels metadata are not supported")
    @pytest.mark.integration
    def test_delete_labels_by_filter(self, ds, labels):
        pass

    @pytest.mark.skip(reason="labels metadata are not supported")
    @pytest.mark.integration
    def test_delete_labels_by_filter_id(self, ds, labels):
        pass

    @pytest.mark.skip(reason="labels metadata are not supported")
    @pytest.mark.integration
    def test_multilabel_filter_aggregations(self):
        pass

    @pytest.mark.skip(reason="labels metadata are not supported")
    @pytest.mark.integration
    def test_multilabel_meta_aggregations(self):
        pass

    @pytest.mark.skip(reason="tested in test_write_index_docs")
    @pytest.mark.integration
    def test_get_embedding_count(self):
        pass

    @pytest.mark.skip(reason="can't store embeddings in SQL")
    @pytest.mark.integration
    def test_custom_embedding_field(self, ds):
        pass
