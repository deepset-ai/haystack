import pytest
import numpy as np
import pandas as pd

from haystack.nodes import EmbeddingRetriever, TableTextRetriever

from ..conftest import document_store


@pytest.mark.parametrize("name", ["elasticsearch", "faiss", "memory"])
def test_update_embeddings(name, tmp_path):
    documents = []
    for i in range(6):
        documents.append({"content": f"text_{i}", "id": str(i), "meta_field": f"value_{i}"})
    documents.append({"content": "text_0", "id": "6", "meta_field": "value_0"})

    with document_store(name, documents, tmp_path) as ds:
        retriever = EmbeddingRetriever(document_store=ds, embedding_model="deepset/sentence_bert", use_gpu=False)

        ds.update_embeddings(retriever, batch_size=3)
        documents = ds.get_all_documents(return_embedding=True)
        assert len(documents) == 7
        for doc in documents:
            assert type(doc.embedding) is np.ndarray

        documents = ds.get_all_documents(filters={"meta_field": ["value_0"]}, return_embedding=True)
        assert len(documents) == 2
        for doc in documents:
            assert doc.meta["meta_field"] == "value_0"
        np.testing.assert_array_almost_equal(documents[0].embedding, documents[1].embedding, decimal=4)

        documents = ds.get_all_documents(filters={"meta_field": ["value_0", "value_5"]}, return_embedding=True)
        documents_with_value_0 = [doc for doc in documents if doc.meta["meta_field"] == "value_0"]
        documents_with_value_5 = [doc for doc in documents if doc.meta["meta_field"] == "value_5"]
        np.testing.assert_raises(
            AssertionError,
            np.testing.assert_array_equal,
            documents_with_value_0[0].embedding,
            documents_with_value_5[0].embedding,
        )

        doc = {
            "content": "text_7",
            "id": "7",
            "meta_field": "value_7",
            "embedding": retriever.embed_queries(queries=["a random string"])[0],
        }
        ds.write_documents([doc])

        documents = []
        for i in range(8, 11):
            documents.append({"content": f"text_{i}", "id": str(i), "meta_field": f"value_{i}"})
        ds.write_documents(documents)

        doc_before_update = ds.get_all_documents(filters={"meta_field": ["value_7"]})[0]
        embedding_before_update = doc_before_update.embedding

        ds.update_embeddings(retriever, batch_size=3, update_existing_embeddings=False)
        doc_after_update = ds.get_all_documents(filters={"meta_field": ["value_7"]})[0]
        embedding_after_update = doc_after_update.embedding
        np.testing.assert_array_equal(embedding_before_update, embedding_after_update)

        # test updating with filters
        if name == "faiss":
            with pytest.raises(Exception):
                ds.update_embeddings(retriever, update_existing_embeddings=True, filters={"meta_field": ["value"]})
        else:
            ds.update_embeddings(retriever, batch_size=3, filters={"meta_field": ["value_0", "value_1"]})
            doc_after_update = ds.get_all_documents(filters={"meta_field": ["value_7"]})[0]
            embedding_after_update = doc_after_update.embedding
            np.testing.assert_array_equal(embedding_before_update, embedding_after_update)

        # test update all embeddings
        ds.update_embeddings(retriever, batch_size=3, update_existing_embeddings=True)
        assert ds.get_embedding_count() == 11
        doc_after_update = ds.get_all_documents(filters={"meta_field": ["value_7"]})[0]
        embedding_after_update = doc_after_update.embedding
        np.testing.assert_raises(
            AssertionError, np.testing.assert_array_equal, embedding_before_update, embedding_after_update
        )

        # test update embeddings for newly added docs
        documents = []
        for i in range(12, 15):
            documents.append({"content": f"text_{i}", "id": str(i), "meta_field": f"value_{i}"})
        ds.write_documents(documents)

        ds.update_embeddings(retriever, batch_size=3, update_existing_embeddings=False)
        assert ds.get_embedding_count() == 14


def test_update_embeddings_table_text_retriever(tmp_path):
    documents = []
    for i in range(3):
        documents.append(
            {"content": f"text_{i}", "id": f"pssg_{i}", "meta_field": f"value_text_{i}", "content_type": "text"}
        )
        documents.append(
            {
                "content": pd.DataFrame(columns=[f"col_{i}", f"col_{i+1}"], data=[[f"cell_{i}", f"cell_{i+1}"]]),
                "id": f"table_{i}",
                "meta_field": f"value_table_{i}",
                "content_type": "table",
            }
        )
    documents.append({"content": "text_0", "id": "pssg_4", "meta_field": "value_text_0", "content_type": "text"})
    documents.append(
        {
            "content": pd.DataFrame(columns=["col_0", "col_1"], data=[["cell_0", "cell_1"]]),
            "id": "table_4",
            "meta_field": "value_table_0",
            "content_type": "table",
        }
    )

    with document_store("elasticsearch", documents, tmp_path, embedding_dim=512) as ds:
        retriever = TableTextRetriever(
            document_store=document_store,
            query_embedding_model="deepset/bert-small-mm_retrieval-question_encoder",
            passage_embedding_model="deepset/bert-small-mm_retrieval-passage_encoder",
            table_embedding_model="deepset/bert-small-mm_retrieval-table_encoder",
            use_gpu=False,
        )
        ds.update_embeddings(retriever, batch_size=3)
        documents = ds.get_all_documents(return_embedding=True)
        assert len(documents) == 8
        for doc in documents:
            assert type(doc.embedding) is np.ndarray

        # Check if Documents with same content (text) get same embedding
        documents = ds.get_all_documents(filters={"meta_field": ["value_text_0"]}, return_embedding=True)
        assert len(documents) == 2
        for doc in documents:
            assert doc.meta["meta_field"] == "value_text_0"
        np.testing.assert_array_almost_equal(documents[0].embedding, documents[1].embedding, decimal=4)

        # Check if Documents with same content (table) get same embedding
        documents = ds.get_all_documents(filters={"meta_field": ["value_table_0"]}, return_embedding=True)
        assert len(documents) == 2
        for doc in documents:
            assert doc.meta["meta_field"] == "value_table_0"
        np.testing.assert_array_almost_equal(documents[0].embedding, documents[1].embedding, decimal=4)

        # Check if Documents wih different content (text) get different embedding
        documents = ds.get_all_documents(
            filters={"meta_field": ["value_text_1", "value_text_2"]}, return_embedding=True
        )
        np.testing.assert_raises(
            AssertionError, np.testing.assert_array_equal, documents[0].embedding, documents[1].embedding
        )

        # Check if Documents with different content (table) get different embeddings
        documents = ds.get_all_documents(
            filters={"meta_field": ["value_table_1", "value_table_2"]}, return_embedding=True
        )
        np.testing.assert_raises(
            AssertionError, np.testing.assert_array_equal, documents[0].embedding, documents[1].embedding
        )

        # Check if Documents with different content (table + text) get different embeddings
        documents = ds.get_all_documents(
            filters={"meta_field": ["value_text_1", "value_table_1"]}, return_embedding=True
        )
        np.testing.assert_raises(
            AssertionError, np.testing.assert_array_equal, documents[0].embedding, documents[1].embedding
        )
