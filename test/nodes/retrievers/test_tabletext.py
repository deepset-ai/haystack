from typing import List

from math import isclose

import pytest
import pandas as pd
import numpy as np
from transformers import DPRContextEncoderTokenizerFast, DPRQuestionEncoderTokenizerFast

from haystack.schema import Document
from haystack.document_stores import BaseDocumentStore, InMemoryDocumentStore
from haystack.nodes.retriever import TableTextRetriever

from test.conftest import SAMPLES_PATH


# TODO Cannot inherit from the base test suites due to the document type
# TODO Is there a way around such limitation?
class TestTableTextRetriever:  # (ABC_TestDenseRetrievers):
    @pytest.fixture
    def table_docs(self):
        table_data = {
            "Mountain": ["Mount Everest", "K2", "Kangchenjunga", "Lhotse", "Makalu"],
            "Height": ["8848m", "8,611 m", "8 586m", "8 516 m", "8,485m"],
        }
        table = pd.DataFrame(table_data)
        return [Document(content=table, content_type="table", id="6")]

    @pytest.fixture
    def mixed_docs(self, docs: List[Document], table_docs: List[Document]):
        return table_docs + docs

    @pytest.fixture
    def docstore(self, mixed_docs: List[Document]):
        docstore = InMemoryDocumentStore(return_embedding=True, embedding_dim=512)
        docstore.write_documents(mixed_docs)
        return docstore

    @pytest.fixture()
    def retriever(self, docstore: BaseDocumentStore):
        retriever = TableTextRetriever(
            document_store=docstore,
            query_embedding_model="deepset/bert-small-mm_retrieval-question_encoder",
            passage_embedding_model="deepset/bert-small-mm_retrieval-passage_encoder",
            table_embedding_model="deepset/bert-small-mm_retrieval-table_encoder",
            use_gpu=False,
        )
        docstore.update_embeddings(retriever=retriever)
        return retriever

    @pytest.fixture()
    def empty_retriever(self):
        return TableTextRetriever(
            document_store=InMemoryDocumentStore(return_embedding=True, embedding_dim=512),
            query_embedding_model="deepset/bert-small-mm_retrieval-question_encoder",
            passage_embedding_model="deepset/bert-small-mm_retrieval-passage_encoder",
            table_embedding_model="deepset/bert-small-mm_retrieval-table_encoder",
            use_gpu=False,
        )

    @pytest.mark.integration
    def test_table_text_retriever_embedding(self, retriever: TableTextRetriever):

        sorted_docs = sorted(retriever.document_store.get_all_documents(), key=lambda d: d.id)
        expected_values = [0.061191384, 0.038075786, 0.27447605, 0.09399721, 0.0959682]

        for doc, expected_value in zip(sorted_docs, expected_values):
            assert len(doc.embedding) == 512
            assert isclose(doc.embedding[0], expected_value, rel_tol=0.001)

    def test_table_text_retriever_saving_and_loading(self, tmp_path, retriever):
        retriever.save(tmp_path / "test_table_text_retriever_save")

        def sum_params(model):
            s = []
            for p in model.parameters():
                n = p.cpu().data.numpy()
                s.append(np.sum(n))
            return sum(s)

        original_sum_query = sum_params(retriever.query_encoder)
        original_sum_passage = sum_params(retriever.passage_encoder)
        original_sum_table = sum_params(retriever.table_encoder)

        loaded_retriever = TableTextRetriever.load(
            tmp_path / "test_table_text_retriever_save", retriever.document_store
        )

        loaded_sum_query = sum_params(loaded_retriever.query_encoder)
        loaded_sum_passage = sum_params(loaded_retriever.passage_encoder)
        loaded_sum_table = sum_params(loaded_retriever.table_encoder)

        assert abs(original_sum_query - loaded_sum_query) < 0.1
        assert abs(original_sum_passage - loaded_sum_passage) < 0.1
        assert abs(original_sum_table - loaded_sum_table) < 0.01

        # attributes
        assert loaded_retriever.processor.embed_meta_fields == retriever.processor.embed_meta_fields
        assert loaded_retriever.batch_size == retriever.batch_size
        assert loaded_retriever.processor.max_seq_len_passage == retriever.processor.max_seq_len_passage
        assert loaded_retriever.processor.max_seq_len_table == retriever.processor.max_seq_len_table
        assert loaded_retriever.processor.max_seq_len_query == retriever.processor.max_seq_len_query

        # Tokenizer
        assert isinstance(loaded_retriever.passage_tokenizer, DPRContextEncoderTokenizerFast)
        assert isinstance(loaded_retriever.table_tokenizer, DPRContextEncoderTokenizerFast)
        assert isinstance(loaded_retriever.query_tokenizer, DPRQuestionEncoderTokenizerFast)
        assert loaded_retriever.passage_tokenizer.do_lower_case == retriever.passage_tokenizer.do_lower_case
        assert loaded_retriever.table_tokenizer.do_lower_case == retriever.table_tokenizer.do_lower_case
        assert loaded_retriever.query_tokenizer.do_lower_case == retriever.query_tokenizer.do_lower_case
        assert loaded_retriever.passage_tokenizer.vocab_size == retriever.passage_tokenizer.vocab_size
        assert loaded_retriever.table_tokenizer.vocab_size == retriever.table_tokenizer.vocab_size
        assert loaded_retriever.query_tokenizer.vocab_size == retriever.query_tokenizer.vocab_size

    def test_table_text_retriever_training(self, tmp_path, empty_retriever: TableTextRetriever):
        empty_retriever.train(
            data_dir=SAMPLES_PATH / "mmr",
            train_filename="sample.json",
            n_epochs=1,
            n_gpu=0,
            save_dir=tmp_path / "test_table_text_retriever_train",
        )

        # Load trained model
        TableTextRetriever.load(
            load_dir=tmp_path / "test_table_text_retriever_train", document_store=empty_retriever.document_store
        )
