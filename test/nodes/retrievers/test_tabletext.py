from typing import List

import pytest

from haystack.nodes.retriever.base import BaseRetriever
from haystack.schema import Document
from haystack.document_stores import ElasticsearchDocumentStore
from haystack.nodes.retriever import DensePassageRetriever

from test.nodes.retrievers.dense import TestDenseRetrievers


class TestTableTextRetriever(TestDenseRetrievers):
    @pytest.fixture()
    def test_retriever(self, docstore):
        pass

    @pytest.mark.integration
    @pytest.mark.parametrize("retriever", ["table_text_retriever"], indirect=True)
    @pytest.mark.parametrize("document_store", ["elasticsearch", "memory"], indirect=True)
    @pytest.mark.embedding_dim(512)
    def test_table_text_retriever_embedding(self, document_store, retriever, docs):

        document_store.return_embedding = True
        document_store.write_documents(docs)
        table_data = {
            "Mountain": ["Mount Everest", "K2", "Kangchenjunga", "Lhotse", "Makalu"],
            "Height": ["8848m", "8,611 m", "8 586m", "8 516 m", "8,485m"],
        }
        table = pd.DataFrame(table_data)
        table_doc = Document(content=table, content_type="table", id="6")
        document_store.write_documents([table_doc])
        document_store.update_embeddings(retriever=retriever)

        docs = document_store.get_all_documents()
        docs = sorted(docs, key=lambda d: d.id)

        expected_values = [0.061191384, 0.038075786, 0.27447605, 0.09399721, 0.0959682]
        for doc, expected_value in zip(docs, expected_values):
            assert len(doc.embedding) == 512
            assert isclose(doc.embedding[0], expected_value, rel_tol=0.001)

    @pytest.mark.parametrize("retriever", ["table_text_retriever"], indirect=True)
    @pytest.mark.embedding_dim(512)
    def test_table_text_retriever_saving_and_loading(self, tmp_path, retriever, document_store):
        retriever.save(f"{tmp_path}/test_table_text_retriever_save")

        def sum_params(model):
            s = []
            for p in model.parameters():
                n = p.cpu().data.numpy()
                s.append(np.sum(n))
            return sum(s)

        original_sum_query = sum_params(retriever.query_encoder)
        original_sum_passage = sum_params(retriever.passage_encoder)
        original_sum_table = sum_params(retriever.table_encoder)
        del retriever

        loaded_retriever = TableTextRetriever.load(f"{tmp_path}/test_table_text_retriever_save", document_store)

        loaded_sum_query = sum_params(loaded_retriever.query_encoder)
        loaded_sum_passage = sum_params(loaded_retriever.passage_encoder)
        loaded_sum_table = sum_params(loaded_retriever.table_encoder)

        assert abs(original_sum_query - loaded_sum_query) < 0.1
        assert abs(original_sum_passage - loaded_sum_passage) < 0.1
        assert abs(original_sum_table - loaded_sum_table) < 0.01

        # attributes
        assert loaded_retriever.processor.embed_meta_fields == ["name", "section_title", "caption"]
        assert loaded_retriever.batch_size == 16
        assert loaded_retriever.processor.max_seq_len_passage == 256
        assert loaded_retriever.processor.max_seq_len_table == 256
        assert loaded_retriever.processor.max_seq_len_query == 64

        # Tokenizer
        assert isinstance(loaded_retriever.passage_tokenizer, DPRContextEncoderTokenizerFast)
        assert isinstance(loaded_retriever.table_tokenizer, DPRContextEncoderTokenizerFast)
        assert isinstance(loaded_retriever.query_tokenizer, DPRQuestionEncoderTokenizerFast)
        assert loaded_retriever.passage_tokenizer.do_lower_case == True
        assert loaded_retriever.table_tokenizer.do_lower_case == True
        assert loaded_retriever.query_tokenizer.do_lower_case == True
        assert loaded_retriever.passage_tokenizer.vocab_size == 30522
        assert loaded_retriever.table_tokenizer.vocab_size == 30522
        assert loaded_retriever.query_tokenizer.vocab_size == 30522

    @pytest.mark.embedding_dim(128)
    def test_table_text_retriever_training(self, document_store):
        retriever = TableTextRetriever(
            document_store=document_store,
            query_embedding_model="deepset/bert-small-mm_retrieval-question_encoder",
            passage_embedding_model="deepset/bert-small-mm_retrieval-passage_encoder",
            table_embedding_model="deepset/bert-small-mm_retrieval-table_encoder",
            use_gpu=False,
        )

        retriever.train(
            data_dir=SAMPLES_PATH / "mmr",
            train_filename="sample.json",
            n_epochs=1,
            n_gpu=0,
            save_dir="test_table_text_retriever_train",
        )

        # Load trained model
        retriever = TableTextRetriever.load(load_dir="test_table_text_retriever_train", document_store=document_store)
