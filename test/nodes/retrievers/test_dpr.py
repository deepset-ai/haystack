from typing import List

import pytest

from haystack.nodes.retriever.base import BaseRetriever
from haystack.schema import Document
from haystack.document_stores import ElasticsearchDocumentStore
from haystack.nodes.retriever import DensePassageRetriever

from test.nodes.retrievers.dense import TestDenseRetrievers


class TestDPRetriever(TestDenseRetrievers):
    @pytest.fixture()
    def test_retriever(self, docstore):
        pass

    @pytest.mark.integration
    @pytest.mark.parametrize(
        "document_store",
        ["elasticsearch", "faiss", "memory", "milvus1", "milvus", "weaviate", "pinecone"],
        indirect=True,
    )
    @pytest.mark.parametrize("retriever", ["dpr"], indirect=True)
    def test_dpr_embedding(self, document_store: BaseDocumentStore, retriever, docs_with_ids):
        document_store.return_embedding = True
        document_store.write_documents(docs_with_ids)
        document_store.update_embeddings(retriever=retriever)

        docs = document_store.get_all_documents()
        docs.sort(key=lambda d: d.id)

        print([doc.id for doc in docs])

        expected_values = [0.00892, 0.00780, 0.00482, -0.00626, 0.010966]
        for doc, expected_value in zip(docs, expected_values):
            embedding = doc.embedding
            # always normalize vector as faiss returns normalized vectors and other document stores do not
            embedding /= np.linalg.norm(embedding)
            assert len(embedding) == 768
            assert isclose(embedding[0], expected_value, rel_tol=0.001)

    @pytest.mark.parametrize("retriever", ["dpr"], indirect=True)
    @pytest.mark.parametrize("document_store", ["memory"], indirect=True)
    def test_dpr_saving_and_loading(self, tmp_path, retriever, document_store):
        retriever.save(f"{tmp_path}/test_dpr_save")

        def sum_params(model):
            s = []
            for p in model.parameters():
                n = p.cpu().data.numpy()
                s.append(np.sum(n))
            return sum(s)

        original_sum_query = sum_params(retriever.query_encoder)
        original_sum_passage = sum_params(retriever.passage_encoder)
        del retriever

        loaded_retriever = DensePassageRetriever.load(f"{tmp_path}/test_dpr_save", document_store)

        loaded_sum_query = sum_params(loaded_retriever.query_encoder)
        loaded_sum_passage = sum_params(loaded_retriever.passage_encoder)

        assert abs(original_sum_query - loaded_sum_query) < 0.1
        assert abs(original_sum_passage - loaded_sum_passage) < 0.1

        # comparison of weights (RAM intense!)
        # for p1, p2 in zip(retriever.query_encoder.parameters(), loaded_retriever.query_encoder.parameters()):
        #     assert (p1.data.ne(p2.data).sum() == 0)
        #
        # for p1, p2 in zip(retriever.passage_encoder.parameters(), loaded_retriever.passage_encoder.parameters()):
        #     assert (p1.data.ne(p2.data).sum() == 0)

        # attributes
        assert loaded_retriever.processor.embed_title == True
        assert loaded_retriever.batch_size == 16
        assert loaded_retriever.processor.max_seq_len_passage == 256
        assert loaded_retriever.processor.max_seq_len_query == 64

        # Tokenizer
        assert isinstance(loaded_retriever.passage_tokenizer, DPRContextEncoderTokenizerFast)
        assert isinstance(loaded_retriever.query_tokenizer, DPRQuestionEncoderTokenizerFast)
        assert loaded_retriever.passage_tokenizer.do_lower_case == True
        assert loaded_retriever.query_tokenizer.do_lower_case == True
        assert loaded_retriever.passage_tokenizer.vocab_size == 30522
        assert loaded_retriever.query_tokenizer.vocab_size == 30522


class TestEmbeddingRetriever(_TestDenseRetrievers):
    @pytest.fixture()
    def test_retriever(self, docstore):
        pass

    def test_embeddings_encoder_of_embedding_retriever_should_warn_about_model_format(self, caplog):
        document_store = InMemoryDocumentStore()

        with caplog.at_level(logging.WARNING):
            EmbeddingRetriever(
                document_store=document_store,
                embedding_model="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
                model_format="farm",
            )

            assert (
                "You may need to set model_format='sentence_transformers' to ensure correct loading of model."
                in caplog.text
            )


class TestTableTextRetriever(_TestDenseRetrievers):
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


class TestMultiHopRetriever(_TestDenseRetrievers):
    @pytest.fixture()
    def test_retriever(self, docstore):
        pass


class TestMultiModalRetriever(_TestDenseRetrievers):
    @pytest.fixture()
    def test_retriever(self, docstore):
        pass
