import pytest
import numpy as np
from math import isclose

from transformers import DPRContextEncoderTokenizerFast, DPRQuestionEncoderTokenizerFast

from haystack.document_stores import BaseDocumentStore
from haystack.nodes.retriever import DensePassageRetriever

from test.nodes.retrievers.base import ABC_TestTextRetrievers


@pytest.mark.integration
class TestDPRetriever(ABC_TestTextRetrievers):
    @pytest.fixture()
    def retriever(self, docstore: BaseDocumentStore):
        retriever = DensePassageRetriever(
            document_store=docstore,
            query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
            passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base",
            use_gpu=False,
            embed_title=True,
        )
        retriever.document_store.update_embeddings(retriever=retriever)
        return retriever

    def test_dpr_embedding(self, retriever: DensePassageRetriever):
        expected_embeddings_start = [0.00892, 0.00780, 0.00482, -0.00626, 0.010966]
        sorted_docs = sorted(retriever.document_store.get_all_documents(), key=lambda d: d.id)

        for doc, expected_value in zip(sorted_docs, expected_embeddings_start):
            embedding = doc.embedding
            # always normalize vector as faiss returns normalized vectors and other document stores do not
            embedding /= np.linalg.norm(embedding)
            assert len(embedding) == 768
            assert isclose(embedding[0], expected_value, rel_tol=0.001)

    def test_dpr_saving_and_loading(self, tmp_path, retriever: DensePassageRetriever, docstore: BaseDocumentStore):
        retriever.save(f"{tmp_path}/test_dpr_save")

        def sum_params(model):
            s = []
            for p in model.parameters():
                n = p.cpu().data.numpy()
                s.append(np.sum(n))
            return sum(s)

        original_sum_query = sum_params(retriever.query_encoder)
        original_sum_passage = sum_params(retriever.passage_encoder)

        loaded_retriever = DensePassageRetriever.load(load_dir=f"{tmp_path}/test_dpr_save", document_store=docstore)

        loaded_sum_query = sum_params(loaded_retriever.query_encoder)
        loaded_sum_passage = sum_params(loaded_retriever.passage_encoder)

        assert abs(original_sum_query - loaded_sum_query) < 0.1
        assert abs(original_sum_passage - loaded_sum_passage) < 0.1

        # # comparison of weights (RAM intense!)
        # # for p1, p2 in zip(retriever.query_encoder.parameters(), loaded_retriever.query_encoder.parameters()):
        # #     assert (p1.data.ne(p2.data).sum() == 0)
        # #
        # # for p1, p2 in zip(retriever.passage_encoder.parameters(), loaded_retriever.passage_encoder.parameters()):
        # #     assert (p1.data.ne(p2.data).sum() == 0)

        # attributes
        assert loaded_retriever.processor.embed_title == retriever.processor.embed_title
        assert loaded_retriever.batch_size == retriever.batch_size
        assert loaded_retriever.processor.max_seq_len_passage == retriever.processor.max_seq_len_passage
        assert loaded_retriever.processor.max_seq_len_query == retriever.processor.max_seq_len_query

        # Tokenizer
        assert isinstance(loaded_retriever.passage_tokenizer, DPRContextEncoderTokenizerFast)
        assert isinstance(loaded_retriever.query_tokenizer, DPRQuestionEncoderTokenizerFast)
        assert loaded_retriever.passage_tokenizer.do_lower_case == retriever.passage_tokenizer.do_lower_case
        assert loaded_retriever.query_tokenizer.do_lower_case == retriever.query_tokenizer.do_lower_case
        assert loaded_retriever.passage_tokenizer.vocab_size == retriever.passage_tokenizer.vocab_size
        assert loaded_retriever.query_tokenizer.vocab_size == retriever.query_tokenizer.vocab_size
