from typing import List

import pytest
import numpy as np
from math import isclose

from transformers import DPRContextEncoderTokenizerFast, DPRQuestionEncoderTokenizerFast

from haystack.nodes.retriever.base import BaseRetriever
from haystack.schema import Document
from haystack.document_stores import BaseDocumentStore, ElasticsearchDocumentStore
from haystack.nodes.retriever import DensePassageRetriever

from test.nodes.retrievers.dense import ABC_TestDenseRetrievers


class TestDPRetriever(ABC_TestDenseRetrievers):
    @pytest.fixture()
    def retriever(self, docstore):
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
