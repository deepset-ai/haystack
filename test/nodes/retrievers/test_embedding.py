from typing import List

import pytest

from haystack.nodes.retriever.base import BaseRetriever
from haystack.schema import Document
from haystack.document_stores import ElasticsearchDocumentStore
from haystack.nodes.retriever import DensePassageRetriever

from test.nodes.retrievers.dense import TestDenseRetrievers


class TestEmbeddingRetriever(TestDenseRetrievers):
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
