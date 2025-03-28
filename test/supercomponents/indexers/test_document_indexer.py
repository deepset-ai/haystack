# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock, ANY
from uuid import UUID
import pytest

from haystack import Document, Pipeline
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.writers import DocumentWriter
from haystack.super_components.indexers import SentenceTransformersDocumentIndexer


class TestDocumentIndexer:
    @pytest.fixture
    def indexer(self) -> SentenceTransformersDocumentIndexer:
        return SentenceTransformersDocumentIndexer(document_store=InMemoryDocumentStore())

    @pytest.fixture
    def embedding_backend(self, indexer: SentenceTransformersDocumentIndexer, monkeypatch: pytest.MonkeyPatch) -> Mock:
        backend = Mock()
        backend.embed.return_value = [[0.3, 0.4, 0.01, 0.7], [0.1, 0.9, 0.87, 0.3]]

        embedder = indexer.pipeline.get_component("embedder")
        monkeypatch.setattr(embedder, "embedding_backend", backend)

        return backend

    def test_init(self, indexer: SentenceTransformersDocumentIndexer) -> None:
        assert isinstance(indexer.pipeline, Pipeline)
        assert indexer.input_mapping == {"documents": ["embedder.documents"]}
        assert indexer.output_mapping == {"writer.documents_written": "documents_written"}

        embedder = indexer.pipeline.get_component("embedder")
        assert isinstance(embedder, SentenceTransformersDocumentEmbedder)

        writer = indexer.pipeline.get_component("writer")
        assert isinstance(writer, DocumentWriter)
        assert isinstance(writer.document_store, InMemoryDocumentStore)
        assert writer.document_store.bm25_tokenization_regex == r"(?u)\b\w\w+\b"
        assert writer.document_store.bm25_algorithm == "BM25L"
        assert writer.document_store.bm25_parameters == {}
        assert writer.document_store.embedding_similarity_function == "dot_product"
        assert UUID(writer.document_store.index, version=4)

    def test_from_dict(self) -> None:
        indexer = SentenceTransformersDocumentIndexer.from_dict(
            {
                "init_parameters": {
                    "document_store": {
                        "init_parameters": {
                            "bm25_algorithm": "BM25L",
                            "bm25_parameters": {},
                            "bm25_tokenization_regex": "(?u)\\b\\w\\w+\\b",
                            "embedding_similarity_function": "dot_product",
                            "index": "28f84766-11b7-4eac-bb75-3ee4e8d56958",
                        },
                        "type": "haystack.document_stores.in_memory.document_store.InMemoryDocumentStore",
                    },
                    "prefix": "",
                    "suffix": "",
                    "batch_size": 32,
                    "embedding_separator": "\n",
                    "meta_fields_to_embed": None,
                    "duplicate_policy": "overwrite",
                },
                "type": "haystack.super_components.indexers.sentence_transformers_document_indexer.SentenceTransformersDocumentIndexer",
            }
        )
        assert isinstance(indexer, SentenceTransformersDocumentIndexer)

    def test_to_dict(self, indexer: SentenceTransformersDocumentIndexer) -> None:
        expected = {
            "init_parameters": {
                "batch_size": 32,
                "config_kwargs": None,
                "device": None,
                "document_store": {
                    "init_parameters": {
                        "bm25_algorithm": "BM25L",
                        "bm25_parameters": {},
                        "bm25_tokenization_regex": "(?u)\\b\\w\\w+\\b",
                        "embedding_similarity_function": "dot_product",
                        "index": ANY,
                    },
                    "type": "haystack.document_stores.in_memory.document_store.InMemoryDocumentStore",
                },
                "duplicate_policy": "overwrite",
                "embedding_separator": "\n",
                "meta_fields_to_embed": None,
                "model": "sentence-transformers/all-mpnet-base-v2",
                "model_kwargs": None,
                "normalize_embeddings": False,
                "precision": "float32",
                "prefix": "",
                "progress_bar": True,
                "suffix": "",
                "token": {"env_vars": ["HF_API_TOKEN", "HF_TOKEN"], "strict": False, "type": "env_var"},
                "tokenizer_kwargs": None,
                "truncate_dim": None,
                "trust_remote_code": False,
            },
            "type": "haystack.super_components.indexers.sentence_transformers_document_indexer.SentenceTransformersDocumentIndexer",
        }
        assert indexer.to_dict() == expected

    def test_warm_up(self, indexer: SentenceTransformersDocumentIndexer, monkeypatch: pytest.MonkeyPatch) -> None:
        with monkeypatch.context() as m:
            m.setattr(indexer.pipeline, "warm_up", Mock())

            indexer.warm_up()

            indexer.pipeline.warm_up.assert_called_once()

    def test_run(self, indexer: SentenceTransformersDocumentIndexer, embedding_backend: Mock) -> None:
        documents = [Document(content="Test document"), Document(content="Another test document")]

        indexer.warm_up()
        result = indexer.run(documents=documents)

        embedding_backend.embed.assert_called_once
        assert result == {"documents_written": len(documents)}
