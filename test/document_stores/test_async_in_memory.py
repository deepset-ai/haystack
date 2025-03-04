# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import logging
from unittest.mock import patch

import pytest
import tempfile

from haystack import Document
from haystack.document_stores.errors import DocumentStoreError, DuplicateDocumentError
from haystack.document_stores.in_memory.async_document_store import InMemoryDocumentStore
from haystack.testing.document_store import DocumentStoreBaseTests


class TestAsyncMemoryDocumentStore(DocumentStoreBaseTests):  # pylint: disable=R0904
    """
    Test AsyncInMemoryDocumentStore's specific features
    """

    @pytest.fixture
    def tmp_dir(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield tmp_dir

    @pytest.fixture
    def document_store(self) -> InMemoryDocumentStore:
        return InMemoryDocumentStore(bm25_algorithm="BM25L")

    def test_to_dict(self):
        store = InMemoryDocumentStore()
        data = store.to_dict()
        assert data == {
            "type": "haystack.document_stores.in_memory.document_store.InMemoryDocumentStore",
            "init_parameters": {
                "bm25_tokenization_regex": r"(?u)\b\w\w+\b",
                "bm25_algorithm": "BM25L",
                "bm25_parameters": {},
                "embedding_similarity_function": "dot_product",
                "index": store.index,
            },
        }

    def test_to_dict_with_custom_init_parameters(self):
        store = InMemoryDocumentStore(
            bm25_tokenization_regex="custom_regex",
            bm25_algorithm="BM25Plus",
            bm25_parameters={"key": "value"},
            embedding_similarity_function="cosine",
            index="my_cool_index",
        )
        data = store.to_dict()
        assert data == {
            "type": "haystack.document_stores.in_memory.document_store.InMemoryDocumentStore",
            "init_parameters": {
                "bm25_tokenization_regex": "custom_regex",
                "bm25_algorithm": "BM25Plus",
                "bm25_parameters": {"key": "value"},
                "embedding_similarity_function": "cosine",
                "index": "my_cool_index",
            },
        }

    @patch("haystack.document_stores.in_memory.document_store.re")
    def test_from_dict(self, mock_regex):
        data = {
            "type": "haystack.document_stores.in_memory.document_store.InMemoryDocumentStore",
            "init_parameters": {
                "bm25_tokenization_regex": "custom_regex",
                "bm25_algorithm": "BM25Plus",
                "bm25_parameters": {"key": "value"},
                "index": "my_cool_index",
            },
        }
        store = InMemoryDocumentStore.from_dict(data)
        mock_regex.compile.assert_called_with("custom_regex")
        assert store.tokenizer
        assert store.bm25_algorithm == "BM25Plus"
        assert store.bm25_parameters == {"key": "value"}
        assert store.index == "my_cool_index"

    def test_save_to_disk_and_load_from_disk(self, tmp_dir: str):
        docs = [Document(content="Hello world"), Document(content="Haystack supports multiple languages")]
        document_store = InMemoryDocumentStore()
        document_store.write_documents(docs)
        tmp_dir = tmp_dir + "/document_store.json"
        document_store.save_to_disk(tmp_dir)
        document_store_loaded = InMemoryDocumentStore.load_from_disk(tmp_dir)

        assert document_store_loaded.count_documents() == 2
        assert list(document_store_loaded.storage.values()) == docs
        assert document_store_loaded.to_dict() == document_store.to_dict()

    def test_invalid_bm25_algorithm(self):
        with pytest.raises(ValueError, match="BM25 algorithm 'invalid' is not supported"):
            InMemoryDocumentStore(bm25_algorithm="invalid")

    @pytest.mark.asyncio
    async def test_write_documents(self, document_store):
        docs = [Document(id="1")]
        assert await document_store.write_documents_async(docs) == 1
        with pytest.raises(DuplicateDocumentError):
            await document_store.write_documents_async(docs)

    @pytest.mark.asyncio
    async def test_bm25_retrieval(self, document_store: InMemoryDocumentStore):
        # Tests if the bm25_retrieval method returns the correct document based on the input query.
        docs = [Document(content="Hello world"), Document(content="Haystack supports multiple languages")]
        await document_store.write_documents_async(docs)
        results = await document_store.bm25_retrieval_async(query="What languages?", top_k=1)
        assert len(results) == 1
        assert results[0].content == "Haystack supports multiple languages"

    @pytest.mark.asyncio
    async def test_bm25_retrieval_with_empty_document_store(self, document_store: InMemoryDocumentStore, caplog):
        caplog.set_level(logging.INFO)
        # Tests if the bm25_retrieval method correctly returns an empty list when there are no documents in the DocumentStore.
        results = await document_store.bm25_retrieval_async(query="How to test this?", top_k=2)
        assert len(results) == 0
        assert "No documents found for BM25 retrieval. Returning empty list." in caplog.text

    @pytest.mark.asyncio
    async def test_bm25_retrieval_empty_query(self, document_store: InMemoryDocumentStore):
        # Tests if the bm25_retrieval method returns a document when the query is an empty string.
        docs = [Document(content="Hello world"), Document(content="Haystack supports multiple languages")]
        await document_store.write_documents_async(docs)
        with pytest.raises(ValueError, match="Query should be a non-empty string"):
            await document_store.bm25_retrieval_async(query="", top_k=1)

    @pytest.mark.asyncio
    async def test_bm25_retrieval_with_different_top_k(self, document_store: InMemoryDocumentStore):
        # Tests if the bm25_retrieval method correctly changes the number of returned documents
        # based on the top_k parameter.
        docs = [
            Document(content="Hello world"),
            Document(content="Haystack supports multiple languages"),
            Document(content="Python is a popular programming language"),
        ]
        await document_store.write_documents_async(docs)

        # top_k = 2
        results = await document_store.bm25_retrieval_async(query="language", top_k=2)
        assert len(results) == 2

        # top_k = 3
        results = await document_store.bm25_retrieval_async(query="languages", top_k=3)
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_bm25_plus_retrieval(self):
        doc_store = InMemoryDocumentStore(bm25_algorithm="BM25Plus")
        docs = [
            Document(content="Hello world"),
            Document(content="Haystack supports multiple languages"),
            Document(content="Python is a popular programming language"),
        ]
        await doc_store.write_documents_async(docs)

        results = await doc_store.bm25_retrieval_async(query="language", top_k=1)
        assert len(results) == 1
        assert results[0].content == "Python is a popular programming language"

    @pytest.mark.asyncio
    async def test_bm25_retrieval_with_two_queries(self, document_store: InMemoryDocumentStore):
        # Tests if the bm25_retrieval method returns different documents for different queries.
        docs = [
            Document(content="Javascript is a popular programming language"),
            Document(content="Java is a popular programming language"),
            Document(content="Python is a popular programming language"),
            Document(content="Ruby is a popular programming language"),
            Document(content="PHP is a popular programming language"),
        ]
        await document_store.write_documents_async(docs)

        results = await document_store.bm25_retrieval_async(query="Java", top_k=1)
        assert results[0].content == "Java is a popular programming language"

        results = await document_store.bm25_retrieval_async(query="Python", top_k=1)
        assert results[0].content == "Python is a popular programming language"

    @pytest.mark.asyncio
    async def test_bm25_retrieval_with_updated_docs(self, document_store: InMemoryDocumentStore):
        # Tests if the bm25_retrieval method correctly updates the retrieved documents when new
        # documents are added to the DocumentStore.
        docs = [Document(content="Hello world")]
        await document_store.write_documents_async(docs)

        results = await document_store.bm25_retrieval_async(query="Python", top_k=1)
        assert len(results) == 0

        await document_store.write_documents_async([Document(content="Python is a popular programming language")])
        results = await document_store.bm25_retrieval_async(query="Python", top_k=1)
        assert len(results) == 1
        assert results[0].content == "Python is a popular programming language"

        await document_store.write_documents_async([Document(content="Java is a popular programming language")])
        results = await document_store.bm25_retrieval_async(query="Python", top_k=1)
        assert len(results) == 1
        assert results[0].content == "Python is a popular programming language"

    @pytest.mark.asyncio
    async def test_bm25_retrieval_with_scale_score(self, document_store: InMemoryDocumentStore):
        docs = [Document(content="Python programming"), Document(content="Java programming")]
        await document_store.write_documents_async(docs)

        results1 = await document_store.bm25_retrieval_async(query="Python", top_k=1, scale_score=True)
        # Confirm that score is scaled between 0 and 1
        assert results1[0].score is not None
        assert 0.0 <= results1[0].score <= 1.0

        # Same query, different scale, scores differ when not scaled
        results = await document_store.bm25_retrieval_async(query="Python", top_k=1, scale_score=False)
        assert results[0].score != results1[0].score

    @pytest.mark.asyncio
    async def test_bm25_retrieval_with_non_scaled_BM25Okapi(self):
        # Highly repetitive documents make BM25Okapi return negative scores, which should not be filtered if the
        # scores are not scaled
        docs = [
            Document(
                content="""Use pip to install a basic version of Haystack's latest release: pip install
                farm-haystack. All the core Haystack components live in the haystack repo. But there's also the
                haystack-extras repo which contains components that are not as widely used, and you need to
                install them separately."""
            ),
            Document(
                content="""Use pip to install a basic version of Haystack's latest release: pip install
                farm-haystack[inference]. All the core Haystack components live in the haystack repo. But there's
                also the haystack-extras repo which contains components that are not as widely used, and you need
                to install them separately."""
            ),
            Document(
                content="""Use pip to install only the Haystack 2.0 code: pip install haystack-ai. The haystack-ai
                package is built on the main branch which is an unstable beta version, but it's useful if you want
                to try the new features as soon as they are merged."""
            ),
        ]
        document_store = InMemoryDocumentStore(bm25_algorithm="BM25Okapi")
        await document_store.write_documents_async(docs)

        results1 = await document_store.bm25_retrieval_async(query="Haystack installation", top_k=10, scale_score=False)
        assert len(results1) == 3
        assert all(res.score < 0.0 for res in results1)

        results2 = await document_store.bm25_retrieval_async(query="Haystack installation", top_k=10, scale_score=True)
        assert len(results2) == 3
        assert all(0.0 <= res.score <= 1.0 for res in results2)

    @pytest.mark.asyncio
    async def test_bm25_retrieval_default_filter(self, document_store: InMemoryDocumentStore):
        docs = [Document(), Document(content="Gardening"), Document(content="Bird watching")]
        await document_store.write_documents_async(docs)
        results = await document_store.bm25_retrieval_async(query="doesn't matter, top_k is 10", top_k=10)
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_embedding_retrieval(self):
        docstore = InMemoryDocumentStore(embedding_similarity_function="cosine")
        # Tests if the embedding retrieval method returns the correct document based on the input query embedding.
        docs = [
            Document(content="Hello world", embedding=[0.1, 0.2, 0.3, 0.4]),
            Document(content="Haystack supports multiple languages", embedding=[1.0, 1.0, 1.0, 1.0]),
        ]
        await docstore.write_documents_async(docs)
        results = await docstore.embedding_retrieval_async(
            query_embedding=[0.1, 0.1, 0.1, 0.1], top_k=1, filters={}, scale_score=False
        )
        assert len(results) == 1
        assert results[0].content == "Haystack supports multiple languages"

    @pytest.mark.asyncio
    async def test_embedding_retrieval_invalid_query(self):
        docstore = InMemoryDocumentStore()
        with pytest.raises(ValueError, match="query_embedding should be a non-empty list of floats"):
            await docstore.embedding_retrieval_async(query_embedding=[])
        with pytest.raises(ValueError, match="query_embedding should be a non-empty list of floats"):
            await docstore.embedding_retrieval_async(query_embedding=["invalid", "list", "of", "strings"])  # type: ignore

    @pytest.mark.asyncio
    async def test_embedding_retrieval_no_embeddings(self, caplog):
        caplog.set_level(logging.WARNING)
        docstore = InMemoryDocumentStore()
        docs = [Document(content="Hello world"), Document(content="Haystack supports multiple languages")]
        await docstore.write_documents_async(docs)
        results = await docstore.embedding_retrieval_async(query_embedding=[0.1, 0.1, 0.1, 0.1])
        assert len(results) == 0
        assert "No Documents found with embeddings. Returning empty list." in caplog.text

    @pytest.mark.asyncio
    async def test_embedding_retrieval_some_documents_wo_embeddings(self, caplog):
        caplog.set_level(logging.INFO)
        docstore = InMemoryDocumentStore()
        docs = [
            Document(content="Hello world", embedding=[0.1, 0.2, 0.3, 0.4]),
            Document(content="Haystack supports multiple languages"),
        ]
        await docstore.write_documents_async(docs)
        await docstore.embedding_retrieval_async(query_embedding=[0.1, 0.1, 0.1, 0.1])
        assert "Skipping some Documents that don't have an embedding." in caplog.text

    @pytest.mark.asyncio
    async def test_embedding_retrieval_documents_different_embedding_sizes(self):
        docstore = InMemoryDocumentStore()
        docs = [
            Document(content="Hello world", embedding=[0.1, 0.2, 0.3, 0.4]),
            Document(content="Haystack supports multiple languages", embedding=[1.0, 1.0]),
        ]
        await docstore.write_documents_async(docs)

        with pytest.raises(DocumentStoreError, match="The embedding size of all Documents should be the same."):
            await docstore.embedding_retrieval_async(query_embedding=[0.1, 0.1, 0.1, 0.1])

    @pytest.mark.asyncio
    async def test_embedding_retrieval_query_documents_different_embedding_sizes(self):
        docstore = InMemoryDocumentStore()
        docs = [Document(content="Hello world", embedding=[0.1, 0.2, 0.3, 0.4])]
        await docstore.write_documents_async(docs)

        with pytest.raises(
            DocumentStoreError,
            match="The embedding size of the query should be the same as the embedding size of the Documents.",
        ):
            await docstore.embedding_retrieval_async(query_embedding=[0.1, 0.1])

    @pytest.mark.asyncio
    async def test_embedding_retrieval_with_different_top_k(self):
        docstore = InMemoryDocumentStore()
        docs = [
            Document(content="Hello world", embedding=[0.1, 0.2, 0.3, 0.4]),
            Document(content="Haystack supports multiple languages", embedding=[1.0, 1.0, 1.0, 1.0]),
            Document(content="Python is a popular programming language", embedding=[0.5, 0.5, 0.5, 0.5]),
        ]
        await docstore.write_documents_async(docs)

        results = await docstore.embedding_retrieval_async(query_embedding=[0.1, 0.1, 0.1, 0.1], top_k=2)
        assert len(results) == 2

        results = await docstore.embedding_retrieval_async(query_embedding=[0.1, 0.1, 0.1, 0.1], top_k=3)
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_embedding_retrieval_with_scale_score(self):
        docstore = InMemoryDocumentStore()
        docs = [
            Document(content="Hello world", embedding=[0.1, 0.2, 0.3, 0.4]),
            Document(content="Haystack supports multiple languages", embedding=[1.0, 1.0, 1.0, 1.0]),
            Document(content="Python is a popular programming language", embedding=[0.5, 0.5, 0.5, 0.5]),
        ]
        await docstore.write_documents_async(docs)

        results1 = await docstore.embedding_retrieval_async(
            query_embedding=[0.1, 0.1, 0.1, 0.1], top_k=1, scale_score=True
        )
        # Confirm that score is scaled between 0 and 1
        assert results1[0].score is not None
        assert 0.0 <= results1[0].score <= 1.0

        # Same query, different scale, scores differ when not scaled
        results = await docstore.embedding_retrieval_async(
            query_embedding=[0.1, 0.1, 0.1, 0.1], top_k=1, scale_score=False
        )
        assert results[0].score != results1[0].score

    @pytest.mark.asyncio
    async def test_embedding_retrieval_return_embedding(self):
        docstore = InMemoryDocumentStore(embedding_similarity_function="cosine")
        docs = [
            Document(content="Hello world", embedding=[0.1, 0.2, 0.3, 0.4]),
            Document(content="Haystack supports multiple languages", embedding=[1.0, 1.0, 1.0, 1.0]),
        ]
        await docstore.write_documents_async(docs)

        results = await docstore.embedding_retrieval_async(
            query_embedding=[0.1, 0.1, 0.1, 0.1], top_k=1, return_embedding=False
        )
        assert results[0].embedding is None

        results = await docstore.embedding_retrieval_async(
            query_embedding=[0.1, 0.1, 0.1, 0.1], top_k=1, return_embedding=True
        )
        assert results[0].embedding == [1.0, 1.0, 1.0, 1.0]

    @pytest.mark.asyncio
    async def test_multiple_document_stores_using_same_index(self):
        index = "test_multiple_document_stores_using_same_index"
        document_store_1 = InMemoryDocumentStore(index=index)
        document_store_2 = InMemoryDocumentStore(index=index)

        assert await document_store_1.count_documents_async() == await document_store_2.count_documents_async() == 0

        doc_1 = Document(content="Hello world")
        await document_store_1.write_documents_async([doc_1])
        assert await document_store_1.count_documents_async() == await document_store_2.count_documents_async() == 1

        assert (
            await document_store_1.filter_documents_async()
            == await document_store_2.filter_documents_async()
            == [doc_1]
        )

        doc_2 = Document(content="Hello another world")
        await document_store_2.write_documents_async([doc_2])
        assert await document_store_1.count_documents_async() == await document_store_2.count_documents_async() == 2

        assert (
            await document_store_1.filter_documents_async()
            == await document_store_2.filter_documents_async()
            == [doc_1, doc_2]
        )

        await document_store_1.delete_documents_async([doc_2.id])
        assert await document_store_1.count_documents_async() == await document_store_2.count_documents_async() == 1

        await document_store_2.delete_documents_async([doc_1.id])
        assert await document_store_1.count_documents_async() == await document_store_2.count_documents_async() == 0
