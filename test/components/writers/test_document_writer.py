# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from haystack import Document, DeserializationError
from haystack.testing.factory import document_store_class
from haystack.components.writers.document_writer import DocumentWriter
from haystack.document_stores.types import DuplicatePolicy
from haystack.document_stores.in_memory import InMemoryDocumentStore


@pytest.fixture
def document_store():
    """
    Create a fresh InMemoryDocumentStore for each test with proper cleanup.

    Using a fixture ensures the ThreadPoolExecutor is shut down immediately after test completion rather than
    during (unpredictable) garbage collection, which can make the CI hang.
    """
    store = InMemoryDocumentStore()
    yield store
    store.shutdown()


class TestDocumentWriter:
    def test_to_dict(self):
        mocked_docstore_class = document_store_class("MockedDocumentStore")
        component = DocumentWriter(document_store=mocked_docstore_class())
        data = component.to_dict()
        assert data == {
            "type": "haystack.components.writers.document_writer.DocumentWriter",
            "init_parameters": {
                "document_store": {"type": "haystack.testing.factory.MockedDocumentStore", "init_parameters": {}},
                "policy": "NONE",
            },
        }

    def test_to_dict_with_custom_init_parameters(self):
        mocked_docstore_class = document_store_class("MockedDocumentStore")
        component = DocumentWriter(document_store=mocked_docstore_class(), policy=DuplicatePolicy.SKIP)
        data = component.to_dict()
        assert data == {
            "type": "haystack.components.writers.document_writer.DocumentWriter",
            "init_parameters": {
                "document_store": {"type": "haystack.testing.factory.MockedDocumentStore", "init_parameters": {}},
                "policy": "SKIP",
            },
        }

    def test_from_dict(self):
        data = {
            "type": "haystack.components.writers.document_writer.DocumentWriter",
            "init_parameters": {
                "document_store": {
                    "type": "haystack.document_stores.in_memory.document_store.InMemoryDocumentStore",
                    "init_parameters": {},
                },
                "policy": "SKIP",
            },
        }
        component = DocumentWriter.from_dict(data)
        assert isinstance(component.document_store, InMemoryDocumentStore)
        assert component.policy == DuplicatePolicy.SKIP

    def test_from_dict_without_docstore(self):
        data = {"type": "DocumentWriter", "init_parameters": {}}
        with pytest.raises(DeserializationError, match="Missing 'document_store' in serialization data"):
            DocumentWriter.from_dict(data)

    def test_from_dict_without_docstore_type(self):
        data = {"type": "DocumentWriter", "init_parameters": {"document_store": {"init_parameters": {}}}}
        with pytest.raises(DeserializationError):
            DocumentWriter.from_dict(data)

    def test_from_dict_nonexisting_docstore(self):
        data = {
            "type": "DocumentWriter",
            "init_parameters": {"document_store": {"type": "Nonexisting.DocumentStore", "init_parameters": {}}},
        }
        with pytest.raises(DeserializationError):
            DocumentWriter.from_dict(data)

    def test_run(self, document_store):
        writer = DocumentWriter(document_store)
        documents = [
            Document(content="This is the text of a document."),
            Document(content="This is the text of another document."),
        ]

        result = writer.run(documents=documents)
        assert result["documents_written"] == 2

    def test_run_skip_policy(self, document_store):
        writer = DocumentWriter(document_store, policy=DuplicatePolicy.SKIP)
        documents = [
            Document(content="This is the text of a document."),
            Document(content="This is the text of another document."),
        ]

        result = writer.run(documents=documents)
        assert result["documents_written"] == 2

        result = writer.run(documents=documents)
        assert result["documents_written"] == 0

    @pytest.mark.asyncio
    async def test_run_async_invalid_docstore(self):
        document_store = document_store_class("MockedDocumentStore")

        writer = DocumentWriter(document_store)
        documents = [
            Document(content="This is the text of a document."),
            Document(content="This is the text of another document."),
        ]

        with pytest.raises(TypeError, match="does not provide async support"):
            await writer.run_async(documents=documents)

    @pytest.mark.asyncio
    async def test_run_async(self, document_store):
        writer = DocumentWriter(document_store)
        documents = [
            Document(content="This is the text of a document."),
            Document(content="This is the text of another document."),
        ]

        result = await writer.run_async(documents=documents)
        assert result["documents_written"] == 2

    @pytest.mark.asyncio
    async def test_run_async_skip_policy(self, document_store):
        writer = DocumentWriter(document_store, policy=DuplicatePolicy.SKIP)
        documents = [
            Document(content="This is the text of a document."),
            Document(content="This is the text of another document."),
        ]

        result = await writer.run_async(documents=documents)
        assert result["documents_written"] == 2

        result = await writer.run_async(documents=documents)
        assert result["documents_written"] == 0
