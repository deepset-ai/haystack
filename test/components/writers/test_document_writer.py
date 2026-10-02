# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock, Mock

import pytest

from haystack import Document
from haystack.components.writers.document_writer import DocumentWriter
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.document_stores.types import DuplicatePolicy
from haystack.testing.factory import document_store_class


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

    def test_from_dict_without_policy(self):
        data = {
            "type": "haystack.components.writers.document_writer.DocumentWriter",
            "init_parameters": {
                "document_store": {
                    "type": "haystack.document_stores.in_memory.document_store.InMemoryDocumentStore",
                    "init_parameters": {},
                }
            },
        }
        component = DocumentWriter.from_dict(data)
        assert isinstance(component.document_store, InMemoryDocumentStore)
        assert component.policy == DuplicatePolicy.NONE

    def test_from_dict_without_docstore(self):
        data = {"type": "haystack.components.writers.document_writer.DocumentWriter", "init_parameters": {}}
        with pytest.raises(TypeError, match="missing 1 required positional argument: 'document_store'"):
            DocumentWriter.from_dict(data)

    def test_from_dict_nonexisting_docstore(self):
        # Use a type whose module passes the deserialization allowlist (haystack.*) but cannot be
        # resolved, so we still exercise the "import failed" code path rather than the allowlist gate.
        data = {
            "type": "haystack.components.writers.document_writer.DocumentWriter",
            "init_parameters": {
                "document_store": {"type": "haystack.does.not.exist.DocumentStore", "init_parameters": {}}
            },
        }
        with pytest.raises(
            ImportError, match=r"Failed to deserialize 'document_store':.*haystack\.does\.not\.exist\.DocumentStore"
        ):
            DocumentWriter.from_dict(data)

    def test_run(self, in_memory_doc_store):
        writer = DocumentWriter(in_memory_doc_store)
        documents = [
            Document(content="This is the text of a document."),
            Document(content="This is the text of another document."),
        ]

        result = writer.run(documents=documents)
        assert result["documents_written"] == 2

    def test_run_skip_policy(self, in_memory_doc_store):
        writer = DocumentWriter(in_memory_doc_store, policy=DuplicatePolicy.SKIP)
        documents = [
            Document(content="This is the text of a document."),
            Document(content="This is the text of another document."),
        ]

        result = writer.run(documents=documents)
        assert result["documents_written"] == 2

        result = writer.run(documents=documents)
        assert result["documents_written"] == 0

    def test_close(self):
        closable_document_store = Mock(spec=["close"])
        writer = DocumentWriter(document_store=closable_document_store)
        writer.close()
        closable_document_store.close.assert_called_once_with()

        nonclosable_document_store = Mock(spec=[])
        writer = DocumentWriter(document_store=nonclosable_document_store)
        writer.close()
        assert nonclosable_document_store.mock_calls == []

    @pytest.mark.asyncio
    async def test_run_async_invalid_docstore(self):
        mocked_docstore_class = document_store_class("MockedDocumentStore")

        writer = DocumentWriter(mocked_docstore_class())
        documents = [
            Document(content="This is the text of a document."),
            Document(content="This is the text of another document."),
        ]

        with pytest.raises(TypeError, match="does not provide async support"):
            await writer.run_async(documents=documents)

    @pytest.mark.asyncio
    async def test_run_async(self, in_memory_doc_store):
        writer = DocumentWriter(in_memory_doc_store)
        documents = [
            Document(content="This is the text of a document."),
            Document(content="This is the text of another document."),
        ]

        result = await writer.run_async(documents=documents)
        assert result["documents_written"] == 2

    @pytest.mark.asyncio
    async def test_run_async_skip_policy(self, in_memory_doc_store):
        writer = DocumentWriter(in_memory_doc_store, policy=DuplicatePolicy.SKIP)
        documents = [
            Document(content="This is the text of a document."),
            Document(content="This is the text of another document."),
        ]

        result = await writer.run_async(documents=documents)
        assert result["documents_written"] == 2

        result = await writer.run_async(documents=documents)
        assert result["documents_written"] == 0

    @pytest.mark.asyncio
    async def test_close_async(self):
        closable_document_store = Mock(spec=["close_async"])
        closable_document_store.close_async = AsyncMock()
        writer = DocumentWriter(document_store=closable_document_store)
        await writer.close_async()
        closable_document_store.close_async.assert_awaited_once_with()

        nonclosable_document_store = Mock(spec=[])
        writer = DocumentWriter(document_store=nonclosable_document_store)
        await writer.close_async()
        assert nonclosable_document_store.mock_calls == []
