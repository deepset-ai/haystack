# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import random
import re

import pytest

from haystack import Document, Pipeline
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.retrievers import InMemoryBM25Retriever
from haystack.components.retrievers.sentence_window_retriever import SentenceWindowRetriever
from haystack.core.pipeline.async_pipeline import AsyncPipeline
from haystack.document_stores.in_memory import InMemoryDocumentStore


class TestSentenceWindowRetrieverAsync:
    async def test_document_without_split_id(self):
        docs = [
            Document(content="This is a text with some words. There is a ", meta={"id": "doc_0"}),
            Document(content="some words. There is a second sentence. And there is ", meta={"id": "doc_1"}),
        ]
        with pytest.raises(ValueError, match="The retrieved documents must have 'split_id_test' in their metadata."):
            retriever = SentenceWindowRetriever(
                document_store=InMemoryDocumentStore(), window_size=3, split_id_meta_field="split_id_test"
            )
            await retriever.run_async(retrieved_documents=docs)

    @pytest.mark.asyncio
    async def test_document_without_source_id(self):
        docs = [
            Document(content="This is a text with some words. There is a ", meta={"id": "doc_0", "split_id": 0}),
            Document(
                content="some words. There is a second sentence. And there is ",
                meta={"id": "doc_1", "split_id": 1, "source_id_test": "source1"},
            ),
        ]
        with pytest.raises(ValueError, match="The retrieved documents must have 'source_id_test' in their metadata."):
            retriever = SentenceWindowRetriever(
                document_store=InMemoryDocumentStore(), window_size=3, source_id_meta_field="source_id_test"
            )
            await retriever.run_async(retrieved_documents=docs)

    @pytest.mark.asyncio
    async def test_document_without_all_source_ids(self):
        docs = [
            Document(
                content="These are words from the first section",
                meta={"id": "doc_1", "split_id": 0, "section_id": "section1"},
            ),
            Document(
                content="These are words from the second section, but missing section_id",
                meta={"id": "doc_0", "split_id": 0},
            ),
        ]
        with pytest.raises(
            ValueError, match=re.escape("The retrieved documents must have '['id', 'section_id']' in their metadata.")
        ):
            retriever = SentenceWindowRetriever(
                document_store=InMemoryDocumentStore(), window_size=3, source_id_meta_field=["id", "section_id"]
            )
            await retriever.run_async(retrieved_documents=docs)

    @pytest.mark.asyncio
    async def test_run_async_invalid_window_size(self):
        docs = [Document(content="This is a text with some words. There is a ", meta={"id": "doc_0", "split_id": 0})]
        with pytest.raises(ValueError):
            retriever = SentenceWindowRetriever(document_store=InMemoryDocumentStore(), window_size=0)
            await retriever.run_async(retrieved_documents=docs)

    @pytest.mark.asyncio
    async def test_constructor_parameter_does_not_change(self):
        retriever = SentenceWindowRetriever(InMemoryDocumentStore(), window_size=5)
        assert retriever.window_size == 5

        doc = {
            "id": "doc_0",
            "content": "This is a text with some words. There is a ",
            "source_id": "c5d7c632affc486d0cfe7b3c0f4dc1d3896ea720da2b538d6d10b104a3df5f99",
            "page_number": 1,
            "split_id": 0,
            "split_idx_start": 0,
            "_split_overlap": [{"doc_id": "doc_1", "range": (0, 23)}],
        }

        await retriever.run_async(retrieved_documents=[Document.from_dict(doc)], window_size=1)
        assert retriever.window_size == 5

    @pytest.mark.asyncio
    async def test_context_documents_returned_are_ordered_by_split_idx_start(self):
        docs = []
        accumulated_length = 0
        for sent in range(10):
            content = f"Sentence {sent}."
            docs.append(
                Document(
                    content=content,
                    meta={
                        "id": f"doc_{sent}",
                        "split_idx_start": accumulated_length,
                        "source_id": "source1",
                        "split_id": sent,
                    },
                )
            )
            accumulated_length += len(content)

        random.shuffle(docs)

        doc_store = InMemoryDocumentStore()
        doc_store.write_documents(docs)
        retriever = SentenceWindowRetriever(document_store=doc_store, window_size=3)

        # run the retriever with a document whose content = "Sentence 4."
        result = await retriever.run_async(retrieved_documents=[doc for doc in docs if doc.content == "Sentence 4."])

        # assert that the context documents are in the correct order
        assert len(result["context_documents"]) == 7
        assert [doc.meta["split_idx_start"] for doc in result["context_documents"]] == [11, 22, 33, 44, 55, 66, 77]

    @pytest.mark.asyncio
    async def test_run_async_custom_fields(self):
        docs = []
        accumulated_length = 0
        for sent in range(10):
            content = f"Sentence {sent}."
            docs.append(
                Document(
                    content=content,
                    meta={
                        "id": f"doc_{sent}",
                        # Missing split_idx_start
                        "source_id_test": "source1",
                        "split_id_test": sent,
                    },
                )
            )
            accumulated_length += len(content)

        random.shuffle(docs)

        doc_store = InMemoryDocumentStore()
        doc_store.write_documents(docs)
        retriever = SentenceWindowRetriever(
            document_store=doc_store,
            window_size=3,
            source_id_meta_field="source_id_test",
            split_id_meta_field="split_id_test",
        )

        # run the retriever with a document whose content = "Sentence 4."
        result = await retriever.run_async(retrieved_documents=[doc for doc in docs if doc.content == "Sentence 4."])
        assert len(result["context_documents"]) == 7

    @pytest.mark.asyncio
    async def test_run_async_with_multiple_source_ids(self):
        docs = [
            Document(content="This is the first chunk.", meta={"section": "1", "split_id": 0, "source_id": "source1"}),
            Document(content="This is the second chunk.", meta={"section": "1", "split_id": 1, "source_id": "source1"}),
            Document(content="This is the third chunk.", meta={"section": "1", "split_id": 2, "source_id": "source1"}),
            Document(
                content="This is a chunk from section 2.", meta={"section": "2", "split_id": 3, "source_id": "source1"}
            ),
        ]
        doc_store = InMemoryDocumentStore()
        doc_store.write_documents(docs)

        retriever = SentenceWindowRetriever(
            document_store=doc_store, window_size=5, source_id_meta_field=["section", "source_id"]
        )
        result = await retriever.run_async(
            retrieved_documents=[
                Document(
                    content="This is the second chunk.", meta={"section": "1", "split_id": 1, "source_id": "source1"}
                )
            ]
        )

        assert len(result["context_windows"]) == 1
        assert len(result["context_documents"]) == 3
        assert all(doc.meta["section"] == "1" for doc in result["context_documents"])

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_run_async_with_pipeline(self):
        splitter = DocumentSplitter(split_length=1, split_overlap=0, split_by="period")
        text = (
            "This is a text with some words. There is a second sentence. And there is also a third sentence. "
            "It also contains a fourth sentence. And a fifth sentence. And a sixth sentence. And a seventh sentence"
        )
        doc = Document(content=text)
        docs = splitter.run([doc])
        doc_store = InMemoryDocumentStore()
        doc_store.write_documents(docs["documents"])

        pipe = AsyncPipeline()
        pipe.add_component("bm25_retriever", InMemoryBM25Retriever(doc_store, top_k=1))
        pipe.add_component(
            "sentence_window_retriever", SentenceWindowRetriever(document_store=doc_store, window_size=2)
        )
        pipe.connect("bm25_retriever", "sentence_window_retriever")
        result = await pipe.run_async({"bm25_retriever": {"query": "third"}})

        assert result["sentence_window_retriever"]["context_windows"] == [
            "This is a text with some words. There is a second sentence. And there is also a third sentence. "
            "It also contains a fourth sentence. And a fifth sentence."
        ]
        assert len(result["sentence_window_retriever"]["context_documents"]) == 5

        result = await pipe.run_async(
            {"bm25_retriever": {"query": "third"}, "sentence_window_retriever": {"window_size": 1}}
        )
        assert result["sentence_window_retriever"]["context_windows"] == [
            " There is a second sentence. And there is also a third sentence. It also contains a fourth sentence."
        ]
        assert len(result["sentence_window_retriever"]["context_documents"]) == 3

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_serialization_deserialization_in_pipeline(self):
        doc_store = InMemoryDocumentStore()
        pipe = AsyncPipeline()
        pipe.add_component("bm25_retriever", InMemoryBM25Retriever(doc_store, top_k=1))
        pipe.add_component(
            "sentence_window_retriever", SentenceWindowRetriever(document_store=doc_store, window_size=2)
        )
        pipe.connect("bm25_retriever", "sentence_window_retriever")

        serialized = pipe.to_dict()
        deserialized = AsyncPipeline.from_dict(serialized)

        assert deserialized == pipe
