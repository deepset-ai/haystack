# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os

import pytest

from haystack import Pipeline
from haystack.components.converters.pypdf import PyPDFToDocument
from haystack.components.joiners import DocumentJoiner
from haystack.components.preprocessors.document_splitter import DocumentSplitter
from haystack.components.writers.document_writer import DocumentWriter
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.extractors.image.llm_document_content_extractor import LLMDocumentContentExtractor
from haystack.components.generators.chat.openai import OpenAIChatGenerator
from haystack.components.routers.document_length_router import DocumentLengthRouter


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY", None),
    reason="Export an env var called OPENAI_API_KEY containing the OpenAI API key to run this test.",
)
def test_pdf_content_extraction_pipeline():
    """
    Test a pipeline that processes PDFs with the following steps:
    1. Convert PDFs to documents
    2. Split documents by page
    3. Route documents by length (short vs long)
    4. Extract content from short documents using LLM
    5. Join documents back together
    6. Write to document store
    """
    document_store = InMemoryDocumentStore()

    pdf_converter = PyPDFToDocument(store_full_path=True)
    pdf_splitter = DocumentSplitter(split_by="page", split_length=1, skip_empty_documents=False)
    doc_length_router = DocumentLengthRouter(threshold=10)
    content_extractor = LLMDocumentContentExtractor(chat_generator=OpenAIChatGenerator(model="gpt-4o-mini"))
    final_doc_joiner = DocumentJoiner(sort_by_score=False)
    document_writer = DocumentWriter(document_store=document_store)

    # Create and configure pipeline
    indexing_pipe = Pipeline()
    indexing_pipe.add_component("pdf_converter", pdf_converter)
    indexing_pipe.add_component("pdf_splitter", pdf_splitter)
    indexing_pipe.add_component("doc_length_router", doc_length_router)
    indexing_pipe.add_component("content_extractor", content_extractor)
    indexing_pipe.add_component("final_doc_joiner", final_doc_joiner)
    indexing_pipe.add_component("document_writer", document_writer)

    # Connect components
    indexing_pipe.connect("pdf_converter.documents", "pdf_splitter.documents")
    indexing_pipe.connect("pdf_splitter.documents", "doc_length_router.documents")
    # The short PDF pages will be enriched/captioned
    indexing_pipe.connect("doc_length_router.short_documents", "content_extractor.documents")
    indexing_pipe.connect("doc_length_router.long_documents", "final_doc_joiner.documents")
    indexing_pipe.connect("content_extractor.documents", "final_doc_joiner.documents")
    indexing_pipe.connect("final_doc_joiner.documents", "document_writer.documents")

    # Test with both text-searchable and non-text-searchable PDFs
    test_files = [
        "test/test_files/pdf/sample_pdf_1.pdf",  # a PDF with 4 pages
        "test/test_files/pdf/non_text_searchable.pdf",  # a non-text searchable PDF with 1 page
    ]

    # Run the indexing pipeline
    indexing_result = indexing_pipe.run(data={"sources": test_files})

    assert indexing_result is not None
    assert "document_writer" in indexing_result

    indexed_documents = document_store.filter_documents()

    # We expect documents from both PDFs
    # sample_pdf_1.pdf has 4 pages, non_text_searchable.pdf has 1 page
    assert len(indexed_documents) == 5

    file_paths = {doc.meta["file_path"] for doc in indexed_documents}
    assert file_paths == set(test_files)

    for doc in indexed_documents:
        assert hasattr(doc, "content")
        assert hasattr(doc, "meta")
        assert "file_path" in doc.meta
        assert "page_number" in doc.meta

    for doc in indexed_documents:
        assert isinstance(doc.meta["page_number"], int)
        assert doc.meta["page_number"] >= 1
