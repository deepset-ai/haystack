# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
from unittest.mock import Mock, patch

import pytest

from haystack import Document, Pipeline
from haystack.components.converters.image.document_to_image import DocumentToImageContent
from haystack.components.extractors.image import LLMDocumentContentExtractor
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.writers import DocumentWriter
from haystack.core.serialization import component_to_dict
from haystack.dataclasses import TextContent
from haystack.dataclasses.chat_message import ChatMessage, ImageContent
from haystack.document_stores.in_memory import InMemoryDocumentStore


class TestLLMDocumentContentExtractor:
    def test_init(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")
        chat_generator = OpenAIChatGenerator(model="gpt-4o-mini", generation_kwargs={"temperature": 0.5})

        extractor = LLMDocumentContentExtractor(
            chat_generator=chat_generator,
            prompt="Extract content from this image",
            file_path_meta_field="file_path",
            root_path="/test/path",
            detail="high",
            size=(800, 600),
            raise_on_failure=True,
            max_workers=5,
        )

        assert isinstance(extractor._chat_generator, OpenAIChatGenerator)
        assert extractor._chat_generator.model == "gpt-4o-mini"
        assert extractor._chat_generator.generation_kwargs == {"temperature": 0.5}
        assert extractor.prompt == "Extract content from this image"
        assert extractor.file_path_meta_field == "file_path"
        assert extractor.root_path == "/test/path"
        assert extractor.detail == "high"
        assert extractor.size == (800, 600)
        assert extractor.raise_on_failure is True
        assert extractor.max_workers == 5

    def test_init_with_defaults(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")
        chat_generator = OpenAIChatGenerator(model="gpt-4o-mini")
        extractor = LLMDocumentContentExtractor(chat_generator=chat_generator)
        assert extractor.prompt.startswith("\nYou are part of an information extraction pipeline")
        assert extractor.file_path_meta_field == "file_path"
        assert extractor.root_path == ""
        assert extractor.detail is None
        assert extractor.size is None
        assert extractor.raise_on_failure is False
        assert extractor.max_workers == 3

    def test_init_with_variables_in_prompt(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")
        chat_generator = OpenAIChatGenerator(model="gpt-4o-mini")

        with pytest.raises(ValueError, match="The prompt must not have any variables"):
            LLMDocumentContentExtractor(
                chat_generator=chat_generator, prompt="Extract content from {{document.content}}"
            )

    def test_init_fails_without_chat_generator(self):
        with pytest.raises(TypeError):
            LLMDocumentContentExtractor()

    def test_to_dict_openai(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")
        chat_generator = OpenAIChatGenerator(model="gpt-4o-mini", generation_kwargs={"temperature": 0.5})

        extractor = LLMDocumentContentExtractor(
            chat_generator=chat_generator,
            prompt="Custom extraction prompt",
            file_path_meta_field="custom_path",
            root_path="/custom/root",
            detail="low",
            size=(1024, 768),
            raise_on_failure=True,
            max_workers=4,
        )

        extractor_dict = extractor.to_dict()

        assert extractor_dict == {
            "type": "haystack.components.extractors.image.llm_document_content_extractor.LLMDocumentContentExtractor",
            "init_parameters": {
                "chat_generator": component_to_dict(chat_generator, "chat_generator"),
                "prompt": "Custom extraction prompt",
                "file_path_meta_field": "custom_path",
                "root_path": "/custom/root",
                "detail": "low",
                "size": (1024, 768),
                "raise_on_failure": True,
                "max_workers": 4,
            },
        }

    def test_from_dict_openai(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")
        chat_generator = OpenAIChatGenerator(model="gpt-4o-mini", generation_kwargs={"temperature": 0.5})

        extractor_dict = {
            "type": "haystack.components.extractors.image.llm_document_content_extractor.LLMDocumentContentExtractor",
            "init_parameters": {
                "chat_generator": component_to_dict(chat_generator, "chat_generator"),
                "prompt": "Custom extraction prompt",
                "file_path_meta_field": "custom_path",
                "root_path": "/custom/root",
                "detail": "low",
                "size": (1024, 768),
                "raise_on_failure": True,
                "max_workers": 4,
            },
        }

        extractor = LLMDocumentContentExtractor.from_dict(extractor_dict)

        assert extractor.prompt == "Custom extraction prompt"
        assert extractor.file_path_meta_field == "custom_path"
        assert extractor.root_path == "/custom/root"
        assert extractor.detail == "low"
        assert extractor.size == (1024, 768)
        assert extractor.raise_on_failure is True
        assert extractor.max_workers == 4
        assert component_to_dict(extractor._chat_generator, "name") == component_to_dict(chat_generator, "name")

    def test_warm_up_with_chat_generator(self, monkeypatch):
        mock_chat_generator = Mock()
        mock_chat_generator.warm_up = Mock()
        extractor = LLMDocumentContentExtractor(chat_generator=mock_chat_generator)
        mock_chat_generator.warm_up.assert_not_called()
        extractor.warm_up()
        mock_chat_generator.warm_up.assert_called_once()

    def test_warm_up_without_warm_up_method(self, monkeypatch):
        mock_chat_generator = Mock()
        extractor = LLMDocumentContentExtractor(chat_generator=mock_chat_generator)
        extractor.warm_up()
        assert extractor._is_warmed_up is True

    def test_run_no_documents(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")
        chat_generator = OpenAIChatGenerator()
        extractor = LLMDocumentContentExtractor(chat_generator=chat_generator)
        result = extractor.run(documents=[])
        assert result["documents"] == []
        assert result["failed_documents"] == []

    @patch.object(DocumentToImageContent, "run")
    def test_run_with_failed_image_conversion(self, mock_doc_to_image_run, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")
        mock_chat_generator = Mock(spec=OpenAIChatGenerator)

        extractor = LLMDocumentContentExtractor(chat_generator=mock_chat_generator)

        # Mock DocumentToImageContent to return None (failed conversion)
        mock_doc_to_image_run.return_value = {"image_contents": [None]}

        doc = Document(content="", meta={"file_path": "/path/to/image.pdf"})
        docs = [doc]
        result = extractor.run(documents=docs)

        # Document should be in failed_documents
        assert len(result["documents"]) == 0
        assert len(result["failed_documents"]) == 1

        failed_doc = result["failed_documents"][0]
        assert failed_doc.id == doc.id
        assert "content_extraction_error" in failed_doc.meta
        assert failed_doc.meta["content_extraction_error"] == "Document has no content, skipping LLM call."

        # Ensure no attempt was made to call the LLM
        mock_chat_generator.run.assert_not_called()

    @patch.object(DocumentToImageContent, "run")
    def test_run_with_llm_success(self, mock_doc_to_image_run):
        # Mock successful LLM response
        mock_chat_generator = Mock(spec=OpenAIChatGenerator)
        mock_chat_generator.run.return_value = {
            "replies": [ChatMessage.from_assistant(text="Extracted content from the image")]
        }
        # Mock DocumentToImageContent to return valid image content
        mock_doc_to_image_run.return_value = {
            "image_contents": [ImageContent.from_file_path("./test/test_files/images/apple.jpg")]
        }

        extractor = LLMDocumentContentExtractor(chat_generator=mock_chat_generator)

        docs = [Document(content="", meta={"file_path": "/path/to/image.pdf"})]
        result = extractor.run(documents=docs)

        # Document should be successfully processed
        assert len(result["documents"]) == 1
        assert len(result["failed_documents"]) == 0

        processed_doc = result["documents"][0]
        assert processed_doc.id == docs[0].id
        assert processed_doc.content == "Extracted content from the image"
        assert "content_extraction_error" not in processed_doc.meta

        # Verify LLM was called
        mock_chat_generator.run.assert_called_once()

    @patch.object(DocumentToImageContent, "run")
    def test_run_with_llm_failure_raise_on_failure_false(self, mock_doc_to_image_run, caplog):
        # Mock LLM failure
        mock_chat_generator = Mock(spec=OpenAIChatGenerator)
        mock_chat_generator.run.side_effect = Exception("LLM API error")
        extractor = LLMDocumentContentExtractor(chat_generator=mock_chat_generator, raise_on_failure=False)

        # Mock DocumentToImageContent to return valid image content
        mock_doc_to_image_run.return_value = {
            "image_contents": [ImageContent.from_file_path("./test/test_files/images/apple.jpg")]
        }

        docs = [Document(content="", meta={"file_path": "/path/to/image.pdf"})]
        result = extractor.run(documents=docs)

        # Document should be in failed_documents
        assert len(result["documents"]) == 0
        assert len(result["failed_documents"]) == 1

        failed_doc = result["failed_documents"][0]
        assert failed_doc.id == docs[0].id
        assert "content_extraction_error" in failed_doc.meta
        assert "LLM failed with exception: LLM API error" in failed_doc.meta["content_extraction_error"]

        # Check that error was logged
        assert "LLM" in caplog.text
        assert "execution failed" in caplog.text

    @patch.object(DocumentToImageContent, "run")
    def test_run_with_llm_failure_raise_on_failure_true(self, mock_doc_to_image_run):
        # Mock LLM failure
        mock_chat_generator = Mock(spec=OpenAIChatGenerator)
        mock_chat_generator.run.side_effect = Exception("LLM API error")
        extractor = LLMDocumentContentExtractor(chat_generator=mock_chat_generator, raise_on_failure=True)
        # Mock DocumentToImageContent to return valid image content
        mock_doc_to_image_run.return_value = {
            "image_contents": [ImageContent.from_file_path("./test/test_files/images/apple.jpg")]
        }
        with pytest.raises(Exception, match="LLM API error"):
            extractor.run(documents=[Document(content="", meta={"file_path": "/path/to/image.pdf"})])

    @patch.object(DocumentToImageContent, "run")
    def test_run_removes_content_extraction_error_from_previous_runs(self, mock_doc_to_image_run):
        # Mock successful LLM response
        mock_chat_generator = Mock(spec=OpenAIChatGenerator)
        mock_chat_generator.run.return_value = {
            "replies": [ChatMessage.from_assistant(text="Successfully extracted content")]
        }
        # Mock DocumentToImageContent to return valid image content
        mock_doc_to_image_run.return_value = {
            "image_contents": [ImageContent.from_file_path("./test/test_files/images/apple.jpg")]
        }

        extractor = LLMDocumentContentExtractor(chat_generator=mock_chat_generator)

        # Document with previous extraction error
        docs = [
            Document(
                content="",
                meta={
                    "file_path": "/path/to/image.pdf",
                    "content_extraction_error": "Previous error",
                    "other_meta": "should_remain",
                },
            )
        ]

        result = extractor.run(documents=docs)

        # Document should be successfully processed
        assert len(result["documents"]) == 1
        assert len(result["failed_documents"]) == 0

        processed_doc = result["documents"][0]
        assert processed_doc.content == "Successfully extracted content"
        assert "content_extraction_error" not in processed_doc.meta
        assert processed_doc.meta["other_meta"] == "should_remain"

    @patch.object(DocumentToImageContent, "run")
    def test_run_with_mixed_success_and_failure(self, mock_doc_to_image_run):
        # Mock successful LLM response for first call, failure for second
        mock_chat_generator = Mock(spec=OpenAIChatGenerator)
        mock_chat_generator.run.side_effect = [
            {"replies": [ChatMessage.from_assistant(text="Successfully extracted content")]},  # Success
            Exception("LLM API error"),  # Failure
        ]

        extractor = LLMDocumentContentExtractor(chat_generator=mock_chat_generator, raise_on_failure=False)

        # Mock DocumentToImageContent - first succeeds, second fails
        mock_doc_to_image_run.return_value = {
            "image_contents": [ImageContent.from_file_path("./test/test_files/images/apple.jpg"), None]
        }

        doc1 = Document(content="", meta={"file_path": "./test/test_files/images/apple.jpg"})
        doc2 = Document(content="", meta={"file_path": "/path/to/image.jpg"})
        docs = [doc1, doc2]

        result = extractor.run(documents=docs)

        # One document should succeed, one should fail
        assert len(result["documents"]) == 1
        assert len(result["failed_documents"]) == 1

        successful_doc = result["documents"][0]
        assert successful_doc.id == doc1.id
        assert successful_doc.content == "Successfully extracted content"

        failed_doc = result["failed_documents"][0]
        assert failed_doc.id == doc2.id
        assert "content_extraction_error" in failed_doc.meta

    def test_run_on_thread_with_none_prompt(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")
        extractor = LLMDocumentContentExtractor(chat_generator=OpenAIChatGenerator())
        result = extractor._run_on_thread(None)
        assert "error" in result
        assert result["error"] == "Document has no content, skipping LLM call."

    def test_run_on_thread_with_valid_prompt(self):
        mock_chat_generator = Mock(spec=OpenAIChatGenerator)
        mock_chat_generator.run.return_value = {"replies": [ChatMessage.from_assistant(text="Extracted content")]}

        extractor = LLMDocumentContentExtractor(chat_generator=mock_chat_generator)

        prompt = ChatMessage.from_user(
            content_parts=[
                TextContent(text="Instructions to extract content"),
                ImageContent.from_file_path("./test/test_files/images/apple.jpg"),
            ]
        )

        result = extractor._run_on_thread(prompt)
        assert "error" not in result
        assert result == {"replies": [ChatMessage.from_assistant(text="Extracted content")]}
        mock_chat_generator.run.assert_called_once_with(messages=[prompt])

    @pytest.mark.integration
    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY", None),
        reason="Export an env var called OPENAI_API_KEY containing the OpenAI API key to run this test.",
    )
    def test_live_run(self):
        docs = [Document(content="", meta={"file_path": "./test/test_files/images/apple.jpg"})]
        doc_store = InMemoryDocumentStore()
        extractor = LLMDocumentContentExtractor(chat_generator=OpenAIChatGenerator(model="gpt-4.1-mini"))
        writer = DocumentWriter(document_store=doc_store)
        pipeline = Pipeline()
        pipeline.add_component("extractor", extractor)
        pipeline.add_component("doc_writer", writer)
        pipeline.connect("extractor.documents", "doc_writer.documents")
        pipeline.run(data={"documents": docs})

        doc_store_docs = doc_store.filter_documents()
        assert len(doc_store_docs) >= 1
        assert len(doc_store_docs[0].content) > 0
