# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
from unittest.mock import Mock, patch

import pytest

from haystack import Document, Pipeline
from haystack.components.converters.image.document_to_image import DocumentToImageContent
from haystack.components.extractors.image import LLMDocumentContentExtractor
from haystack.components.extractors.image.llm_document_content_extractor import (
    DEFAULT_METADATA_PROMPT_TEMPLATE,
    DOCUMENT_CONTENT_KEY,
)
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.writers import DocumentWriter
from haystack.core.serialization import component_to_dict
from haystack.dataclasses import TextContent
from haystack.dataclasses.chat_message import ChatMessage, ImageContent
from haystack.document_stores.in_memory import InMemoryDocumentStore


class TestLLMDocumentContentExtractor:
    def test_init(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")
        chat_generator = OpenAIChatGenerator(generation_kwargs={"temperature": 0.5})

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
        # Not testing specific model name, just that it's set
        assert extractor._chat_generator.model is not None
        assert extractor._chat_generator.generation_kwargs == {"temperature": 0.5}
        assert extractor.prompt == "Extract content from this image"
        assert extractor.file_path_meta_field == "file_path"
        assert extractor.root_path == "/test/path"
        assert extractor.detail == "high"
        assert extractor.size == (800, 600)
        assert extractor.raise_on_failure is True
        assert extractor.max_workers == 5
        assert extractor.extraction_mode == "content"

    def test_init_with_defaults(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")
        chat_generator = OpenAIChatGenerator()
        extractor = LLMDocumentContentExtractor(chat_generator=chat_generator)
        assert extractor.extraction_mode == "content"
        assert extractor.prompt.startswith("\nYou are part of an information extraction pipeline")
        assert extractor.metadata_prompt == DEFAULT_METADATA_PROMPT_TEMPLATE
        assert extractor.file_path_meta_field == "file_path"
        assert extractor.root_path == ""
        assert extractor.detail is None
        assert extractor.size is None
        assert extractor.raise_on_failure is False
        assert extractor.max_workers == 3

    def test_init_extraction_mode_both(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")
        chat_generator = OpenAIChatGenerator()
        extractor = LLMDocumentContentExtractor(chat_generator=chat_generator, extraction_mode="both")
        assert extractor.extraction_mode == "both"

    def test_init_expected_keys(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")
        chat_generator = OpenAIChatGenerator()
        extractor = LLMDocumentContentExtractor(
            chat_generator=chat_generator,
            extraction_mode="metadata",
            expected_keys=["title", "author", "document_type"],
        )
        assert extractor.expected_keys == ["title", "author", "document_type"]
        extractor_default = LLMDocumentContentExtractor(chat_generator=chat_generator)
        assert extractor_default.expected_keys == []

    def test_init_with_variables_in_prompt(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")
        chat_generator = OpenAIChatGenerator()

        with pytest.raises(ValueError, match="The prompt must not have any variables"):
            LLMDocumentContentExtractor(
                chat_generator=chat_generator, prompt="Extract content from {{document.content}}"
            )

    def test_init_fails_without_chat_generator(self):
        with pytest.raises(TypeError):
            LLMDocumentContentExtractor()

    def test_to_dict_openai(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")
        chat_generator = OpenAIChatGenerator(generation_kwargs={"temperature": 0.5})

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
                "extraction_mode": "content",
                "prompt": "Custom extraction prompt",
                "metadata_prompt": DEFAULT_METADATA_PROMPT_TEMPLATE,
                "expected_keys": [],
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
        chat_generator = OpenAIChatGenerator(generation_kwargs={"temperature": 0.5})

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

        assert extractor.extraction_mode == "content"
        assert extractor.prompt == "Custom extraction prompt"
        assert extractor.metadata_prompt == DEFAULT_METADATA_PROMPT_TEMPLATE
        assert extractor.file_path_meta_field == "custom_path"
        assert extractor.root_path == "/custom/root"
        assert extractor.detail == "low"
        assert extractor.size == (1024, 768)
        assert extractor.raise_on_failure is True
        assert extractor.max_workers == 4
        assert component_to_dict(extractor._chat_generator, "name") == component_to_dict(chat_generator, "name")

    def test_to_dict_and_from_dict_with_extraction_mode_both(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")
        chat_generator = OpenAIChatGenerator()
        extractor = LLMDocumentContentExtractor(chat_generator=chat_generator, extraction_mode="both")
        extractor_dict = extractor.to_dict()
        assert extractor_dict["init_parameters"]["extraction_mode"] == "both"
        assert "metadata_key" not in extractor_dict["init_parameters"]
        restored = LLMDocumentContentExtractor.from_dict(extractor_dict)
        assert restored.extraction_mode == "both"

    def test_to_dict_and_from_dict_with_expected_keys(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")
        chat_generator = OpenAIChatGenerator()
        extractor = LLMDocumentContentExtractor(
            chat_generator=chat_generator, extraction_mode="metadata", expected_keys=["title", "author"]
        )
        extractor_dict = extractor.to_dict()
        assert extractor_dict["init_parameters"]["expected_keys"] == ["title", "author"]
        restored = LLMDocumentContentExtractor.from_dict(extractor_dict)
        assert restored.expected_keys == ["title", "author"]

    def test_from_dict_backward_compatible_without_new_keys(self, monkeypatch):
        """Old serialized dicts without extraction_mode/metadata_prompt still deserialize."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")
        chat_generator = OpenAIChatGenerator(generation_kwargs={"temperature": 0.5})
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
        assert extractor.extraction_mode == "content"
        assert extractor.prompt == "Custom extraction prompt"
        assert extractor.expected_keys == []

    def test_from_dict_backward_compatible_with_metadata_key(self, monkeypatch):
        """Old serialized dicts that contain metadata_key still deserialize; metadata_key is ignored."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")
        chat_generator = OpenAIChatGenerator(generation_kwargs={"temperature": 0.5})
        extractor_dict = {
            "type": "haystack.components.extractors.image.llm_document_content_extractor.LLMDocumentContentExtractor",
            "init_parameters": {
                "chat_generator": component_to_dict(chat_generator, "chat_generator"),
                "prompt": "Custom extraction prompt",
                "metadata_key": "legacy_key",
                "file_path_meta_field": "custom_path",
                "raise_on_failure": False,
                "max_workers": 4,
            },
        }
        extractor = LLMDocumentContentExtractor.from_dict(extractor_dict)
        assert extractor.prompt == "Custom extraction prompt"
        assert not hasattr(extractor, "metadata_key")

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
        # Mock successful LLM response (JSON with document_content)
        mock_chat_generator = Mock(spec=OpenAIChatGenerator)
        mock_chat_generator.run.return_value = {
            "replies": [ChatMessage.from_assistant(text='{"document_content": "Extracted content from the image"}')]
        }
        mock_doc_to_image_run.return_value = {
            "image_contents": [ImageContent.from_file_path("./test/test_files/images/apple.jpg")]
        }

        extractor = LLMDocumentContentExtractor(chat_generator=mock_chat_generator)

        docs = [Document(content="", meta={"file_path": "/path/to/image.pdf"})]
        result = extractor.run(documents=docs)

        assert len(result["documents"]) == 1
        assert len(result["failed_documents"]) == 0
        processed_doc = result["documents"][0]
        assert processed_doc.id == docs[0].id
        assert processed_doc.content == "Extracted content from the image"
        assert "content_extraction_error" not in processed_doc.meta
        mock_chat_generator.run.assert_called_once()

    @patch.object(DocumentToImageContent, "run")
    def test_run_content_mode_invalid_json_fails(self, mock_doc_to_image_run):
        """When content mode gets invalid JSON from LLM, document goes to failed_documents."""
        mock_chat_generator = Mock(spec=OpenAIChatGenerator)
        mock_chat_generator.run.return_value = {"replies": [ChatMessage.from_assistant(text="Plain text, not JSON")]}
        mock_doc_to_image_run.return_value = {
            "image_contents": [ImageContent.from_file_path("./test/test_files/images/apple.jpg")]
        }
        extractor = LLMDocumentContentExtractor(chat_generator=mock_chat_generator)
        docs = [Document(content="", meta={"file_path": "/path/to/image.pdf"})]
        result = extractor.run(documents=docs)
        assert len(result["documents"]) == 0
        assert len(result["failed_documents"]) == 1
        failed = result["failed_documents"][0]
        assert "content_extraction_error" in failed.meta
        assert "JSON" in failed.meta["content_extraction_error"]

    @patch.object(DocumentToImageContent, "run")
    def test_run_content_mode_missing_document_content_key_fails(self, mock_doc_to_image_run):
        """When content mode gets JSON without document_content key, document goes to failed_documents."""
        mock_chat_generator = Mock(spec=OpenAIChatGenerator)
        mock_chat_generator.run.return_value = {
            "replies": [ChatMessage.from_assistant(text='{"title": "Only metadata, no document_content"}')]
        }
        mock_doc_to_image_run.return_value = {
            "image_contents": [ImageContent.from_file_path("./test/test_files/images/apple.jpg")]
        }
        extractor = LLMDocumentContentExtractor(chat_generator=mock_chat_generator)
        docs = [Document(content="", meta={"file_path": "/path/to/image.pdf"})]
        result = extractor.run(documents=docs)
        assert len(result["documents"]) == 0
        assert len(result["failed_documents"]) == 1
        failed = result["failed_documents"][0]
        assert "content_extraction_error" in failed.meta
        assert DOCUMENT_CONTENT_KEY in failed.meta["content_extraction_error"]

    @patch.object(DocumentToImageContent, "run")
    def test_run_content_mode_extra_json_keys_merged_into_metadata(self, mock_doc_to_image_run):
        """When content mode gets JSON with document_content and other keys, other keys are merged into metadata."""
        mock_chat_generator = Mock(spec=OpenAIChatGenerator)
        mock_chat_generator.run.return_value = {
            "replies": [
                ChatMessage.from_assistant(text='{"document_content": "Main text", "title": "Doc Title", "page": "1"}')
            ]
        }
        mock_doc_to_image_run.return_value = {
            "image_contents": [ImageContent.from_file_path("./test/test_files/images/apple.jpg")]
        }
        extractor = LLMDocumentContentExtractor(chat_generator=mock_chat_generator)
        docs = [Document(content="", meta={"file_path": "/path/to/image.pdf"})]
        result = extractor.run(documents=docs)
        assert len(result["documents"]) == 1
        processed = result["documents"][0]
        assert processed.content == "Main text"
        assert processed.meta["title"] == "Doc Title"
        assert processed.meta["page"] == "1"

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
        mock_chat_generator = Mock(spec=OpenAIChatGenerator)
        mock_chat_generator.run.return_value = {
            "replies": [ChatMessage.from_assistant(text='{"document_content": "Successfully extracted content"}')]
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
            {"replies": [ChatMessage.from_assistant(text='{"document_content": "Successfully extracted content"}')]},
            Exception("LLM API error"),
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

    @patch.object(DocumentToImageContent, "run")
    def test_run_extraction_mode_metadata_at_init(self, mock_doc_to_image_run):
        """When extraction_mode is 'metadata', JSON output is parsed and merged into document meta."""
        mock_chat_generator = Mock(spec=OpenAIChatGenerator)
        mock_chat_generator.run.return_value = {
            "replies": [
                ChatMessage.from_assistant(text='{"title": "Sample Doc", "author": "Test", "document_type": "report"}')
            ]
        }
        mock_doc_to_image_run.return_value = {
            "image_contents": [ImageContent.from_file_path("./test/test_files/images/apple.jpg")]
        }
        extractor = LLMDocumentContentExtractor(chat_generator=mock_chat_generator, extraction_mode="metadata")
        original_content = "Original content"
        docs = [Document(content=original_content, meta={"file_path": "/path/to/image.pdf"})]
        result = extractor.run(documents=docs)
        assert len(result["documents"]) == 1
        assert len(result["failed_documents"]) == 0
        processed = result["documents"][0]
        assert processed.content == original_content
        assert processed.meta["title"] == "Sample Doc"
        assert processed.meta["author"] == "Test"
        assert processed.meta["document_type"] == "report"
        assert "metadata_extraction_error" not in processed.meta
        assert "metadata_extraction_response" not in processed.meta

    @patch.object(DocumentToImageContent, "run")
    def test_run_extraction_mode_metadata_invalid_json_fails(self, mock_doc_to_image_run):
        """
        When extraction_mode is 'metadata' and LLM returns invalid JSON, document goes to failed_documents with
        metadata_extraction_response.
        """
        raw_reply = "title: Foo\nauthor: Bar"
        mock_reply = ChatMessage.from_assistant(text=raw_reply)
        mock_chat_generator = Mock(spec=OpenAIChatGenerator)
        mock_chat_generator.run.return_value = {"replies": [mock_reply]}
        mock_doc_to_image_run.return_value = {
            "image_contents": [ImageContent.from_file_path("./test/test_files/images/apple.jpg")]
        }
        extractor = LLMDocumentContentExtractor(chat_generator=mock_chat_generator, extraction_mode="metadata")
        docs = [Document(content="", meta={"file_path": "/path/to/image.pdf"})]
        result = extractor.run(documents=docs)
        assert len(result["documents"]) == 0
        assert len(result["failed_documents"]) == 1
        failed = result["failed_documents"][0]
        assert "metadata_extraction_error" in failed.meta
        assert "Response is not valid JSON" in failed.meta["metadata_extraction_error"]
        assert "JSONDecodeError" in failed.meta["metadata_extraction_error"]
        assert "metadata_extraction_response" in failed.meta
        assert failed.meta["metadata_extraction_response"] is mock_reply

    @patch.object(DocumentToImageContent, "run")
    def test_run_extraction_mode_metadata_with_expected_keys_all_present(self, mock_doc_to_image_run):
        """When expected_keys is set and LLM returns all keys, document is updated and no warning path is triggered."""
        mock_chat_generator = Mock(spec=OpenAIChatGenerator)
        mock_chat_generator.run.return_value = {
            "replies": [
                ChatMessage.from_assistant(text='{"title": "Sample Doc", "author": "Test", "document_type": "report"}')
            ]
        }
        mock_doc_to_image_run.return_value = {
            "image_contents": [ImageContent.from_file_path("./test/test_files/images/apple.jpg")]
        }
        extractor = LLMDocumentContentExtractor(
            chat_generator=mock_chat_generator,
            extraction_mode="metadata",
            expected_keys=["title", "author", "document_type"],
        )
        docs = [Document(content="", meta={"file_path": "/path/to/image.pdf"})]
        result = extractor.run(documents=docs)
        assert len(result["documents"]) == 1
        assert result["documents"][0].meta["title"] == "Sample Doc"
        assert result["documents"][0].meta["author"] == "Test"
        assert result["documents"][0].meta["document_type"] == "report"
        assert "metadata_extraction_response" not in result["documents"][0].meta

    @patch.object(DocumentToImageContent, "run")
    def test_run_extraction_mode_metadata_with_expected_keys_missing_key_still_merges(
        self, mock_doc_to_image_run, caplog
    ):
        """When expected_keys is set and LLM response is missing a key, a warning is logged but extraction continues."""
        mock_chat_generator = Mock(spec=OpenAIChatGenerator)
        mock_chat_generator.run.return_value = {
            "replies": [ChatMessage.from_assistant(text='{"title": "Sample Doc", "author": "Test"}')]
        }
        mock_doc_to_image_run.return_value = {
            "image_contents": [ImageContent.from_file_path("./test/test_files/images/apple.jpg")]
        }
        extractor = LLMDocumentContentExtractor(
            chat_generator=mock_chat_generator,
            extraction_mode="metadata",
            expected_keys=["title", "author", "document_type"],
        )
        docs = [Document(content="", meta={"file_path": "/path/to/image.pdf"})]
        result = extractor.run(documents=docs)
        assert len(result["documents"]) == 1
        assert result["documents"][0].meta["title"] == "Sample Doc"
        assert result["documents"][0].meta["author"] == "Test"
        assert "document_type" not in result["documents"][0].meta
        assert "metadata_extraction_response" not in result["documents"][0].meta
        assert "Expected response from LLM to be a JSON with keys" in caplog.text

    @patch.object(DocumentToImageContent, "run")
    def test_run_extraction_mode_metadata_removes_previous_errors_and_response(self, mock_doc_to_image_run):
        """
        When extraction_mode is 'metadata' and run succeeds, previous metadata_extraction_error and
        metadata_extraction_response are removed.
        """
        mock_chat_generator = Mock(spec=OpenAIChatGenerator)
        mock_chat_generator.run.return_value = {
            "replies": [ChatMessage.from_assistant(text='{"title": "New Title", "author": "New Author"}')]
        }
        mock_doc_to_image_run.return_value = {
            "image_contents": [ImageContent.from_file_path("./test/test_files/images/apple.jpg")]
        }
        extractor = LLMDocumentContentExtractor(chat_generator=mock_chat_generator, extraction_mode="metadata")
        old_reply = ChatMessage.from_assistant(text="old failed reply")
        doc = Document(
            content="",
            meta={
                "file_path": "/path/to/image.pdf",
                "metadata_extraction_error": "Old error",
                "metadata_extraction_response": old_reply,
                "other": "kept",
            },
        )
        result = extractor.run(documents=[doc])
        assert len(result["documents"]) == 1
        processed = result["documents"][0]
        assert "metadata_extraction_error" not in processed.meta
        assert "metadata_extraction_response" not in processed.meta
        assert processed.meta["other"] == "kept"
        assert processed.meta["title"] == "New Title"
        assert processed.meta["author"] == "New Author"

    @patch.object(DocumentToImageContent, "run")
    def test_run_extraction_mode_override_at_runtime(self, mock_doc_to_image_run):
        """extraction_mode in run() overrides init for that run only."""
        mock_chat_generator = Mock(spec=OpenAIChatGenerator)
        mock_chat_generator.run.side_effect = [
            {"replies": [ChatMessage.from_assistant(text='{"document_content": "Extracted text"}')]},
            {"replies": [ChatMessage.from_assistant(text='{"title": "Doc", "author": "Jane"}')]},
        ]
        mock_doc_to_image_run.return_value = {
            "image_contents": [ImageContent.from_file_path("./test/test_files/images/apple.jpg")]
        }
        extractor = LLMDocumentContentExtractor(chat_generator=mock_chat_generator, extraction_mode="content")
        docs = [Document(content="", meta={"file_path": "/path/to/image.pdf"})]
        result_content = extractor.run(documents=docs, extraction_mode="content")
        assert result_content["documents"][0].content == "Extracted text"
        result_metadata = extractor.run(documents=docs, extraction_mode="metadata")
        assert result_metadata["documents"][0].content == ""
        assert result_metadata["documents"][0].meta["title"] == "Doc"
        assert result_metadata["documents"][0].meta["author"] == "Jane"

    @patch.object(DocumentToImageContent, "run")
    def test_run_extraction_mode_both_success(self, mock_doc_to_image_run):
        """
        When extraction_mode is 'both', document gets content from document_content key and metadata
        from second JSON.
        """
        mock_chat_generator = Mock(spec=OpenAIChatGenerator)
        content_reply = '{"document_content": "Extracted document content as markdown."}'
        metadata_reply = '{"title": "Sample", "author": "Jane", "document_type": "report"}'
        mock_chat_generator.run.side_effect = [
            {"replies": [ChatMessage.from_assistant(text=content_reply)]},
            {"replies": [ChatMessage.from_assistant(text=metadata_reply)]},
        ]
        mock_doc_to_image_run.return_value = {
            "image_contents": [ImageContent.from_file_path("./test/test_files/images/apple.jpg")]
        }
        extractor = LLMDocumentContentExtractor(chat_generator=mock_chat_generator, extraction_mode="both")
        docs = [Document(content="", meta={"file_path": "/path/to/image.pdf"})]
        result = extractor.run(documents=docs)
        assert len(result["documents"]) == 1
        assert len(result["failed_documents"]) == 0
        processed = result["documents"][0]
        assert processed.content == "Extracted document content as markdown."
        assert processed.meta["title"] == "Sample"
        assert processed.meta["author"] == "Jane"
        assert processed.meta["document_type"] == "report"
        assert "content_extraction_error" not in processed.meta
        assert "metadata_extraction_error" not in processed.meta
        assert "metadata_extraction_response" not in processed.meta
        assert mock_chat_generator.run.call_count == 2

    @patch.object(DocumentToImageContent, "run")
    def test_run_extraction_mode_both_metadata_invalid_json(self, mock_doc_to_image_run):
        """
        When extraction_mode is 'both' and metadata LLM returns invalid JSON, document has metadata_extraction_error
        and metadata_extraction_response.
        """
        content_reply = '{"document_content": "Extracted content."}'
        raw_metadata_reply = "not valid json"
        mock_metadata_reply = ChatMessage.from_assistant(text=raw_metadata_reply)
        mock_chat_generator = Mock(spec=OpenAIChatGenerator)
        mock_chat_generator.run.side_effect = [
            {"replies": [ChatMessage.from_assistant(text=content_reply)]},
            {"replies": [mock_metadata_reply]},
        ]
        mock_doc_to_image_run.return_value = {
            "image_contents": [ImageContent.from_file_path("./test/test_files/images/apple.jpg")]
        }
        extractor = LLMDocumentContentExtractor(chat_generator=mock_chat_generator, extraction_mode="both")
        doc = Document(content="", meta={"file_path": "/path/to/image.pdf"})
        result = extractor.run(documents=[doc])
        assert len(result["documents"]) == 1
        assert len(result["failed_documents"]) == 0
        processed = result["documents"][0]
        assert processed.content == "Extracted content."
        assert "metadata_extraction_error" in processed.meta
        assert "Response is not valid JSON" in processed.meta["metadata_extraction_error"]
        assert processed.meta["metadata_extraction_response"] is mock_metadata_reply

    @patch.object(DocumentToImageContent, "run")
    def test_run_extraction_mode_both_both_fail(self, mock_doc_to_image_run):
        """When extraction_mode is 'both' and both LLM calls fail, document goes to failed_documents."""
        mock_chat_generator = Mock(spec=OpenAIChatGenerator)
        mock_chat_generator.run.side_effect = [Exception("Content API error"), Exception("Metadata API error")]
        mock_doc_to_image_run.return_value = {
            "image_contents": [ImageContent.from_file_path("./test/test_files/images/apple.jpg")]
        }
        extractor = LLMDocumentContentExtractor(
            chat_generator=mock_chat_generator, extraction_mode="both", raise_on_failure=False
        )
        docs = [Document(content="", meta={"file_path": "/path/to/image.pdf"})]
        result = extractor.run(documents=docs)
        assert len(result["documents"]) == 0
        assert len(result["failed_documents"]) == 1
        failed = result["failed_documents"][0]
        assert "content_extraction_error" in failed.meta
        assert "metadata_extraction_error" in failed.meta
        assert "Content API error" in failed.meta["content_extraction_error"]
        assert "Metadata API error" in failed.meta["metadata_extraction_error"]

    @patch.object(DocumentToImageContent, "run")
    def test_run_extraction_mode_both_content_fails_metadata_succeeds(self, mock_doc_to_image_run):
        """
        When extraction_mode is 'both', content fails and metadata succeeds: document in documents with
        error in meta.
        """
        mock_chat_generator = Mock(spec=OpenAIChatGenerator)
        metadata_reply = '{"title": "Doc", "document_type": "letter"}'
        mock_chat_generator.run.side_effect = [
            Exception("Content API error"),
            {"replies": [ChatMessage.from_assistant(text=metadata_reply)]},
        ]
        mock_doc_to_image_run.return_value = {
            "image_contents": [ImageContent.from_file_path("./test/test_files/images/apple.jpg")]
        }
        extractor = LLMDocumentContentExtractor(
            chat_generator=mock_chat_generator, extraction_mode="both", raise_on_failure=False
        )
        doc = Document(content="original", meta={"file_path": "/path/to/image.pdf"})
        result = extractor.run(documents=[doc])
        assert len(result["documents"]) == 1
        assert len(result["failed_documents"]) == 0
        processed = result["documents"][0]
        assert processed.content == "original"
        assert processed.meta["title"] == "Doc"
        assert processed.meta["document_type"] == "letter"
        assert "content_extraction_error" in processed.meta
        assert "metadata_extraction_error" not in processed.meta
        assert "metadata_extraction_response" not in processed.meta

    @patch.object(DocumentToImageContent, "run")
    def test_run_extraction_mode_both_metadata_fails_content_succeeds(self, mock_doc_to_image_run):
        """
        When extraction_mode is 'both', metadata fails and content succeeds: document in documents with content
        and error in meta.
        """
        mock_chat_generator = Mock(spec=OpenAIChatGenerator)
        content_reply = '{"document_content": "Extracted text here."}'
        mock_chat_generator.run.side_effect = [
            {"replies": [ChatMessage.from_assistant(text=content_reply)]},
            Exception("Metadata API error"),
        ]
        mock_doc_to_image_run.return_value = {
            "image_contents": [ImageContent.from_file_path("./test/test_files/images/apple.jpg")]
        }
        extractor = LLMDocumentContentExtractor(
            chat_generator=mock_chat_generator, extraction_mode="both", raise_on_failure=False
        )
        doc = Document(content="", meta={"file_path": "/path/to/image.pdf"})
        result = extractor.run(documents=[doc])
        assert len(result["documents"]) == 1
        assert len(result["failed_documents"]) == 0
        processed = result["documents"][0]
        assert processed.content == "Extracted text here."
        assert "content_extraction_error" not in processed.meta
        assert "metadata_extraction_error" in processed.meta

    @patch.object(DocumentToImageContent, "run")
    def test_run_extraction_mode_both_removes_previous_errors(self, mock_doc_to_image_run):
        """When extraction_mode is 'both' and run succeeds, previous content/metadata extraction errors are removed."""
        mock_chat_generator = Mock(spec=OpenAIChatGenerator)
        mock_chat_generator.run.side_effect = [
            {"replies": [ChatMessage.from_assistant(text='{"document_content": "New content"}')]},
            {"replies": [ChatMessage.from_assistant(text='{"title": "New", "document_type": "doc"}')]},
        ]
        mock_doc_to_image_run.return_value = {
            "image_contents": [ImageContent.from_file_path("./test/test_files/images/apple.jpg")]
        }
        extractor = LLMDocumentContentExtractor(chat_generator=mock_chat_generator, extraction_mode="both")
        old_reply = ChatMessage.from_assistant(text="old metadata reply")
        doc = Document(
            content="",
            meta={
                "file_path": "/path/to/image.pdf",
                "content_extraction_error": "Old content error",
                "metadata_extraction_error": "Old metadata error",
                "metadata_extraction_response": old_reply,
                "other": "kept",
            },
        )
        result = extractor.run(documents=[doc])
        assert len(result["documents"]) == 1
        processed = result["documents"][0]
        assert "content_extraction_error" not in processed.meta
        assert "metadata_extraction_error" not in processed.meta
        assert "metadata_extraction_response" not in processed.meta
        assert processed.meta["other"] == "kept"
        assert processed.content == "New content"
        assert processed.meta["title"] == "New"
        assert processed.meta["document_type"] == "doc"

    @patch.object(DocumentToImageContent, "run")
    def test_run_extraction_mode_both_with_failed_image_conversion(self, mock_doc_to_image_run):
        """
        When extraction_mode is 'both' and image conversion fails, document is in failed_documents with
        both errors.
        """
        mock_chat_generator = Mock(spec=OpenAIChatGenerator)
        mock_doc_to_image_run.return_value = {"image_contents": [None]}
        extractor = LLMDocumentContentExtractor(chat_generator=mock_chat_generator, extraction_mode="both")
        doc = Document(content="", meta={"file_path": "/path/to/image.pdf"})
        result = extractor.run(documents=[doc])
        assert len(result["documents"]) == 0
        assert len(result["failed_documents"]) == 1
        failed = result["failed_documents"][0]
        assert failed.meta["content_extraction_error"] == "Document has no content, skipping LLM call."
        assert failed.meta["metadata_extraction_error"] == "Document has no content, skipping LLM call."
        mock_chat_generator.run.assert_not_called()

    @patch.object(DocumentToImageContent, "run")
    def test_run_extraction_mode_override_to_both_at_runtime(self, mock_doc_to_image_run):
        """extraction_mode='both' can be passed at runtime to override init."""
        mock_chat_generator = Mock(spec=OpenAIChatGenerator)
        mock_chat_generator.run.side_effect = [
            {"replies": [ChatMessage.from_assistant(text='{"document_content": "Content"}')]},
            {"replies": [ChatMessage.from_assistant(text='{"title": "X", "author": "Y"}')]},
        ]
        mock_doc_to_image_run.return_value = {
            "image_contents": [ImageContent.from_file_path("./test/test_files/images/apple.jpg")]
        }
        extractor = LLMDocumentContentExtractor(chat_generator=mock_chat_generator, extraction_mode="content")
        doc = Document(content="", meta={"file_path": "/path/to/image.pdf"})
        result = extractor.run(documents=[doc], extraction_mode="both")
        assert len(result["documents"]) == 1
        assert result["documents"][0].content == "Content"  # from document_content key
        assert result["documents"][0].meta["title"] == "X"
        assert result["documents"][0].meta["author"] == "Y"
        assert mock_chat_generator.run.call_count == 2

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
        extractor = LLMDocumentContentExtractor(chat_generator=OpenAIChatGenerator(model="gpt-4.1-nano"))
        writer = DocumentWriter(document_store=doc_store)
        pipeline = Pipeline()
        pipeline.add_component("extractor", extractor)
        pipeline.add_component("doc_writer", writer)
        pipeline.connect("extractor.documents", "doc_writer.documents")
        pipeline.run(data={"documents": docs})

        doc_store_docs = doc_store.filter_documents()
        assert len(doc_store_docs) >= 1
        assert len(doc_store_docs[0].content) > 0

    @pytest.mark.integration
    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY", None),
        reason="Export an env var called OPENAI_API_KEY containing the OpenAI API key to run this test.",
    )
    def test_live_run_metadata_extraction(self):
        docs = [Document(content="", meta={"file_path": "./test/test_files/images/apple.jpg"})]
        doc_store = InMemoryDocumentStore()
        extractor = LLMDocumentContentExtractor(
            chat_generator=OpenAIChatGenerator(model="gpt-4.1-nano"), extraction_mode="metadata"
        )
        writer = DocumentWriter(document_store=doc_store)
        pipeline = Pipeline()
        pipeline.add_component("extractor", extractor)
        pipeline.add_component("doc_writer", writer)
        pipeline.connect("extractor.documents", "doc_writer.documents")
        pipeline.run(data={"documents": docs})

        doc_store_docs = doc_store.filter_documents()
        assert len(doc_store_docs) >= 1
        assert "metadata_extraction_error" not in doc_store_docs[0].meta
        # Metadata extraction merges JSON into meta; we expect at least file_path + extracted keys
        assert len(doc_store_docs[0].meta) >= 1

    @pytest.mark.integration
    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY", None),
        reason="Export an env var called OPENAI_API_KEY containing the OpenAI API key to run this test.",
    )
    def test_live_run_both_extraction(self):
        docs = [Document(content="", meta={"file_path": "./test/test_files/images/apple.jpg"})]
        doc_store = InMemoryDocumentStore()
        extractor = LLMDocumentContentExtractor(
            chat_generator=OpenAIChatGenerator(model="gpt-4.1-nano"), extraction_mode="both"
        )
        writer = DocumentWriter(document_store=doc_store)
        pipeline = Pipeline()
        pipeline.add_component("extractor", extractor)
        pipeline.add_component("doc_writer", writer)
        pipeline.connect("extractor.documents", "doc_writer.documents")
        pipeline.run(data={"documents": docs})

        doc_store_docs = doc_store.filter_documents()
        assert len(doc_store_docs) >= 1
        assert len(doc_store_docs[0].content) > 0
        assert "metadata_extraction_error" not in doc_store_docs[0].meta
        assert len(doc_store_docs[0].meta) >= 1

    @pytest.mark.integration
    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY", None),
        reason="Export an env var called OPENAI_API_KEY containing the OpenAI API key to run this test.",
    )
    def test_live_run_both_extraction_image_metadata_png(self):
        """
        Live test using image_metadata.png: extracts both metadata (author, date, document_type, topic)
        and content (image description of the document) via extraction_mode='both'.
        """
        image_path = "./test/test_files/images/image_metadata.png"
        docs = [Document(content="", meta={"file_path": image_path})]
        doc_store = InMemoryDocumentStore()
        extractor = LLMDocumentContentExtractor(
            chat_generator=OpenAIChatGenerator(model="gpt-4.1-nano"), extraction_mode="both"
        )
        writer = DocumentWriter(document_store=doc_store)
        pipeline = Pipeline()
        pipeline.add_component("extractor", extractor)
        pipeline.add_component("doc_writer", writer)
        pipeline.connect("extractor.documents", "doc_writer.documents")
        pipeline.run(data={"documents": docs})

        doc_store_docs = doc_store.filter_documents()
        assert len(doc_store_docs) >= 1
        doc = doc_store_docs[0]
        assert len(doc.content) > 0, "Expected non-empty content (image/document description)"
        assert "content_extraction_error" not in doc.meta
        assert "metadata_extraction_error" not in doc.meta
        assert len(doc.meta) >= 1, "Expected at least one metadata key"
        expected_metadata_keys = {"author", "date", "document_type", "topic"}
        found = expected_metadata_keys & set(doc.meta.keys())
        assert len(found) >= 1, (
            f"Expected at least one of {expected_metadata_keys} in metadata, got keys: {list(doc.meta.keys())}"
        )

        print("Extracted content:", doc.content)
        print("Extracted metadata:", {k: v for k, v in doc.meta.items() if k in expected_metadata_keys})
