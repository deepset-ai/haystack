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

    def test_init_with_defaults(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")
        chat_generator = OpenAIChatGenerator()
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

        assert extractor.prompt == "Custom extraction prompt"
        assert extractor.file_path_meta_field == "custom_path"
        assert extractor.root_path == "/custom/root"
        assert extractor.detail == "low"
        assert extractor.size == (1024, 768)
        assert extractor.raise_on_failure is True
        assert extractor.max_workers == 4
        assert component_to_dict(extractor._chat_generator, "name") == component_to_dict(chat_generator, "name")

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
        assert extractor.prompt == "Custom extraction prompt"

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
    def test_run_plain_string_response_goes_to_content(self, mock_doc_to_image_run):
        """When LLM returns plain string (non-JSON), it is written to document content."""
        mock_chat_generator = Mock(spec=OpenAIChatGenerator)
        mock_chat_generator.run.return_value = {"replies": [ChatMessage.from_assistant(text="Plain text, not JSON")]}
        mock_doc_to_image_run.return_value = {
            "image_contents": [ImageContent.from_file_path("./test/test_files/images/apple.jpg")]
        }
        extractor = LLMDocumentContentExtractor(chat_generator=mock_chat_generator)
        docs = [Document(content="", meta={"file_path": "/path/to/image.pdf"})]
        result = extractor.run(documents=docs)
        assert len(result["documents"]) == 1
        assert len(result["failed_documents"]) == 0
        assert result["documents"][0].content == "Plain text, not JSON"

    @patch.object(DocumentToImageContent, "run")
    def test_run_valid_json_not_object_reports_error(self, mock_doc_to_image_run):
        """When LLM returns valid JSON that is not an object (e.g. array or primitive), report error."""
        mock_chat_generator = Mock(spec=OpenAIChatGenerator)
        mock_chat_generator.run.return_value = {
            "replies": [ChatMessage.from_assistant(text='["array", "not", "object"]')]
        }
        mock_doc_to_image_run.return_value = {
            "image_contents": [ImageContent.from_file_path("./test/test_files/images/apple.jpg")]
        }
        extractor = LLMDocumentContentExtractor(chat_generator=mock_chat_generator)
        docs = [Document(content="", meta={"file_path": "/path/to/image.pdf"})]
        result = extractor.run(documents=docs)
        assert len(result["documents"]) == 0
        assert len(result["failed_documents"]) == 1
        assert "content_extraction_error" in result["failed_documents"][0].meta
        assert "JSON object" in result["failed_documents"][0].meta["content_extraction_error"]

    @patch.object(DocumentToImageContent, "run")
    def test_run_json_single_key_metadata_only_merged_into_meta(self, mock_doc_to_image_run):
        """When LLM returns JSON with a single key that is not document_content, it goes to metadata only."""
        mock_chat_generator = Mock(spec=OpenAIChatGenerator)
        mock_chat_generator.run.return_value = {
            "replies": [ChatMessage.from_assistant(text='{"title": "Only metadata, no document_content"}')]
        }
        mock_doc_to_image_run.return_value = {
            "image_contents": [ImageContent.from_file_path("./test/test_files/images/apple.jpg")]
        }
        extractor = LLMDocumentContentExtractor(chat_generator=mock_chat_generator)
        docs = [Document(content="original", meta={"file_path": "/path/to/image.pdf"})]
        result = extractor.run(documents=docs)
        assert len(result["documents"]) == 1
        assert len(result["failed_documents"]) == 0
        processed = result["documents"][0]
        assert processed.content == "original"
        assert processed.meta["title"] == "Only metadata, no document_content"

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
    def test_run_json_multiple_keys_metadata_merged(self, mock_doc_to_image_run):
        """When LLM returns JSON with multiple keys and no document_content, all keys are merged into metadata."""
        mock_chat_generator = Mock(spec=OpenAIChatGenerator)
        mock_chat_generator.run.return_value = {
            "replies": [
                ChatMessage.from_assistant(text='{"title": "Sample Doc", "author": "Test", "document_type": "report"}')
            ]
        }
        mock_doc_to_image_run.return_value = {
            "image_contents": [ImageContent.from_file_path("./test/test_files/images/apple.jpg")]
        }
        extractor = LLMDocumentContentExtractor(chat_generator=mock_chat_generator)
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

    @patch.object(DocumentToImageContent, "run")
    def test_run_removes_previous_metadata_errors_on_success(self, mock_doc_to_image_run):
        """When run succeeds, previous metadata_extraction_error and metadata_extraction_response are removed."""
        mock_chat_generator = Mock(spec=OpenAIChatGenerator)
        mock_chat_generator.run.return_value = {
            "replies": [ChatMessage.from_assistant(text='{"title": "New Title", "author": "New Author"}')]
        }
        mock_doc_to_image_run.return_value = {
            "image_contents": [ImageContent.from_file_path("./test/test_files/images/apple.jpg")]
        }
        extractor = LLMDocumentContentExtractor(chat_generator=mock_chat_generator)
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

    def test_run_on_thread_with_none_prompt(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")
        extractor = LLMDocumentContentExtractor(chat_generator=OpenAIChatGenerator())
        result = extractor._run_on_thread(None)
        assert "error" in result
        assert result["error"] == "Document has no content, skipping LLM call."

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
        """Single prompt run; if LLM returns JSON with keys, they are merged into metadata."""
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
        assert "metadata_extraction_error" not in doc_store_docs[0].meta
        # Metadata extraction merges JSON into meta; we expect at least file_path + extracted keys
        assert len(doc_store_docs[0].meta) >= 1

    @pytest.mark.integration
    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY", None),
        reason="Export an env var called OPENAI_API_KEY containing the OpenAI API key to run this test.",
    )
    def test_live_run_both_extraction(self):
        """Single prompt run; if LLM returns JSON with document_content and other keys, content and metadata are set."""
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
        assert "metadata_extraction_error" not in doc_store_docs[0].meta
        assert len(doc_store_docs[0].meta) >= 1

    @pytest.mark.integration
    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY", None),
        reason="Export an env var called OPENAI_API_KEY containing the OpenAI API key to run this test.",
    )
    def test_live_run_on_image_with_metadata(self):
        """
        Live test using image_metadata.png: single prompt; LLM can return JSON with document_content
        and metadata keys (author, date, document_type, topic) in one response.
        """
        image_path = "./test/test_files/images/image_metadata.png"
        docs = [Document(content="", meta={"file_path": image_path})]
        doc_store = InMemoryDocumentStore()
        extractor = LLMDocumentContentExtractor(
            chat_generator=OpenAIChatGenerator(
                model="gpt-4.1-nano", generation_kwargs={"temperature": 0.2, "response_format": {"type": "json_object"}}
            )
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
