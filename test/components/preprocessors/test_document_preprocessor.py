# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import inspect
from unittest.mock import patch

import pytest

from haystack import Document, Pipeline
from haystack.components.preprocessors.document_cleaner import DocumentCleaner
from haystack.components.preprocessors.document_preprocessor import DocumentPreprocessor
from haystack.components.preprocessors.document_splitter import DocumentSplitter


class TestDocumentPreprocessor:
    @pytest.fixture
    def preprocessor(self) -> DocumentPreprocessor:
        return DocumentPreprocessor(
            # Cleaner parameters
            remove_empty_lines=True,
            remove_extra_whitespaces=True,
            remove_repeated_substrings=False,
            keep_id=True,
            # Splitter parameters
            split_by="word",
            split_length=3,
            split_overlap=1,
            respect_sentence_boundary=False,
            language="en",
        )

    def test_init(self, preprocessor: DocumentPreprocessor) -> None:
        assert isinstance(preprocessor.pipeline, Pipeline)
        assert preprocessor.input_mapping == {"documents": ["splitter.documents"]}
        assert preprocessor.output_mapping == {"cleaner.documents": "documents"}

        cleaner = preprocessor.pipeline.get_component("cleaner")
        assert isinstance(cleaner, DocumentCleaner)
        assert cleaner.remove_empty_lines is True
        assert cleaner.remove_extra_whitespaces is True
        assert cleaner.remove_repeated_substrings is False
        assert cleaner.keep_id is True

        splitter = preprocessor.pipeline.get_component("splitter")
        assert isinstance(splitter, DocumentSplitter)
        assert splitter.split_by == "word"
        assert splitter.split_length == 3
        assert splitter.split_overlap == 1
        assert splitter.respect_sentence_boundary is False
        assert splitter.language == "en"

    def test_from_dict(self) -> None:
        data = {
            "init_parameters": {
                "remove_empty_lines": True,
                "remove_extra_whitespaces": True,
                "remove_repeated_substrings": False,
                "keep_id": True,
                "remove_substrings": None,
                "remove_regex": None,
                "unicode_normalization": None,
                "ascii_only": False,
                "split_by": "word",
                "split_length": 3,
                "split_overlap": 1,
                "split_threshold": 0,
                "splitting_function": None,
                "respect_sentence_boundary": False,
                "language": "en",
                "use_split_rules": True,
                "extend_abbreviations": True,
                "tokenizer_encoding": "o200k_base",
            },
            "type": "haystack.components.preprocessors.document_preprocessor.DocumentPreprocessor",
        }
        preprocessor = DocumentPreprocessor.from_dict(data)
        assert isinstance(preprocessor, DocumentPreprocessor)

    def test_to_dict(self, preprocessor: DocumentPreprocessor) -> None:
        expected = {
            "init_parameters": {
                "remove_empty_lines": True,
                "remove_extra_whitespaces": True,
                "remove_repeated_substrings": False,
                "keep_id": True,
                "remove_substrings": None,
                "remove_regex": None,
                "unicode_normalization": None,
                "ascii_only": False,
                "strip_whitespaces": False,
                "replace_regexes": None,
                "min_content_length": 0,
                "split_by": "word",
                "split_length": 3,
                "split_overlap": 1,
                "split_threshold": 0,
                "splitting_function": None,
                "respect_sentence_boundary": False,
                "language": "en",
                "use_split_rules": True,
                "extend_abbreviations": True,
                "tokenizer_encoding": "o200k_base",
                "skip_empty_documents": True,
            },
            "type": "haystack.components.preprocessors.document_preprocessor.DocumentPreprocessor",
        }
        assert preprocessor.to_dict() == expected

    def test_serialization_round_trip_keeps_passthrough_params(self) -> None:
        preprocessor = DocumentPreprocessor(
            skip_empty_documents=False, strip_whitespaces=True, replace_regexes={"foo": "bar"}, min_content_length=5
        )
        restored = DocumentPreprocessor.from_dict(preprocessor.to_dict())

        splitter = restored.pipeline.get_component("splitter")
        cleaner = restored.pipeline.get_component("cleaner")
        assert isinstance(splitter, DocumentSplitter)
        assert isinstance(cleaner, DocumentCleaner)
        assert splitter.skip_empty_documents is False
        assert cleaner.strip_whitespaces is True
        assert cleaner.replace_regexes == {"foo": "bar"}
        assert cleaner.min_content_length == 5

    def test_from_dict_without_passthrough_params_uses_defaults(self, preprocessor: DocumentPreprocessor) -> None:
        data = preprocessor.to_dict()
        for key in ("skip_empty_documents", "strip_whitespaces", "replace_regexes", "min_content_length"):
            del data["init_parameters"][key]

        restored = DocumentPreprocessor.from_dict(data)

        assert restored.skip_empty_documents is True
        assert restored.strip_whitespaces is False
        assert restored.replace_regexes is None
        assert restored.min_content_length == 0

    def test_exposes_all_wrapped_component_params(self) -> None:
        exposed = set(inspect.signature(DocumentPreprocessor.__init__).parameters)
        assert set(inspect.signature(DocumentSplitter.__init__).parameters) <= exposed
        assert set(inspect.signature(DocumentCleaner.__init__).parameters) <= exposed

    def test_init_rejects_negative_min_content_length(self) -> None:
        with pytest.raises(ValueError, match="min_content_length must be greater than or equal to 0."):
            DocumentPreprocessor(min_content_length=-1)

    def test_init_token(self) -> None:
        preprocessor = DocumentPreprocessor(split_by="token", split_length=10, tokenizer_encoding="cl100k_base")
        splitter = preprocessor.pipeline.get_component("splitter")
        assert isinstance(splitter, DocumentSplitter)
        assert splitter.split_by == "token"
        assert splitter.split_length == 10
        assert splitter.tokenizer_encoding == "cl100k_base"

    def test_warm_up(self, preprocessor: DocumentPreprocessor) -> None:
        with patch.object(preprocessor.pipeline, "warm_up") as mock_warm_up:
            preprocessor.warm_up()
            mock_warm_up.assert_called_once()

    def test_run(self, preprocessor: DocumentPreprocessor) -> None:
        documents = [
            Document(content="This is a test document. It has multiple sentences."),
            Document(content="Another test document with some content."),
        ]

        result = preprocessor.run(documents=documents)

        # Check that we got processed documents back
        assert "documents" in result
        processed_docs = result["documents"]
        assert len(processed_docs) > len(documents)  # Should have more docs due to splitting

        # Check that the content was cleaned and split
        for doc in processed_docs:
            assert doc.content is not None
            assert doc.content.strip() == doc.content
            assert len(doc.content.split()) <= 3  # Split length of 3 words
            assert doc.id is not None

    def test_run_skips_empty_document_by_default(self) -> None:
        assert DocumentPreprocessor().run(documents=[Document(content="")])["documents"] == []

    def test_run_keeps_empty_document_when_skipping_disabled(self) -> None:
        preprocessor = DocumentPreprocessor(skip_empty_documents=False)
        result = preprocessor.run(documents=[Document(content="", meta={"file_path": "scan.pdf"})])

        assert [doc.content for doc in result["documents"]] == [""]
        assert result["documents"][0].meta["file_path"] == "scan.pdf"

    def test_run_min_content_length_drops_short_document(self) -> None:
        preprocessor = DocumentPreprocessor(min_content_length=5)
        result = preprocessor.run(documents=[Document(content="abc"), Document(content="long enough")])

        assert [doc.content for doc in result["documents"]] == ["long enough"]

    def test_run_replace_regexes_replaces_matches(self) -> None:
        preprocessor = DocumentPreprocessor(replace_regexes={"foo": "bar"})
        result = preprocessor.run(documents=[Document(content="foo")])

        assert [doc.content for doc in result["documents"]] == ["bar"]

    def test_run_strip_whitespaces_strips_content(self) -> None:
        preprocessor = DocumentPreprocessor(
            remove_empty_lines=False, remove_extra_whitespaces=False, strip_whitespaces=True
        )
        result = preprocessor.run(documents=[Document(content="  some text  ")])

        assert [doc.content for doc in result["documents"]] == ["some text"]

    @pytest.mark.integration
    def test_run_token(self) -> None:
        preprocessor = DocumentPreprocessor(split_by="token", split_length=5)
        docs = [Document(content="one two three four five six seven eight nine ten")]
        result = preprocessor.run(documents=docs)
        assert len(result["documents"]) > 1

    def test_run_with_custom_splitting_function(self) -> None:
        def custom_split(text: str) -> list[str]:
            return [t for t in text.split(".") if t.strip() != ""]

        preprocessor = DocumentPreprocessor(split_by="function", splitting_function=custom_split, split_length=1)

        documents = [Document(content="First sentence. Second sentence. Third sentence.")]
        result = preprocessor.run(documents=documents)

        processed_docs = result["documents"]
        assert len(processed_docs) == 3  # Should be split into 3 sentences
        for doc in processed_docs:
            assert doc.content is not None
            assert "." not in doc.content  # Each doc should be a single sentence
