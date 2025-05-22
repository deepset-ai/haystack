# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch
import pytest

from haystack import Document, Pipeline
from haystack.components.preprocessors.document_preprocessor import DocumentPreprocessor


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
        assert cleaner.remove_empty_lines is True
        assert cleaner.remove_extra_whitespaces is True
        assert cleaner.remove_repeated_substrings is False
        assert cleaner.keep_id is True

        splitter = preprocessor.pipeline.get_component("splitter")
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
                "split_by": "word",
                "split_length": 3,
                "split_overlap": 1,
                "split_threshold": 0,
                "splitting_function": None,
                "respect_sentence_boundary": False,
                "language": "en",
                "use_split_rules": True,
                "extend_abbreviations": True,
            },
            "type": "haystack.components.preprocessors.document_preprocessor.DocumentPreprocessor",
        }
        assert preprocessor.to_dict() == expected

    def test_warm_up(self, preprocessor: DocumentPreprocessor) -> None:
        with patch.object(preprocessor.pipeline, "warm_up") as mock_warm_up:
            preprocessor.warm_up()
            mock_warm_up.assert_called_once()

    def test_run(self, preprocessor: DocumentPreprocessor) -> None:
        documents = [
            Document(content="This is a test document. It has multiple sentences."),
            Document(content="Another test document with some content."),
        ]

        preprocessor.warm_up()
        result = preprocessor.run(documents=documents)

        # Check that we got processed documents back
        assert "documents" in result
        processed_docs = result["documents"]
        assert len(processed_docs) > len(documents)  # Should have more docs due to splitting

        # Check that the content was cleaned and split
        for doc in processed_docs:
            assert doc.content.strip() == doc.content
            assert len(doc.content.split()) <= 3  # Split length of 3 words
            assert doc.id is not None

    def test_run_with_custom_splitting_function(self) -> None:
        def custom_split(text: str) -> list[str]:
            return [t for t in text.split(".") if t.strip() != ""]

        preprocessor = DocumentPreprocessor(split_by="function", splitting_function=custom_split, split_length=1)

        documents = [Document(content="First sentence. Second sentence. Third sentence.")]
        preprocessor.warm_up()
        result = preprocessor.run(documents=documents)

        processed_docs = result["documents"]
        assert len(processed_docs) == 3  # Should be split into 3 sentences
        assert all("." not in doc.content for doc in processed_docs)  # Each doc should be a single sentence


def test_import_document_preprocessor() -> None:
    # test if the DocumentPreprocessor.run() doesn't trigger any type or static analyzer errors
    doc_processor = DocumentPreprocessor()
    doc_processor.run(documents=[])
