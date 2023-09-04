import logging
from unittest.mock import patch

import pytest
from pathlib import Path

from canals.errors import PipelineRuntimeError
from langdetect import LangDetectException

from haystack.preview.components.file_converters.txt import TextFileToDocument


class TestTextfileToDocument:
    @pytest.mark.unit
    def test_to_dict(self):
        component = TextFileToDocument()
        data = component.to_dict()
        assert data == {
            "type": "TextFileToDocument",
            "init_parameters": {
                "encoding": "utf-8",
                "remove_numeric_tables": False,
                "numeric_row_threshold": 0.4,
                "valid_languages": [],
                "id_hash_keys": [],
                "progress_bar": True,
            },
        }

    @pytest.mark.unit
    def test_to_dict_with_custom_init_parameters(self):
        component = TextFileToDocument(
            encoding="latin-1",
            remove_numeric_tables=True,
            numeric_row_threshold=0.7,
            valid_languages=["en", "de"],
            id_hash_keys=["name"],
            progress_bar=False,
        )
        data = component.to_dict()
        assert data == {
            "type": "TextFileToDocument",
            "init_parameters": {
                "encoding": "latin-1",
                "remove_numeric_tables": True,
                "numeric_row_threshold": 0.7,
                "valid_languages": ["en", "de"],
                "id_hash_keys": ["name"],
                "progress_bar": False,
            },
        }

    @pytest.mark.unit
    def test_from_dict(self):
        data = {
            "type": "TextFileToDocument",
            "init_parameters": {
                "encoding": "latin-1",
                "remove_numeric_tables": True,
                "numeric_row_threshold": 0.7,
                "valid_languages": ["en", "de"],
                "id_hash_keys": ["name"],
                "progress_bar": False,
            },
        }
        component = TextFileToDocument.from_dict(data)
        assert component.encoding == "latin-1"
        assert component.remove_numeric_tables
        assert component.numeric_row_threshold == 0.7
        assert component.valid_languages == ["en", "de"]
        assert component.id_hash_keys == ["name"]
        assert not component.progress_bar

    @pytest.mark.unit
    def test_run(self, preview_samples_path):
        """
        Test if the component runs correctly.
        """
        paths = [preview_samples_path / "txt" / "doc_1.txt", preview_samples_path / "txt" / "doc_2.txt"]
        converter = TextFileToDocument()
        output = converter.run(paths=paths)
        docs = output["documents"]
        assert len(docs) == 2
        assert docs[0].content == "Some text for testing.\nTwo lines in here."
        assert docs[1].content == "This is a test line.\n123 456 789\n987 654 321."
        assert docs[0].metadata["file_path"] == str(paths[0])
        assert docs[1].metadata["file_path"] == str(paths[1])

    @pytest.mark.unit
    def test_run_warning_for_invalid_language(self, preview_samples_path, caplog):
        file_path = preview_samples_path / "txt" / "doc_1.txt"
        converter = TextFileToDocument()
        with patch("haystack.preview.components.file_converters.txt.langdetect.detect", return_value="en"):
            with caplog.at_level(logging.WARNING):
                output = converter.run(paths=[file_path], valid_languages=["de"])
                assert (
                    f"Text from file {file_path} is not in one of the valid languages: ['de']. "
                    f"The file may have been decoded incorrectly." in caplog.text
                )

        docs = output["documents"]
        assert len(docs) == 1
        assert docs[0].content == "Some text for testing.\nTwo lines in here."

    @pytest.mark.unit
    def test_run_error_handling(self, preview_samples_path, caplog):
        """
        Test if the component correctly handles errors.
        """
        paths = [preview_samples_path / "txt" / "doc_1.txt", "non_existing_file.txt"]
        converter = TextFileToDocument()
        with caplog.at_level(logging.WARNING):
            output = converter.run(paths=paths)
            assert (
                "Could not read file non_existing_file.txt. Skipping it. Error message: File at path non_existing_file.txt does not exist."
                in caplog.text
            )
        docs = output["documents"]
        assert len(docs) == 1
        assert docs[0].metadata["file_path"] == str(paths[0])

    @pytest.mark.unit
    def test_prepare_metadata_no_metadata(self):
        """
        Test if the metadata is correctly prepared when no custom metadata is provided.
        """
        converter = TextFileToDocument()
        meta = converter._prepare_metadata(
            metadata=None, paths=["data/sample_path_1.txt", Path("data/sample_path_2.txt")]
        )
        assert len(meta) == 2
        assert meta[0]["file_path"] == "data/sample_path_1.txt"
        assert meta[1]["file_path"] == str(Path("data/sample_path_2.txt"))

    @pytest.mark.unit
    def test_prepare_metadata_single_dict(self):
        """
        Test if the metadata is correctly prepared when a single dict is provided.
        """
        converter = TextFileToDocument()
        meta = converter._prepare_metadata(
            metadata={"name": "test"}, paths=["data/sample_path_1.txt", Path("data/sample_path_2.txt")]
        )
        assert len(meta) == 2
        assert meta[0]["file_path"] == "data/sample_path_1.txt"
        assert meta[1]["file_path"] == str(Path("data/sample_path_2.txt"))
        assert meta[0]["name"] == "test"
        assert meta[1]["name"] == "test"

    @pytest.mark.unit
    def test_prepare_metadata_list_of_dicts(self):
        """
        Test if the metadata is correctly prepared when a list of dicts is provided.
        """
        converter = TextFileToDocument()
        meta = converter._prepare_metadata(
            metadata=[{"name": "test1"}, {"name": "test2"}],
            paths=["data/sample_path_1.txt", Path("data/sample_path_2.txt")],
        )
        assert len(meta) == 2
        assert meta[0]["file_path"] == "data/sample_path_1.txt"
        assert meta[1]["file_path"] == str(Path("data/sample_path_2.txt"))
        assert meta[0]["name"] == "test1"
        assert meta[1]["name"] == "test2"

    @pytest.mark.unit
    def test_prepare_metadata_unmatching_list_len(self):
        """
        Test if an error is raised when the number of metadata dicts is not equal to the number of
        file paths.
        """
        converter = TextFileToDocument()
        with pytest.raises(
            PipelineRuntimeError,
            match="The number of metadata entries must match the number of paths if metadata is a list.",
        ):
            converter._prepare_metadata(
                metadata=[{"name": "test1"}, {"name": "test2"}],
                paths=["data/sample_path_1.txt", Path("data/sample_path_2.txt"), "data/sample_path_3.txt"],
            )

    @pytest.mark.unit
    def test_read_and_clean_file(self, preview_samples_path):
        """
        Test if the file is correctly read.
        """
        file_path = preview_samples_path / "txt" / "doc_1.txt"
        converter = TextFileToDocument()
        text = converter._read_and_clean_file(path=file_path, encoding="utf-8", remove_numeric_tables=False)
        assert text == "Some text for testing.\nTwo lines in here."

    @pytest.mark.unit
    def test_read_and_clean_file_non_existing_file(self):
        """
        Test if an error is raised when the file does not exist.
        """
        converter = TextFileToDocument()
        file_path = "non_existing_file.txt"
        with pytest.raises(PipelineRuntimeError, match=f"File at path {file_path} does not exist."):
            converter._read_and_clean_file(path=file_path, encoding="utf-8", remove_numeric_tables=False)

    @pytest.mark.unit
    def test_read_and_clean_file_remove_numeric_tables(self, preview_samples_path):
        """
        Test if the file is correctly read and numeric tables are removed.
        """
        file_path = preview_samples_path / "txt" / "doc_2.txt"
        converter = TextFileToDocument()
        text = converter._read_and_clean_file(path=file_path, encoding="utf-8", remove_numeric_tables=True)
        assert text == "This is a test line.\n987 654 321."

    @pytest.mark.unit
    def test_clean_page_without_remove_numeric_tables(self):
        """
        Test if the page is not changed when remove_numeric_tables is False.
        """
        converter = TextFileToDocument()
        page = "This is a test line.\n123 456 789"
        cleaned_page = converter._clean_page(page=page, remove_numeric_tables=False)
        assert cleaned_page == page

    @pytest.mark.unit
    def test_clean_page_with_remove_numeric_tables(self):
        """
        Test if the page is correctly cleaned when remove_numeric_tables is True.
        """
        converter = TextFileToDocument()
        page = "This is a test line.\n123 456 789"
        cleaned_page = converter._clean_page(page=page, remove_numeric_tables=True)
        assert cleaned_page == "This is a test line."

    @pytest.mark.unit
    def test_is_numeric_row_only_numbers(self):
        """
        Test if the line is correctly identified as a numeric row when it only contains numbers.
        """
        converter = TextFileToDocument()
        line = "123 456 789"
        assert converter._is_numeric_row(line=line)

    @pytest.mark.unit
    def test_is_numeric_row_only_text(self):
        """
        Test if the line is correctly identified as a non-numeric row when it only contains text.
        """
        converter = TextFileToDocument()
        line = "This is a test line."
        assert not converter._is_numeric_row(line=line)

    @pytest.mark.unit
    def test_is_numeric_row_only_numbers_with_period(self):
        """
        Test if the line is correctly identified as a non-numeric row when it only contains numbers and a period at
        the end.
        """
        converter = TextFileToDocument()
        line = "123 456 789."
        assert not converter._is_numeric_row(line=line)

    @pytest.mark.unit
    def test_is_numeric_row_more_numbers_than_text(self):
        """
        Test if the line is correctly identified as a numeric row when it consists of more than 40% of numbers than.
        """
        converter = TextFileToDocument()
        line = "123 456 789 This is a test"
        assert converter._is_numeric_row(line=line)

    @pytest.mark.unit
    def test_is_numeric_row_less_numbers_than_text(self):
        """
        Test if the line is correctly identified as a non-numeric row when it consists of less than 40% of numbers than.
        """
        converter = TextFileToDocument()
        line = "123 456 789 This is a test line"
        assert not converter._is_numeric_row(line=line)

    @pytest.mark.unit
    def test_is_numeric_row_words_consist_of_numbers_and_text(self):
        """
        Test if the line is correctly identified as a numeric row when the words consist of numbers and text.
        """
        converter = TextFileToDocument()
        line = "123eur 456usd"
        assert converter._is_numeric_row(line=line)

    @pytest.mark.unit
    def test_validate_language(self):
        """
        Test if the language is correctly validated.
        """
        converter = TextFileToDocument()
        with patch("haystack.preview.components.file_converters.txt.langdetect.detect", return_value="en"):
            assert converter._validate_language(text="This is an english text.", valid_languages=["en"])
            assert not converter._validate_language(text="This is an english text.", valid_languages=["de"])

    @pytest.mark.unit
    def test_validate_language_no_languages_specified(self):
        """
        Test if _validate_languages returns True when no languages are specified.
        """
        converter = TextFileToDocument()
        assert converter._validate_language(text="This is an english test.", valid_languages=[])

    @pytest.mark.unit
    def test_validate_language_lang_detect_exception(self):
        """
        Test if _validate_languages returns False when langdetect throws an exception.
        """
        converter = TextFileToDocument()
        with patch(
            "haystack.preview.components.file_converters.txt.langdetect.detect",
            side_effect=LangDetectException(code=0, message="Test"),
        ):
            assert not converter._validate_language(text="This is an english text.", valid_languages=["en"])
