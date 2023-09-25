from unittest.mock import patch

import pytest

from haystack.preview.components.file_converters.tika import TikaDocumentConverter


class TestTikaDocumentConverter:
    @pytest.mark.unit
    def test_to_dict(self):
        component = TikaDocumentConverter()
        data = component.to_dict()
        assert data == {
            "type": "TikaDocumentConverter",
            "init_parameters": {"tika_url": "http://localhost:9998/tika", "id_hash_keys": []},
        }

    @pytest.mark.unit
    def test_to_dict_with_custom_init_parameters(self):
        component = TikaDocumentConverter(tika_url="http://localhost:1234/tika", id_hash_keys=["text", "category"])
        data = component.to_dict()
        assert data == {
            "type": "TikaDocumentConverter",
            "init_parameters": {"tika_url": "http://localhost:1234/tika", "id_hash_keys": ["text", "category"]},
        }

    @pytest.mark.unit
    def test_from_dict(self):
        data = {
            "type": "TikaDocumentConverter",
            "init_parameters": {"tika_url": "http://localhost:9998/tika", "id_hash_keys": ["text", "category"]},
        }
        component = TikaDocumentConverter.from_dict(data)
        assert component.tika_url == "http://localhost:9998/tika"
        assert component.id_hash_keys == ["text", "category"]

    @pytest.mark.unit
    def test_run(self):
        component = TikaDocumentConverter()
        with patch("haystack.preview.components.file_converters.tika.tika_parser.from_file") as mock_tika_parser:
            mock_tika_parser.return_value = {"content": "Content of mock_file.pdf"}
            documents = component.run(paths=["mock_file.pdf"])["documents"]

        assert len(documents) == 1
        assert documents[0].text == "Content of mock_file.pdf"

    @pytest.mark.unit
    def test_run_logs_warning_if_content_empty(self, caplog):
        component = TikaDocumentConverter()
        with patch("haystack.preview.components.file_converters.tika.tika_parser.from_file") as mock_tika_parser:
            mock_tika_parser.return_value = {"content": ""}
            with caplog.at_level("WARNING"):
                component.run(paths=["mock_file.pdf"])
                assert "Skipping file at 'mock_file.pdf' as Tika was not able to extract any content." in caplog.text

    @pytest.mark.unit
    def test_run_logs_error(self, caplog):
        component = TikaDocumentConverter()
        with patch("haystack.preview.components.file_converters.tika.tika_parser.from_file") as mock_tika_parser:
            mock_tika_parser.side_effect = Exception("Some error")
            with caplog.at_level("ERROR"):
                component.run(paths=["mock_file.pdf"])
                assert "Could not convert file at 'mock_file.pdf' to Document. Error: Some error" in caplog.text

    @pytest.mark.integration
    def test_run_with_txt_files(self, preview_samples_path):
        component = TikaDocumentConverter()
        output = component.run(
            paths=[preview_samples_path / "txt" / "doc_1.txt", preview_samples_path / "txt" / "doc_2.txt"]
        )
        documents = output["documents"]
        assert len(documents) == 2
        assert "Some text for testing.\nTwo lines in here." in documents[0].text
        assert "This is a test line.\n123 456 789\n987 654 321" in documents[1].text

    @pytest.mark.integration
    def test_run_with_pdf_file(self, preview_samples_path):
        component = TikaDocumentConverter()
        output = component.run(
            paths=[preview_samples_path / "pdf" / "sample_pdf_1.pdf", preview_samples_path / "pdf" / "sample_pdf_2.pdf"]
        )
        documents = output["documents"]
        assert len(documents) == 2
        assert "A sample PDF file" in documents[0].text
        assert "Page 2 of Sample PDF" in documents[0].text
        assert "Page 4 of Sample PDF" in documents[0].text
        assert "First Page" in documents[1].text
        assert (
            "Wiki engines usually allow content to be written using a simplified markup language" in documents[1].text
        )
        assert "This section needs additional citations for verification." in documents[1].text
        assert "This would make it easier for other users to find the article." in documents[1].text

    @pytest.mark.integration
    def test_run_with_docx_file(self, preview_samples_path):
        component = TikaDocumentConverter()
        output = component.run(paths=[preview_samples_path / "docx" / "sample_docx.docx"])
        documents = output["documents"]
        assert len(documents) == 1
        assert "Sample Docx File" in documents[0].text
        assert "Now we are in Page 2" in documents[0].text
        assert "Page 3 was empty this is page 4" in documents[0].text
