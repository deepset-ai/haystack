import logging
import pytest

from haystack.components.converters import JSONToDocument


class TestHTMLToDocument:
    def test_run(self, test_files_path):
        """
        Test if the component runs correctly.
        """
        example_file_path = test_files_path / "json" / "azure_sample_pdf_1.json"
        sources = [example_file_path]
        converter = JSONToDocument(jq_schema=".api_version")
        results = converter.run(sources=sources)
        docs = results["documents"]
        assert len(docs) == 1
        assert "2023-02-28-preview" == docs[0].content
        assert example_file_path == docs[0].meta["source"]
        assert 1 == docs[0].meta["row_idx"]

    def test_run_wrong_file_type(self, test_files_path, caplog):
        """
        Test if the component runs correctly when an input file is not of the expected type.
        """
        sources = [test_files_path / "audio" / "answer.wav"]
        converter = JSONToDocument(jq_schema=".api_version")
        with caplog.at_level(logging.WARNING):
            results = converter.run(sources=sources)
            assert "Could not read" in caplog.text

        assert results["documents"] == []

    def test_run_wrong_json_load(self, test_files_path, caplog):
        """
        Test if the component runs correctly when there is no JSON content within an input file.
        """
        sources = [test_files_path / "txt" / "doc_1.txt"]
        converter = JSONToDocument(jq_schema=".api_version")
        with caplog.at_level(logging.WARNING):
            results = converter.run(sources=sources)
            assert "Failed to extract JSON content from" in caplog.text

        assert results["documents"] == []
