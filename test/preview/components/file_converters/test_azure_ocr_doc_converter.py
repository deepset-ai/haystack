import os
from unittest.mock import patch, Mock

import pytest

from haystack.preview.components.file_converters.azure import AzureOCRDocumentConverter


class TestAzureOCRDocumentConverter:
    @pytest.mark.unit
    def test_to_dict(self):
        component = AzureOCRDocumentConverter(endpoint="test_endpoint", api_key="test_credential_key")
        data = component.to_dict()
        assert data == {
            "type": "AzureOCRDocumentConverter",
            "init_parameters": {
                "api_key": "test_credential_key",
                "endpoint": "test_endpoint",
                "id_hash_keys": [],
                "model_id": "prebuilt-read",
                "save_json": False,
            },
        }

    @pytest.mark.unit
    def test_from_dict(self):
        data = {
            "type": "AzureOCRDocumentConverter",
            "init_parameters": {
                "api_key": "test_credential_key",
                "endpoint": "test_endpoint",
                "id_hash_keys": [],
                "model_id": "prebuilt-read",
                "save_json": False,
            },
        }
        component = AzureOCRDocumentConverter.from_dict(data)
        assert component.endpoint == "test_endpoint"
        assert component.api_key == "test_credential_key"
        assert component.id_hash_keys == []
        assert component.model_id == "prebuilt-read"
        assert component.save_json is False

    @pytest.mark.unit
    def test_run(self, preview_samples_path):
        with patch("haystack.preview.components.file_converters.azure.DocumentAnalysisClient") as mock_azure_client:
            mock_result = Mock(pages=[Mock(lines=[Mock(content="mocked line 1"), Mock(content="mocked line 2")])])
            mock_azure_client.return_value.begin_analyze_document.return_value.result.return_value = mock_result

            component = AzureOCRDocumentConverter(endpoint="test_endpoint", api_key="test_credential_key")
            output = component.run(paths=[preview_samples_path / "pdf" / "sample_pdf_1.pdf"])
            document = output["documents"][0]
            assert document.text == "mocked line 1\nmocked line 2\n\f"

    @pytest.mark.integration
    @pytest.mark.skipif(
        "AZURE_FORMRECOGNIZER_ENDPOINT" not in os.environ and "AZURE_FORMRECOGNIZER_KEY" not in os.environ,
        reason="Azure credentials not available",
    )
    def test_run_with_pdf_file(self, preview_samples_path):
        component = AzureOCRDocumentConverter(
            endpoint=os.environ["AZURE_FORMRECOGNIZER_ENDPOINT"], api_key=os.environ["AZURE_FORMRECOGNIZER_KEY"]
        )
        output = component.run(paths=[preview_samples_path / "pdf" / "sample_pdf_1.pdf"])
        documents = output["documents"]
        assert len(documents) == 1
        assert "A sample PDF file" in documents[0].text
        assert "Page 2 of Sample PDF" in documents[0].text
        # Azure free tier limits the extraction to 2 pages
        # assert "Page 4 of Sample PDF" in documents[0].text

    @pytest.mark.integration
    @pytest.mark.skipif(
        "AZURE_FORMRECOGNIZER_ENDPOINT" not in os.environ and "AZURE_FORMRECOGNIZER_KEY" not in os.environ,
        reason="Azure credentials not available",
    )
    def test_with_image_file(self, preview_samples_path):
        component = AzureOCRDocumentConverter(
            endpoint=os.environ["AZURE_FORMRECOGNIZER_ENDPOINT"], api_key=os.environ["AZURE_FORMRECOGNIZER_KEY"]
        )
        output = component.run(paths=[preview_samples_path / "images" / "haystack-logo.png"])
        documents = output["documents"]
        assert len(documents) == 1
        assert "haystack\nby deepset" in documents[0].text

    @pytest.mark.integration
    @pytest.mark.skipif(
        "AZURE_FORMRECOGNIZER_ENDPOINT" not in os.environ and "AZURE_FORMRECOGNIZER_KEY" not in os.environ,
        reason="Azure credentials not available",
    )
    def test_run_with_docx_file(self, preview_samples_path):
        component = AzureOCRDocumentConverter(
            endpoint=os.environ["AZURE_FORMRECOGNIZER_ENDPOINT"], api_key=os.environ["AZURE_FORMRECOGNIZER_KEY"]
        )
        output = component.run(paths=[preview_samples_path / "docx" / "sample_docx.docx"])
        documents = output["documents"]
        assert len(documents) == 1
        assert "Sample Docx File" in documents[0].text
        assert "Now we are in Page 2" in documents[0].text
        assert "Page 3 was empty this is page 4" in documents[0].text
