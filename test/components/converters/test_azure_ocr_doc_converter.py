import os
from unittest.mock import Mock, patch

import pytest

from haystack.components.converters.azure import AzureOCRDocumentConverter
from haystack.dataclasses import ByteStream
from haystack.utils import Secret


class TestAzureOCRDocumentConverter:
    def test_init_fail_wo_api_key(self, monkeypatch):
        monkeypatch.delenv("AZURE_AI_API_KEY", raising=False)
        with pytest.raises(ValueError):
            AzureOCRDocumentConverter(endpoint="test_endpoint")

    @patch("haystack.utils.auth.EnvVarSecret.resolve_value")
    def test_to_dict(self, mock_resolve_value):
        mock_resolve_value.return_value = "test_api_key"
        component = AzureOCRDocumentConverter(endpoint="test_endpoint")
        data = component.to_dict()
        assert data == {
            "type": "haystack.components.converters.azure.AzureOCRDocumentConverter",
            "init_parameters": {
                "api_key": {"env_vars": ["AZURE_AI_API_KEY"], "strict": True, "type": "env_var"},
                "endpoint": "test_endpoint",
                "model_id": "prebuilt-read",
            },
        }

    @pytest.mark.integration
    @pytest.mark.skipif(not os.environ.get("CORE_AZURE_CS_ENDPOINT", None), reason="Azure endpoint not available")
    @pytest.mark.skipif(not os.environ.get("CORE_AZURE_CS_API_KEY", None), reason="Azure credentials not available")
    def test_run_with_pdf_file(self, test_files_path):
        component = AzureOCRDocumentConverter(
            endpoint=os.environ["CORE_AZURE_CS_ENDPOINT"], api_key=Secret.from_env_var("CORE_AZURE_CS_API_KEY")
        )
        output = component.run(sources=[test_files_path / "pdf" / "sample_pdf_1.pdf"])
        documents = output["documents"]
        assert len(documents) == 1
        assert "A sample PDF file" in documents[0].content
        assert "Page 2 of Sample PDF" in documents[0].content
        assert "Page 4 of Sample PDF" in documents[0].content

    @pytest.mark.integration
    @pytest.mark.skipif(not os.environ.get("CORE_AZURE_CS_ENDPOINT", None), reason="Azure endpoint not available")
    @pytest.mark.skipif(not os.environ.get("CORE_AZURE_CS_API_KEY", None), reason="Azure credentials not available")
    def test_with_image_file(self, test_files_path):
        component = AzureOCRDocumentConverter(
            endpoint=os.environ["CORE_AZURE_CS_ENDPOINT"], api_key=Secret.from_env_var("CORE_AZURE_CS_API_KEY")
        )
        output = component.run(sources=[test_files_path / "images" / "haystack-logo.png"])
        documents = output["documents"]
        assert len(documents) == 1
        assert "haystack" in documents[0].content
        assert "by deepset" in documents[0].content

    @pytest.mark.integration
    @pytest.mark.skipif(not os.environ.get("CORE_AZURE_CS_ENDPOINT", None), reason="Azure endpoint not available")
    @pytest.mark.skipif(not os.environ.get("CORE_AZURE_CS_API_KEY", None), reason="Azure credentials not available")
    def test_run_with_docx_file(self, test_files_path):
        component = AzureOCRDocumentConverter(
            endpoint=os.environ["CORE_AZURE_CS_ENDPOINT"], api_key=Secret.from_env_var("CORE_AZURE_CS_API_KEY")
        )
        output = component.run(sources=[test_files_path / "docx" / "sample_docx.docx"])
        documents = output["documents"]
        assert len(documents) == 1
        assert "Sample Docx File" in documents[0].content
        assert "Now we are in Page 2" in documents[0].content
        assert "Page 3 was empty this is page 4" in documents[0].content
