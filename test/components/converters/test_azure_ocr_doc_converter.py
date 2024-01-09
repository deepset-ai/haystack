import os
from unittest.mock import patch, Mock

import pytest

from haystack.components.converters.azure import AzureOCRDocumentConverter
from haystack.dataclasses import ByteStream


class TestAzureOCRDocumentConverter:
    def test_init_fail_wo_api_key(self, monkeypatch):
        monkeypatch.delenv("AZURE_AI_API_KEY", raising=False)
        with pytest.raises(ValueError):
            AzureOCRDocumentConverter(endpoint="test_endpoint")

    def test_to_dict(self):
        component = AzureOCRDocumentConverter(endpoint="test_endpoint", api_key="test_credential_key")
        data = component.to_dict()
        assert data == {
            "type": "haystack.components.converters.azure.AzureOCRDocumentConverter",
            "init_parameters": {"endpoint": "test_endpoint", "model_id": "prebuilt-read"},
        }

    def test_run(self, test_files_path):
        with patch("haystack.components.converters.azure.DocumentAnalysisClient") as mock_azure_client:
            mock_result = Mock(pages=[Mock(lines=[Mock(content="mocked line 1"), Mock(content="mocked line 2")])])
            mock_result.to_dict.return_value = {
                "api_version": "2023-02-28-preview",
                "model_id": "prebuilt-read",
                "content": "mocked line 1\nmocked line 2\n\f",
                "pages": [{"lines": [{"content": "mocked line 1"}, {"content": "mocked line 2"}]}],
            }
            mock_azure_client.return_value.begin_analyze_document.return_value.result.return_value = mock_result

            component = AzureOCRDocumentConverter(endpoint="test_endpoint", api_key="test_credential_key")
            output = component.run(sources=[test_files_path / "pdf" / "sample_pdf_1.pdf"])
            document = output["documents"][0]
            assert document.content == "mocked line 1\nmocked line 2\n\f"
            assert "raw_azure_response" in output
            assert output["raw_azure_response"][0] == {
                "api_version": "2023-02-28-preview",
                "model_id": "prebuilt-read",
                "content": "mocked line 1\nmocked line 2\n\f",
                "pages": [{"lines": [{"content": "mocked line 1"}, {"content": "mocked line 2"}]}],
            }

    def test_run_with_meta(self, test_files_path):
        bytestream = ByteStream(data=b"test", meta={"author": "test_author", "language": "en"})
        with patch("haystack.components.converters.azure.DocumentAnalysisClient"):
            component = AzureOCRDocumentConverter(endpoint="test_endpoint", api_key="test_credential_key")

        output = component.run(
            sources=[bytestream, test_files_path / "pdf" / "sample_pdf_1.pdf"], meta={"language": "it"}
        )

        # check that the metadata from the bytestream is merged with that from the meta parameter
        assert output["documents"][0].meta["author"] == "test_author"
        assert output["documents"][0].meta["language"] == "it"
        assert output["documents"][1].meta["language"] == "it"

    @pytest.mark.integration
    @pytest.mark.skipif(not os.environ.get("CORE_AZURE_CS_ENDPOINT", None), reason="Azure credentials not available")
    @pytest.mark.skipif(not os.environ.get("CORE_AZURE_CS_API_KEY", None), reason="Azure credentials not available")
    def test_run_with_pdf_file(self, test_files_path):
        component = AzureOCRDocumentConverter(
            endpoint=os.environ["CORE_AZURE_CS_ENDPOINT"], api_key=os.environ["CORE_AZURE_CS_API_KEY"]
        )
        output = component.run(sources=[test_files_path / "pdf" / "sample_pdf_1.pdf"])
        documents = output["documents"]
        assert len(documents) == 1
        assert "A sample PDF file" in documents[0].content
        assert "Page 2 of Sample PDF" in documents[0].content
        assert "Page 4 of Sample PDF" in documents[0].content

    @pytest.mark.integration
    @pytest.mark.skipif(not os.environ.get("CORE_AZURE_CS_ENDPOINT", None), reason="Azure credentials not available")
    @pytest.mark.skipif(not os.environ.get("CORE_AZURE_CS_API_KEY", None), reason="Azure credentials not available")
    def test_with_image_file(self, test_files_path):
        component = AzureOCRDocumentConverter(
            endpoint=os.environ["CORE_AZURE_CS_ENDPOINT"], api_key=os.environ["CORE_AZURE_CS_API_KEY"]
        )
        output = component.run(sources=[test_files_path / "images" / "haystack-logo.png"])
        documents = output["documents"]
        assert len(documents) == 1
        assert "haystack" in documents[0].content
        assert "by deepset" in documents[0].content

    @pytest.mark.integration
    @pytest.mark.skipif(not os.environ.get("CORE_AZURE_CS_ENDPOINT", None), reason="Azure credentials not available")
    @pytest.mark.skipif(not os.environ.get("CORE_AZURE_CS_API_KEY", None), reason="Azure credentials not available")
    def test_run_with_docx_file(self, test_files_path):
        component = AzureOCRDocumentConverter(
            endpoint=os.environ["CORE_AZURE_CS_ENDPOINT"], api_key=os.environ["CORE_AZURE_CS_API_KEY"]
        )
        output = component.run(sources=[test_files_path / "docx" / "sample_docx.docx"])
        documents = output["documents"]
        assert len(documents) == 1
        assert "Sample Docx File" in documents[0].content
        assert "Now we are in Page 2" in documents[0].content
        assert "Page 3 was empty this is page 4" in documents[0].content
