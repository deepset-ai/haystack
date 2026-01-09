# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os

import pytest

from haystack.components.converters.azure import AzureDocumentIntelligenceConverter
from haystack.utils import Secret


class TestAzureDocumentIntelligenceConverter:
    def test_init(self):
        """Test basic initialization with defaults"""
        converter = AzureDocumentIntelligenceConverter(
            endpoint="https://test.cognitiveservices.azure.com/", api_key=Secret.from_token("test_api_key")
        )

        assert converter.endpoint == "https://test.cognitiveservices.azure.com/"
        assert converter.model_id == "prebuilt-read"
        assert converter.output_format == "markdown"
        assert converter.table_format == "markdown"
        assert converter.store_full_path is False

    def test_to_dict(self):
        """Test serialization with Secret handling"""
        converter = AzureDocumentIntelligenceConverter(
            endpoint="https://test.cognitiveservices.azure.com/",
            api_key=Secret.from_env_var("AZURE_AI_API_KEY"),
            model_id="prebuilt-layout",
            output_format="text",
            table_format="csv",
            store_full_path=True,
        )

        data = converter.to_dict()

        assert data == {
            "type": "haystack.components.converters.azure.AzureDocumentIntelligenceConverter",
            "init_parameters": {
                "api_key": {"type": "env_var", "env_vars": ["AZURE_AI_API_KEY"], "strict": True},
                "endpoint": "https://test.cognitiveservices.azure.com/",
                "model_id": "prebuilt-layout",
                "output_format": "text",
                "table_format": "csv",
                "store_full_path": True,
            },
        }

    def test_from_dict(self):
        """Test deserialization"""
        data = {
            "type": "haystack.components.converters.azure.AzureDocumentIntelligenceConverter",
            "init_parameters": {
                "api_key": {"type": "env_var", "env_vars": ["AZURE_AI_API_KEY"], "strict": True},
                "endpoint": "https://test.cognitiveservices.azure.com/",
                "model_id": "prebuilt-layout",
                "output_format": "markdown",
                "table_format": "markdown",
                "store_full_path": False,
            },
        }

        converter = AzureDocumentIntelligenceConverter.from_dict(data)

        assert converter.endpoint == "https://test.cognitiveservices.azure.com/"
        assert converter.model_id == "prebuilt-layout"
        assert converter.output_format == "markdown"

    @pytest.mark.integration
    @pytest.mark.skipif(not os.environ.get("AZURE_DI_ENDPOINT", None), reason="Azure endpoint not available")
    @pytest.mark.skipif(not os.environ.get("AZURE_AI_API_KEY", None), reason="Azure credentials not available")
    @pytest.mark.flaky(reruns=5, reruns_delay=5)
    def test_run_with_markdown_output(self, test_files_path):
        """Integration test with real Azure API - markdown mode"""
        converter = AzureDocumentIntelligenceConverter(
            endpoint=os.environ["AZURE_DI_ENDPOINT"],
            api_key=Secret.from_env_var("AZURE_AI_API_KEY"),
            output_format="markdown",
        )

        results = converter.run(sources=[test_files_path / "pdf" / "sample_pdf_1.pdf"])

        assert "documents" in results
        assert len(results["documents"]) == 1
        assert len(results["documents"][0].content) > 0
        assert results["documents"][0].meta["content_format"] == "markdown"
        assert "A sample PDF file" in results["documents"][0].content

    @pytest.mark.integration
    @pytest.mark.skipif(not os.environ.get("AZURE_DI_ENDPOINT", None), reason="Azure endpoint not available")
    @pytest.mark.skipif(not os.environ.get("AZURE_AI_API_KEY", None), reason="Azure credentials not available")
    @pytest.mark.flaky(reruns=5, reruns_delay=5)
    def test_run_with_text_output_csv_tables(self, test_files_path):
        """Integration test with real Azure API - text mode with CSV tables"""
        converter = AzureDocumentIntelligenceConverter(
            endpoint=os.environ["AZURE_DI_ENDPOINT"],
            api_key=Secret.from_env_var("AZURE_AI_API_KEY"),
            output_format="text",
            table_format="csv",
        )

        results = converter.run(sources=[test_files_path / "pdf" / "sample_pdf_1.pdf"])

        assert "documents" in results
        assert len(results["documents"]) >= 1

        # Check that we have text document
        text_docs = [d for d in results["documents"] if d.meta.get("content_format") == "text"]
        assert len(text_docs) == 1
        assert "A sample PDF file" in text_docs[0].content

    @pytest.mark.integration
    @pytest.mark.skipif(not os.environ.get("AZURE_DI_ENDPOINT", None), reason="Azure endpoint not available")
    @pytest.mark.skipif(not os.environ.get("AZURE_AI_API_KEY", None), reason="Azure credentials not available")
    @pytest.mark.flaky(reruns=5, reruns_delay=5)
    def test_run_with_metadata(self, test_files_path):
        """Integration test - verify metadata handling"""
        converter = AzureDocumentIntelligenceConverter(
            endpoint=os.environ["AZURE_DI_ENDPOINT"],
            api_key=Secret.from_env_var("AZURE_AI_API_KEY"),
            store_full_path=False,
        )

        results = converter.run(
            sources=[test_files_path / "pdf" / "sample_pdf_1.pdf"], meta={"custom_key": "custom_value"}
        )

        doc = results["documents"][0]
        assert doc.meta["custom_key"] == "custom_value"
        # Should be basename only
        assert doc.meta["file_path"] == "sample_pdf_1.pdf"

    @pytest.mark.integration
    @pytest.mark.skipif(not os.environ.get("AZURE_DI_ENDPOINT", None), reason="Azure endpoint not available")
    @pytest.mark.skipif(not os.environ.get("AZURE_AI_API_KEY", None), reason="Azure credentials not available")
    @pytest.mark.flaky(reruns=5, reruns_delay=5)
    def test_run_with_multiple_files(self, test_files_path):
        """Integration test - process multiple files"""
        converter = AzureDocumentIntelligenceConverter(
            endpoint=os.environ["AZURE_DI_ENDPOINT"], api_key=Secret.from_env_var("AZURE_AI_API_KEY")
        )

        results = converter.run(
            sources=[test_files_path / "pdf" / "sample_pdf_1.pdf", test_files_path / "pdf" / "sample_pdf_2.pdf"]
        )

        assert "documents" in results
        assert len(results["documents"]) == 2
        assert "A sample PDF file" in results["documents"][0].content
        assert "wiki" in results["documents"][1].content.lower()
