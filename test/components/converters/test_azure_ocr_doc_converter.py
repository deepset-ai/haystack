# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501
import json
import os
import os.path
from typing import Literal
from unittest.mock import patch

import pandas as pd
import pytest
from azure.ai.formrecognizer import AnalyzeResult

from haystack.components.converters.azure import AzureOCRDocumentConverter
from haystack.dataclasses.byte_stream import ByteStream
from haystack.utils import Secret


def get_sample_pdf_1_text(page_layout: Literal["natural", "single_column"]) -> str:
    if page_layout == "natural":
        return (
            "A sample PDF file\nHistory and standardization\nFormat (PDF) Adobe Systems made the PDF specification "
            "available free of charge in 1993. In the early years PDF was popular mainly in desktop publishing "
            "workflows, and competed with a variety of formats such as DjVu, Envoy, Common Ground Digital Paper, "
            "Farallon Replica and even Adobe's own PostScript format. PDF was a proprietary format controlled by "
            "Adobe until it was released as an open standard on July 1, 2008, and published by the International "
            "Organization for Standardization as ISO 32000-1:2008, at which time control of the specification "
            "passed to an ISO Committee of volunteer industry experts. In 2008, Adobe published a Public Patent "
            "License to ISO 32000-1 granting royalty-free rights for all patents owned by Adobe that are necessary "
            "to make, use, sell, and distribute PDF-compliant implementations. PDF 1.7, the sixth edition of the PDF "
            "specification that became ISO 32000-1, includes some proprietary technologies defined only by Adobe, "
            "such as Adobe XML Forms Architecture (XFA) and JavaScript extension for Acrobat, which are referenced "
            "by ISO 32000-1 as normative and indispensable for the full implementation of the ISO 32000-1 "
            "specification. These proprietary technologies are not standardized and their specification is published "
            "only on Adobe's website. Many of them are also not supported by popular third-party implementations of "
            "PDF.\n\x0cPage 2 of Sample PDF\n\x0c\x0cPage 4 of Sample PDF\n... the page 3 is empty.\n"
        )
    else:
        return (
            "A sample PDF file\nHistory and standardization\nFormat (PDF) Adobe Systems made the PDF specification "
            "available free of\ncharge in 1993. In the early years PDF was popular mainly in desktop\npublishing "
            "workflows, and competed with a variety of formats such as DjVu,\nEnvoy, Common Ground Digital Paper, "
            "Farallon Replica and even Adobe's\nown PostScript format. PDF was a proprietary format controlled by "
            "Adobe\nuntil it was released as an open standard on July 1, 2008, and published by\nthe International "
            "Organization for Standardization as ISO 32000-1:2008, at\nwhich time control of the specification passed "
            "to an ISO Committee of\nvolunteer industry experts. In 2008, Adobe published a Public Patent License\nto "
            "ISO 32000-1 granting royalty-free rights for all patents owned by Adobe\nthat are necessary to make, use, "
            "sell, and distribute PDF-compliant\nimplementations. PDF 1.7, the sixth edition of the PDF specification "
            "that\nbecame ISO 32000-1, includes some proprietary technologies defined only by\nAdobe, such as Adobe "
            "XML Forms Architecture (XFA) and JavaScript\nextension for Acrobat, which are referenced by ISO 32000-1 "
            "as normative\nand indispensable for the full implementation of the ISO 32000-1\nspecification. These "
            "proprietary technologies are not standardized and their\nspecification is published only on Adobe's "
            "website. Many of them are also not\nsupported by popular third-party implementations of PDF.\n\x0cPage 2 "
            "of Sample PDF\n\x0c\x0cPage 4 of Sample PDF\n... the page 3 is empty.\n"
        )


def get_sample_pdf_2_text(page_layout: Literal["natural", "single_column"]) -> str:
    if page_layout == "natural":
        return (
            "A Simple PDF File\nThis is a small demonstration .pdf file -\njust for use in the Virtual Mechanics "
            "tutorials. More text. And more text. And more text. And more text. And more text.\nAnd more text. And more "
            "text. And more text. And more text. And more text. And more text. Boring, zzzzz. And more text. And more "
            "text. And more text. And more text. And more text. And more text. And more text. And more text. And more "
            "text.\nAnd more text. And more text. And more text. And more text. And more text. And more text. And more "
            "text. Even more. Continued on page 2 ...\n\x0cSimple PDF File 2\n... continued from page 1. Yet more text. "
            "And more text. And more text. And more text. And more text. And more text. And more text. And more text. "
            "Oh, how boring typing this stuff. But not as boring as watching paint dry. And more text. And more text. "
            "And more text. And more text. Boring. More, a little more text. The end, and just as well.\n"
        )
    else:
        return (
            "A Simple PDF File\nThis is a small demonstration .pdf file -\njust for use in the Virtual Mechanics "
            "tutorials. More text. And more\ntext. And more text. And more text. And more text.\nAnd more text. And "
            "more text. And more text. And more text. And more\ntext. And more text. Boring, zzzzz. And more text. "
            "And more text. And\nmore text. And more text. And more text. And more text. And more text.\nAnd more text. "
            "And more text.\nAnd more text. And more text. And more text. And more text. And more\ntext. And more text. "
            "And more text. Even more. Continued on page 2 ...\n\x0cSimple PDF File 2\n... continued from page 1. "
            "Yet more text. And more text. And more text.\nAnd more text. And more text. And more text. And more text. "
            "And more\ntext. Oh, how boring typing this stuff. But not as boring as watching\npaint dry. And more text. "
            "And more text. And more text. And more text.\nBoring. More, a little more text. The end, and just as well.\n"
        )


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
                "following_context_len": 3,
                "merge_multiple_column_headers": True,
                "model_id": "prebuilt-read",
                "page_layout": "natural",
                "preceding_context_len": 3,
                "threshold_y": 0.05,
                "store_full_path": False,
            },
        }

    @patch("haystack.utils.auth.EnvVarSecret.resolve_value")
    def test_azure_converter_with_pdf(self, mock_resolve_value, test_files_path) -> None:
        mock_resolve_value.return_value = "test_api_key"

        class MockPoller:
            def result(self) -> AnalyzeResult:
                with open(test_files_path / "json" / "azure_sample_pdf_2.json", encoding="utf-8") as azure_file:
                    result = json.load(azure_file)
                return AnalyzeResult.from_dict(result)

        with patch("azure.ai.formrecognizer.DocumentAnalysisClient.begin_analyze_document") as azure_mock:
            azure_mock.return_value = MockPoller()
            ocr_node = AzureOCRDocumentConverter(endpoint="")
            out = ocr_node.run(sources=[test_files_path / "pdf" / "sample_pdf_2.pdf"])
        assert len(out["documents"]) == 1
        assert out["documents"][0].content == get_sample_pdf_2_text(page_layout="natural")
        assert out["documents"][0].content.count("\f") == 1

    @pytest.mark.parametrize("page_layout", ["natural", "single_column"])
    @patch("haystack.utils.auth.EnvVarSecret.resolve_value")
    def test_azure_converter_with_table(
        self, mock_resolve_value, page_layout: Literal["natural", "single_column"], test_files_path
    ) -> None:
        mock_resolve_value.return_value = "test_api_key"

        class MockPoller:
            def result(self) -> AnalyzeResult:
                with open(test_files_path / "json" / "azure_sample_pdf_1.json", encoding="utf-8") as azure_file:
                    result = json.load(azure_file)
                return AnalyzeResult.from_dict(result)

        with patch("azure.ai.formrecognizer.DocumentAnalysisClient.begin_analyze_document") as azure_mock:
            azure_mock.return_value = MockPoller()
            ocr_node = AzureOCRDocumentConverter(endpoint="", page_layout=page_layout)
            out = ocr_node.run(sources=[test_files_path / "pdf" / "sample_pdf_1.pdf"])

        docs = out["documents"]
        assert len(docs) == 2
        # Checking the table doc extracted
        assert docs[0].content_type == "table"
        assert docs[0].dataframe.shape[0] == 4  # number of rows
        assert docs[0].dataframe.shape[1] == 4  # number of columns
        assert list(docs[0].dataframe.columns) == ["", "Column 1", "Column 2", "Column 3"]
        assert list(docs[0].dataframe.iloc[3]) == ["D", "$54.35", "$6345.", ""]
        assert (
            docs[0].meta["preceding_context"] == "specification. These proprietary technologies are not "
            "standardized and their\nspecification is published only on "
            "Adobe's website. Many of them are also not\nsupported by "
            "popular third-party implementations of PDF."
        )
        assert docs[0].meta["following_context"] == ""
        assert docs[0].meta["page"] == 1

        # Checking the text extracted
        assert docs[1].content_type == "text"
        assert docs[1].content.startswith("A sample PDF file")
        assert docs[1].content.count("\f") == 3  # There should be three page separations
        pages = docs[1].content.split("\f")
        gold_pages = get_sample_pdf_1_text(page_layout=page_layout).split("\f")
        assert pages[0] == gold_pages[0]
        assert pages[1] == gold_pages[1]
        assert pages[2] == gold_pages[2]
        assert pages[3] == gold_pages[3]

    @patch("haystack.utils.auth.EnvVarSecret.resolve_value")
    def test_azure_converter_with_table_no_bounding_region(self, mock_resolve_value, test_files_path) -> None:
        mock_resolve_value.return_value = "test_api_key"

        class MockPoller:
            def result(self) -> AnalyzeResult:
                with open(test_files_path / "json" / "azure_sample_pdf_1.json", encoding="utf-8") as azure_file:
                    result = json.load(azure_file)
                return AnalyzeResult.from_dict(result)

        with patch("azure.ai.formrecognizer.DocumentAnalysisClient.begin_analyze_document") as azure_mock:
            azure_mock.return_value = MockPoller()
            ocr_node = AzureOCRDocumentConverter(endpoint="")
            out = ocr_node.run(sources=[test_files_path / "pdf" / "sample_pdf_1.pdf"])

        docs = out["documents"]
        assert len(docs) == 2
        # Checking the table doc extracted that is missing bounding info
        assert docs[0].content_type == "table"
        assert docs[0].dataframe.shape[0] == 4  # number of rows
        assert docs[0].dataframe.shape[1] == 4  # number of columns
        assert list(docs[0].dataframe.columns) == ["", "Column 1", "Column 2", "Column 3"]
        assert list(docs[0].dataframe.iloc[3]) == ["D", "$54.35", "$6345.", ""]
        # TODO below assert fails
        # assert docs[0].meta["preceding_context"] == ""

    @patch("haystack.utils.auth.EnvVarSecret.resolve_value")
    def test_azure_converter_with_multicolumn_header_table(self, mock_resolve_value, test_files_path) -> None:
        mock_resolve_value.return_value = "test_api_key"

        class MockPoller:
            def result(self) -> AnalyzeResult:
                with open(test_files_path / "json" / "azure_sample_pdf_3.json", encoding="utf-8") as azure_file:
                    result = json.load(azure_file)
                return AnalyzeResult.from_dict(result)

        with patch("azure.ai.formrecognizer.DocumentAnalysisClient.begin_analyze_document") as azure_mock:
            azure_mock.return_value = MockPoller()
            ocr_node = AzureOCRDocumentConverter(endpoint="")

            # TODO: fails because of non-unique column names, azure_sample_pdf_3.json has duplicate column names
            out = ocr_node.run(sources=[test_files_path / "pdf" / "sample_pdf_3.pdf"])

        docs = out["documents"]
        assert len(docs) == 2
        assert docs[0].content_type == "table"
        assert docs[0].dataframe.shape[0] == 1  # number of rows
        assert docs[0].dataframe.shape[1] == 3  # number of columns
        assert list(docs[0].dataframe.columns) == ["This is a subheader", "This is a subheader", "This is a subheader"]
        assert list(docs[0].dataframe.iloc[0]) == ["Value 1", "Value 2", "Val 3"]
        assert (
            docs[0].meta["preceding_context"]
            == "Table 1. This is an example table with two multicolumn headers\nHeader 1"
        )

    @patch("haystack.utils.auth.EnvVarSecret.resolve_value")
    def test_table_pdf_with_non_empty_meta(self, mock_resolve_value, test_files_path) -> None:
        mock_resolve_value.return_value = "test_api_key"

        class MockPoller:
            def result(self) -> AnalyzeResult:
                with open(test_files_path / "json" / "azure_sample_pdf_1.json", encoding="utf-8") as azure_file:
                    result = json.load(azure_file)
                return AnalyzeResult.from_dict(result)

        with patch("azure.ai.formrecognizer.DocumentAnalysisClient.begin_analyze_document") as azure_mock:
            azure_mock.return_value = MockPoller()
            ocr_node = AzureOCRDocumentConverter(endpoint="")
            out = ocr_node.run(sources=[test_files_path / "pdf" / "sample_pdf_1.pdf"], meta=[{"test": "value_1"}])

        docs = out["documents"]
        # TODO assert below changed from the original test
        assert docs[1].meta["test"] == "value_1"

    @pytest.mark.integration
    @pytest.mark.skipif(not os.environ.get("CORE_AZURE_CS_ENDPOINT", None), reason="Azure endpoint not available")
    @pytest.mark.skipif(not os.environ.get("CORE_AZURE_CS_API_KEY", None), reason="Azure credentials not available")
    @pytest.mark.flaky(reruns=5, reruns_delay=5)
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
        """
        Test if the component runs correctly with store_full_path=False
        """
        component = AzureOCRDocumentConverter(
            endpoint=os.environ["CORE_AZURE_CS_ENDPOINT"], api_key=Secret.from_env_var("CORE_AZURE_CS_API_KEY")
        )
        output = component.run(sources=[test_files_path / "docx" / "sample_docx.docx"])
        documents = output["documents"]
        assert len(documents) == 1
        assert "Sample Docx File" in documents[0].content
        assert "Now we are in Page 2" in documents[0].content
        assert "Page 3 was empty this is page 4" in documents[0].content

    @pytest.mark.integration
    @pytest.mark.skipif(not os.environ.get("CORE_AZURE_CS_ENDPOINT", None), reason="Azure endpoint not available")
    @pytest.mark.skipif(not os.environ.get("CORE_AZURE_CS_API_KEY", None), reason="Azure credentials not available")
    def test_run_with_store_full_path_false(self, test_files_path):
        component = AzureOCRDocumentConverter(
            endpoint=os.environ["CORE_AZURE_CS_ENDPOINT"],
            api_key=Secret.from_env_var("CORE_AZURE_CS_API_KEY"),
            store_full_path=False,
        )
        output = component.run(sources=[test_files_path / "docx" / "sample_docx.docx"])
        documents = output["documents"]
        assert len(documents) == 1
        assert "Sample Docx File" in documents[0].content
        assert documents[0].meta["file_path"] == "sample_docx.docx"

    @patch("haystack.utils.auth.EnvVarSecret.resolve_value")
    def test_hashing_dataframe(self, mock_resolve_value):
        mock_resolve_value.return_value = "test_api_key"
        component = AzureOCRDocumentConverter(endpoint="")
        hash_length = 32

        df = pd.DataFrame({"A": [1, 2, 3]})
        hash_string_1 = component._hash_dataframe(df)
        assert len(hash_string_1) == hash_length

        df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        hash_string_2 = component._hash_dataframe(df)
        assert len(hash_string_2) == hash_length

        df = pd.DataFrame({"B": [4, 5, 6], "A": [1, 2, 3], "D": [7, 8, 9]})
        hash_string_3 = component._hash_dataframe(df)
        assert len(hash_string_3) == hash_length

        # doesn't mean much, more for sanity check
        assert hash_string_1 != hash_string_2 != hash_string_3

    @patch("haystack.utils.auth.EnvVarSecret.resolve_value")
    def test_meta_from_byte_stream(self, mock_resolve_value, test_files_path) -> None:
        mock_resolve_value.return_value = "test_api_key"

        class MockPoller:
            def result(self) -> AnalyzeResult:
                with open(test_files_path / "json" / "azure_sample_pdf_1.json", encoding="utf-8") as azure_file:
                    result = json.load(azure_file)
                return AnalyzeResult.from_dict(result)

        with patch("azure.ai.formrecognizer.DocumentAnalysisClient.begin_analyze_document") as azure_mock:
            azure_mock.return_value = MockPoller()
            ocr_node = AzureOCRDocumentConverter(endpoint="")
            bytes = (test_files_path / "pdf" / "sample_pdf_1.pdf").read_bytes()
            byte_stream = ByteStream(data=bytes, meta={"test_from": "byte_stream"})
            out = ocr_node.run(sources=[byte_stream], meta=[{"test": "value_1"}])

        docs = out["documents"]
        assert docs[1].meta["test"] == "value_1"
        assert docs[1].meta["test_from"] == "byte_stream"
