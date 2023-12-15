from pathlib import Path
from typing import List, Union, Dict, Any, Optional
import os
import logging

from haystack.lazy_imports import LazyImport
from haystack import component, Document, default_to_dict
from haystack.dataclasses import ByteStream
from haystack.components.converters.utils import get_bytestream_from_source

logger = logging.getLogger(__name__)

with LazyImport(message="Run 'pip install \"azure-ai-formrecognizer>=3.2.0b2\"'") as azure_import:
    from azure.ai.formrecognizer import DocumentAnalysisClient, AnalyzeResult
    from azure.core.credentials import AzureKeyCredential


@component
class AzureOCRDocumentConverter:
    """
    A component for converting files to Documents using Azure's Document Intelligence service.
    Supported file formats are: PDF, JPEG, PNG, BMP, TIFF, DOCX, XLSX, PPTX, and HTML.

    In order to be able to use this component, you need an active Azure account
    and a Document Intelligence or Cognitive Services resource. Please follow the steps described in the
    [Azure documentation](https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/quickstarts/get-started-sdks-rest-api)
    to set up your resource.

    Usage example:
    ```python
    from haystack.components.converters.azure import AzureOCRDocumentConverter

    converter = AzureOCRDocumentConverter()
    results = converter.run(sources=["image-based-document.pdf"])
    documents = results["documents"]
    print(documents[0].content)
    # 'This is a text from the PDF file.'
    ```
    """

    def __init__(self, endpoint: str, api_key: Optional[str] = None, model_id: str = "prebuilt-read"):
        """
        Create an AzureOCRDocumentConverter component.

        :param endpoint: The endpoint of your Azure resource.
        :param api_key: The key of your Azure resource. It can be
        explicitly provided or automatically read from the
        environment variable AZURE_AI_API_KEY (recommended).
        :param model_id: The model ID of the model you want to use. Please refer to [Azure documentation](https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/choose-model-feature)
            for a list of available models. Default: `"prebuilt-read"`.
        """
        azure_import.check()

        if api_key is None:
            try:
                api_key = os.environ["AZURE_AI_API_KEY"]
            except KeyError as e:
                raise ValueError(
                    "AzureOCRDocumentConverter expects an Azure Credential key. "
                    "Set the AZURE_AI_API_KEY environment variable (recommended) or pass it explicitly."
                ) from e

        self.api_key = api_key
        self.document_analysis_client = DocumentAnalysisClient(
            endpoint=endpoint, credential=AzureKeyCredential(api_key)
        )
        self.endpoint = endpoint
        self.model_id = model_id

    @component.output_types(documents=List[Document], raw_azure_response=List[Dict])
    def run(self, sources: List[Union[str, Path, ByteStream]], meta: Optional[List[Dict[str, Any]]] = None):
        """
        Convert files to Documents using Azure's Document Intelligence service.

        This component creates two outputs: `documents` and `raw_azure_response`. The `documents` output contains
        a list of Documents that were created from the files. The `raw_azure_response` output contains a list of
        the raw responses from Azure's Document Intelligence service.

        :param sources: List of file paths or ByteStream objects.
        :param meta: Optional list of metadata to attach to the Documents.
          The length of the list must match the number of sources. Defaults to `None`.
        :return: A dictionary containing a list of Document objects under the 'documents' key
          and the raw Azure response under the 'raw_azure_response' key.
        """
        documents = []
        azure_output = []

        if meta is None:
            meta = [{}] * len(sources)
        elif len(sources) != len(meta):
            raise ValueError("The length of the metadata list must match the number of sources.")

        for source, metadata in zip(sources, meta):
            try:
                bytestream = get_bytestream_from_source(source=source)
            except Exception as e:
                logger.warning("Could not read %s. Skipping it. Error: %s", source, e)
                continue

            poller = self.document_analysis_client.begin_analyze_document(
                model_id=self.model_id, document=bytestream.data
            )
            result = poller.result()
            azure_output.append(result.to_dict())

            file_suffix = None
            if "file_path" in bytestream.metadata:
                file_suffix = Path(bytestream.metadata["file_path"]).suffix

            document = AzureOCRDocumentConverter._convert_azure_result_to_document(result, file_suffix)
            merged_metadata = {**bytestream.metadata, **metadata}
            document.meta = merged_metadata
            documents.append(document)

        return {"documents": documents, "raw_azure_response": azure_output}

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize this component to a dictionary.
        """
        return default_to_dict(self, endpoint=self.endpoint, model_id=self.model_id)

    @staticmethod
    def _convert_azure_result_to_document(result: "AnalyzeResult", file_suffix: Optional[str] = None) -> Document:
        """
        Convert the result of Azure OCR to a Haystack text Document.
        """
        if file_suffix == ".pdf":
            text = ""
            for page in result.pages:
                lines = page.lines if page.lines else []
                for line in lines:
                    text += f"{line.content}\n"

                text += "\f"
        else:
            text = result.content

        document = Document(content=text)

        return document
