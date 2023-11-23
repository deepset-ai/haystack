from pathlib import Path
from typing import List, Union, Dict, Any, Optional
import os

from haystack.preview.lazy_imports import LazyImport
from haystack.preview import component, Document, default_to_dict


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

    @component.output_types(documents=List[Document], azure=List[Dict])
    def run(self, paths: List[Union[str, Path]]):
        """
        Convert files to Documents using Azure's Document Intelligence service.

        This component creates two outputs: `documents` and `raw_azure_response`. The `documents` output contains
        a list of Documents that were created from the files. The `raw_azure_response` output contains a list of
        the raw responses from Azure's Document Intelligence service.

        :param paths: Paths to the files to convert.
        """
        documents = []
        azure_output = []
        for path in paths:
            path = Path(path)
            with open(path, "rb") as file:
                poller = self.document_analysis_client.begin_analyze_document(model_id=self.model_id, document=file)
                result = poller.result()
                azure_output.append(result.to_dict())

            file_suffix = path.suffix
            document = AzureOCRDocumentConverter._convert_azure_result_to_document(result, file_suffix)
            documents.append(document)

        return {"documents": documents, "raw_azure_response": azure_output}

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize this component to a dictionary.
        """
        return default_to_dict(self, endpoint=self.endpoint, model_id=self.model_id)

    @staticmethod
    def _convert_azure_result_to_document(result: "AnalyzeResult", file_suffix: str) -> Document:
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
