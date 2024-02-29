from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from haystack import Document, component, default_from_dict, default_to_dict, logging
from haystack.components.converters.utils import get_bytestream_from_source, normalize_metadata
from haystack.dataclasses import ByteStream
from haystack.lazy_imports import LazyImport
from haystack.utils import Secret, deserialize_secrets_inplace

logger = logging.getLogger(__name__)

with LazyImport(message="Run 'pip install \"azure-ai-formrecognizer>=3.2.0b2\"'") as azure_import:
    from azure.ai.formrecognizer import AnalyzeResult, DocumentAnalysisClient
    from azure.core.credentials import AzureKeyCredential


@component
class AzureOCRDocumentConverter:
    """
    A component for converting files to Documents using Azure's Document Intelligence service.
    Supported file formats are: PDF, JPEG, PNG, BMP, TIFF, DOCX, XLSX, PPTX, and HTML.

    In order to be able to use this component, you need an active Azure account
    and a Document Intelligence or Cognitive Services resource. Follow the steps described in the
    [Azure documentation](https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/quickstarts/get-started-sdks-rest-api)
    to set up your resource.

    Usage example:
    ```python
    from haystack.components.converters import AzureOCRDocumentConverter
    from haystack.utils import Secret

    converter = AzureOCRDocumentConverter(endpoint="<url>", api_key=Secret.from_token("<your-api-key>"))
    results = converter.run(sources=["path/to/document_with_images.pdf"], meta={"date_added": datetime.now().isoformat()})
    documents = results["documents"]
    print(documents[0].content)
    # 'This is a text from the PDF file.'
    ```
    """

    def __init__(
        self, endpoint: str, api_key: Secret = Secret.from_env_var("AZURE_AI_API_KEY"), model_id: str = "prebuilt-read"
    ):
        """
        Create an AzureOCRDocumentConverter component.

        :param endpoint:
            The endpoint of your Azure resource.
        :param api_key:
            The key of your Azure resource.
        :param model_id:
            The model ID of the model you want to use. Please refer to [Azure documentation](https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/choose-model-feature)
            for a list of available models. Default: `"prebuilt-read"`.
        """
        azure_import.check()

        self.document_analysis_client = DocumentAnalysisClient(endpoint=endpoint, credential=AzureKeyCredential(api_key.resolve_value()))  # type: ignore
        self.endpoint = endpoint
        self.model_id = model_id
        self.api_key = api_key

    @component.output_types(documents=List[Document], raw_azure_response=List[Dict])
    def run(self, sources: List[Union[str, Path, ByteStream]], meta: Optional[List[Dict[str, Any]]] = None):
        """
        Convert a list of files to Documents using Azure's Document Intelligence service.

        :param sources:
            List of file paths or ByteStream objects.
        :param meta:
            Optional metadata to attach to the Documents.
            This value can be either a list of dictionaries or a single dictionary.
            If it's a single dictionary, its content is added to the metadata of all produced Documents.
            If it's a list, the length of the list must match the number of sources, because the two lists will be zipped.
            If `sources` contains ByteStream objects, their `meta` will be added to the output Documents.

        :returns:
            A dictionary with the following keys:
            - `documents`: List of created Documents
            - `raw_azure_response`: List of raw Azure responses used to create the Documents
        """
        documents = []
        azure_output = []
        meta_list = normalize_metadata(meta=meta, sources_count=len(sources))

        for source, metadata in zip(sources, meta_list):
            try:
                bytestream = get_bytestream_from_source(source=source)
            except Exception as e:
                logger.warning("Could not read {source}. Skipping it. Error: {error}", source=source, error=e)
                continue

            poller = self.document_analysis_client.begin_analyze_document(
                model_id=self.model_id, document=bytestream.data
            )
            result = poller.result()
            azure_output.append(result.to_dict())

            file_suffix = None
            if "file_path" in bytestream.meta:
                file_suffix = Path(bytestream.meta["file_path"]).suffix

            document = AzureOCRDocumentConverter._convert_azure_result_to_document(result, file_suffix)
            merged_metadata = {**bytestream.meta, **metadata}
            document.meta = merged_metadata
            documents.append(document)

        return {"documents": documents, "raw_azure_response": azure_output}

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(self, api_key=self.api_key.to_dict(), endpoint=self.endpoint, model_id=self.model_id)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AzureOCRDocumentConverter":
        """
        Deserializes the component from a dictionary.

        :param data:
            The dictionary to deserialize from.
        :returns:
            The deserialized component.
        """
        deserialize_secrets_inplace(data["init_parameters"], keys=["api_key"])
        return default_from_dict(cls, data)

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
