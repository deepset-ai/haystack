from typing import List, Optional, Dict, Any
import os

from haystack.preview import component, default_to_dict, default_from_dict
from haystack.preview.lazy_imports import LazyImport

with LazyImport(message="Run 'pip install cohere'") as cohere_import:
    from cohere import Client, AsyncClient, CohereError


API_BASE_URL = "https://api.cohere.ai/v1/embed"


@component
class CohereTextEmbedder:
    """
    A component for embedding strings using Cohere models.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "embed-english-v2.0",
        api_base_url: str = API_BASE_URL,
        truncate: str = "END",
        use_async_client: bool = False,
        max_retries: Optional[int] = 3,
        timeout: Optional[int] = 120,
    ):
        """
        Create a CohereTextEmbedder component.

        :param api_key: The Cohere API key. It can be explicitly provided or automatically read from the environment variable COHERE_API_KEY (recommended).
        :param model_name: The name of the model to use, defaults to `"embed-english-v2.0"`. Supported Models are `"embed-english-v2.0"`/ `"large"`, `"embed-english-light-v2.0"`/ `"small"`, `"embed-multilingual-v2.0"`/ `"multilingual-22-12"`.
        :param api_base_url: The Cohere API Base url, defaults to `https://api.cohere.ai/v1/embed`.
        :param truncate: Truncate embeddings that are too long from start or end, ("NONE"|"START"|"END"), defaults to `"END"`. Passing START will discard the start of the input. END will discard the end of the input. In both cases, input is discarded until the remaining input is exactly the maximum input token length for the model. If NONE is selected, when the input exceeds the maximum input token length an error will be returned.
        :param use_async_client: Flag to select the AsyncClient, defaults to `False`. It is recommended to use AsyncClient for applications with many concurrent calls.
        :param max_retries: Maximum number of retries for requests, defaults to `3`.
        :param timeout: Request timeout in seconds, defaults to `120`.
        """

        if api_key is None:
            try:
                api_key = os.environ["COHERE_API_KEY"]
            except KeyError as error_msg:
                raise ValueError(
                    "CohereTextEmbedder expects an Cohere API key. "
                    "Please provide one by setting the environment variable COHERE_API_KEY (recommended) or by passing it explicitly."
                ) from error_msg

        self.api_key = api_key
        self.model_name = model_name
        self.api_base_url = api_base_url
        self.truncate = truncate
        self.use_async_client = use_async_client
        self.max_retries = max_retries
        self.timeout = timeout

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize this component to a dictionary.
        """
        return default_to_dict(
            self,
            api_key=self.api_key,
            model_name=self.model_name,
            api_base_url=self.api_base_url,
            truncate=self.truncate,
            use_async_client=self.use_async_client,
            max_retries=self.max_retries,
            timeout=self.timeout,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CohereTextEmbedder":
        """
        Deserialize this component from a dictionary.
        """
        return default_from_dict(cls, data)

    async def _get_async_response(self, cohere_async_client: AsyncClient, text: str):
        try:
            response = await cohere_async_client.embed(texts=[text], model=self.model_name, truncate=self.truncate)
            metadata = response.meta
            embedding = [list(map(float, emb)) for emb in response.embeddings][0]

        except CohereError as error_response:
            print(error_response.message)

        return embedding, metadata

    @component.output_types(embedding=List[float], metadata=Dict[str, Any])
    def run(self, text: str):
        """Embed a string."""
        if not isinstance(text, str):
            raise TypeError(
                "CohereTextEmbedder expects a string as input."
                "In case you want to embed a list of Documents, please use the CohereDocumentEmbedder."
            )

        # Establish connection to API

        if self.use_async_client == True:
            cohere_client = AsyncClient(
                self.api_key, api_url=self.api_base_url, max_retries=self.max_retries, timeout=self.timeout
            )
            embedding, metadata = self._get_async_response(cohere_client, text)

        else:
            cohere_client = Client(
                self.api_key, api_url=self.api_base_url, max_retries=self.max_retries, timeout=self.timeout
            )

            try:
                response = cohere_client.embed(texts=[text], model=self.model_name, truncate=self.truncate)
                metadata = response.meta
                embedding = [list(map(float, emb)) for emb in response.embeddings][0]

            except CohereError as error_response:
                print(error_response.message)

        return {"embedding": embedding, "metadata": metadata}
