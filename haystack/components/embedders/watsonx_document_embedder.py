# SPDX-FileCopyrightText: 2023-present IBM Corporation
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, List, Optional

from haystack import Document, component, default_from_dict, default_to_dict
from haystack.utils import Secret, deserialize_secrets_inplace
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import Embeddings


@component
class WatsonXDocumentEmbedder:
    """
    Computes document embeddings using IBM watsonx.ai models.

    ### Usage example

    ```python
    from haystack import Document
    from haystack.components.embedders import WatsonXDocumentEmbedder

    documents = [Document(content="I love pizza!"), Document(content="Pasta is great too")]

    document_embedder = WatsonXDocumentEmbedder(
        model="ibm/slate-30m-english-rtrvr",
        api_key=Secret.from_env_var("WATSONX_API_KEY"),
        url="https://us-south.ml.cloud.ibm.com",
        project_id="your-project-id"
    )

    result = document_embedder.run(documents=documents)
    print(result['documents'][0].embedding)

    # [0.017020374536514282, -0.023255806416273117, ...]
    ```
    """

    def __init__(
        self,
        model: str = "ibm/slate-30m-english-rtrvr",
        api_key: Secret = Secret.from_env_var("WATSONX_API_KEY"),
        url: str = "https://us-south.ml.cloud.ibm.com",
        project_id: Optional[str] = None,
        space_id: Optional[str] = None,
        truncate_input_tokens: Optional[int] = None,
        prefix: str = "",
        suffix: str = "",
        batch_size: int = 1000,
        concurrency_limit: int = 5,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
    ):
        """
        Creates a WatsonXDocumentEmbedder component.

        :param model:
            The name of the model to use for calculating embeddings.
            Default is "ibm/slate-30m-english-rtrvr".
        :param api_key:
            The WATSONX API key. Can be set via environment variable WATSONX_API_KEY.
        :param url:
            The WATSONX URL for the watsonx.ai service.
            Default is "https://us-south.ml.cloud.ibm.com".
        :param project_id:
            The ID of the Watson Studio project. Either project_id or space_id must be provided.
        :param space_id:
            The ID of the Watson Studio space. Either project_id or space_id must be provided.
        :param truncate_input_tokens:
            Maximum number of tokens to use from the input text.
        :param prefix:
            A string to add at the beginning of each text.
        :param suffix:
            A string to add at the end of each text.
        :param batch_size:
            Number of documents to embed in one API call. Default is 1000.
        :param concurrency_limit:
            Number of parallel requests to make. Default is 5.
        :param timeout:
            Timeout for API requests in seconds.
        :param max_retries:
            Maximum number of retries for API requests.
        """
        if not project_id and not space_id:
            raise ValueError("Either project_id or space_id must be provided")

        self.model = model
        self.api_key = api_key
        self.url = url
        self.project_id = project_id
        self.space_id = space_id
        self.truncate_input_tokens = truncate_input_tokens
        self.prefix = prefix
        self.suffix = suffix
        self.batch_size = batch_size
        self.concurrency_limit = concurrency_limit
        self.timeout = timeout
        self.max_retries = max_retries

        # Initialize the embeddings client
        credentials = Credentials(
            api_key=api_key.resolve_value(),
            url=url
        )

        params = {}
        if truncate_input_tokens is not None:
            params["truncate_input_tokens"] = truncate_input_tokens

        self.embedder = Embeddings(
            model_id=model,
            credentials=credentials,
            project_id=project_id,
            space_id=space_id,
            params=params if params else None,
            batch_size=batch_size,
            concurrency_limit=concurrency_limit,
            max_retries=max_retries or 10,
        )

    def _get_telemetry_data(self) -> Dict[str, Any]:
        """
        Data that is sent to Posthog for usage analytics.
        """
        return {"model": self.model}

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the component to a dictionary.
        """
        return default_to_dict(
            self,
            model=self.model,
            api_key=self.api_key.to_dict(),
            url=self.url,
            project_id=self.project_id,
            space_id=self.space_id,
            truncate_input_tokens=self.truncate_input_tokens,
            prefix=self.prefix,
            suffix=self.suffix,
            batch_size=self.batch_size,
            concurrency_limit=self.concurrency_limit,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WatsonXDocumentEmbedder":
        """
        Deserializes the component from a dictionary.
        """
        deserialize_secrets_inplace(data["init_parameters"], keys=["api_key"])
        return default_from_dict(cls, data)

    def _prepare_text(self, text: str) -> str:
        """
        Prepares text for embedding by adding prefix and suffix.
        """
        return self.prefix + text + self.suffix

    @component.output_types(documents=List[Document], meta=Dict[str, Any])
    def run(self, documents: List[Document]):
        """
        Embeds a list of documents.

        :param documents:
            A list of documents to embed.
        :returns:
            A dictionary with:
            - 'documents': List of Documents with embeddings added
            - 'meta': Information about the model usage
        """
        if not isinstance(documents, list) or documents and not isinstance(documents[0], Document):
            raise TypeError(
                "WatsonXDocumentEmbedder expects a list of Documents as input."
                "In case you want to embed a string, please use the WatsonXTextEmbedder."
            )

        texts_to_embed = [self._prepare_text(doc.content or "") for doc in documents]
        embeddings = self.embedder.embed_documents(texts_to_embed)

        for doc, emb in zip(documents, embeddings):
            doc.embedding = emb

        return {
            "documents": documents,
            "meta": {
                "model": self.model,
                "truncate_input_tokens": self.truncate_input_tokens,
                "batch_size": self.batch_size,
            },
        }