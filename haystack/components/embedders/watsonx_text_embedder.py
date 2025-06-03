import os
from typing import Any, Dict, List, Optional

from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import Embeddings
from ibm_watsonx_ai.foundation_models.utils.enums import EmbeddingTypes

from haystack import component, default_from_dict, default_to_dict
from haystack.utils import Secret, deserialize_secrets_inplace


@component
class WatsonXTextEmbedder:
    """
    Embeds strings using IBM watsonx.ai foundation models.

    You can use it to embed user query and send it to an embedding Retriever.

    ### Usage example

    ```python
    from haystack.components.embedders import WatsonXTextEmbedder

    text_to_embed = "I love pizza!"

    text_embedder = WatsonXTextEmbedder(
        model="ibm/slate-30m-english-rtrvr",
        api_key=Secret.from_env_var("WATSONX_API_KEY"),
        url="https://us-south.ml.cloud.ibm.com",
        project_id="your-project-id"
    )

    print(text_embedder.run(text_to_embed))

    # {'embedding': [0.017020374536514282, -0.023255806416273117, ...],
    #  'meta': {'model': 'ibm/slate-30m-english-rtrvr',
    #           'truncated_input_tokens': 3}}
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
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
    ):
        """
        Creates an WatsonXTextEmbedder component.

        :param model:
            The name of the IBM watsonx model to use for calculating embeddings.
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
            A string to add at the beginning of each text to embed.
        :param suffix:
            A string to add at the end of each text to embed.
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
        self.timeout = timeout
        self.max_retries = max_retries

        # Initialize the embeddings client
        credentials = Credentials(api_key=api_key.resolve_value(), url=url)

        params = {}
        if truncate_input_tokens is not None:
            params["truncate_input_tokens"] = truncate_input_tokens

        self.embedder = Embeddings(
            model_id=model,
            credentials=credentials,
            project_id=project_id,
            space_id=space_id,
            params=params if params else None,
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
            timeout=self.timeout,
            max_retries=self.max_retries,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WatsonXTextEmbedder":
        """
        Deserializes the component from a dictionary.
        """
        deserialize_secrets_inplace(data["init_parameters"], keys=["api_key"])
        return default_from_dict(cls, data)

    def _prepare_input(self, text: str) -> str:
        if not isinstance(text, str):
            raise TypeError(
                "WatsonXTextEmbedder expects a string as an input. "
                "In case you want to embed a list of Documents, please use the WatsonXDocumentEmbedder."
            )
        return self.prefix + text + self.suffix

    def _prepare_output(self, embedding: List[float]) -> Dict[str, Any]:
        return {
            "embedding": embedding,
            "meta": {"model": self.model, "truncated_input_tokens": self.truncate_input_tokens},
        }

    @component.output_types(embedding=List[float], meta=Dict[str, Any])
    def run(self, text: str):
        """
        Embeds a single string.

        :param text: Text to embed.
        :returns: A dictionary with:
            - 'embedding': The embedding of the input text
            - 'meta': Information about the model usage
        """
        text_to_embed = self._prepare_input(text=text)
        embedding = self.embedder.embed_query(text_to_embed)
        return self._prepare_output(embedding)
