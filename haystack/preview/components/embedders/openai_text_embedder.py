from typing import List, Optional, Dict, Any

import openai
import os
from canals.errors import DeserializationError

from haystack.preview import component, default_to_dict, default_from_dict


API_BASE_URL = "https://api.openai.com/v1"


@component
class OpenAITextEmbedder:
    """
    A component for embedding strings using OpenAI models.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "text-embedding-ada-002",
        api_base_url: str = API_BASE_URL,
        organization: Optional[str] = None,
        prefix: str = "",
        suffix: str = "",
    ):
        """
        Create a OpenAITextEmbedder component.

        :param api_key: The OpenAI API key.
                        If not expressly provided, the API key will be read from the environment variable OPENAI_API_KEY.
        :param model_name: The name of the model to use.
        :param api_base_url: The OpenAI API Base url, defaults to `https://api.openai.com/v1`.
        :param organization: The OpenAI-Organization ID, defaults to `None`. For more details, see OpenAI
        [documentation](https://platform.openai.com/docs/api-reference/requesting-organization).
        :param prefix: A string to add to the beginning of each text.
        :param suffix: A string to add to the end of each text.
        """

        if api_key is None:
            try:
                api_key = os.environ["OPENAI_API_KEY"]
            except KeyError:
                raise ValueError(
                    "OpenAITextEmbedder expects an OpenAI API key. "
                    "Please provide one using the api_key parameter or set the environment variable OPENAI_API_KEY."
                )

        self.api_key = api_key
        self.model_name = model_name
        self.api_base_url = api_base_url
        self.organization = organization
        self.prefix = prefix
        self.suffix = suffix

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize this component to a dictionary.
        """
        return default_to_dict(
            self,
            api_key="OPENAI_API_KEY",
            model_name=self.model_name,
            api_base_url=self.api_base_url,
            organization=self.organization,
            prefix=self.prefix,
            suffix=self.suffix,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OpenAITextEmbedder":
        """
        Deserialize this component from a dictionary.
        """
        env_var_name = data["init_parameters"]["api_key"]
        try:
            data["init_parameters"]["api_key"] = os.environ[env_var_name]
        except KeyError as e:
            raise DeserializationError(
                f"For deserialization, the OpenAITextEmbedder expects the api_key to be set as an environment variable named {env_var_name}"
            ) from e

        return default_from_dict(cls, data)

    @component.output_types(embedding=List[float], metadata=Dict[str, Any])
    def run(self, text: str):
        """Embed a string."""
        if not isinstance(text, str):
            raise TypeError(
                "OpenAITextEmbedder expects a string as input."
                "In case you want to embed a list of Documents, please use the OpenAIDocumentEmbedder."
            )

        text_to_embed = self.prefix + text + self.suffix

        # copied from OpenAI embedding_utils (https://github.com/openai/openai-python/blob/main/openai/embeddings_utils.py)
        # replace newlines, which can negatively affect performance.
        text_to_embed = text_to_embed.replace("\n", " ")

        openai.api_key = self.api_key
        if self.organization is not None:
            openai.organization = self.organization

        response = openai.Embedding.create(model=self.model_name, input=text_to_embed)

        metadata = {"model": response.model, "usage": dict(response.usage.items())}
        embedding = response.data[0]["embedding"]

        return {"embedding": embedding, "metadata": metadata}
