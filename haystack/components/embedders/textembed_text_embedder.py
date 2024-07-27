"""TextEmbed: Embedding Inference Server.

TextEmbed offers a high-throughput, low-latency service for generating embeddings using various sentence-transformer models.
It now also supports image embedding models, providing flexibility and scalability for diverse applications.

Maintained by Keval Dekivadiya, TextEmbed is licensed under Apache-2.0.
"""

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import requests

from haystack import component, logging
from haystack.utils import Secret

logger = logging.getLogger(__name__)

DEFAULT_URL = "http://0.0.0.0:8000/v1"


@component
class TextEmbedEmbedder:
    """
    A component that embeds text using the TextEmbed API.
    """

    def __init__(
        self,
        model_name: str,
        base_url: str = DEFAULT_URL,
        embed_batch_size: int = 32,
        timeout: float = 60.0,
        api_key: Optional[Secret] = Secret.from_env_var(
            ["textembed_api_key"], strict=False
        ),
    ):
        """
        Initializes the TextEmbedEmbedder object with specified parameters.

        Args:
            model_name (str): The name of the model to be used for embeddings.
            base_url (str): The base URL of the embedding service.
            embed_batch_size (int): The batch size for embedding requests.
            timeout (float): Timeout for requests.
            api_key (Optional[str]): Authentication token for generating it.
        """
        self.model_name = model_name
        self.base_url = base_url
        self.embed_batch_size = embed_batch_size
        self.timeout = timeout
        self.api_key = api_key

    def _permute(
        self, texts: List[str], sorter: Callable = len
    ) -> Tuple[List[str], Callable]:
        """Sorts texts in ascending order and provides a function to restore the original order.

        Args:
            texts (List[str]): List of texts to sort.
            sorter (Callable, optional): Sorting function, defaults to length.

        Returns:
            Tuple[List[str], Callable]: Sorted texts and a function to restore original order.
        """
        if len(texts) == 1:
            return texts, lambda t: t
        length_sorted_idx = np.argsort([-sorter(sen) for sen in texts])
        texts_sorted = [texts[idx] for idx in length_sorted_idx]

        return texts_sorted, lambda unsorted_embeddings: [
            unsorted_embeddings[idx] for idx in np.argsort(length_sorted_idx)
        ]

    def _batch(self, texts: List[str]) -> List[List[str]]:
        """Splits a list of texts into batches of size max `self.embed_batch_size`.

        Args:
            texts (List[str]): List of texts to split.

        Returns:
            List[List[str]]: List of batches of texts.
        """
        if len(texts) == 1:
            return [texts]
        batches = []
        for start_index in range(0, len(texts), self.embed_batch_size):
            batches.append(texts[start_index : start_index + self.embed_batch_size])
        return batches

    def _unbatch(self, batch_of_texts: List[List[Any]]) -> List[Any]:
        """
        Merges batches of texts into a single list.

        Args:
            batch_of_texts (List[List[Any]]): List of batches of texts.

        Returns:
            List[Any]: Merged list of texts.
        """
        if len(batch_of_texts) == 1 and len(batch_of_texts[0]) == 1:
            return batch_of_texts[0]
        texts = []
        for sublist in batch_of_texts:
            texts.extend(sublist)
        return texts

    def _kwargs_post_request(self, texts: List[str]) -> Dict[str, Any]:
        """
        Builds the kwargs for the POST request, used by sync method.

        Args:
            texts (List[str]): List of texts to embed.

        Returns:
            Dict[str, Any]: Dictionary of POST request parameters.
        """
        return dict(
            url=f"{self.base_url}/embedding",
            headers={
                "accept": "application/json",
                "content-type": "application/json",
                "Authorization": (f"Bearer {self.api_key}" if self.api_key else None),
            },
            json=dict(
                input=texts,
                model=self.model_name,
            ),
            timeout=self.timeout,
        )

    def _sync_request_embed(self, batch_texts: List[str]) -> List[List[float]]:
        """Sends a synchronous request to the embedding endpoint.

        Args:
            batch_texts (List[str]): Batch of texts to embed.

        Returns:
            List[List[float]]: List of embeddings for the batch.

        Raises:
            Exception: If the response status is not 200.
        """
        response = requests.post(**self._kwargs_post_request(texts=batch_texts))
        if response.status_code != 200:
            raise Exception(
                f"TextEmbed responded with an unexpected status message "
                f"{response.status_code}: {response.text}"
            )
        return [e["embedding"] for e in response.json()["data"]]

    def _call_api(self, texts: List[str]) -> List[List[float]]:
        """Calls the TextEmbed API to get embeddings for a list of texts.

        Args:
            texts (List[str]): A list of texts to get embeddings for.

        Returns:
            List[List[float]]: A list of embeddings for the input texts.

        Raises:
            Exception: If the API responds with a status code other than 200.
        """
        perm_texts, unpermute_func = self._permute(texts)
        perm_texts_batched = self._batch(perm_texts)

        # Request
        map_args = (
            self._sync_request_embed,
            perm_texts_batched,
        )
        if len(perm_texts_batched) == 1:
            embeddings_batch_perm = list(map(*map_args))
        else:
            with ThreadPoolExecutor(32) as p:
                embeddings_batch_perm = list(p.map(*map_args))

        embeddings_perm = self._unbatch(embeddings_batch_perm)
        embeddings = unpermute_func(embeddings_perm)
        return embeddings

    def embed(
        self, text: Union[str, List[str]]
    ) -> Union[List[float], List[List[float]]]:
        """Get the embedding for a single text or a list of texts.

        Args:
            text (Union[str, List[str]]): The text(s) to get the embedding(s) for.

        Returns:
            Union[List[float], List[List[float]]]: The embedding(s) for the input text(s).
        """
        if isinstance(text, str):
            return self._call_api([text])[0]
        elif isinstance(text, list):
            return self._call_api(text)
        else:
            raise TypeError("Input must be a string or a list of strings.")

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the component to a dictionary.

        Returns:
            Dict[str, Any]: Dictionary with serialized data.
        """
        return {
            "model_name": self.model_name,
            "base_url": self.base_url,
            "embed_batch_size": self.embed_batch_size,
            "timeout": self.timeout,
            "api_key": self.api_key,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TextEmbedEmbedder":
        """Deserializes the component from a dictionary.

        Args:
            data (Dict[str, Any]): Dictionary to deserialize from.

        Returns:
            TextEmbedEmbedder: Deserialized component.
        """
        return cls(
            model_name=data["model_name"],
            base_url=data.get("base_url", DEFAULT_URL),
            embed_batch_size=data.get("embed_batch_size", 32),
            timeout=data.get("timeout", 60.0),
            api_key=data.get("api_key"),
        )

    @component.output_types(embedding=List[float])
    def run(self, text: str):
        """Embed a single string.

        Args:
            text (str): Text to embed.

        Returns:
            dict: A dictionary with the following keys:
            - `embedding`: The embedding of the input text.
        """
        if not isinstance(text, str):
            raise TypeError("TextEmbedEmbedder expects a string as an input.")

        return {"embedding": self.embed(text)}
