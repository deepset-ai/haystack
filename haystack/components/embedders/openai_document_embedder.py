from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI
from tqdm import tqdm

from haystack import Document, component, default_from_dict, default_to_dict
from haystack.utils import Secret, deserialize_secrets_inplace


@component
class OpenAIDocumentEmbedder:
    """
    A component for computing Document embeddings using OpenAI models.

    Usage example:
    ```python
    from haystack import Document
    from haystack.components.embedders import OpenAIDocumentEmbedder

    doc = Document(content="I love pizza!")

    document_embedder = OpenAIDocumentEmbedder()

    result = document_embedder.run([doc])
    print(result['documents'][0].embedding)

    # [0.017020374536514282, -0.023255806416273117, ...]
    ```
    """

    def __init__(
        self,
        api_key: Secret = Secret.from_env_var("OPENAI_API_KEY"),
        model: str = "text-embedding-ada-002",
        dimensions: Optional[int] = None,
        api_base_url: Optional[str] = None,
        organization: Optional[str] = None,
        prefix: str = "",
        suffix: str = "",
        batch_size: int = 32,
        progress_bar: bool = True,
        meta_fields_to_embed: Optional[List[str]] = None,
        embedding_separator: str = "\n",
    ):
        """
        Create a OpenAIDocumentEmbedder component.

        :param api_key:
            The OpenAI API key.
        :param model:
            The name of the model to use.
        :param dimensions:
            The number of dimensions the resulting output embeddings should have. Only supported in `text-embedding-3` and later models.
        :param api_base_url:
            Overrides default base url for all HTTP requests.
        :param organization:
            The Organization ID. See OpenAI's
            [production best practices](https://platform.openai.com/docs/guides/production-best-practices/setting-up-your-organization)
            for more information.
        :param prefix:
            A string to add at the beginning of each text.
        :param suffix:
            A string to add at the end of each text.
        :param batch_size:
            Number of Documents to encode at once.
        :param progress_bar:
            If True shows a progress bar when running.
        :param meta_fields_to_embed:
            List of meta fields that will be embedded along with the Document text.
        :param embedding_separator:
            Separator used to concatenate the meta fields to the Document text.
        """
        self.api_key = api_key
        self.model = model
        self.dimensions = dimensions
        self.api_base_url = api_base_url
        self.organization = organization
        self.prefix = prefix
        self.suffix = suffix
        self.batch_size = batch_size
        self.progress_bar = progress_bar
        self.meta_fields_to_embed = meta_fields_to_embed or []
        self.embedding_separator = embedding_separator

        self.client = OpenAI(api_key=api_key.resolve_value(), organization=organization, base_url=api_base_url)

    def _get_telemetry_data(self) -> Dict[str, Any]:
        """
        Data that is sent to Posthog for usage analytics.
        """
        return {"model": self.model}

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            model=self.model,
            dimensions=self.dimensions,
            organization=self.organization,
            api_base_url=self.api_base_url,
            prefix=self.prefix,
            suffix=self.suffix,
            batch_size=self.batch_size,
            progress_bar=self.progress_bar,
            meta_fields_to_embed=self.meta_fields_to_embed,
            embedding_separator=self.embedding_separator,
            api_key=self.api_key.to_dict(),
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OpenAIDocumentEmbedder":
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :returns:
            Deserialized component.
        """
        deserialize_secrets_inplace(data["init_parameters"], keys=["api_key"])
        return default_from_dict(cls, data)

    def _prepare_texts_to_embed(self, documents: List[Document]) -> List[str]:
        """
        Prepare the texts to embed by concatenating the Document text with the metadata fields to embed.
        """
        texts_to_embed = []
        for doc in documents:
            meta_values_to_embed = [
                str(doc.meta[key]) for key in self.meta_fields_to_embed if key in doc.meta and doc.meta[key] is not None
            ]

            text_to_embed = (
                self.prefix + self.embedding_separator.join(meta_values_to_embed + [doc.content or ""]) + self.suffix
            )

            # copied from OpenAI embedding_utils (https://github.com/openai/openai-python/blob/main/openai/embeddings_utils.py)
            # replace newlines, which can negatively affect performance.
            text_to_embed = text_to_embed.replace("\n", " ")
            texts_to_embed.append(text_to_embed)
        return texts_to_embed

    def _embed_batch(self, texts_to_embed: List[str], batch_size: int) -> Tuple[List[List[float]], Dict[str, Any]]:
        """
        Embed a list of texts in batches.
        """

        all_embeddings = []
        meta: Dict[str, Any] = {}
        for i in tqdm(
            range(0, len(texts_to_embed), batch_size), disable=not self.progress_bar, desc="Calculating embeddings"
        ):
            batch = texts_to_embed[i : i + batch_size]
            if self.dimensions is not None:
                response = self.client.embeddings.create(model=self.model, dimensions=self.dimensions, input=batch)
            else:
                response = self.client.embeddings.create(model=self.model, input=batch)
            embeddings = [el.embedding for el in response.data]
            all_embeddings.extend(embeddings)

            if "model" not in meta:
                meta["model"] = response.model
            if "usage" not in meta:
                meta["usage"] = dict(response.usage)
            else:
                meta["usage"]["prompt_tokens"] += response.usage.prompt_tokens
                meta["usage"]["total_tokens"] += response.usage.total_tokens

        return all_embeddings, meta

    @component.output_types(documents=List[Document], meta=Dict[str, Any])
    def run(self, documents: List[Document]):
        """
        Embed a list of Documents.

        :param documents:
            Documents to embed.

        :returns:
            A dictionary with the following keys:
            - `documents`: Documents with embeddings
            - `meta`: Information about the usage of the model.
        """
        if not isinstance(documents, list) or documents and not isinstance(documents[0], Document):
            raise TypeError(
                "OpenAIDocumentEmbedder expects a list of Documents as input."
                "In case you want to embed a string, please use the OpenAITextEmbedder."
            )

        texts_to_embed = self._prepare_texts_to_embed(documents=documents)

        embeddings, meta = self._embed_batch(texts_to_embed=texts_to_embed, batch_size=self.batch_size)

        for doc, emb in zip(documents, embeddings):
            doc.embedding = emb

        return {"documents": documents, "meta": meta}
