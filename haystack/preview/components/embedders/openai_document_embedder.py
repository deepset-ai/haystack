from typing import List, Optional, Dict, Any, Tuple

import openai
from tqdm import tqdm


from haystack.preview import component, Document, default_to_dict, default_from_dict


API_BASE_URL = "https://api.openai.com/v1"


@component
class OpenAIDocumentEmbedder:
    """
    A component for computing Document embeddings using OpenAI models.
    The embedding of each Document is stored in the `embedding` field of the Document.
    """

    def __init__(
        self,
        api_key: str,
        model_name: str = "text-embedding-ada-002",
        api_base_url: str = API_BASE_URL,
        organization: Optional[str] = None,
        prefix: str = "",
        suffix: str = "",
        batch_size: int = 32,
        progress_bar: bool = True,
        metadata_fields_to_embed: Optional[List[str]] = None,
        embedding_separator: str = "\n",
    ):
        """
        Create a OpenAIDocumentEmbedder component.
        :param api_key: The OpenAI API key.
        :param model_name: The name of the model to use.
        :param api_base_url: The OpenAI API Base url, defaults to `https://api.openai.com/v1`.
        :param organization: The OpenAI-Organization ID, defaults to `None`. For more details, see OpenAI
        [documentation](https://platform.openai.com/docs/api-reference/requesting-organization).
        :param prefix: A string to add to the beginning of each text.
        :param suffix: A string to add to the end of each text.
        :param batch_size: Number of strings to encode at once.
        :param progress_bar: If true, displays progress bar during embedding.
        :param metadata_fields_to_embed: List of meta fields that should be embedded along with the Document content.
        :param embedding_separator: Separator used to concatenate the meta fields to the Document content.
        """

        self.api_key = api_key
        self.model_name = model_name
        self.api_base_url = api_base_url
        self.organization = organization
        self.prefix = prefix
        self.suffix = suffix
        self.batch_size = batch_size
        self.progress_bar = progress_bar
        self.metadata_fields_to_embed = metadata_fields_to_embed or []
        self.embedding_separator = embedding_separator

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize this component to a dictionary.
        """
        return default_to_dict(
            self,
            api_key=self.api_key,
            model_name=self.model_name,
            api_base_url=self.api_base_url,
            organization=self.organization,
            prefix=self.prefix,
            suffix=self.suffix,
            batch_size=self.batch_size,
            progress_bar=self.progress_bar,
            metadata_fields_to_embed=self.metadata_fields_to_embed,
            embedding_separator=self.embedding_separator,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OpenAIDocumentEmbedder":
        """
        Deserialize this component from a dictionary.
        """
        return default_from_dict(cls, data)

    def _prepare_texts_to_embed(self, documents: List[Document]) -> List[str]:
        """
        Prepare the texts to embed by concatenating the Document text with the metadata fields to embed.
        """
        texts_to_embed = []
        for doc in documents:
            meta_values_to_embed = [
                str(doc.metadata[key])
                for key in self.metadata_fields_to_embed
                if key in doc.metadata and doc.metadata[key]
            ]

            text_to_embed = (
                self.prefix + self.embedding_separator.join(meta_values_to_embed + [doc.text or ""]) + self.suffix
            )

            # copied from OpenAI embedding_utils (https://github.com/openai/openai-python/blob/main/openai/embeddings_utils.py)
            # replace newlines, which can negatively affect performance.
            text_to_embed = text_to_embed.replace("\n", " ")
            texts_to_embed.append(text_to_embed)
        return texts_to_embed

    def _embed_batch(self, texts_to_embed: List[str], batch_size: int) -> Tuple[List[str], Dict[str, Any]]:
        """
        Embed a list of texts in batches.
        """

        all_embeddings = []
        metadata = {}
        for i in tqdm(
            range(0, len(texts_to_embed), batch_size), disable=not self.progress_bar, desc="Calculating embeddings"
        ):
            batch = texts_to_embed[i : i + batch_size]
            response = openai.Embedding.create(model=self.model_name, input=batch)
            embeddings = [el["embedding"] for el in response.data]
            all_embeddings.extend(embeddings)

            if "model" not in metadata:
                metadata["model"] = response.model
            if "usage" not in metadata:
                metadata["usage"] = dict(response.usage.items())
            else:
                metadata["usage"]["prompt_tokens"] += response.usage.prompt_tokens
                metadata["usage"]["total_tokens"] += response.usage.total_tokens

        return all_embeddings, metadata

    @component.output_types(documents=List[Document], metadata=Dict[str, Any])
    def run(self, documents: List[Document]):
        """
        Embed a list of Documents.
        The embedding of each Document is stored in the `embedding` field of the Document.
        """
        if not isinstance(documents, list) or not isinstance(documents[0], Document):
            raise TypeError(
                "OpenAIDocumentEmbedder expects a list of Documents as input."
                "In case you want to embed a string, please use the OpenAITextEmbedder."
            )

        openai.api_key = self.api_key
        if self.organization is not None:
            openai.organization = self.organization

        texts_to_embed = self._prepare_texts_to_embed(documents=documents)

        embeddings, metadata = self._embed_batch(texts_to_embed=texts_to_embed, batch_size=self.batch_size)

        documents_with_embeddings = []
        for doc, emb in zip(documents, embeddings):
            doc_as_dict = doc.to_dict()
            doc_as_dict["embedding"] = emb
            documents_with_embeddings.append(Document.from_dict(doc_as_dict))

        return {"documents": documents_with_embeddings, "metadata": metadata}
