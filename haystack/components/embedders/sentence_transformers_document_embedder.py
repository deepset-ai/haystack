# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, List, Optional

from haystack import Document, component, default_from_dict, default_to_dict
from haystack.components.embedders.backends.sentence_transformers_backend import (
    _SentenceTransformersEmbeddingBackendFactory,
)
from haystack.utils import ComponentDevice, Secret, deserialize_secrets_inplace


@component
class SentenceTransformersDocumentEmbedder:
    """
    A component for computing Document embeddings using Sentence Transformers models.

    Usage example:
    ```python
    from haystack import Document
    from haystack.components.embedders import SentenceTransformersDocumentEmbedder
    doc = Document(content="I love pizza!")
    doc_embedder = SentenceTransformersDocumentEmbedder()
    doc_embedder.warm_up()

    result = doc_embedder.run([doc])
    print(result['documents'][0].embedding)

    # [-0.07804739475250244, 0.1498992145061493, ...]
    ```
    """

    def __init__(
        self,
        model: str = "sentence-transformers/all-mpnet-base-v2",
        device: Optional[ComponentDevice] = None,
        token: Optional[Secret] = Secret.from_env_var("HF_API_TOKEN", strict=False),
        prefix: str = "",
        suffix: str = "",
        batch_size: int = 32,
        progress_bar: bool = True,
        normalize_embeddings: bool = False,
        meta_fields_to_embed: Optional[List[str]] = None,
        embedding_separator: str = "\n",
        trust_remote_code: bool = False,
    ):
        """
        Create a SentenceTransformersDocumentEmbedder component.

        :param model:
            Local path or ID of the model on HuggingFace Hub.
        :param device:
            Overrides the default device used to load the model.
        :param token:
            The API token used to download private models from Hugging Face.
        :param prefix:
            A string to add at the beginning of each text.
            Can be used to prepend the text with an instruction, as required by some embedding models,
            such as E5 and bge.
        :param suffix:
            A string to add at the end of each text.
        :param batch_size:
            Number of Documents to encode at once.
        :param progress_bar:
            If True shows a progress bar when running.
        :param normalize_embeddings:
            If True returned vectors will have length 1.
        :param meta_fields_to_embed:
            List of meta fields that will be embedded along with the Document text.
        :param embedding_separator:
            Separator used to concatenate the meta fields to the Document text.
        :param trust_remote_code:
            If `False`, only Hugging Face verified model architectures are allowed.
            If `True`, custom models and scripts are allowed.
        """

        self.model = model
        self.device = ComponentDevice.resolve_device(device)
        self.token = token
        self.prefix = prefix
        self.suffix = suffix
        self.batch_size = batch_size
        self.progress_bar = progress_bar
        self.normalize_embeddings = normalize_embeddings
        self.meta_fields_to_embed = meta_fields_to_embed or []
        self.embedding_separator = embedding_separator
        self.trust_remote_code = trust_remote_code

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
            device=self.device.to_dict(),
            token=self.token.to_dict() if self.token else None,
            prefix=self.prefix,
            suffix=self.suffix,
            batch_size=self.batch_size,
            progress_bar=self.progress_bar,
            normalize_embeddings=self.normalize_embeddings,
            meta_fields_to_embed=self.meta_fields_to_embed,
            embedding_separator=self.embedding_separator,
            trust_remote_code=self.trust_remote_code,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SentenceTransformersDocumentEmbedder":
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :returns:
            Deserialized component.
        """
        init_params = data["init_parameters"]
        if init_params["device"] is not None:
            init_params["device"] = ComponentDevice.from_dict(init_params["device"])
        deserialize_secrets_inplace(data["init_parameters"], keys=["token"])
        return default_from_dict(cls, data)

    def warm_up(self):
        """
        Initializes the component.
        """
        if not hasattr(self, "embedding_backend"):
            self.embedding_backend = _SentenceTransformersEmbeddingBackendFactory.get_embedding_backend(
                model=self.model,
                device=self.device.to_torch_str(),
                auth_token=self.token,
                trust_remote_code=self.trust_remote_code,
            )

    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document]):
        """
        Embed a list of Documents.

        :param documents:
            Documents to embed.

        :returns:
            A dictionary with the following keys:
            - `documents`: Documents with embeddings
        """
        if not isinstance(documents, list) or documents and not isinstance(documents[0], Document):
            raise TypeError(
                "SentenceTransformersDocumentEmbedder expects a list of Documents as input."
                "In case you want to embed a list of strings, please use the SentenceTransformersTextEmbedder."
            )
        if not hasattr(self, "embedding_backend"):
            raise RuntimeError("The embedding model has not been loaded. Please call warm_up() before running.")

        # TODO: once non textual Documents are properly supported, we should also prepare them for embedding here

        texts_to_embed = []
        for doc in documents:
            meta_values_to_embed = [
                str(doc.meta[key]) for key in self.meta_fields_to_embed if key in doc.meta and doc.meta[key]
            ]
            text_to_embed = (
                self.prefix + self.embedding_separator.join(meta_values_to_embed + [doc.content or ""]) + self.suffix
            )
            texts_to_embed.append(text_to_embed)

        embeddings = self.embedding_backend.embed(
            texts_to_embed,
            batch_size=self.batch_size,
            show_progress_bar=self.progress_bar,
            normalize_embeddings=self.normalize_embeddings,
        )

        for doc, emb in zip(documents, embeddings):
            doc.embedding = emb

        return {"documents": documents}
