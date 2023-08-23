from typing import List, Optional, Union

from haystack.preview import component
from haystack.preview import Document
from haystack.preview.embedding_backends.sentence_transformers_backend import (
    SentenceTransformersEmbeddingBackendFactory,
)


@component
class SentenceTransformersDocumentEmbedder:
    """
    A component for computing Document embeddings using Sentence Transformers models.
    The embedding of each Document is stored in the `embedding` field of the Document.
    """

    def __init__(
        self,
        model_name_or_path: str,
        device: Optional[str] = None,
        use_auth_token: Union[bool, str, None] = None,
        batch_size: int = 32,
        progress_bar: bool = True,
        normalize_embeddings: bool = False,
        embed_meta_fields: Optional[List[str]] = None,
        embed_separator: str = "\n",
    ):
        """
        Create a SentenceTransformersDocumentEmbedder component.

        :param model_name_or_path: Local path or name of model in Hugging Face's model hub such as ``'sentence-transformers/all-MiniLM-L6-v2'``.
        :param device: Device (like 'cuda' / 'cpu') that should be used for computation. If None, checks if a GPU can be used.
        :param use_auth_token: The API token used to download private models from Hugging Face.
                        If this parameter is set to `True`, then the token generated when running
                        `transformers-cli login` (stored in ~/.huggingface) will be used.
        :param batch_size: Number of strings to encode at once.
        :param progress_bar: If true displays progress bar during embedding.
        :param normalize_embeddings: If set to true, returned vectors will have length 1.
        :param embed_meta_fields: List of meta fields that should be embedded along with the Document content.
        :param embed_separator: Separator used to concatenate the meta fields to the Document content.
        """

        self.model_name_or_path = model_name_or_path
        # TODO: remove device parameter and use Haystack's device management once migrated
        self.device = device
        self.use_auth_token = use_auth_token
        self.batch_size = batch_size
        self.progress_bar = progress_bar
        self.normalize_embeddings = normalize_embeddings
        self.embed_meta_fields = embed_meta_fields or []
        self.embed_separator = embed_separator

    def warm_up(self):
        """
        Load the embedding backend.
        """
        if not hasattr(self, "embedding_backend"):
            self.embedding_backend = SentenceTransformersEmbeddingBackendFactory.get_embedding_backend(
                model_name_or_path=self.model_name_or_path, device=self.device, use_auth_token=self.use_auth_token
            )

    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document]):
        """
        Embed a list of Documents.
        The embedding of each Document is stored in the `embedding` field of the Document.
        """
        self.warm_up()

        # TODO: once non textual Documents are properly supported, we should also prepare them for embedding here

        texts_to_embed = []
        for doc in documents:
            meta_values_to_embed = [
                doc.metadata[key] for key in self.embed_meta_fields if key in doc.metadata and doc.metadata[key]
            ]
            text_to_embed = self.embed_separator.join(meta_values_to_embed + [doc.content])
            texts_to_embed.append(text_to_embed)

        embeddings = self.embedding_backend.embed(
            texts_to_embed,
            batch_size=self.batch_size,
            show_progress_bar=self.progress_bar,
            normalize_embeddings=self.normalize_embeddings,
        )

        documents_with_embeddings = []
        for doc, emb in zip(documents, embeddings):
            doc_as_dict = doc.to_dict()
            doc_as_dict["embedding"] = emb
            documents_with_embeddings.append(Document.from_dict(doc_as_dict))

        return {"documents": documents_with_embeddings}
