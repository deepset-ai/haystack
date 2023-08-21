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
        """

        self.model_name_or_path = model_name_or_path
        # TODO: remove device parameter and use Haystack's device management once migrated
        self.device = device
        self.use_auth_token = use_auth_token
        self.batch_size = batch_size
        self.progress_bar = progress_bar
        self.normalize_embeddings = normalize_embeddings

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

        # TODO: we should find a proper strategy for supporting the embedding of meta fields, also supporting templates
        # E.g.: This article talks about {{doc.meta["company"]}}, it was published on {{doc.meta["publication_date"]}}. Here is the article's content: {{doc.content}}
        texts_to_embed = [doc.content for doc in documents]

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
