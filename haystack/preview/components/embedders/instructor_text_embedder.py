from typing import List, Optional, Union, Dict, Any

from haystack.preview import component, default_to_dict, default_from_dict
from haystack.preview.embedding_backends.instructor_backend import _InstructorEmbeddingBackendFactory


@component
class InstructorTextEmbedder:
    """
    A component for embedding strings using Sentence Transformers models.
    """

    def __init__(
        self,
        model_name_or_path: str = "hkunlp/instructor-base",
        device: Optional[str] = None,
        use_auth_token: Union[bool, str, None] = None,
        instruction: str = "Represent the 'domain' 'text_type' for 'task_objective'",
        batch_size: int = 32,
        progress_bar: bool = True,
        normalize_embeddings: bool = False,
    ):
        """
        Create a InstructorTextEmbedder component.

        :param model_name_or_path: Local path or name of the model in Hugging Face's model hub, such as ``'sentence-transformers/all-mpnet-base-v2'``.
        :param device: Device (like 'cuda' / 'cpu') that should be used for computation. If None, checks if a GPU can be used.
        :param use_auth_token: The API token used to download private models from Hugging Face.
                        If this parameter is set to `True`, then the token generated when running
                        `transformers-cli login` (stored in ~/.huggingface) will be used.
        :param instruction: The instruction string to be used while computing domain specific embeddings. The instruction follows the unified template of the form: "Represent the 'domain' 'text_type' for 'task_objective'", where
        - "domain" is optional, and it specifies the domain of the text, e.g., science, finance, medicine, etc.
        - "text_type" is required, and it specifies the encoding unit, e.g., sentence, document, paragraph, etc.
        - "task_objective" is optional, and it specifies the objective of embedding, e.g., retrieve a document, classify the sentence, etc.
        :param batch_size: Number of strings to encode at once.
        :param progress_bar: If true, displays progress bar during embedding.
        :param normalize_embeddings: If set to true, returned vectors will have length 1.
        """

        self.model_name_or_path = model_name_or_path
        # TODO: remove device parameter and use Haystack's device management once migrated
        self.device = device or "cpu"
        self.use_auth_token = use_auth_token
        self.instruction = instruction
        self.batch_size = batch_size
        self.progress_bar = progress_bar
        self.normalize_embeddings = normalize_embeddings

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize this component to a dictionary.
        """
        return default_to_dict(
            self,
            model_name_or_path=self.model_name_or_path,
            device=self.device,
            use_auth_token=self.use_auth_token,
            instruction=self.instruction,
            batch_size=self.batch_size,
            progress_bar=self.progress_bar,
            normalize_embeddings=self.normalize_embeddings,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InstructorTextEmbedder":
        """
        Deserialize this component from a dictionary.
        """
        return default_from_dict(cls, data)

    def warm_up(self):
        """
        Load the embedding backend.
        """
        if not hasattr(self, "embedding_backend"):
            self.embedding_backend = _InstructorEmbeddingBackendFactory.get_embedding_backend(
                model_name_or_path=self.model_name_or_path, device=self.device, use_auth_token=self.use_auth_token
            )

    @component.output_types(embedding=List[float])
    def run(self, text: str):
        """Embed a string."""
        if not isinstance(text, str):
            raise TypeError(
                "InstructorTextEmbedder expects a string as input."
                "In case you want to embed a list of Documents, please use the InstructorDocumentEmbedder."
            )
        if not hasattr(self, "embedding_backend"):
            raise RuntimeError("The embedding model has not been loaded. Please call warm_up() before running.")

        text_to_embed = [self.instruction, text]
        embedding = self.embedding_backend.embed(
            [text_to_embed],
            batch_size=self.batch_size,
            show_progress_bar=self.progress_bar,
            normalize_embeddings=self.normalize_embeddings,
        )[0]
        return {"embedding": embedding}
