from typing import Any, Dict, List, Optional

from haystack.preview import component, default_to_dict
from haystack.preview.lazy_imports import LazyImport

with LazyImport(message="Run 'pip install gradientai'") as gradientai_import:
    from gradientai import Gradient


@component
class GradientTextEmbedder:
    """
    A component for embedding strings using models hosted on Gradient AI (https://gradient.ai).

    ```python
    embedder = GradientTextEmbedder(
        access_token=gradient_access_token,
        workspace_id=gradient_workspace_id,
        model_name="bge_large")
    p = Pipeline()
    p.add_component(instance=embedder, name="text_embedder")
    p.add_component(instance=InMemoryEmbeddingRetriever(document_store=InMemoryDocumentStore()), name="retriever")
    p.connect("text_embedder", "retriever")
    p.run("embed me!!!")
    ```
    """

    def __init__(
        self,
        *,
        model_name: str = "bge-large",
        access_token: Optional[str] = None,
        workspace_id: Optional[str] = None,
        host: Optional[str] = None,
    ) -> None:
        """
        Create a GradientTextEmbedder component.

        :param model_name: The name of the model to use.
        :param access_token: The Gradient access token. If not provided it's read from the environment
                             variable GRADIENT_ACCESS_TOKEN.
        :param workspace_id: The Gradient workspace ID. If not provided it's read from the environment
                             variable GRADIENT_WORKSPACE_ID.
        :param host: The Gradient host. By default it uses https://api.gradient.ai/.
        """
        gradientai_import.check()
        self._host = host
        self._model_name = model_name

        self._gradient = Gradient(access_token=access_token, host=host, workspace_id=workspace_id)

    def _get_telemetry_data(self) -> Dict[str, Any]:
        """
        Data that is sent to Posthog for usage analytics.
        """
        return {"model": self._model_name}

    def to_dict(self) -> dict:
        """
        Serialize the component to a Python dictionary.
        """
        return default_to_dict(self, workspace_id=self._gradient.workspace_id, model_name=self._model_name)

    def warm_up(self) -> None:
        """
        Load the embedding model.
        """
        if not hasattr(self, "_embedding_model"):
            self._embedding_model = self._gradient.get_embeddings_model(slug=self._model_name)

    @component.output_types(embedding=List[float])
    def run(self, text: str):
        """Generates an embedding for a single text."""
        if not isinstance(text, str):
            raise TypeError(
                "GradientTextEmbedder expects a string as an input."
                "In case you want to embed a list of Documents, please use the GradientDocumentEmbedder."
            )

        if not hasattr(self, "_embedding_model"):
            raise RuntimeError("The embedding model has not been loaded. Please call warm_up() before running.")

        result = self._embedding_model.generate_embeddings(inputs=[{"input": text}])

        if (not result) or (result.embeddings is None) or (len(result.embeddings) == 0):
            raise RuntimeError("The embedding model did not return any embeddings.")

        return {"embedding": result.embeddings[0].embedding}
