import inspect
from typing import Any

from haystack import component, default_from_dict, default_to_dict
from haystack.components.embedders.types.protocol import TextEmbedder
from haystack.components.retrievers.types.protocol import EmbeddingRetriever, TextRetriever
from haystack.dataclasses import Document
from haystack.utils.misc import _deduplicate_documents

RETRIEVERS_TYPE = list[TextRetriever] | list[EmbeddingRetriever] | list[TextRetriever | EmbeddingRetriever]

# NOTES:
# Draft version of my basic idea
# Some open questions:
# - Do we need the flexibility to provide multiple text embedders --> E.g. if different EmbeddingRetrievers need
#   different embedders
#   - If yes we may need to create a new component (e.g. VectorSearch) that takes in a TextEmbedder + EmbeddingRetriever
#     that would ideally satisfy the TextRetriever protocol.
#     Then the updated type would just be RETRIEVERS_TYPE = list[TextRetriever]


@component
class MultiRetriever:
    def __init__(self, retrievers: RETRIEVERS_TYPE, text_embedder: TextEmbedder | None) -> None:
        self.retrievers = retrievers
        self.text_embedder = text_embedder
        self._is_warmed_up = False

    def warm_up(self) -> None:
        """
        Warm up the Agent.
        """
        if self._is_warmed_up:
            return
        for retriever in self.retrievers:
            if hasattr(retriever, "warm_up") and callable(retriever.warm_up):
                retriever.warm_up()
        if hasattr(self.text_embedder, "warm_up") and callable(self.text_embedder.warm_up):
            self.text_embedder.warm_up()
        self._is_warmed_up = True

    @component.output_types(documents=list[Document])
    def run(self, query: str, filters: dict[str, Any] | None = None, top_k: int = 10) -> dict[str, list[Document]]:
        """
        Runs the retriever on the given query and filters.

        :param query:
            The query to run the retriever on.
        :param filters:
            The filters to apply to the retriever.
        :param top_k:
            The number of documents to return per retriever.

        :returns:
            A dictionary with the keys:
                - "documents": A list of retrieved documents.
        """
        all_documents = []
        # TODO Update to run in parallel
        for retriever in self.retrievers:
            sig = inspect.signature(retriever.run)

            # TODO One option to distinguish between the two types.
            #      Could also use isinstance if we add runtime_checkable to the protocols.
            if "query_embedding" in sig.parameters:
                embedding = self.text_embedder.run(text=query)["embedding"]
                result = retriever.run(query_embedding=embedding, filters=filters, top_k=top_k)
            elif "query" in sig.parameters:
                result = retriever.run(query=query, filters=filters, top_k=top_k)
            else:
                raise ValueError(f"Unknown retriever type: {type(retriever)}")

            documents = result.get("documents", [])
            all_documents.extend(documents)

        all_documents = _deduplicate_documents(all_documents)
        return {"documents": all_documents}

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            retrievers=[default_to_dict(retriever) for retriever in self.retrievers],
            text_embedder=default_to_dict(self.text_embedder) if self.text_embedder else None,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MultiRetriever":
        """
        Creates an instance of the component from a dictionary.

        :param data:
            Dictionary with the data to create the component.
        """
        # TODO Finish need to deserialize each one individually
        return default_from_dict(cls, data)
