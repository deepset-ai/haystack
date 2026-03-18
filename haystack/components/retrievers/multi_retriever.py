from typing import Any

from haystack import component, default_from_dict, default_to_dict
from haystack.components.retrievers.types.protocol import TextRetriever
from haystack.core.serialization import component_from_dict, component_to_dict, import_class_by_name
from haystack.dataclasses import Document
from haystack.utils.misc import _deduplicate_documents


@component
class MultiRetriever:
    def __init__(
        self, *, retrievers: list[TextRetriever], filters: dict[str, Any] | None = None, top_k: int = 10
    ) -> None:
        """
        Create the MultiRetriever component.

        :param retrievers:
            A list of retriever components to run in parallel.
        :param filters:
            A dictionary of filters to apply when retrieving documents.
        :param top_k:
            The maximum number of documents to return per retriever.
        """
        self.retrievers = retrievers
        self.filters = filters
        self.top_k = top_k
        self._is_warmed_up = False

    def warm_up(self) -> None:
        """
        Warm up the retrievers if any has a warm_up method.
        """
        if self._is_warmed_up:
            return
        for retriever in self.retrievers:
            if hasattr(retriever, "warm_up") and callable(retriever.warm_up):
                retriever.warm_up()
        self._is_warmed_up = True

    @component.output_types(documents=list[Document])
    def run(
        self, query: str, filters: dict[str, Any] | None = None, top_k: int | None = None
    ) -> dict[str, list[Document]]:
        """
        Runs the retriever on the given query and filters.

        :param query:
            The query to run the retriever on.
        :param filters:
            The filters to apply to the retriever. If not provided, the filters from the initialization of the
            component will be used. If those are also not provided, no filters will be applied.
        :param top_k:
            The number of documents to return per retriever. If not provided, the top_k from the initialization of
            the component will be used. If that is also not provided, all retrieved documents will be returned.

        :returns:
            A dictionary with the keys:
                - "documents": A list of retrieved documents.
        """
        resolved_top_k = top_k if top_k is not None else self.top_k
        resolved_filters = filters if filters is not None else self.filters

        all_documents: list[Document] = []
        # TODO Update to run in parallel
        for retriever in self.retrievers:
            result = retriever.run(query=query, filters=resolved_filters, top_k=resolved_top_k)
            all_documents.extend(result.get("documents", []))
        all_documents = _deduplicate_documents(all_documents)
        return {"documents": all_documents}

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(
            self, retrievers=[component_to_dict(obj=retriever, name="retriever") for retriever in self.retrievers]
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MultiRetriever":
        """
        Creates an instance of the component from a dictionary.

        :param data:
            Dictionary with the data to create the component.
        """
        # TODO This is a bit messy. We could update default_from_dict to handle lists of components to simplify the
        #      implementation here.
        retrievers_data = data.get("init_parameters", {}).get("retrievers", [])
        if retrievers_data:
            retrievers = []
            for retriever_data in retrievers_data:
                try:
                    imported_class = import_class_by_name(retriever_data["type"])
                except ImportError as e:
                    raise ImportError(
                        f"Could not import class {retriever_data['type']} for retriever. Error: {str(e)}"
                    ) from e
                retriever = component_from_dict(
                    cls=imported_class, data=retriever_data["init_parameters"], name="retriever"
                )
                retrievers.append(retriever)
            data["init_parameters"]["retrievers"] = retrievers
        return default_from_dict(cls, data)
