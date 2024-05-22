# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import importlib
from typing import Any, Dict, List

from haystack import DeserializationError, Document, component, default_from_dict, default_to_dict, logging

logger = logging.getLogger(__name__)


@component
class SimilarDocumentsRetriever:
    """
    Retrieves similar documents for each of the given documents for each preset retrievers.

    Usage example:
    ```python
    from haystack import Document
    from haystack.components.retrievers import SimilarDocumentsRetriever
    from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
    from haystack.document_stores.in_memory import InMemoryDocumentStore

    docs = [
        Document(content="Javascript is a popular programming language"),
        Document(content="Python is a popular programming language"),
        Document(content="A chromosome is a package of DNA"),
        Document(content="DNA carries genetic information"),
    ]

    doc_store = InMemoryDocumentStore()
    doc_store.write_documents(docs)
    retriever = InMemoryBM25Retriever(doc_store, top_k=1)
    sim_docs_retriever = SimilarDocumentsRetriever(retrievers=[retriever])

    result = sim_docs_retriever.run(
        documents=[
            Document(content="A DNA document"),
            Document(content="A programming document"),
        ]
    )

    print(result["document_lists"])
    # [
    #   [Document(..., content: 'DNA carries genetic information', ...)],
    #   [Document(..., content: 'Javascript is a popular programming language', ...)]
    # ]
    ```
    """

    def __init__(self, retrievers: List):
        """
        Create the SimilarDocumentsRetriever component.

        :param retrievers:
            List of Retrievers to be used to retrieve similar documents.
        """
        if len(retrievers) == 0:
            raise ValueError("Empty retrievers list provided")
        self.retrievers = retrievers

    @component.output_types(document_lists=List[List[Document]])
    def run(self, documents: List[Document]):
        """
        Run the InMemoryBM25Retriever on the given input data.

        :param documents:
            List of Document for which to find similar documents.
            Every retriever is run for every document provided.
        """
        retrieved_docs = []
        for r in self.retrievers:
            for d in documents:
                retrieved_docs.append(r.run(query=d.content)["documents"])
        return {"document_lists": retrieved_docs}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SimilarDocumentsRetriever":
        """
        Deserializes the component from a dictionary.

        :param data:
            The dictionary to deserialize from.
        :returns:
            The deserialized component.
        """
        init_params = data.get("init_parameters", {})
        if "retrievers" not in init_params:
            raise DeserializationError("Missing 'retrievers' in serialization data")
        retrievers = []
        for retriever_params in init_params["retrievers"]:
            if "type" not in retriever_params:
                raise DeserializationError("Missing 'type' in retriever's serialization data")
            try:
                module_name, type_ = retriever_params["type"].rsplit(".", 1)
                logger.debug("Trying to import module '{module_name}'", module_name=module_name)
                module = importlib.import_module(module_name)
            except (ImportError, DeserializationError) as e:
                raise DeserializationError(
                    f"DocumentStore of type '{retriever_params['type']}' not correctly imported"
                ) from e
            retriever_class = getattr(module, type_)
            retrievers.append(retriever_class.from_dict(retriever_params))

        data["init_parameters"]["retrievers"] = retrievers

        return default_from_dict(cls, data)

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        retrievers = [r.to_dict() for r in self.retrievers]
        return default_to_dict(self, retrievers=retrievers)
