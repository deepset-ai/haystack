from typing import List

from haystack import Document, component, logging
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever

# Initialize a logger for the component
logger = logging.getLogger(__name__)


@component
class FilterByNumWords:
    """

    FilterByNumWords is a component that filters a list of documents
    based on the maximum number of words allowed.

    This component iterates through a list of Document objects, counting the total number of words.
    It stops adding documents to the final list once the cumulative word count exceeds a specified maximum (max_size).

    This is particularly useful for limiting the size of documents passed to downstream components,
    ensuring that the total word count does not exceed a predefined threshold.

    ### usage example:

    ```python
    from haystack.document_stores.in_memory import InMemoryDocumentStore
    from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
    from haystack.components.preprocessors.filter_by_num_words import FilterByNumWords

    rag_pipeline = Pipeline()
    rag_pipeline.add_component(instance=InMemoryBM25Retriever(document_store=InMemoryDocumentStore()), name="retriever")
    rag_pipeline.add_component(instance=FilterByNumWords(), name="filter_by_num_words")
    rag_pipeline.connect("retriever", "filter_by_num_words.documents")
    ```

    """

    def __init__(self, max_size: int = 40000):
        """
        Initializes the FilterByNumWords component.

        :param max_size: (int) The maximum number of words allowed. Default is 40,000 words.
        """
        self.max_size = max_size

    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document]) -> dict:
        """
        Filters the provided list of documents based on the maximum word count allowed.

        :param documents: A list of Document objects to be filtered.
        :raises TypeError: if documents is not a list of Documents.
        :returns: A dictionary with a single key "documents",
        containing the filtered list of Document objects.
        The list includes only those documents that keep
        the cumulative word count within the specified max_size.
        """

        if not isinstance(documents, list) or documents and not isinstance(documents[0], Document):
            raise TypeError("FilterByNumWords expects a List of Documents as input.")

        current_tokens = 0  # Initialize a counter to keep track of the total word count
        final_docs = []  # Initialize an empty list to store the filtered documents

        # Iterate over each document in the input list
        for doc in documents:
            # Increment the word count by the number of words in the current document
            current_tokens += len(doc.content.split(" "))

            # If adding the current document exceeds the max_size, stop the iteration
            if current_tokens > self.max_size:
                break

            # Add the current document to the final list
            final_docs.append(doc)

        # Return the filtered list of documents as a dictionary
        return {"documents": final_docs}
