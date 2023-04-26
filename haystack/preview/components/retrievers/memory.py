from typing import Optional, Dict, Any, List

from haystack.preview import Document, component


@component
class MemoryRetriever:
    def __init__(
        self,
        input: str = "queries",
        output: str = "documents",
        default_top_k: int = 5,
        default_filters: Optional[Dict[str, Any]] = None,
    ):
        self.inputs = [input]
        self.outputs = [output]
        self.top_k = default_top_k
        self.filters = default_filters

    def retrieve(self, query: str, filters: Dict[str, Any], top_k: int = 5) -> List[Document]:
        """
        Performs BM25 retrieval using `rank_bm25`.

        :param query: the query, as a string
        :param filters: perform retrieval only on the subset defined by this filter
        :param top_k: how many hits to return. Note that it might return less than top_k if the store
            contains less than top_k documents, or the filters returnes less than top_k documents.
        :returns: a list of documents in order of relevance. The documents have the score field populated
            with the value computed by bm25 against the given query.
        """
        if not argsort:
            raise ImportError(
                "numpy could not be imported, local retrieval can't work. " "Run 'pip install numpy' to fix this issue."
            )
        if not query:
            raise ValueError("The query can't empty.")

        filtered_document_ids = self.filter_documents(filters={**filters, "content_type": "text"})
        tokenized_query = self.bm25.bm25_tokenization_regex(query.content.lower())
        docs_scores = self.bm25.bm25_ranking.get_scores(tokenized_query)
        most_relevant_ids = argsort(docs_scores)[::-1]

        all_ids = [doc.id for doc in self.filter_documents()]

        current_position = 0
        returned_docs = 0
        while returned_docs < top_k:
            try:
                id = all_ids[most_relevant_ids[current_position]]
            except IndexError as e:
                logger.debug(f"Returning less than top_k results: the filters returned less than {top_k} documents.")
                return

            if id not in filtered_document_ids:
                current_position += 1
            else:
                document_data = self.storage[id].to_dict()
                document_data["score"] = docs_scores[most_relevant_ids[current_position]]
                doc = Document.from_dict(document_data)

                yield doc

                returned_docs += 1
                current_position += 1
