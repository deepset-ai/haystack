from typing import Literal, Any, Dict, List, Optional, Union, Iterable

import logging

from haystack.preview.dataclasses import Document
from haystack.preview.document_stores._utils import DuplicateDocumentError, MissingDocumentError
from haystack.preview.document_stores.memory._bm25 import BM25Representation, BM25RepresentationMissing


logger = logging.getLogger(__name__)
DuplicatePolicy = Literal["skip", "overwrite", "fail"]


try:
    from numpy import argsort, ndarray
except ImportError as exc:
    logger.debug("numpy could not be imported. MemoryDocumentStore won't support retrieval.")
    argsort, ndarray = None, None


try:
    from torch import device
except ImportError as exc:
    logger.debug("torch could not be imported. MemoryDocumentStore won't support retrieval by embedding.")


class MemoryDocumentStore:
    """
    Stores data in-memory. It's ephemeral and cannot be saved to disk.
    """

    def __init__(
        self,
        use_bm25: bool = True,
        bm25_parameters: Optional[Dict[str, Any]] = None,
        devices: Optional[List[Union[str, device]]] = None,
    ):
        """
        Initializes the store.

        :param use_bm25: whether to support bm25 retrieval. It might slow down this document store on high volumes.
        :param bm25_parameters: parameters for rank_bm25.
        :param devices: which devices to use for embedding retrieval. Leave empty to support embedding retrieval on CPU only.
        """
        self.storage: Dict[str, Document] = {}

        # For BM25 retrieval
        self.bm25 = BM25Representation(**(bm25_parameters or {})) if use_bm25 else None

        # For embedding retrieval
        if devices:
            if not device:
                raise ImportError(
                    "torch could not be imported: embedding retrieval won't work. "
                    "To fix this error, run `pip install torch`."
                )
            self.device = [device(device) for device in devices]

    def count_documents(self) -> int:
        """
        Returns the number of how many documents match the given filters.
        """
        return len(self.storage.keys())

    def filter_documents(self, filters: Optional[Dict[str, Any]] = None) -> List[Document]:
        """
        Returns the documents that match the filters provided.

        :param filters: the filters to apply to the document list.
        """
        # TODO apply filters
        return list(self.storage.values())

    def write_documents(self, documents: List[Document], duplicates: DuplicatePolicy = "fail") -> None:
        """
        Writes (or overwrites) documents into the store.

        :param documents: a list of documents.
        :param duplicates: documents with the same ID count as duplicates. When duplicates are met,
            the store can:
             - skip: keep the existing document and ignore the new one.
             - overwrite: remove the old document and write the new one.
             - fail: an error is raised
        :raises DuplicateError: Exception trigger on duplicate document
        :return: None
        """
        if (
            not isinstance(documents, Iterable)
            or isinstance(documents, str)
            or any(not isinstance(doc, Document) for doc in documents)
        ):
            raise ValueError("Please provide a list of Documents.")

        for document in documents:
            if document.id in self.storage.keys():
                if duplicates == "fail":
                    raise DuplicateDocumentError(f"ID '{document.id}' already exists.")
                if duplicates == "skip":
                    logger.warning("ID '%s' already exists", document.id)
            self.storage[document.id] = document

        if self.bm25:
            self.bm25.update_bm25(self.filter_documents(filters={}))

    def delete_documents(self, object_ids: List[str]) -> None:
        """
        Deletes all object_ids from the given pool.
        Fails with `MissingDocumentError` if no object with this id is present in the store.

        :param object_ids: the object_ids to delete
        """
        for object_id in object_ids:
            if not object_id in self.storage.keys():
                raise MissingDocumentError(f"ID '{object_id}' not found, cannot delete it.")
            del self.storage[object_id]

    def bm25_retrieval(self, query: str, filters: Dict[str, Any], top_k: int = 5) -> List[Document]:
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
                logging.debug(f"Returning less than top_k results: the filters returned less than {top_k} documents.")
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

    def embedding_retrieval(
        self,
        queries: List[ndarray],
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 10,
        similarity: str = "dot_product",
        scoring_batch_size: int = 500000,
        scale_score: bool = True,
    ) -> List[Document]:
        """
        Performs retrieval by embedding similarity.
        """
        pass

        # results: Dict[str, List[Document]] = {}
        # for query in queries:
        #     if query is None or not query.content:
        #         logger.info(
        #             "You tried to perform retrieval on an empty query (%s). No documents returned for it.", query
        #         )
        #         results[query] = []
        #         continue
        #     if query.embedding is None:
        #         raise MissingEmbeddingError(
        #             "You tried to perform retrieval by embedding similarity on a query without embedding (%s). "
        #             "Please compute the embeddings for all your queries before using this method.",
        #             query,
        #         )

        #     filtered_documents = self.document_store.get_items(filters=filters)
        #     try:
        #         ids, embeddings = zip(*[(doc["id"], doc["embedding"]) for doc in filtered_documents])
        #     except KeyError:
        #         # FIXME make it nodeable
        #         raise MissingEmbeddingError(
        #             "Some of the documents don't have embeddings. Use the Embedder to compute them."
        #         )

        #     # At this stage the iterable gets consumed.
        #     if self.device and self.device.type == "cuda":
        #         scores = get_scores_torch(
        #             query=query.embedding,
        #             documents=embeddings,
        #             similarity=similarity,
        #             batch_size=batch_size,
        #             device=self.device,
        #         )
        #     else:
        #         embeddings = np.array(embeddings)
        #         scores = get_scores_numpy(query.embedding, embeddings, similarity=similarity)

        #     top_k_ids = sorted(list(zip(ids, scores)), key=lambda x: x[1] if x[1] is not None else 0.0, reverse=True)[
        #         :top_k
        #     ]

        #     relevant_documents = []
        #     for id, score in top_k_ids:
        #         document_data = self.document_store.get_item(id=id)
        #         if scale_score:
        #             score = scale_to_unit_interval(score, similarity)
        #         document_data["score"] = score
        #         document = TextDocument.from_dict(dictionary=document_data)
        #         relevant_documents.append(document)

        #     results[query] = relevant_documents

        # return results
