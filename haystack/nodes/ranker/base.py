from typing import List, Optional, Union

import logging
from abc import abstractmethod
from functools import wraps
from time import perf_counter
from copy import deepcopy

from haystack.schema import Document
from haystack.nodes.base import BaseComponent


logger = logging.getLogger(__name__)


class BaseRanker(BaseComponent):
    return_no_answers: bool
    outgoing_edges = 1
    query_count = 0
    query_time = 0

    @abstractmethod
    def predict(self, query: str, documents: List[Document], top_k: Optional[int] = None):
        pass

    @abstractmethod
    def predict_batch(
        self,
        queries: List[str],
        documents: Union[List[Document], List[List[Document]]],
        top_k: Optional[int] = None,
        batch_size: Optional[int] = None,
    ) -> Union[List[Document], List[List[Document]]]:
        pass

    def _add_meta_fields_to_docs(
        self, documents: List[Document], embed_meta_fields: Optional[List[str]] = None
    ) -> List[Document]:
        """
        Concatenates specified metadata fields with the text representations.

        :param documents: List of documents to add metadata to.
        :param embed_meta_fields: Concatenate the provided meta fields and into the text passage that is then used in
            reranking.
        :return: List of documents with metadata.
        """
        if not embed_meta_fields:
            return documents

        docs_with_meta = []
        for doc in documents:
            doc = deepcopy(doc)
            # Gather all relevant metadata fields
            meta_data_fields = []
            for key in embed_meta_fields:
                if key in doc.meta and doc.meta[key]:
                    if isinstance(doc.meta[key], list):
                        meta_data_fields.extend(list(doc.meta[key]))
                    else:
                        meta_data_fields.append(doc.meta[key])
            # Convert to type string (e.g. for ints or floats)
            meta_data_fields = [str(field) for field in meta_data_fields]
            doc.content = "\n".join(meta_data_fields + [doc.content])
            docs_with_meta.append(doc)
        return docs_with_meta

    def run(self, query: str, documents: List[Document], top_k: Optional[int] = None):  # type: ignore
        """
        :param query: Query string.
        :param documents: List of Documents to process.
        :param top_k: The maximum number of Documents to return.
        """
        self.query_count += 1
        if documents:
            predict = self.timing(self.predict, "query_time")
            results = predict(query=query, documents=documents, top_k=top_k)
        else:
            results = []

        document_ids = [doc.id for doc in results]
        logger.debug("Retrieved documents with IDs: %s", document_ids)
        output = {"documents": results}

        return output, "output_1"

    def run_batch(  # type: ignore
        self,
        queries: List[str],
        documents: Union[List[Document], List[List[Document]]],
        top_k: Optional[int] = None,
        batch_size: Optional[int] = None,
    ):
        """
        :param queries: List of query strings.
        :param documents: List of list of Documents to process.
        :param top_k: The maximum number of answers to return.
        :param batch_size: Number of Documents to process at a time.
        """
        self.query_count = +len(queries)
        predict_batch = self.timing(self.predict_batch, "query_time")
        results = predict_batch(queries=queries, documents=documents, top_k=top_k, batch_size=batch_size)

        for doc_list in results:
            document_ids = [doc.id for doc in doc_list]
            logger.debug("Ranked documents with IDs: %s", document_ids)

        output = {"documents": results}

        return output, "output_1"

    def timing(self, fn, attr_name):
        """Wrapper method used to time functions."""

        @wraps(fn)
        def wrapper(*args, **kwargs):
            if attr_name not in self.__dict__:
                self.__dict__[attr_name] = 0
            tic = perf_counter()
            ret = fn(*args, **kwargs)
            toc = perf_counter()
            self.__dict__[attr_name] += toc - tic
            return ret

        return wrapper

    def print_time(self):
        print("Ranker (Speed)")
        print("---------------")
        if not self.query_count:
            print("No querying performed via Retriever.run()")
        else:
            print(f"Queries Performed: {self.query_count}")
            print(f"Query time: {self.query_time}s")
            print(f"{self.query_time / self.query_count} seconds per query")

    def eval(
        self,
        label_index: str = "label",
        doc_index: str = "eval_document",
        label_origin: str = "gold_label",
        top_k: int = 10,
        open_domain: bool = False,
        return_preds: bool = False,
    ) -> dict:
        """
        Performs evaluation of the Ranker.
        Ranker is evaluated in the same way as a Retriever based on whether it finds the correct document given the query string and at which
        position in the ranking of documents the correct document is.

        Returns a dict containing the following metrics:

            - "recall": Proportion of questions for which correct document is among retrieved documents
            - "mrr": Mean of reciprocal rank. Rewards retrievers that give relevant documents a higher rank.
              Only considers the highest ranked relevant document.
            - "map": Mean of average precision for each question. Rewards retrievers that give relevant
              documents a higher rank. Considers all retrieved relevant documents. If ``open_domain=True``,
              average precision is normalized by the number of retrieved relevant documents per query.
              If ``open_domain=False``, average precision is normalized by the number of all relevant documents
              per query.

        :param label_index: Index/Table in DocumentStore where labeled questions are stored
        :param doc_index: Index/Table in DocumentStore where documents that are used for evaluation are stored
        :param top_k: How many documents to return per query
        :param open_domain: If ``True``, retrieval will be evaluated by checking if the answer string to a question is
                            contained in the retrieved docs (common approach in open-domain QA).
                            If ``False``, retrieval uses a stricter evaluation that checks if the retrieved document ids
                            are within ids explicitly stated in the labels.
        :param return_preds: Whether to add predictions in the returned dictionary. If True, the returned dictionary
                             contains the keys "predictions" and "metrics".
        """
        raise NotImplementedError
