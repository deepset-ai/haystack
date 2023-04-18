from typing import List, Optional, Union

import logging
from abc import abstractmethod
from functools import wraps
from time import perf_counter

from haystack.schema import Document
from haystack.nodes.base import BaseComponent


logger = logging.getLogger(__name__)


class BaseSampler(BaseComponent):
    return_no_answers: bool
    outgoing_edges = 1
    query_count = 0
    query_time = 0

    @abstractmethod
    def predict(self, query: str, documents: List[Document], top_p: Optional[float] = None):
        pass

    @abstractmethod
    def predict_batch(
        self,
        queries: List[str],
        documents: Union[List[Document], List[List[Document]]],
        top_p: Optional[int] = None,
        batch_size: Optional[int] = None,
    ) -> Union[List[Document], List[List[Document]]]:
        pass

    def run(self, query: str, documents: List[Document], top_p: Optional[float] = None):  # type: ignore
        self.query_count += 1
        if documents:
            predict = self.timing(self.predict, "query_time")
            results = predict(query=query, documents=documents, top_p=top_p)
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
        top_p: Optional[float] = None,
        batch_size: Optional[int] = None,
    ):
        self.query_count += len(queries)
        predict_batch = self.timing(self.predict_batch, "query_time")
        results = predict_batch(queries=queries, documents=documents, top_p=top_p, batch_size=batch_size)

        for doc_list in results:
            document_ids = [doc.id for doc in doc_list]
            logger.debug("Ranked documents with IDs: %s", document_ids)

        output = {"documents": results}

        return output, "output_1"

    def timing(self, fn, attr_name):
        """A wrapper method used for time functions."""

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
        print("Sampler (Speed)")
        print("---------------")
        if not self.query_count:
            print("No querying performed with Retriever.run()")
        else:
            print(f"Queries Performed: {self.query_count}")
            print(f"Query time: {self.query_time}s")
            print(f"{self.query_time / self.query_count} seconds per query")
