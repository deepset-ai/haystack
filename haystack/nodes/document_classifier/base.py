from typing import List, Union, Optional

import logging
from abc import abstractmethod
from functools import wraps
from time import perf_counter

from haystack.schema import Document
from haystack.nodes.base import BaseComponent


logger = logging.getLogger(__name__)


class BaseDocumentClassifier(BaseComponent):
    outgoing_edges = 1
    query_count = 0
    query_time = 0

    @abstractmethod
    def predict(self, documents: List[Document]):
        pass

    @abstractmethod
    def predict_batch(
        self, documents: Union[List[Document], List[List[Document]]], batch_size: Optional[int] = None
    ) -> Union[List[Document], List[List[Document]]]:
        pass

    def run(self, documents: Union[List[dict], List[Document]], root_node: str):  # type: ignore
        self.query_count += 1
        if documents:
            predict = self.timing(self.predict, "query_time")
            documents = [Document.from_dict(doc) if isinstance(doc, dict) else doc for doc in documents]
            results = predict(documents=documents)
        else:
            results = []

        document_ids = [doc.id for doc in results]
        logger.debug("Classified documents with IDs: %s", document_ids)

        # convert back to dicts if we are in an indexing pipeline
        if root_node == "File":
            results = [doc.to_dict() for doc in results]

        output = {"documents": results}

        return output, "output_1"

    def run_batch(self, documents: Union[List[Document], List[List[Document]]], batch_size: Optional[int] = None):  # type: ignore
        predict_batch = self.timing(self.predict_batch, "query_time")
        results = predict_batch(documents=documents, batch_size=batch_size)
        output = {"documents": results}

        if isinstance(documents[0], Document):
            document_ids = [doc.id for doc in results]
            logger.debug("Classified documents with IDs: %s", document_ids)
        else:
            for doc_list in results:
                document_ids = [doc.id for doc in doc_list]
                logger.debug("Classified documents with IDs: %s", document_ids)

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
        print("Classifier (Speed)")
        print("---------------")
        if not self.query_count:
            print("No querying performed via Classifier.run()")
        else:
            print(f"Queries Performed: {self.query_count}")
            print(f"Query time: {self.query_time}s")
            print(f"{self.query_time / self.query_count} seconds per query")
