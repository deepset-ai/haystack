from abc import abstractmethod
from typing import List, Optional
import logging
from time import perf_counter
from functools import wraps
from tqdm import tqdm
from copy import deepcopy
from haystack import Document, BaseComponent
from haystack.document_store.base import BaseDocumentStore

logger = logging.getLogger(__name__)


class BaseRetriever(BaseComponent):
    document_store: BaseDocumentStore
    outgoing_edges = 1
    query_count = 0
    index_count = 0
    query_time = 0.0
    index_time = 0.0
    retrieve_time = 0.0

    @abstractmethod
    def retrieve(self, query: str, filters: dict = None, top_k: Optional[int] = None, index: str = None) -> List[Document]:
        """
        Scan through documents in DocumentStore and return a small number documents
        that are most relevant to the query.

        :param query: The query
        :param filters: A dictionary where the keys specify a metadata field and the value is a list of accepted values for that field
        :param top_k: How many documents to return per query.
        :param index: The name of the index in the DocumentStore from which to retrieve documents
        """
        pass

    def timing(self, fn, attr_name):
        """Wrapper method used to time functions. """
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
        Performs evaluation on the Retriever.
        Retriever is evaluated based on whether it finds the correct document given the query string and at which
        position in the ranking of documents the correct document is.

        |  Returns a dict containing the following metrics:

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

        # Extract all questions for evaluation
        filters = {"origin": [label_origin]}

        timed_retrieve = self.timing(self.retrieve, "retrieve_time")

        labels = self.document_store.get_all_labels_aggregated(index=label_index, filters=filters, open_domain=open_domain)

        correct_retrievals = 0
        summed_avg_precision = 0.0
        summed_reciprocal_rank = 0.0

        # Collect questions and corresponding answers/document_ids in a dict
        question_label_dict = {}
        for label in labels:
            id_question_tuple = (label.multiple_document_ids[0], label.question)
            if open_domain:
                question_label_dict[id_question_tuple] = label.multiple_answers
            else:
                deduplicated_doc_ids = list(set([str(x) for x in label.multiple_document_ids]))
                question_label_dict[id_question_tuple] = deduplicated_doc_ids

        predictions = []

        # Option 1: Open-domain evaluation by checking if the answer string is in the retrieved docs
        logger.info("Performing eval queries...")
        if open_domain:
            for (_, question), gold_answers in tqdm(question_label_dict.items()):
                retrieved_docs = timed_retrieve(question, top_k=top_k, index=doc_index)
                if return_preds:
                    predictions.append({"question": question, "retrieved_docs": retrieved_docs})
                # check if correct doc in retrieved docs
                found_relevant_doc = False
                relevant_docs_found = 0
                current_avg_precision = 0.0
                for doc_idx, doc in enumerate(retrieved_docs):
                    for gold_answer in gold_answers:
                        if gold_answer in doc.text:
                            relevant_docs_found += 1
                            if not found_relevant_doc:
                                correct_retrievals += 1
                                summed_reciprocal_rank += 1 / (doc_idx + 1)
                            current_avg_precision += relevant_docs_found / (doc_idx + 1)
                            found_relevant_doc = True
                            break
                if found_relevant_doc:
                    summed_avg_precision += current_avg_precision / relevant_docs_found
        # Option 2: Strict evaluation by document ids that are listed in the labels
        else:
            for (_, question), gold_ids in tqdm(question_label_dict.items()):
                retrieved_docs = timed_retrieve(question, top_k=top_k, index=doc_index)
                if return_preds:
                    predictions.append({"question": question, "retrieved_docs": retrieved_docs})
                # check if correct doc in retrieved docs
                found_relevant_doc = False
                relevant_docs_found = 0
                current_avg_precision = 0.0
                for doc_idx, doc in enumerate(retrieved_docs):
                    for gold_id in gold_ids:
                        if str(doc.id) == gold_id:
                            relevant_docs_found += 1
                            if not found_relevant_doc:
                                correct_retrievals += 1
                                summed_reciprocal_rank += 1 / (doc_idx + 1)
                            current_avg_precision += relevant_docs_found / (doc_idx + 1)
                            found_relevant_doc = True
                            break
                if found_relevant_doc:
                    all_relevant_docs = len(set(gold_ids))
                    summed_avg_precision += current_avg_precision / all_relevant_docs
        # Metrics
        number_of_questions = len(question_label_dict)
        recall = correct_retrievals / number_of_questions
        mean_reciprocal_rank = summed_reciprocal_rank / number_of_questions
        mean_avg_precision = summed_avg_precision / number_of_questions

        logger.info((f"For {correct_retrievals} out of {number_of_questions} questions ({recall:.2%}), the answer was in"
                     f" the top-{top_k} candidate passages selected by the retriever."))

        metrics =  {
            "recall": recall,
            "map": mean_avg_precision,
            "mrr": mean_reciprocal_rank,
            "retrieve_time": self.retrieve_time,
            "n_questions": number_of_questions,
            "top_k": top_k
        }

        if return_preds:
            return {"metrics": metrics, "predictions": predictions}
        else:
            return metrics

    def run(
        self,
        root_node: str,
        query: Optional[str],
        filters: Optional[dict] = None,
        top_k: Optional[int] = None,
        documents: Optional[List[dict]] = None
    ):  # type: ignore
        if root_node == "Query":
            self.query_count += 1
            run_query_timed = self.timing(self.run_query, "query_time")
            output, stream = run_query_timed(query=query, filters=filters, top_k=top_k)
        elif root_node == "File":
            self.index_count += len(documents)
            run_indexing = self.timing(self.run_indexing, "index_time")
            output, stream = run_indexing(documents=documents)
        else:
            raise Exception(f"Invalid root_node '{root_node}'.")
        return output, stream

    def run_query(
        self,
        query: str,
        filters: Optional[dict] = None,
        top_k: Optional[int] = None,
        **kwargs,
    ):
        index = kwargs.get("index", None)
        documents = self.retrieve(query=query, filters=filters, top_k=top_k, index=index)
        document_ids = [doc.id for doc in documents]
        logger.debug(f"Retrieved documents with IDs: {document_ids}")
        output = {
            "query": query,
            "documents": documents,
        }

        return output, "output_1"

    def run_indexing(self, documents: List[dict]):
        if self.__class__.__name__ in ["DensePassageRetriever", "EmbeddingRetriever"]:
            documents = deepcopy(documents)
            document_objects = [Document.from_dict(doc) for doc in documents]
            embeddings = self.embed_passages(document_objects)  # type: ignore
            for doc, emb in zip(documents, embeddings):
                doc["embedding"] = emb
        output = {"documents": documents}
        return output, "output_1"

    def print_time(self):
        print("Retriever (Speed)")
        print("---------------")
        if not self.index_count:
            print("No indexing performed via Retriever.run()")
        else:
            print(f"Documents indexed: {self.index_count}")
            print(f"Index time: {self.index_time}s")
            print(f"{self.query_time / self.query_count} seconds per document")
        if not self.query_count:
            print("No querying performed via Retriever.run()")
        else:
            print(f"Queries Performed: {self.query_count}")
            print(f"Query time: {self.query_time}s")
            print(f"{self.query_time / self.query_count} seconds per query")
