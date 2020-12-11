import logging
from abc import abstractmethod, ABC
from typing import Any, Optional, Dict, List, Union
from haystack import Document, Label, MultiLabel

logger = logging.getLogger(__name__)


class BaseDocumentStore(ABC):
    """
    Base class for implementing Document Stores.
    """
    index: Optional[str]
    label_index: Optional[str]
    similarity: Optional[str]

    @abstractmethod
    def write_documents(self, documents: Union[List[dict], List[Document]], index: Optional[str] = None):
        """
        Indexes documents for later queries.

        :param documents: a list of Python dictionaries or a list of Haystack Document objects.
                          For documents as dictionaries, the format is {"text": "<the-actual-text>"}.
                          Optionally: Include meta data via {"text": "<the-actual-text>",
                          "meta":{"name": "<some-document-name>, "author": "somebody", ...}}
                          It can be used for filtering and is accessible in the responses of the Finder.
        :param index: Optional name of index where the documents shall be written to.
                      If None, the DocumentStore's default index (self.index) will be used.

        :return: None
        """
        pass

    @abstractmethod
    def get_all_documents(
            self,
            index: Optional[str] = None,
            filters: Optional[Dict[str, List[str]]] = None,
            return_embedding: Optional[bool] = None
    ) -> List[Document]:
        """
        Get documents from the document store.

        :param index: Name of the index to get the documents from. If None, the
                      DocumentStore's default index (self.index) will be used.
        :param filters: Optional filters to narrow down the documents to return.
                        Example: {"name": ["some", "more"], "category": ["only_one"]}
        :param return_embedding: Whether to return the document embeddings.
        """
        pass

    @abstractmethod
    def get_all_labels(self, index: Optional[str] = None, filters: Optional[Dict[str, List[str]]] = None) -> List[Label]:
        pass

    def get_all_labels_aggregated(self,
                                  index: Optional[str] = None,
                                  filters: Optional[Dict[str, List[str]]] = None) -> List[MultiLabel]:
        aggregated_labels = []
        all_labels = self.get_all_labels(index=index, filters=filters)

        # Collect all answers to a question in a dict
        question_ans_dict = {} # type: ignore
        for l in all_labels:
            # only aggregate labels with correct answers, as only those can be currently used in evaluation
            if not l.is_correct_answer:
                continue

            if l.question in question_ans_dict:
                question_ans_dict[l.question].append(l)
            else:
                question_ans_dict[l.question] = [l]

        # Aggregate labels
        for q, ls in question_ans_dict.items():
            ls = list(set(ls))  # get rid of exact duplicates
            # check if there are both text answer and "no answer" present
            t_present = False
            no_present = False
            no_idx = []
            for idx, l in enumerate(ls):
                if len(l.answer) == 0:
                    no_present = True
                    no_idx.append(idx)
                else:
                    t_present = True
            # if both text and no answer are present, remove no answer labels
            if t_present and no_present:
                logger.warning(
                    f"Both text label and 'no answer possible' label is present for question: {ls[0].question}")
                for remove_idx in no_idx[::-1]:
                    ls.pop(remove_idx)

            # construct Aggregated_label
            for i, l in enumerate(ls):
                if i == 0:
                    agg_label = MultiLabel(question=l.question,
                                           multiple_answers=[l.answer],
                                           is_correct_answer=l.is_correct_answer,
                                           is_correct_document=l.is_correct_document,
                                           origin=l.origin,
                                           multiple_document_ids=[l.document_id],
                                           multiple_offset_start_in_docs=[l.offset_start_in_doc],
                                           no_answer=l.no_answer,
                                           model_id=l.model_id,
                                           )
                else:
                    agg_label.multiple_answers.append(l.answer)
                    agg_label.multiple_document_ids.append(l.document_id)
                    agg_label.multiple_offset_start_in_docs.append(l.offset_start_in_doc)
            aggregated_labels.append(agg_label)
        return aggregated_labels

    @abstractmethod
    def get_document_by_id(self, id: str, index: Optional[str] = None) -> Optional[Document]:
        pass

    @abstractmethod
    def get_document_count(self, filters: Optional[Dict[str, List[str]]] = None, index: Optional[str] = None) -> int:
        pass

    @abstractmethod
    def query_by_embedding(self,
                           query_emb: List[float],
                           filters: Optional[Optional[Dict[str, List[str]]]] = None,
                           top_k: int = 10,
                           index: Optional[str] = None,
                           return_embedding: Optional[bool] = None) -> List[Document]:
        pass

    @abstractmethod
    def get_label_count(self, index: Optional[str] = None) -> int:
        pass

    @abstractmethod
    def add_eval_data(self, filename: str, doc_index: str = "document", label_index: str = "label"):
        pass

    @abstractmethod
    def delete_all_documents(self, index: str, filters: Optional[Dict[str, List[str]]] = None):
        pass

