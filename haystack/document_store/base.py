import logging
from abc import abstractmethod
from pathlib import Path
from typing import Optional, Dict, List, Union

import numpy as np

from haystack import Document, Label, MultiLabel, BaseComponent
from haystack.errors import DuplicateDocumentError
from haystack.preprocessor.preprocessor import PreProcessor
from haystack.preprocessor.utils import eval_data_from_json, eval_data_from_jsonl, squad_json_to_jsonl

logger = logging.getLogger(__name__)


class BaseDocumentStore(BaseComponent):
    """
    Base class for implementing Document Stores.
    """
    index: Optional[str]
    label_index: Optional[str]
    similarity: Optional[str]
    duplicate_documents_options: tuple = ('skip', 'overwrite', 'fail')

    @abstractmethod
    def write_documents(self, documents: Union[List[dict], List[Document]], index: Optional[str] = None,
                        batch_size: int = 10_000, duplicate_documents: Optional[str] = None):
        """
        Indexes documents for later queries.

        :param documents: a list of Python dictionaries or a list of Haystack Document objects.
                          For documents as dictionaries, the format is {"text": "<the-actual-text>"}.
                          Optionally: Include meta data via {"text": "<the-actual-text>",
                          "meta":{"name": "<some-document-name>, "author": "somebody", ...}}
                          It can be used for filtering and is accessible in the responses of the Finder.
        :param index: Optional name of index where the documents shall be written to.
                      If None, the DocumentStore's default index (self.index) will be used.
        :param batch_size: Number of documents that are passed to bulk function at a time.
        :param duplicate_documents: Handle duplicates document based on parameter options.
                                    Parameter options : ( 'skip','overwrite','fail')
                                    skip: Ignore the duplicates documents
                                    overwrite: Update any existing documents with the same ID when adding documents.
                                    fail: an error is raised if the document ID of the document being added already
                                    exists.

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
                                  filters: Optional[Dict[str, List[str]]] = None,
                                  open_domain: bool=True,
                                  aggregate_by_meta: Optional[Union[str, list]]=None) -> List[MultiLabel]:
        """
        Return all labels in the DocumentStore, aggregated into MultiLabel objects. 
        This aggregation step helps, for example, if you collected multiple possible answers for one question and you
        want now all answers bundled together in one place for evaluation.
        How they are aggregated is defined by the open_domain and aggregate_by_meta parameters.
        If the questions are being asked to a single document (i.e. SQuAD style), you should set open_domain=False to aggregate by question and document.
        If the questions are being asked to your full collection of documents, you should set open_domain=True to aggregate just by question.
        If the questions are being asked to a subslice of your document set (e.g. product review use cases),
        you should set open_domain=True and populate aggregate_by_meta with the names of Label meta fields to aggregate by question and your custom meta fields.
        For example, in a product review use case, you might set aggregate_by_meta=["product_id"] so that Labels
        with the same question but different answers from different documents are aggregated into the one MultiLabel
        object, provided that they have the same product_id (to be found in Label.meta["product_id"])

        :param index: Name of the index to get the labels from. If None, the
                      DocumentStore's default index (self.index) will be used.
        :param filters: Optional filters to narrow down the labels to return.
                        Example: {"name": ["some", "more"], "category": ["only_one"]}
        :param open_domain: When True, labels are aggregated purely based on the question text alone.
                            When False, labels are aggregated in a closed domain fashion based on the question text
                            and also the id of the document that the label is tied to. In this setting, this function
                            might return multiple MultiLabel objects with the same question string.
        :param aggregate_by_meta: The names of the Label meta fields by which to aggregate. For example: ["product_id"]

        """
        aggregated_labels = []
        all_labels = self.get_all_labels(index=index, filters=filters)

        # Collect all answers to a question in a dict
        question_ans_dict: dict = {}
        for l in all_labels:
            # This group_by_id determines the key by which we aggregate labels. Its contents depend on
            # whether we are in an open / closed domain setting,
            # or if there are fields in the meta data that we should group by (set using group_by_meta)
            group_by_id_list: list = []
            if open_domain:
                group_by_id_list = [l.question]
            else:
                group_by_id_list = [l.document_id, l.question]
            if aggregate_by_meta:
                if type(aggregate_by_meta) == str:
                    aggregate_by_meta = [aggregate_by_meta]
                for meta_key in aggregate_by_meta:
                    curr_meta = l.meta.get(meta_key, None)
                    if curr_meta:
                        group_by_id_list.append(curr_meta)
            group_by_id = tuple(group_by_id_list)

            # only aggregate labels with correct answers, as only those can be currently used in evaluation
            if not l.is_correct_answer:
                continue

            if group_by_id in question_ans_dict:
                question_ans_dict[group_by_id].append(l)
            else:
                question_ans_dict[group_by_id] = [l]

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
                    # Keep only the label metadata that we are aggregating by
                    if aggregate_by_meta:
                        meta_new = {k: v for k, v in l.meta.items() if k in aggregate_by_meta}
                    else:
                        meta_new = {}

                    agg_label = MultiLabel(question=l.question,
                                           multiple_answers=[l.answer],
                                           is_correct_answer=l.is_correct_answer,
                                           is_correct_document=l.is_correct_document,
                                           origin=l.origin,
                                           multiple_document_ids=[l.document_id],
                                           multiple_offset_start_in_docs=[l.offset_start_in_doc],
                                           no_answer=l.no_answer,
                                           model_id=l.model_id,
                                           meta=meta_new)
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
                           query_emb: np.ndarray,
                           filters: Optional[Optional[Dict[str, List[str]]]] = None,
                           top_k: int = 10,
                           index: Optional[str] = None,
                           return_embedding: Optional[bool] = None) -> List[Document]:
        pass

    @abstractmethod
    def get_label_count(self, index: Optional[str] = None) -> int:
        pass

    @abstractmethod
    def write_labels(self, labels: Union[List[Label], List[dict]], index: Optional[str] = None):
        pass

    def add_eval_data(self, filename: str, doc_index: str = "eval_document", label_index: str = "label",
                      batch_size: Optional[int] = None, preprocessor: Optional[PreProcessor] = None,
                      max_docs: Union[int, bool] = None, open_domain: bool = False):
        """
        Adds a SQuAD-formatted file to the DocumentStore in order to be able to perform evaluation on it.
        If a jsonl file and a batch_size is passed to the function, documents are loaded batchwise
        from disk and also indexed batchwise to the DocumentStore in order to prevent out of memory errors.

        :param filename: Name of the file containing evaluation data (json or jsonl)
        :param doc_index: Elasticsearch index where evaluation documents should be stored
        :param label_index: Elasticsearch index where labeled questions should be stored
        :param batch_size: Optional number of documents that are loaded and processed at a time.
                           When set to None (default) all documents are processed at once.
        :param preprocessor: Optional PreProcessor to preprocess evaluation documents.
                             It can be used for splitting documents into passages (and assigning labels to corresponding passages).
                             Currently the PreProcessor does not support split_by sentence, cleaning nor split_overlap != 0.
                             When set to None (default) preprocessing is disabled.
        :param max_docs: Optional number of documents that will be loaded.
                         When set to None (default) all available eval documents are used.
        :param open_domain: Set this to True if your file is an open domain dataset where two different answers to the
                            same question might be found in different contexts.

        """
        # TODO improve support for PreProcessor when adding eval data
        if preprocessor is not None:
            assert preprocessor.split_by != "sentence", f"Split by sentence not supported.\n" \
                                                    f"Please set 'split_by' to either 'word' or 'passage' in the supplied PreProcessor."
            assert preprocessor.split_respect_sentence_boundary == False, \
                f"split_respect_sentence_boundary not supported yet.\n" \
                f"Please set 'split_respect_sentence_boundary' to False in the supplied PreProcessor."
            assert preprocessor.split_overlap == 0, f"Overlapping documents are currently not supported when adding eval data.\n" \
                                                    f"Please set 'split_overlap=0' in the supplied PreProcessor."
            assert preprocessor.clean_empty_lines == False, f"clean_empty_lines currently not supported when adding eval data.\n" \
                                                    f"Please set 'clean_empty_lines=False' in the supplied PreProcessor."
            assert preprocessor.clean_whitespace == False, f"clean_whitespace is currently not supported when adding eval data.\n" \
                                                    f"Please set 'clean_whitespace=False' in the supplied PreProcessor."
            assert preprocessor.clean_header_footer == False, f"clean_header_footer is currently not supported when adding eval data.\n" \
                                                    f"Please set 'clean_header_footer=False' in the supplied PreProcessor."

        file_path = Path(filename)
        if file_path.suffix == ".json":
            if batch_size is None:
                docs, labels = eval_data_from_json(filename, max_docs=max_docs, preprocessor=preprocessor, open_domain=open_domain)
                self.write_documents(docs, index=doc_index)
                self.write_labels(labels, index=label_index)
            else:
                jsonl_filename = (file_path.parent / (file_path.stem + '.jsonl')).as_posix()
                logger.info(f"Adding evaluation data batch-wise is not compatible with json-formatted SQuAD files. "
                            f"Converting json to jsonl to: {jsonl_filename}")
                squad_json_to_jsonl(filename, jsonl_filename)
                self.add_eval_data(jsonl_filename, doc_index, label_index, batch_size, open_domain=open_domain)

        elif file_path.suffix == ".jsonl":
            for docs, labels in eval_data_from_jsonl(filename, batch_size, max_docs=max_docs, preprocessor=preprocessor, open_domain=open_domain):
                if docs:
                    self.write_documents(docs, index=doc_index)
                if labels:
                    self.write_labels(labels, index=label_index)

        else:
            logger.error("File needs to be in json or jsonl format.")

    def delete_all_documents(self, index: Optional[str] = None, filters: Optional[Dict[str, List[str]]] = None):
        pass

    @abstractmethod
    def delete_documents(self, index: Optional[str] = None, filters: Optional[Dict[str, List[str]]] = None):
        pass

    def run(self, documents: List[dict], index: Optional[str] = None, **kwargs):  # type: ignore
        self.write_documents(documents=documents, index=index)
        return kwargs, "output_1"

    @abstractmethod
    def get_documents_by_id(self, ids: List[str], index: Optional[str] = None,
                            batch_size: int = 10_000) -> List[Document]:
        pass

    def _drop_duplicate_documents(self, documents: List[Document]) -> List[Document]:
        """
         Drop duplicates documents based on same hash ID

         :param documents: A list of Haystack Document objects.
         :return: A list of Haystack Document objects.
        """
        _hash_ids: list = []
        _documents: List[Document] = []

        for document in documents:
            if document.id in _hash_ids:
                logger.warning(f"Duplicate Documents: Document with id '{document.id}' already exists in index "
                               f"'{self.index}'")
                continue
            _documents.append(document)
            _hash_ids.append(document.id)

        return _documents

    def _handle_duplicate_documents(self, documents: List[Document], duplicate_documents: Optional[str] = None):
        """
        Handle duplicates documents

        :param documents: A list of Haystack Document objects.
        :param duplicate_documents: Handle duplicates document based on parameter options.
                                    Parameter options : ( 'skip','overwrite','fail')
                                    skip (default option): Ignore the duplicates documents
                                    overwrite: Update any existing documents with the same ID when adding documents.
                                    fail: an error is raised if the document ID of the document being added already
                                    exists.
        :return: A list of Haystack Document objects.
       """
        if duplicate_documents in ('skip', 'fail'):
            documents = self._drop_duplicate_documents(documents)
            documents_found = self.get_documents_by_id(ids=[doc.id for doc in documents], index=self.index)
            ids_exist_in_db = [doc.id for doc in documents_found]

            if len(ids_exist_in_db) > 0 and duplicate_documents == 'fail':
                raise DuplicateDocumentError(f"Document with ids '{', '.join(ids_exist_in_db)} already exists"
                                             f" in index = '{self.index}'.")

            documents = list(filter(lambda doc: doc.id not in ids_exist_in_db, documents))

        return documents
