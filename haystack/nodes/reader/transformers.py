from typing import List, Optional, Union, Dict, Any, Tuple

import logging
import itertools

import torch
from transformers import pipeline
from transformers.data.processors.squad import SquadExample

from haystack.errors import HaystackError
from haystack.schema import Document, Answer, Span
from haystack.nodes.reader.base import BaseReader
from haystack.modeling.utils import initialize_device_settings


logger = logging.getLogger(__name__)


class TransformersReader(BaseReader):
    """
    Transformer based model for extractive Question Answering using the HuggingFace's transformers framework
    (https://github.com/huggingface/transformers).
    While the underlying model can vary (BERT, Roberta, DistilBERT ...), the interface remains the same.
    With this reader, you can directly get predictions via predict()
    """

    def __init__(
        self,
        model_name_or_path: str = "distilbert-base-uncased-distilled-squad",
        model_version: Optional[str] = None,
        tokenizer: Optional[str] = None,
        context_window_size: int = 70,
        use_gpu: bool = True,
        top_k: int = 10,
        top_k_per_candidate: int = 3,
        return_no_answers: bool = False,
        max_seq_len: int = 256,
        doc_stride: int = 128,
        batch_size: int = 16,
        use_auth_token: Optional[Union[str, bool]] = None,
        devices: Optional[List[Union[str, torch.device]]] = None,
    ):
        """
        Load a QA model from Transformers.
        Available models include:

        - ``'distilbert-base-uncased-distilled-squad`'``
        - ``'bert-large-cased-whole-word-masking-finetuned-squad``'
        - ``'bert-large-uncased-whole-word-masking-finetuned-squad``'

        See https://huggingface.co/models for full list of available QA models

        :param model_name_or_path: Directory of a saved model or the name of a public model e.g. 'bert-base-cased',
        'deepset/bert-base-cased-squad2', 'deepset/bert-base-cased-squad2', 'distilbert-base-uncased-distilled-squad'.
        See https://huggingface.co/models for full list of available models.
        :param model_version: The version of model to use from the HuggingFace model hub. Can be tag name, branch name, or commit hash.
        :param tokenizer: Name of the tokenizer (usually the same as model)
        :param context_window_size: Num of chars (before and after the answer) to return as "context" for each answer.
                                    The context usually helps users to understand if the answer really makes sense.
        :param use_gpu: Whether to use GPU (if available).
        :param top_k: The maximum number of answers to return
        :param top_k_per_candidate: How many answers to extract for each candidate doc that is coming from the retriever (might be a long text).
        Note that this is not the number of "final answers" you will receive
        (see `top_k` in TransformersReader.predict() or Finder.get_answers() for that)
        and that no_answer can be included in the sorted list of predictions.
        :param return_no_answers: If True, the HuggingFace Transformers model could return a "no_answer" (i.e. when there is an unanswerable question)
        If False, it cannot return a "no_answer". Note that `no_answer_boost` is unfortunately not available with TransformersReader.
        If you would like to set no_answer_boost, use a `FARMReader`.
        :param max_seq_len: max sequence length of one input text for the model
        :param doc_stride: length of striding window for splitting long texts (used if len(text) > max_seq_len)
        :param batch_size: Number of documents to process at a time.
        :param use_auth_token: The API token used to download private models from Huggingface.
                               If this parameter is set to `True`, then the token generated when running
                               `transformers-cli login` (stored in ~/.huggingface) will be used.
                               Additional information can be found here
                               https://huggingface.co/transformers/main_classes/model.html#transformers.PreTrainedModel.from_pretrained

        :param devices: List of torch devices (e.g. cuda, cpu, mps) to limit inference to specific devices.
                        A list containing torch device objects and/or strings is supported (For example
                        [torch.device('cuda:0'), "mps", "cuda:1"]). When specifying `use_gpu=False` the devices
                        parameter is not used and a single cpu device is used for inference.
        """
        super().__init__()

        self.devices, _ = initialize_device_settings(devices=devices, use_cuda=use_gpu, multi_gpu=False)

        if len(self.devices) > 1:
            logger.warning(
                f"Multiple devices are not supported in {self.__class__.__name__} inference, "
                f"using the first device {self.devices[0]}."
            )

        self.model = pipeline(
            "question-answering",
            model=model_name_or_path,
            tokenizer=tokenizer,
            device=self.devices[0],
            revision=model_version,
            use_auth_token=use_auth_token,
        )
        self.context_window_size = context_window_size
        self.top_k = top_k
        self.top_k_per_candidate = top_k_per_candidate
        self.return_no_answers = return_no_answers
        self.max_seq_len = max_seq_len
        self.doc_stride = doc_stride
        self.batch_size = batch_size

        # TODO context_window_size behaviour different from behavior in FARMReader

    def predict(self, query: str, documents: List[Document], top_k: Optional[int] = None):
        """
        Use loaded QA model to find answers for a query in the supplied list of Document.

        Returns dictionaries containing answers sorted by (desc.) score.
        Example:

         ```python
         {
             'query': 'Who is the father of Arya Stark?',
             'answers':[
                          {'answer': 'Eddard,',
                          'context': " She travels with her father, Eddard, to King's Landing when he is ",
                          'offset_answer_start': 147,
                          'offset_answer_end': 154,
                          'score': 0.9787139466668613,
                          'document_id': '1337'
                          },...
                       ]
         }
         ```

        :param query: Query string
        :param documents: List of Document in which to search for the answer
        :param top_k: The maximum number of answers to return
        :return: Dict containing query and answers
        """
        if top_k is None:
            top_k = self.top_k

        inputs = []
        all_docs = {}
        for doc in documents:
            cur = self.model.create_sample(question=query, context=doc.content)
            cur.doc_id = doc.id
            all_docs[doc.id] = doc
            inputs.append(cur)

        predictions = self.model(
            inputs,
            top_k=self.top_k_per_candidate,
            handle_impossible_answer=self.return_no_answers,
            max_seq_len=self.max_seq_len,
            doc_stride=self.doc_stride,
        )
        # Transformers gives different output dependiing on top_k_per_candidate and number of inputs
        if isinstance(predictions, dict):
            predictions = [[predictions]]
        elif len(inputs) == 1:
            predictions = [predictions]
        else:
            predictions = [p if isinstance(p, list) else [p] for p in predictions]

        # Add Document ID to predictions to be able to construct Answer objects
        for preds_for_single_doc, inp in zip(predictions, inputs):
            cur_doc_id = inp.doc_id
            for pred in preds_for_single_doc:
                pred["doc_id"] = cur_doc_id
        predictions = list(itertools.chain.from_iterable(predictions))

        answers, max_no_ans_gap = self._extract_answers_of_predictions(predictions, all_docs, top_k)

        results = {"query": query, "answers": answers}
        return results

    def predict_batch(
        self,
        queries: List[str],
        documents: Union[List[Document], List[List[Document]]],
        top_k: Optional[int] = None,
        batch_size: Optional[int] = None,
    ):
        """
        Use loaded QA model to find answers for the queries in the Documents.

        - If you provide a list containing a single query...

            - ... and a single list of Documents, the query will be applied to each Document individually.
            - ... and a list of lists of Documents, the query will be applied to each list of Documents and the Answers
              will be aggregated per Document list.

        - If you provide a list of multiple queries...

            - ... and a single list of Documents, each query will be applied to each Document individually.
            - ... and a list of lists of Documents, each query will be applied to its corresponding list of Documents
              and the Answers will be aggregated per query-Document pair.

        :param queries: Single query or list of queries.
        :param documents: Related documents (e.g. coming from a retriever) that the answer shall be conditioned on.
                          Can be a single list of Documents or a list of lists of Documents.
        :param top_k: Number of returned answers per query.
        :param batch_size: Number of query-document pairs to be processed at a time.
        """
        if top_k is None:
            top_k = self.top_k

        if batch_size is None:
            batch_size = self.batch_size

        inputs, number_of_docs, all_docs, single_doc_list = self._preprocess_batch_queries_and_docs(
            queries=queries, documents=documents
        )

        # Inference
        predictions = self.model(
            inputs,
            top_k=self.top_k_per_candidate,
            handle_impossible_answer=self.return_no_answers,
            max_seq_len=self.max_seq_len,
            doc_stride=self.doc_stride,
            batch_size=batch_size,
        )

        # Transformers flattens lists of length 1. This restores the original list structure.
        if isinstance(predictions, dict):
            predictions = [[predictions]]
        elif len(number_of_docs) == 1:
            predictions = [predictions]
        else:
            predictions = [p if isinstance(p, list) else [p] for p in predictions]

        # Group predictions together
        grouped_predictions = []
        grouped_inputs = []
        left_idx = 0
        right_idx = 0
        for number in number_of_docs:
            right_idx = left_idx + number
            grouped_predictions.append(predictions[left_idx:right_idx])
            grouped_inputs.append(inputs[left_idx:right_idx])
            left_idx = right_idx

        results: Dict = {"queries": queries, "answers": [], "no_ans_gaps": []}
        for grouped_pred, grouped_inp in zip(grouped_predictions, grouped_inputs):
            # Add Document ID to predictions to be able to construct Answer objects
            for preds_for_single_doc, inp in zip(grouped_pred, grouped_inp):
                for pred in preds_for_single_doc:
                    cur_doc_id = inp.doc_id
                    pred["doc_id"] = cur_doc_id
            if isinstance(grouped_pred[0], list):
                group = list(itertools.chain.from_iterable(grouped_pred))
            answers, max_no_ans_gap = self._extract_answers_of_predictions(group, all_docs, top_k)
            results["answers"].append(answers)
            results["no_ans_gaps"].append(max_no_ans_gap)

        # Group answers by question in case of list of queries and single doc list
        if single_doc_list and len(queries) > 1:
            answers_per_query = int(len(results["answers"]) / len(queries))
            answers = []
            for i in range(0, len(results["answers"]), answers_per_query):
                answer_group = results["answers"][i : i + answers_per_query]
                answers.append(answer_group)
            results["answers"] = answers

        return results

    def _extract_answers_of_predictions(
        self, predictions: List[Dict[str, Any]], docs: Dict[str, Document], top_k: int
    ) -> Tuple[List[Answer], float]:
        answers = []
        no_ans_gaps = []
        best_overall_score = 0

        cur_doc_id = predictions[0]["doc_id"]
        cur_doc = docs[cur_doc_id]
        no_ans_doc_score = 0
        best_doc_score = 0

        # TODO add no answer bias on haystack side after getting "no answer" scores from transformers
        for pred in predictions:
            # Update best_overall_score based on best_doc_score of predictions of previous Document
            # + add no_ans_gap for previous Document + update cur_doc
            if cur_doc_id != pred["doc_id"]:
                if best_doc_score > best_overall_score:
                    best_overall_score = best_doc_score
                no_ans_gaps.append(no_ans_doc_score - best_doc_score)
                cur_doc_id = pred["doc_id"]
                cur_doc = docs[cur_doc_id]
                no_ans_doc_score = 0
                best_doc_score = 0
            if pred["answer"]:
                if pred["score"] > best_doc_score:
                    best_doc_score = pred["score"]
                context_start = max(0, pred["start"] - self.context_window_size)
                context_end = min(len(cur_doc.content), pred["end"] + self.context_window_size)
                answers.append(
                    Answer(
                        answer=pred["answer"],
                        type="extractive",
                        score=pred["score"],
                        context=cur_doc.content[context_start:context_end],
                        offsets_in_document=[Span(start=pred["start"], end=pred["end"])],
                        offsets_in_context=[Span(start=pred["start"] - context_start, end=pred["end"] - context_start)],
                        document_id=cur_doc.id,
                        meta=cur_doc.meta,
                    )
                )
            # "no answer" prediction
            else:
                no_ans_doc_score = pred["score"]

        # Update best_overall_score based on best_doc_score of predictions of last Document
        # + add no_ans_gap for last Document
        if best_doc_score > best_overall_score:
            best_overall_score = best_doc_score
        no_ans_gaps.append(no_ans_doc_score - best_doc_score)

        # Calculate the score for predicting "no answer", relative to our best positive answer score
        no_ans_prediction, max_no_ans_gap = self._calc_no_answer(no_ans_gaps, best_overall_score)

        if self.return_no_answers:
            answers.append(no_ans_prediction)
        # Sort answers by score and select top-k
        answers = sorted(answers, reverse=True)
        answers = answers[:top_k]

        return answers, max_no_ans_gap

    def _preprocess_batch_queries_and_docs(
        self, queries: List[str], documents: Union[List[Document], List[List[Document]]]
    ) -> Tuple[List[SquadExample], List[int], Dict[str, Document], bool]:
        # Convert input to transformers format
        inputs = []
        number_of_docs = []
        all_docs = {}
        single_doc_list = False

        # Docs case 1: single list of Documents -> apply each query to all Documents
        if len(documents) > 0 and isinstance(documents[0], Document):
            single_doc_list = True
            for query in queries:
                for doc in documents:
                    number_of_docs.append(1)
                    if not isinstance(doc, Document):
                        raise HaystackError(f"doc was of type {type(doc)}, but expected a Document.")
                    cur = self.model.create_sample(question=query, context=doc.content)
                    cur.doc_id = doc.id
                    all_docs[doc.id] = doc
                    inputs.append(cur)

        # Docs case 2: list of lists of Documents -> apply each query to corresponding list of Documents, if queries
        # contains only one query, apply it to each list of Documents
        elif len(documents) > 0 and isinstance(documents[0], list):
            single_doc_list = False
            if len(queries) == 1:
                queries = queries * len(documents)
            if len(queries) != len(documents):
                raise HaystackError("Number of queries must be equal to number of provided Document lists.")
            for query, cur_docs in zip(queries, documents):
                if not isinstance(cur_docs, list):
                    raise HaystackError(f"cur_docs was of type {type(cur_docs)}, but expected a list of Documents.")
                number_of_docs.append(len(cur_docs))
                for doc in cur_docs:
                    if not isinstance(doc, Document):
                        raise HaystackError(f"doc was of type {type(doc)}, but expected a Document.")
                    cur = self.model.create_sample(question=query, context=doc.content)
                    cur.doc_id = doc.id
                    all_docs[doc.id] = doc
                    inputs.append(cur)

        return inputs, number_of_docs, all_docs, single_doc_list
