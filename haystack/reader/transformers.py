from typing import List, Optional, Tuple, Dict
import logging
from statistics import mean

from transformers import pipeline, TapasTokenizer, TapasForQuestionAnswering, BatchEncoding
import torch
import numpy as np
import pandas as pd
from quantulum3 import parser

from haystack import Document, Answer, Span
from haystack.reader.base import BaseReader


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
        use_gpu: int = 0,
        top_k: int = 10,
        top_k_per_candidate: int = 4,
        return_no_answers: bool = True,
        max_seq_len: int = 256,
        doc_stride: int = 128
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
        :param use_gpu: If < 0, then use cpu. If >= 0, this is the ordinal of the gpu to use
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

        """

        # save init parameters to enable export of component config as YAML
        self.set_config(
            model_name_or_path=model_name_or_path, model_version=model_version, tokenizer=tokenizer,
            context_window_size=context_window_size, use_gpu=use_gpu, top_k=top_k, doc_stride=doc_stride,
            top_k_per_candidate=top_k_per_candidate, return_no_answers=return_no_answers, max_seq_len=max_seq_len,
        )

        self.model = pipeline('question-answering', model=model_name_or_path, tokenizer=tokenizer, device=use_gpu,
                              revision=model_version)
        self.context_window_size = context_window_size
        self.top_k = top_k
        self.top_k_per_candidate = top_k_per_candidate
        self.return_no_answers = return_no_answers
        self.max_seq_len = max_seq_len
        self.doc_stride = doc_stride

        # TODO context_window_size behaviour different from behavior in FARMReader

    def predict(self, query: str, documents: List[Document], top_k: Optional[int] = None):
        """
        Use loaded QA model to find answers for a query in the supplied list of Document.

        Returns dictionaries containing answers sorted by (desc.) score.
        Example:

         ```python
            |{
            |    'query': 'Who is the father of Arya Stark?',
            |    'answers':[
            |                 {'answer': 'Eddard,',
            |                 'context': " She travels with her father, Eddard, to King's Landing when he is ",
            |                 'offset_answer_start': 147,
            |                 'offset_answer_end': 154,
            |                 'score': 0.9787139466668613,
            |                 'document_id': '1337'
            |                 },...
            |              ]
            |}
         ```

        :param query: Query string
        :param documents: List of Document in which to search for the answer
        :param top_k: The maximum number of answers to return
        :return: Dict containing query and answers

        """
        if top_k is None:
            top_k = self.top_k
        # get top-answers for each candidate passage
        answers = []
        no_ans_gaps = []
        best_overall_score = 0
        for doc in documents:
            transformers_query = {"context": doc.content, "question": query}
            predictions = self.model(transformers_query,
                                     topk=self.top_k_per_candidate,
                                     handle_impossible_answer=self.return_no_answers,
                                     max_seq_len=self.max_seq_len,
                                     doc_stride=self.doc_stride)
            # for single preds (e.g. via top_k=1) transformers returns a dict instead of a list
            if type(predictions) == dict:
                predictions = [predictions]

            # assemble and format all answers
            best_doc_score = 0
            # because we cannot ensure a "no answer" prediction coming back from transformers we initialize it here with 0
            no_ans_doc_score = 0
            # TODO add no answer bias on haystack side after getting "no answer" scores from transformers
            for pred in predictions:
                if pred["answer"]:
                    if pred["score"] > best_doc_score:
                        best_doc_score = pred["score"]
                    context_start = max(0, pred["start"] - self.context_window_size)
                    context_end = min(len(doc.content), pred["end"] + self.context_window_size)
                    answers.append(Answer(answer=pred["answer"],
                                          type="extractive",
                                          score=pred["score"],
                                          context=doc.content[context_start:context_end],
                                          offsets_in_document=[Span(start=pred["start"],end=pred["end"])],
                                          offsets_in_context=[Span(start=pred["start"]-context_start, end=pred["end"]-context_start)],
                                          document_id=doc.id,
                                          meta=doc.meta
                                          ))
                else:
                    no_ans_doc_score = pred["score"]

                if best_doc_score > best_overall_score:
                    best_overall_score = best_doc_score

            no_ans_gaps.append(no_ans_doc_score - best_doc_score)

        # Calculate the score for predicting "no answer", relative to our best positive answer score
        no_ans_prediction, max_no_ans_gap = self._calc_no_answer(no_ans_gaps, best_overall_score)

        if self.return_no_answers:
            answers.append(no_ans_prediction)
        # sort answers by their `score` and select top-k
        answers = sorted(answers, reverse=True)
        answers = answers[:top_k]

        results = {"query": query,
                   "answers": answers}

        return results

    def predict_batch(self, query_doc_list: List[dict], top_k: Optional[int] = None, batch_size: Optional[int] = None):

        raise NotImplementedError("Batch prediction not yet available in TransformersReader.")


class TableReader(BaseReader):
    """
    Transformer-based model for extractive Question Answering on Tables with TaPas
    using the HuggingFace's transformers framework (https://github.com/huggingface/transformers).
    With this reader, you can directly get predictions via predict()

    Example:
    ```python
    from haystack import Document
    from haystack.reader import TableReader
    import pandas as pd

    table_reader = TableReader(model_name_or_path="google/tapas-base-finetuned-wtq")
    data = {
        "actors": ["brad pitt", "leonardo di caprio", "george clooney"],
        "age": ["57", "46", "60"],
        "number of movies": ["87", "53", "69"],
        "date of birth": ["7 february 1967", "10 june 1996", "28 november 1967"],
    }
    table = pd.DataFrame(data)
    document = Document(content=table, content_type="table")
    query = "When was DiCaprio born?"
    prediction = table_reader.predict(query=query, documents=[document])
    answer = prediction["answers"][0].answer  # "10 june 1996"
    ```
    """

    def __init__(
            self,
            model_name_or_path: str = "google/tapas-base-finetuned-wtq",
            model_version: Optional[str] = None,
            tokenizer: Optional[str] = None,
            use_gpu: bool = True,
            top_k: int = 10,
            max_seq_len: int = 256,

    ):
        """
        Load a TableQA model from Transformers.
        Available models include:

        - ``'google/tapas-base-finetuned-wtq`'``
        - ``'google/tapas-base-finetuned-wikisql-supervised``'

        See https://huggingface.co/models?pipeline_tag=table-question-answering
        for full list of available TableQA models.

        :param model_name_or_path: Directory of a saved model or the name of a public model e.g.
        See https://huggingface.co/models?pipeline_tag=table-question-answering for full list of available models.
        :param model_version: The version of model to use from the HuggingFace model hub. Can be tag name, branch name, or commit hash.
        :param tokenizer: Name of the tokenizer (usually the same as model)
        :param use_gpu: Whether to make use of a GPU (if available).
        :param top_k: The maximum number of answers to return
        :param max_seq_len: Max sequence length of one input text for the model.
        """

        self.model = TapasForQuestionAnswering.from_pretrained(model_name_or_path, revision=model_version)
        if use_gpu and torch.cuda.is_available():
                self.model.to("cuda")
        if tokenizer is None:
            self.tokenizer = TapasTokenizer.from_pretrained(model_name_or_path)
        else:
            self.tokenizer = TapasTokenizer.from_pretrained(tokenizer)
        self.top_k = top_k
        self.max_seq_len = max_seq_len
        self.return_no_answers = False

    def predict(self, query: str, documents: List[Document], top_k: Optional[int] = None) -> Dict:
        """
        Use loaded TableQA model to find answers for a query in the supplied list of Documents
        of content_type ``'table'``.

        Returns dictionary containing query and list of Answer objects sorted by (desc.) score.
        WARNING: The answer scores are not reliable, as they are always extremely high, even if
                 a question cannot be answered by a given table.

        :param query: Query string
        :param documents: List of Document in which to search for the answer. Documents should be
                          of content_type ``'table'``.
        :param top_k: The maximum number of answers to return
        :return: Dict containing query and answers

        """

        if top_k is None:
            top_k = self.top_k

        answers = []
        for document in documents:
            if document.content_type != "table":
                logger.warning(f"Skipping document with id {document.id} in TableReader, as it is not of type table.")
                continue

            table: pd.DataFrame = document.content
            # Tokenize query and current table
            inputs = self.tokenizer(table=table,
                                    queries=query,
                                    max_length=self.max_seq_len,
                                    return_tensors="pt")
            inputs.to(self.model.device)
            # Forward query and table through model and convert logits to predictions
            outputs = self.model(**inputs)
            inputs.to("cpu")
            predicted_answer_coordinates, predicted_aggregation_indices = self.tokenizer.convert_logits_to_predictions(
                inputs,
                outputs.logits.cpu().detach(),
                outputs.logits_aggregation.cpu().detach()
            )

            # Get cell values
            current_answer_coordinates = predicted_answer_coordinates[0]
            current_answer_cells = []
            for coordinate in current_answer_coordinates:
                current_answer_cells.append(table.iat[coordinate])

            # Get aggregation operator
            current_aggregation_operator = self.model.config.aggregation_labels[predicted_aggregation_indices[0]]
            
            # Calculate answer score
            current_score = self._calculate_answer_score(outputs.logits.cpu().detach(), inputs, current_answer_coordinates)

            if current_aggregation_operator == "NONE":
                answer_str = ", ".join(current_answer_cells)
            else:
                answer_str = self._aggregate_answers(current_aggregation_operator, current_answer_cells)

            answer_offsets = self._calculate_answer_offsets(current_answer_coordinates, table)

            answers.append(
                Answer(
                    answer=answer_str,
                    type="extractive",
                    score=current_score,
                    context=table,
                    offsets_in_document=answer_offsets,
                    offsets_in_context=answer_offsets,
                    document_id=document.id,
                    meta={"aggregation_operator": current_aggregation_operator,
                          "answer_cells": current_answer_cells}
                )
            )

        # Sort answers by score and select top-k answers
        answers = sorted(answers, reverse=True)
        answers = answers[:top_k]

        results = {"query": query,
                   "answers": answers}

        return results
    
    def _calculate_answer_score(self, logits: torch.Tensor, inputs: BatchEncoding,
                                answer_coordinates: List[Tuple[int, int]]) -> float:
        """
        Calculates the answer score by computing each cell's probability of being part of the answer
        and taking the mean probability of the answer cells.
        """

        # Calculate answer score
        # Values over 88.72284 will overflow when passed through exponential, so logits are truncated.
        logits[logits < -88.7] = -88.7
        token_probabilities = 1 / (1 + np.exp(-logits)) * inputs.attention_mask

        segment_ids = inputs.token_type_ids[0, :, 0].tolist()
        column_ids = inputs.token_type_ids[0, :, 1].tolist()
        row_ids = inputs.token_type_ids[0, :, 2].tolist()
        all_cell_probabilities = self.tokenizer._get_mean_cell_probs(token_probabilities[0].tolist(), segment_ids,
                                                                     row_ids, column_ids)
        # _get_mean_cell_probs seems to index cells by (col, row). DataFrames are, however, indexed by (row, col).
        all_cell_probabilities = {(row, col): prob for (col, row), prob in all_cell_probabilities.items()}
        answer_cell_probabilities = [all_cell_probabilities[coord] for coord in answer_coordinates]
        
        return np.mean(answer_cell_probabilities)

    @staticmethod
    def _aggregate_answers(agg_operator: str, answer_cells: List[str]) -> str:
        if agg_operator == "COUNT":
            return str(len(answer_cells))

        # No aggregation needed as only one cell selected as answer_cells
        if len(answer_cells) == 1:
            return answer_cells[0]

        # Parse answer cells in order to aggregate numerical values
        parsed_answer_cells = [parser.parse(cell) for cell in answer_cells]
        # Check if all cells contain at least one numerical value and that all values share the same unit
        if all(parsed_answer_cells) and all(cell[0].unit.name == parsed_answer_cells[0][0].unit.name
                                            for cell in parsed_answer_cells):
            numerical_values = [cell[0].value for cell in parsed_answer_cells]
            unit = parsed_answer_cells[0][0].unit.symbols[0] if parsed_answer_cells[0][0].unit.symbols else ""

            if agg_operator == "SUM":
                answer_value = sum(numerical_values)
            elif agg_operator == "AVERAGE":
                answer_value = mean(numerical_values)
            else:
                return f"{agg_operator} > {', '.join(answer_cells)}"

            if unit:
                return f"{str(answer_value)} {unit}"
            else:
                return str(answer_value)

        # Not all selected answer cells contain a numerical value or answer cells don't share the same unit
        else:
            return f"{agg_operator} > {', '.join(answer_cells)}"

    @staticmethod
    def _calculate_answer_offsets(answer_coordinates: List[Tuple[int, int]], table: pd.DataFrame) -> List[Span]:
        """
        Calculates the answer cell offsets of the linearized table based on the
        answer cell coordinates.
        """
        answer_offsets = []
        n_rows, n_columns = table.shape
        for coord in answer_coordinates:
            answer_cell_offset = (coord[0] * n_columns) + coord[1]
            answer_offsets.append(Span(start=answer_cell_offset, end=answer_cell_offset + 1))
            
        return answer_offsets

    def predict_batch(self, query_doc_list: List[dict], top_k: Optional[int] = None, batch_size: Optional[int] = None):

        raise NotImplementedError("Batch prediction not yet available in TableReader.")
