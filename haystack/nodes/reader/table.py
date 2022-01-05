from typing import List, Optional, Tuple, Dict

import logging
from statistics import mean
import torch
import numpy as np
import pandas as pd
from quantulum3 import parser
from transformers import TapasTokenizer, TapasForQuestionAnswering, AutoTokenizer, AutoModelForSequenceClassification, \
    BatchEncoding, AutoConfig

from haystack.schema import Document, Answer, Span
from haystack.nodes.reader.base import BaseReader
from haystack.modeling.utils import initialize_device_settings


logger = logging.getLogger(__name__)


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
        :param use_gpu: Whether to use GPU or CPU. Falls back on CPU if no GPU is available.
        :param top_k: The maximum number of answers to return
        :param max_seq_len: Max sequence length of one input table for the model. If the number of tokens of
                            query + table exceed max_seq_len, the table will be truncated by removing rows until the
                            input size fits the model.
        """

        self.devices, _ = initialize_device_settings(use_cuda=use_gpu, multi_gpu=False)
        self.model = TapasForQuestionAnswering.from_pretrained(model_name_or_path, revision=model_version)
        self.model.to(str(self.devices[0]))
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
                                    return_tensors="pt",
                                    truncation=True)
            inputs.to(self.devices[0])
            # Forward query and table through model and convert logits to predictions
            outputs = self.model(**inputs)
            inputs.to("cpu")
            if self.model.config.num_aggregation_labels > 0:
                aggregation_logits = outputs.logits_aggregation.cpu().detach()
            else:
                aggregation_logits = None

            predicted_output = self.tokenizer.convert_logits_to_predictions(
                inputs,
                outputs.logits.cpu().detach(),
                aggregation_logits
            )
            if len(predicted_output) == 1:
                predicted_answer_coordinates = predicted_output[0]
            else:
                predicted_answer_coordinates, predicted_aggregation_indices = predicted_output

            # Get cell values
            current_answer_coordinates = predicted_answer_coordinates[0]
            current_answer_cells = []
            for coordinate in current_answer_coordinates:
                current_answer_cells.append(table.iat[coordinate])

            # Get aggregation operator
            if self.model.config.aggregation_labels is not None:
                current_aggregation_operator = self.model.config.aggregation_labels[predicted_aggregation_indices[0]]
            else:
                current_aggregation_operator = "NONE"
            
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
        # Return empty string if model did not select any cell as answer
        if len(answer_cells) == 0:
            return ""

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


class RCIReader(BaseReader):
    """
    Table Reader model based on Glass et al. (2021)'s Row-Column-Intersection model.
    See the original paper for more details:
    Glass, Michael, et al. (2021): "Capturing Row and Column Semantics in Transformer Based Question Answering over Tables"
    (https://aclanthology.org/2021.naacl-main.96/)

    Each row and each column is given a score with regard to the query by two separate models. The score of each cell
    is then calculated as the sum of the corresponding row score and column score. Accordingly, the predicted answer is
    the cell with the highest score.

    Pros and Cons of RCIReader compared to TableReader:
    + Provides meaningful confidence scores
    + Allows larger tables as input
    - Does not support aggregation over table cells
    - Slower
    """

    def __init__(self,
                 row_model_name_or_path: str = "michaelrglass/albert-base-rci-wikisql-row",
                 column_model_name_or_path: str = "michaelrglass/albert-base-rci-wikisql-col",
                 row_model_version: Optional[str] = None,
                 column_model_version: Optional[str] = None,
                 row_tokenizer: Optional[str] = None,
                 column_tokenizer: Optional[str] = None,
                 use_gpu: bool = True,
                 top_k: int = 10,
                 max_seq_len: int = 256,
    ):
        """
        Load an RCI model from Transformers.
        Available models include:

        - ``'michaelrglass/albert-base-rci-wikisql-row'`` + ``'michaelrglass/albert-base-rci-wikisql-col'``
        - ``'michaelrglass/albert-base-rci-wtq-row'`` + ``'michaelrglass/albert-base-rci-wtq-col'``



        :param row_model_name_or_path: Directory of a saved row scoring model or the name of a public model
        :param column_model_name_or_path: Directory of a saved column scoring model or the name of a public model
        :param row_model_version: The version of row model to use from the HuggingFace model hub.
                                  Can be tag name, branch name, or commit hash.
        :param column_model_version: The version of column model to use from the HuggingFace model hub.
                                     Can be tag name, branch name, or commit hash.
        :param row_tokenizer: Name of the tokenizer for the row model (usually the same as model)
        :param column_tokenizer: Name of the tokenizer for the column model (usually the same as model)
        :param use_gpu: Whether to use GPU or CPU. Falls back on CPU if no GPU is available.
        :param top_k: The maximum number of answers to return
        :param max_seq_len: Max sequence length of one input table for the model. If the number of tokens of
                            query + table exceed max_seq_len, the table will be truncated by removing rows until the
                            input size fits the model.
        """
        # Save init parameters to enable export of component config as YAML
        self.set_config(row_model_name_or_path=row_model_name_or_path,
                        column_model_name_or_path=column_model_name_or_path, row_model_version=row_model_version,
                        column_model_version=column_model_version, row_tokenizer=row_tokenizer,
                        column_tokenizer=column_tokenizer, use_gpu=use_gpu, top_k=top_k, max_seq_len=max_seq_len)

        self.devices, _ = initialize_device_settings(use_cuda=use_gpu, multi_gpu=False)
        self.row_model = AutoModelForSequenceClassification.from_pretrained(row_model_name_or_path,
                                                                            revision=row_model_version)
        self.column_model = AutoModelForSequenceClassification.from_pretrained(row_model_name_or_path,
                                                                               revision=column_model_version)
        self.row_model.to(str(self.devices[0]))
        self.column_model.to(str(self.devices[0]))

        if row_tokenizer is None:
            try:
                self.row_tokenizer = AutoTokenizer.from_pretrained(row_model_name_or_path)
            # The existing RCI models on the model hub don't come with tokenizer vocab files.
            except TypeError:
                self.row_tokenizer = AutoTokenizer.from_pretrained("albert-base-v2")
        else:
            self.row_tokenizer = AutoTokenizer.from_pretrained(row_tokenizer)

        if column_tokenizer is None:
            try:
                self.column_tokenizer = AutoTokenizer.from_pretrained(column_model_name_or_path)
            # The existing RCI models on the model hub don't come with tokenizer vocab files.
            except TypeError:
                self.column_tokenizer = AutoTokenizer.from_pretrained("albert-base-v2")
        else:
            self.column_tokenizer = AutoTokenizer.from_pretrained(column_tokenizer)

        self.top_k = top_k
        self.max_seq_len = max_seq_len
        self.return_no_answers = False

    def predict(self, query: str, documents: List[Document], top_k: Optional[int] = None) -> Dict:
        """
        Use loaded RCI models to find answers for a query in the supplied list of Documents
        of content_type ``'table'``.

        Returns dictionary containing query and list of Answer objects sorted by (desc.) score.
        The existing RCI models on the HF model hub don"t allow aggregation, therefore, the answer will always be
        composed of a single cell.

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
                logger.warning(f"Skipping document with id {document.id} in RCIReader, as it is not of type table.")
                continue

            table: pd.DataFrame = document.content
            table = table.astype(str)
            # Create row and column representations
            row_reps, column_reps = self._create_row_column_representations(table)

            # Get row logits
            row_inputs = self.row_tokenizer.batch_encode_plus(
                batch_text_or_text_pairs=[(query, row_rep) for row_rep in row_reps],
                max_length=self.max_seq_len,
                return_tensors="pt",
                add_special_tokens=True,
                truncation=True,
                padding=True
            )
            row_inputs.to(self.devices[0])
            row_logits = self.row_model(**row_inputs)[0].detach().cpu().numpy()[:, 1]

            # Get column logits
            column_inputs = self.column_tokenizer.batch_encode_plus(
                batch_text_or_text_pairs=[(query, column_rep) for column_rep in column_reps],
                max_length=self.max_seq_len,
                return_tensors="pt",
                add_special_tokens=True,
                truncation=True,
                padding=True
            )
            column_inputs.to(self.devices[0])
            column_logits = self.column_model(**column_inputs)[0].detach().cpu().numpy()[:, 1]

            # Calculate cell scores
            current_answers: List[Answer] = []
            cell_scores_table: List[List[float]] = []
            for row_idx, row_score in enumerate(row_logits):
                cell_scores_table.append([])
                for col_idx, col_score in enumerate(column_logits):
                    current_cell_score = float(row_score + col_score)
                    cell_scores_table[-1].append(current_cell_score)

                    answer_str = table.iloc[row_idx, col_idx]
                    answer_offsets = self._calculate_answer_offsets(row_idx, col_idx, table)
                    current_answers.append(
                        Answer(
                            answer=answer_str,
                            type="extractive",
                            score=current_cell_score,
                            context=table,
                            offsets_in_document=[answer_offsets],
                            offsets_in_context=[answer_offsets],
                            document_id=document.id,
                        )
                    )

            # Add cell scores to Answers' meta to be able to use as heatmap
            for answer in current_answers:
                answer.meta = {"table_scores": cell_scores_table}
            answers.extend(current_answers)

        # Sort answers by score and select top-k answers
        answers = sorted(answers, reverse=True)
        answers = answers[:top_k]

        results = {"query": query,
                   "answers": answers}

        return results

    @staticmethod
    def _create_row_column_representations(table: pd.DataFrame) -> Tuple[List[str], List[str]]:
        row_reps = []
        column_reps = []
        columns = table.columns

        for idx, row in table.iterrows():
            current_row_rep = " * ".join([header + " : " + cell for header, cell in zip(columns, row)])
            row_reps.append(current_row_rep)

        for col_name in columns:
            current_column_rep = f"{col_name} * "
            current_column_rep += " * ".join(table[col_name])
            column_reps.append(current_column_rep)

        return row_reps, column_reps

    @staticmethod
    def _calculate_answer_offsets(row_idx, column_index, table) -> Span:
        n_rows, n_columns = table.shape
        answer_cell_offset = (row_idx * n_columns) + column_index

        return Span(start=answer_cell_offset, end=answer_cell_offset + 1)

    def predict_batch(self, query_doc_list: List[dict], top_k: Optional[int] = None, batch_size: Optional[int] = None):
        raise NotImplementedError("Batch prediction not yet available in RCIReader.")
