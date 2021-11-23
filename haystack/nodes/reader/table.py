from typing import List, Optional, Tuple, Dict

import logging
from statistics import mean
import torch
import numpy as np
import pandas as pd
from quantulum3 import parser
from transformers import TapasTokenizer, TapasForQuestionAnswering, BatchEncoding

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
