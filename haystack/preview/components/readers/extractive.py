from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import math
import warnings
import os

from haystack.preview import component, default_to_dict, ComponentError, Document, ExtractedAnswer
from haystack.preview.lazy_imports import LazyImport

with LazyImport("Run 'pip install transformers[torch,sentencepiece]'") as torch_and_transformers_import:
    from transformers import AutoModelForQuestionAnswering, AutoTokenizer
    from tokenizers import Encoding
    import torch


@component
class ExtractiveReader:
    """
    A component for performing extractive QA.
    Every possible answer span is assigned a confidence score independent of other answer spans. This fixes a common issue of other implementations which make comparisons across documents harder by normalising each document's answers independently.

    Example usage:
    ```python
    p = Pipeline()
    p.add_component(instance=InMemoryBM25Retriever(document_store=InMemoryDocumentStore()), name="retriever")
    p.add_component(instance=ExtractiveReader(), name="reader")
    p.connect("retriever", "reader")
    question = "Who lives in Berlin?"
    p.run({"retriever": {"query": question}, "reader": {"query": question}})
    ```
    """

    def __init__(
        self,
        model_name_or_path: Union[Path, str] = "deepset/roberta-base-squad2-distilled",
        device: Optional[str] = None,
        token: Union[bool, str, None] = None,
        top_k: int = 20,
        confidence_threshold: Optional[float] = None,
        max_seq_length: int = 384,
        stride: int = 128,
        max_batch_size: Optional[int] = None,
        answers_per_seq: Optional[int] = None,
        no_answer: bool = True,
        calibration_factor: float = 0.1,
        model_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Creates an ExtractiveReader
        :param model_name_or_path: A HuggingFace transformers question answering model.
            Can either be a path to a folder containing the model files or an identifier for the HF hub
            Default: `'deepset/roberta-base-squad2-distilled'`
        :param device: Pytorch device string. Uses GPU by default if available
        :param token: The API token used to download private models from Hugging Face.
            If this parameter is set to `True`, then the token generated when running
            `transformers-cli login` (stored in ~/.huggingface) will be used.
        :param top_k: Number of answers to return per query.
            It is required even if confidence_threshold is set. Defaults to 20.
            An additional answer is returned if no_answer is set to True (default).
        :param confidence_threshold: Answers with a confidence score below this value will not be returned
        :param max_seq_length: Maximum number of tokens.
            If exceeded by a sequence, the sequence will be split.
            Default: 384
        :param stride: Number of tokens that overlap when sequence is split because it exceeds max_seq_length
            Default: 128
        :param max_batch_size: Maximum number of samples that are fed through the model at the same time
        :param answers_per_seq: Number of answer candidates to consider per sequence.
            This is relevant when a document has been split into multiple sequence due to max_seq_length.
        :param no_answer: Whether to return no answer scores
        :param calibration_factor: Factor used for calibrating confidence scores
        :param model_kwargs: Additional keyword arguments passed to `AutoModelForQuestionAnswering.from_pretrained`
            when loading the model specified in `model_name_or_path`.
        """
        torch_and_transformers_import.check()
        self.model_name_or_path = str(model_name_or_path)
        self.model = None
        self.device = device
        self.token = token
        self.max_seq_length = max_seq_length
        self.top_k = top_k
        self.confidence_threshold = confidence_threshold
        self.stride = stride
        self.max_batch_size = max_batch_size
        self.answers_per_seq = answers_per_seq
        self.no_answer = no_answer
        self.calibration_factor = calibration_factor
        self.model_kwargs = model_kwargs or {}

    def _get_telemetry_data(self) -> Dict[str, Any]:
        """
        Data that is sent to Posthog for usage analytics.
        """
        return {"model": self.model_name_or_path}

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize this component to a dictionary.
        """
        return default_to_dict(
            self,
            model_name_or_path=self.model_name_or_path,
            device=self.device,
            token=self.token if not isinstance(self.token, str) else None,
            max_seq_length=self.max_seq_length,
            top_k=self.top_k,
            confidence_threshold=self.confidence_threshold,
            stride=self.stride,
            max_batch_size=self.max_batch_size,
            answers_per_seq=self.answers_per_seq,
            no_answer=self.no_answer,
            calibration_factor=self.calibration_factor,
            model_kwargs=self.model_kwargs,
        )

    def warm_up(self):
        if self.model is None:
            if torch.cuda.is_available():
                self.device = self.device or "cuda:0"
            elif (
                hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available()
                and os.getenv("HAYSTACK_MPS_ENABLED", "true") != "false"
            ):
                self.device = self.device or "mps:0"
            else:
                self.device = self.device or "cpu:0"

            self.model = AutoModelForQuestionAnswering.from_pretrained(
                self.model_name_or_path, token=self.token, **self.model_kwargs
            ).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, token=self.token)

    def _flatten_documents(
        self, queries: List[str], documents: List[List[Document]]
    ) -> Tuple[List[str], List[Document], List[int]]:
        """
        Flattens queries and documents so all query-document pairs are arranged along one batch axis.
        """
        flattened_queries = [query for documents_, query in zip(documents, queries) for _ in documents_]
        flattened_documents = [document for documents_ in documents for document in documents_]
        query_ids = [i for i, documents_ in enumerate(documents) for _ in documents_]
        return flattened_queries, flattened_documents, query_ids

    def _preprocess(
        self, queries: List[str], documents: List[Document], max_seq_length: int, query_ids: List[int], stride: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[Encoding], List[int], List[int]]:
        """
        Split and tokenise documents and preserve structures by returning mappings to query and document ids.
        """
        texts = []
        document_ids = []
        for i, doc in enumerate(documents):
            if doc.content is None:
                warnings.warn(
                    f"Document with id {doc.id} was passed to ExtractiveReader, but does not contain any text. It will be ignored."
                )
                continue
            texts.append(doc.content)
            document_ids.append(i)
        encodings_pt = self.tokenizer(
            queries,
            [document.content for document in documents],
            padding=True,
            truncation=True,
            max_length=max_seq_length,
            return_tensors="pt",
            return_overflowing_tokens=True,
            stride=stride,
        )

        input_ids = encodings_pt.input_ids.to(self.device)
        attention_mask = encodings_pt.attention_mask.to(self.device)

        query_ids = [query_ids[index] for index in encodings_pt.overflow_to_sample_mapping]
        document_ids = [document_ids[sample_id] for sample_id in encodings_pt.overflow_to_sample_mapping]

        encodings = encodings_pt.encodings
        sequence_ids = torch.tensor(
            [[id_ if id_ is not None else -1 for id_ in encoding.sequence_ids] for encoding in encodings]
        ).to(self.device)

        return input_ids, attention_mask, sequence_ids, encodings, query_ids, document_ids

    def _postprocess(
        self,
        start: torch.Tensor,
        end: torch.Tensor,
        sequence_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        answers_per_seq: int,
        encodings: List[Encoding],
    ) -> Tuple[List[List[int]], List[List[int]], torch.Tensor]:
        """
        Turn start and end logits into confidence scores for each answer span. Unlike most other implementations, there is no normalisation in each split to make the scores more comparable across different splits. The top k answer spans are returned.
        """
        mask = sequence_ids == 1
        mask = torch.logical_and(mask, attention_mask == 1)
        start = torch.where(mask, start, -torch.inf)
        end = torch.where(mask, end, -torch.inf)
        start = start.unsqueeze(-1)
        end = end.unsqueeze(-2)

        logits = start + end  # shape: (batch_size, seq_length (start), seq_length (end))
        mask = torch.ones(logits.shape[-2:], dtype=torch.bool, device=self.device)
        mask = torch.triu(mask)  # End shouldn't be before start
        masked_logits = torch.where(mask, logits, -torch.inf)
        probabilities = torch.sigmoid(masked_logits * self.calibration_factor)

        flat_probabilities = probabilities.flatten(-2, -1)  # necessary for topk
        candidates = torch.topk(flat_probabilities, answers_per_seq)
        seq_length = logits.shape[-1]
        start_candidates = candidates.indices // seq_length  # Recover indices from flattening
        end_candidates = candidates.indices % seq_length
        start_candidates = start_candidates.cpu()
        end_candidates = end_candidates.cpu()

        start_candidates_char_indices = [
            [encoding.token_to_chars(start)[0] for start in candidates]
            for candidates, encoding in zip(start_candidates, encodings)
        ]
        end_candidates_char_indices = [
            [encoding.token_to_chars(end)[1] for end in candidates]
            for candidates, encoding in zip(end_candidates, encodings)
        ]
        probabilities = candidates.values.cpu()

        return start_candidates_char_indices, end_candidates_char_indices, probabilities

    def _nest_answers(
        self,
        start: List[List[int]],
        end: List[List[int]],
        probabilities: torch.Tensor,
        flattened_documents: List[Document],
        queries: List[str],
        answers_per_seq: int,
        top_k: Optional[int],
        confidence_threshold: Optional[float],
        query_ids: List[int],
        document_ids: List[int],
        no_answer: bool,
    ) -> List[List[ExtractedAnswer]]:
        """
        Reconstructs the nested structure that existed before flattening. Also computes a no answer probability. This probability is different from most other implementations because it does not consider the no answer logit introduced with SQuAD 2. Instead, it just computes the probability that the answer does not exist in the top k or top p.
        """
        flat_answers_without_queries = []
        for document_id, start_candidates_, end_candidates_, probabilities_ in zip(
            document_ids, start, end, probabilities
        ):
            for start_, end_, probability in zip(start_candidates_, end_candidates_, probabilities_):
                doc = flattened_documents[document_id]
                flat_answers_without_queries.append({"data": doc.content[start_:end_], "document": doc, "probability": probability.item(), "start": start_, "end": end_, "metadata": {}})  # type: ignore # doc.content cannot be None, because those documents are filtered when preprocessing. However, mypy doesn't know that.
        i = 0
        nested_answers = []
        for query_id in range(query_ids[-1] + 1):
            current_answers = []
            while i < len(flat_answers_without_queries) and query_ids[i // answers_per_seq] == query_id:
                answer = flat_answers_without_queries[i]
                answer["query"] = queries[query_id]
                current_answers.append(ExtractedAnswer(**answer))
                i += 1
            current_answers = sorted(current_answers, key=lambda answer: answer.probability, reverse=True)
            current_answers = current_answers[:top_k]
            if no_answer:
                no_answer_probability = math.prod(1 - answer.probability for answer in current_answers)
                answer_ = ExtractedAnswer(
                    data=None, query=queries[query_id], metadata={}, document=None, probability=no_answer_probability
                )
                current_answers.append(answer_)
            current_answers = sorted(current_answers, key=lambda answer: answer.probability, reverse=True)
            if confidence_threshold is not None:
                current_answers = [answer for answer in current_answers if answer.probability >= confidence_threshold]
            nested_answers.append(current_answers)

        return nested_answers

    @component.output_types(answers=List[ExtractedAnswer])
    def run(
        self,
        query: str,
        documents: List[Document],
        top_k: Optional[int] = None,
        confidence_threshold: Optional[float] = None,
        max_seq_length: Optional[int] = None,
        stride: Optional[int] = None,
        max_batch_size: Optional[int] = None,
        answers_per_seq: Optional[int] = None,
        no_answer: Optional[bool] = None,
    ):
        """
        Performs extractive QA on the given documents using the given query.

        :param query: Query string.
        :param documents: List of Documents to search for an answer to the query.
        :param top_k: The maximum number of answers to return.
            An additional answer is returned if no_answer is set to True (default).
        :return: List of ExtractedAnswers sorted by (desc.) answer score.
        """
        queries = [query]  # Temporary solution until we have decided what batching should look like in v2
        nested_documents = [documents]
        if self.model is None:
            raise ComponentError("The component was not warmed up. Run 'warm_up()' before calling 'run()'.")

        top_k = top_k or self.top_k
        confidence_threshold = confidence_threshold or self.confidence_threshold
        max_seq_length = max_seq_length or self.max_seq_length
        stride = stride or self.stride
        max_batch_size = max_batch_size or self.max_batch_size
        answers_per_seq = answers_per_seq or self.answers_per_seq or top_k or 20
        no_answer = no_answer if no_answer is not None else self.no_answer

        flattened_queries, flattened_documents, query_ids = self._flatten_documents(queries, nested_documents)
        input_ids, attention_mask, sequence_ids, encodings, query_ids, document_ids = self._preprocess(
            flattened_queries, flattened_documents, max_seq_length, query_ids, stride
        )

        num_batches = math.ceil(input_ids.shape[0] / max_batch_size) if max_batch_size else 1
        batch_size = max_batch_size or input_ids.shape[0]

        start_logits_list = []
        end_logits_list = []

        for i in range(num_batches):
            start_index = i * batch_size
            end_index = start_index + batch_size
            cur_input_ids = input_ids[start_index:end_index]
            cur_attention_mask = attention_mask[start_index:end_index]

            output = self.model(input_ids=cur_input_ids, attention_mask=cur_attention_mask)
            cur_start_logits = output.start_logits
            cur_end_logits = output.end_logits
            if num_batches != 1:
                cur_start_logits = cur_start_logits.cpu()
                cur_end_logits = cur_end_logits.cpu()
            start_logits_list.append(cur_start_logits)
            end_logits_list.append(cur_end_logits)

        start_logits = torch.cat(start_logits_list)
        end_logits = torch.cat(end_logits_list)

        start, end, probabilities = self._postprocess(
            start_logits, end_logits, sequence_ids, attention_mask, answers_per_seq, encodings
        )

        answers = self._nest_answers(
            start,
            end,
            probabilities,
            flattened_documents,
            queries,
            answers_per_seq,
            top_k,
            confidence_threshold,
            query_ids,
            document_ids,
            no_answer,
        )

        return {"answers": answers[0]}  # same temporary batching fix as above
