# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import math
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from haystack import Document, ExtractedAnswer, component, default_from_dict, default_to_dict, logging
from haystack.lazy_imports import LazyImport
from haystack.utils import ComponentDevice, DeviceMap, Secret, deserialize_secrets_inplace
from haystack.utils.hf import deserialize_hf_model_kwargs, resolve_hf_device_map, serialize_hf_model_kwargs

with LazyImport("Run 'pip install transformers[torch,sentencepiece]'") as torch_and_transformers_import:
    import accelerate  # pylint: disable=unused-import # the library is used but not directly referenced
    import torch
    from tokenizers import Encoding
    from transformers import AutoModelForQuestionAnswering, AutoTokenizer


logger = logging.getLogger(__name__)


@component
class ExtractiveReader:
    """
    Locates and extracts answers to a given query from Documents.

    The ExtractiveReader component performs extractive question answering.
    It assigns a score to every possible answer span independently of other answer spans.
    This fixes a common issue of other implementations which make comparisons across documents harder by normalizing
    each document's answers independently.

    Example usage:
    ```python
    from haystack import Document
    from haystack.components.readers import ExtractiveReader

    docs = [
        Document(content="Python is a popular programming language"),
        Document(content="python ist eine beliebte Programmiersprache"),
    ]

    reader = ExtractiveReader()
    reader.warm_up()

    question = "What is a popular programming language?"
    result = reader.run(query=question, documents=docs)
    assert "Python" in result["answers"][0].data
    ```
    """

    def __init__(
        self,
        model: Union[Path, str] = "deepset/roberta-base-squad2-distilled",
        device: Optional[ComponentDevice] = None,
        token: Optional[Secret] = Secret.from_env_var(["HF_API_TOKEN", "HF_TOKEN"], strict=False),
        top_k: int = 20,
        score_threshold: Optional[float] = None,
        max_seq_length: int = 384,
        stride: int = 128,
        max_batch_size: Optional[int] = None,
        answers_per_seq: Optional[int] = None,
        no_answer: bool = True,
        calibration_factor: float = 0.1,
        overlap_threshold: Optional[float] = 0.01,
        model_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Creates an instance of ExtractiveReader.

        :param model:
            A Hugging Face transformers question answering model.
            Can either be a path to a folder containing the model files or an identifier for the Hugging Face hub.
        :param device:
            The device on which the model is loaded. If `None`, the default device is automatically selected.
        :param token:
            The API token used to download private models from Hugging Face.
        :param top_k:
            Number of answers to return per query. It is required even if score_threshold is set.
            An additional answer with no text is returned if no_answer is set to True (default).
        :param score_threshold:
            Returns only answers with the probability score above this threshold.
        :param max_seq_length:
            Maximum number of tokens. If a sequence exceeds it, the sequence is split.
        :param stride:
            Number of tokens that overlap when sequence is split because it exceeds max_seq_length.
        :param max_batch_size:
            Maximum number of samples that are fed through the model at the same time.
        :param answers_per_seq:
            Number of answer candidates to consider per sequence.
            This is relevant when a Document was split into multiple sequences because of max_seq_length.
        :param no_answer:
            Whether to return an additional `no answer` with an empty text and a score representing the
            probability that the other top_k answers are incorrect.
        :param calibration_factor:
            Factor used for calibrating probabilities.
        :param overlap_threshold:
            If set this will remove duplicate answers if they have an overlap larger than the
            supplied threshold. For example, for the answers "in the river in Maine" and "the river" we would remove
            one of these answers since the second answer has a 100% (1.0) overlap with the first answer.
            However, for the answers "the river in" and "in Maine" there is only a max overlap percentage of 25% so
            both of these answers could be kept if this variable is set to 0.24 or lower.
            If None is provided then all answers are kept.
        :param model_kwargs:
            Additional keyword arguments passed to `AutoModelForQuestionAnswering.from_pretrained`
            when loading the model specified in `model`. For details on what kwargs you can pass,
            see the model's documentation.
        """
        torch_and_transformers_import.check()
        self.model_name_or_path = str(model)
        self.model = None
        self.tokenizer = None
        self.device = None
        self.token = token
        self.max_seq_length = max_seq_length
        self.top_k = top_k
        self.score_threshold = score_threshold
        self.stride = stride
        self.max_batch_size = max_batch_size
        self.answers_per_seq = answers_per_seq
        self.no_answer = no_answer
        self.calibration_factor = calibration_factor
        self.overlap_threshold = overlap_threshold

        model_kwargs = resolve_hf_device_map(device=device, model_kwargs=model_kwargs)
        self.model_kwargs = model_kwargs

    def _get_telemetry_data(self) -> Dict[str, Any]:
        """
        Data that is sent to Posthog for usage analytics.
        """
        return {"model": self.model_name_or_path}

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        serialization_dict = default_to_dict(
            self,
            model=self.model_name_or_path,
            device=None,
            token=self.token.to_dict() if self.token else None,
            max_seq_length=self.max_seq_length,
            top_k=self.top_k,
            score_threshold=self.score_threshold,
            stride=self.stride,
            max_batch_size=self.max_batch_size,
            answers_per_seq=self.answers_per_seq,
            no_answer=self.no_answer,
            calibration_factor=self.calibration_factor,
            model_kwargs=self.model_kwargs,
        )

        serialize_hf_model_kwargs(serialization_dict["init_parameters"]["model_kwargs"])
        return serialization_dict

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExtractiveReader":
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :returns:
            Deserialized component.
        """
        init_params = data["init_parameters"]
        deserialize_secrets_inplace(init_params, keys=["token"])
        if init_params.get("device") is not None:
            init_params["device"] = ComponentDevice.from_dict(init_params["device"])
        if init_params.get("model_kwargs") is not None:
            deserialize_hf_model_kwargs(init_params["model_kwargs"])

        return default_from_dict(cls, data)

    def warm_up(self):
        """
        Initializes the component.
        """
        # Take the first device used by `accelerate`. Needed to pass inputs from the tokenizer to the correct device.
        if self.model is None:
            self.model = AutoModelForQuestionAnswering.from_pretrained(
                self.model_name_or_path, token=self.token.resolve_value() if self.token else None, **self.model_kwargs
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name_or_path, token=self.token.resolve_value() if self.token else None
            )
            self.device = ComponentDevice.from_multiple(device_map=DeviceMap.from_hf(self.model.hf_device_map))

    def _flatten_documents(
        self, queries: List[str], documents: List[List[Document]]
    ) -> Tuple[List[str], List[Document], List[int]]:
        """
        Flattens queries and Documents so all query-document pairs are arranged along one batch axis.
        """
        flattened_queries = [query for documents_, query in zip(documents, queries) for _ in documents_]
        flattened_documents = [document for documents_ in documents for document in documents_]
        query_ids = [i for i, documents_ in enumerate(documents) for _ in documents_]
        return flattened_queries, flattened_documents, query_ids

    def _preprocess(
        self, queries: List[str], documents: List[Document], max_seq_length: int, query_ids: List[int], stride: int
    ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", List["Encoding"], List[int], List[int]]:
        """
        Splits and tokenizes Documents and preserves structures by returning mappings to query and Document IDs.
        """
        texts = []
        document_ids = []
        document_contents = []
        for i, doc in enumerate(documents):
            if doc.content is None:
                warnings.warn(
                    f"Document with id {doc.id} was passed to ExtractiveReader. The Document doesn't "
                    f"contain any text and it will be ignored."
                )
                continue
            texts.append(doc.content)
            document_ids.append(i)
            document_contents.append(doc.content)

        encodings_pt = self.tokenizer(  # type: ignore
            queries,
            document_contents,
            padding=True,
            truncation=True,
            max_length=max_seq_length,
            return_tensors="pt",
            return_overflowing_tokens=True,
            stride=stride,
        )

        # To make mypy happy even though self.device is set in warm_up()
        assert self.device is not None
        assert self.device.first_device is not None

        # Take the first device used by `accelerate`. Needed to pass inputs from the tokenizer to the correct device.
        first_device = self.device.first_device.to_torch()

        input_ids = encodings_pt.input_ids.to(first_device)
        attention_mask = encodings_pt.attention_mask.to(first_device)

        query_ids = [query_ids[index] for index in encodings_pt.overflow_to_sample_mapping]
        document_ids = [document_ids[sample_id] for sample_id in encodings_pt.overflow_to_sample_mapping]

        encodings = encodings_pt.encodings
        sequence_ids = torch.tensor(
            [[id_ if id_ is not None else -1 for id_ in encoding.sequence_ids] for encoding in encodings]
        ).to(first_device)

        return input_ids, attention_mask, sequence_ids, encodings, query_ids, document_ids

    def _postprocess(
        self,
        start: "torch.Tensor",
        end: "torch.Tensor",
        sequence_ids: "torch.Tensor",
        attention_mask: "torch.Tensor",
        answers_per_seq: int,
        encodings: List["Encoding"],
    ) -> Tuple[List[List[int]], List[List[int]], "torch.Tensor"]:
        """
        Turns start and end logits into probabilities for each answer span.

        Unlike most other implementations, it doesn't normalize the scores in each split to make them easier to
        compare across different splits. Returns the top k answer spans.
        """
        mask = sequence_ids == 1  # Only keep tokens from the context (should ignore special tokens)
        mask = torch.logical_and(mask, attention_mask == 1)  # Definitely remove special tokens
        start = torch.where(mask, start, -torch.inf)  # Apply the mask on the start logits
        end = torch.where(mask, end, -torch.inf)  # Apply the mask on the end logits
        start = start.unsqueeze(-1)
        end = end.unsqueeze(-2)

        logits = start + end  # shape: (batch_size, seq_length (start), seq_length (end))

        # The mask here onwards is the same for all instances in the batch
        # As such we do away with the batch dimension
        mask = torch.ones(logits.shape[-2:], dtype=torch.bool, device=logits.device)
        mask = torch.triu(mask)  # End shouldn't be before start
        masked_logits = torch.where(mask, logits, -torch.inf)
        probabilities = torch.sigmoid(masked_logits * self.calibration_factor)

        flat_probabilities = probabilities.flatten(-2, -1)  # necessary for topk

        # topk can return invalid candidates as well if answers_per_seq > num_valid_candidates
        # We only keep probability > 0 candidates later on
        candidates = torch.topk(flat_probabilities, answers_per_seq)
        seq_length = logits.shape[-1]
        start_candidates = candidates.indices // seq_length  # Recover indices from flattening
        end_candidates = candidates.indices % seq_length
        candidates_values = candidates.values.cpu()
        start_candidates = start_candidates.cpu()
        end_candidates = end_candidates.cpu()

        start_candidates_tokens_to_chars = []
        end_candidates_tokens_to_chars = []
        for i, (s_candidates, e_candidates, encoding) in enumerate(zip(start_candidates, end_candidates, encodings)):
            # Those with probabilities > 0 are valid
            valid = candidates_values[i] > 0
            s_char_spans = []
            e_char_spans = []
            for start_token, end_token in zip(s_candidates[valid], e_candidates[valid]):
                # token_to_chars returns `None` for special tokens
                # But we shouldn't have special tokens in the answers at this point
                # The whole span is given by the start of the start_token (index 0)
                # and the end of the end token (index 1)
                s_char_spans.append(encoding.token_to_chars(start_token)[0])
                e_char_spans.append(encoding.token_to_chars(end_token)[1])
            start_candidates_tokens_to_chars.append(s_char_spans)
            end_candidates_tokens_to_chars.append(e_char_spans)

        return start_candidates_tokens_to_chars, end_candidates_tokens_to_chars, candidates_values

    def _add_answer_page_number(self, answer: ExtractedAnswer) -> ExtractedAnswer:
        if answer.meta is None:
            answer.meta = {}

        if answer.document_offset is None:
            return answer

        if not answer.document or "page_number" not in answer.document.meta:
            return answer

        if not isinstance(answer.document.meta["page_number"], int):
            logger.warning(
                f"Document's page_number must be int but is {type(answer.document.meta['page_number'])}. "
                f"No page number will be added to the answer."
            )
            return answer

        # Calculate the answer page number
        if answer.document.content:
            ans_start = answer.document_offset.start
            answer_page_number = answer.document.meta["page_number"] + answer.document.content[:ans_start].count("\f")
            answer.meta.update({"answer_page_number": answer_page_number})

        return answer

    def _nest_answers(
        self,
        start: List[List[int]],
        end: List[List[int]],
        probabilities: "torch.Tensor",
        flattened_documents: List[Document],
        queries: List[str],
        answers_per_seq: int,
        top_k: Optional[int],
        score_threshold: Optional[float],
        query_ids: List[int],
        document_ids: List[int],
        no_answer: bool,
        overlap_threshold: Optional[float],
    ) -> List[List[ExtractedAnswer]]:
        """
        Reconstructs the nested structure that existed before flattening.

        Also computes a no answer score. This score is different from most other implementations because it does not
        consider the no answer logit introduced with SQuAD 2. Instead, it just computes the probability that the
        answer does not exist in the top k or top p.
        """
        answers_without_query = []
        for document_id, start_candidates_, end_candidates_, probabilities_ in zip(
            document_ids, start, end, probabilities
        ):
            for start_, end_, probability in zip(start_candidates_, end_candidates_, probabilities_):
                doc = flattened_documents[document_id]
                answers_without_query.append(
                    ExtractedAnswer(
                        query="",  # Can't be None but we'll add it later
                        data=doc.content[start_:end_],  # type: ignore
                        document=doc,
                        score=probability.item(),
                        document_offset=ExtractedAnswer.Span(start_, end_),
                        meta={},
                    )
                )
        i = 0
        nested_answers = []
        for query_id in range(query_ids[-1] + 1):
            current_answers = []
            while i < len(answers_without_query) and query_ids[i // answers_per_seq] == query_id:
                answer = answers_without_query[i]
                answer.query = queries[query_id]
                current_answers.append(answer)
                i += 1
            current_answers = sorted(current_answers, key=lambda ans: ans.score, reverse=True)
            current_answers = self.deduplicate_by_overlap(current_answers, overlap_threshold=overlap_threshold)
            current_answers = current_answers[:top_k]

            # Calculate the answer page number and add it to meta
            current_answers = [self._add_answer_page_number(answer=answer) for answer in current_answers]

            if no_answer:
                no_answer_score = math.prod(1 - answer.score for answer in current_answers)
                answer_ = ExtractedAnswer(
                    data=None, query=queries[query_id], meta={}, document=None, score=no_answer_score
                )
                current_answers.append(answer_)
            current_answers = sorted(current_answers, key=lambda ans: ans.score, reverse=True)
            if score_threshold is not None:
                current_answers = [answer for answer in current_answers if answer.score >= score_threshold]
            nested_answers.append(current_answers)

        return nested_answers

    def _calculate_overlap(self, answer1_start: int, answer1_end: int, answer2_start: int, answer2_end: int) -> int:
        """
        Calculates the amount of overlap (in number of characters) between two answer offsets.

        This Stack overflow
        [post](https://stackoverflow.com/questions/325933/determine-whether-two-date-ranges-overlap/325964#325964)
        explains how to calculate the overlap between two ranges.
        """
        # Check for overlap: (StartA <= EndB) and (StartB <= EndA)
        if answer1_start <= answer2_end and answer2_start <= answer1_end:
            return min(
                answer1_end - answer1_start,
                answer1_end - answer2_start,
                answer2_end - answer1_start,
                answer2_end - answer2_start,
            )
        return 0

    def _should_keep(
        self, candidate_answer: ExtractedAnswer, current_answers: List[ExtractedAnswer], overlap_threshold: float
    ) -> bool:
        """
        Determines if the answer should be kept based on how much it overlaps with previous answers.

        NOTE: We might want to avoid throwing away answers that only have a few character (or word) overlap:
            - E.g. The answers "the river in" and "in Maine" from the context "I want to go to the river in Maine."
            might both want to be kept.

        :param candidate_answer:
            Candidate answer that will be checked if it should be kept.
        :param current_answers:
            Current list of answers that will be kept.
        :param overlap_threshold:
            If the overlap between two answers is greater than this threshold then return False.
        """
        keep = True

        # If the candidate answer doesn't have a document keep it
        if not candidate_answer.document:
            return keep

        for ans in current_answers:
            # If an answer in current_answers doesn't have a document skip the comparison
            if not ans.document:
                continue

            # If offset is missing then keep both
            if ans.document_offset is None:
                continue

            # If offset is missing then keep both
            if candidate_answer.document_offset is None:
                continue

            # If the answers come from different documents then keep both
            if candidate_answer.document.id != ans.document.id:
                continue

            overlap_len = self._calculate_overlap(
                answer1_start=ans.document_offset.start,
                answer1_end=ans.document_offset.end,
                answer2_start=candidate_answer.document_offset.start,
                answer2_end=candidate_answer.document_offset.end,
            )

            # If overlap is 0 then keep
            if overlap_len == 0:
                continue

            overlap_frac_answer1 = overlap_len / (ans.document_offset.end - ans.document_offset.start)
            overlap_frac_answer2 = overlap_len / (
                candidate_answer.document_offset.end - candidate_answer.document_offset.start
            )

            if overlap_frac_answer1 > overlap_threshold or overlap_frac_answer2 > overlap_threshold:
                keep = False
                break

        return keep

    def deduplicate_by_overlap(
        self, answers: List[ExtractedAnswer], overlap_threshold: Optional[float]
    ) -> List[ExtractedAnswer]:
        """
        De-duplicates overlapping Extractive Answers.

        De-duplicates overlapping Extractive Answers from the same document based on how much the spans of the
        answers overlap.

        :param answers:
            List of answers to be deduplicated.
        :param overlap_threshold:
            If set this will remove duplicate answers if they have an overlap larger than the
            supplied threshold. For example, for the answers "in the river in Maine" and "the river" we would remove
            one of these answers since the second answer has a 100% (1.0) overlap with the first answer.
            However, for the answers "the river in" and "in Maine" there is only a max overlap percentage of 25% so
            both of these answers could be kept if this variable is set to 0.24 or lower.
            If None is provided then all answers are kept.
        :returns:
            List of deduplicated answers.
        """
        if overlap_threshold is None:
            return answers

        # Initialize with the first answer and its offsets_in_document
        deduplicated_answers = [answers[0]]

        # Loop over remaining answers to check for overlaps
        for ans in answers[1:]:
            keep = self._should_keep(
                candidate_answer=ans, current_answers=deduplicated_answers, overlap_threshold=overlap_threshold
            )
            if keep:
                deduplicated_answers.append(ans)

        return deduplicated_answers

    @component.output_types(answers=List[ExtractedAnswer])
    def run(
        self,
        query: str,
        documents: List[Document],
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        max_seq_length: Optional[int] = None,
        stride: Optional[int] = None,
        max_batch_size: Optional[int] = None,
        answers_per_seq: Optional[int] = None,
        no_answer: Optional[bool] = None,
        overlap_threshold: Optional[float] = None,
    ):
        """
        Locates and extracts answers from the given Documents using the given query.

        :param query:
            Query string.
        :param documents:
            List of Documents in which you want to search for an answer to the query.
        :param top_k:
            The maximum number of answers to return.
            An additional answer is returned if no_answer is set to True (default).
        :param score_threshold:
            Returns only answers with the score above this threshold.
        :param max_seq_length:
            Maximum number of tokens. If a sequence exceeds it, the sequence is split.
        :param stride:
            Number of tokens that overlap when sequence is split because it exceeds max_seq_length.
        :param max_batch_size:
            Maximum number of samples that are fed through the model at the same time.
        :param answers_per_seq:
            Number of answer candidates to consider per sequence.
            This is relevant when a Document was split into multiple sequences because of max_seq_length.
        :param no_answer:
            Whether to return no answer scores.
        :param overlap_threshold:
            If set this will remove duplicate answers if they have an overlap larger than the
            supplied threshold. For example, for the answers "in the river in Maine" and "the river" we would remove
            one of these answers since the second answer has a 100% (1.0) overlap with the first answer.
            However, for the answers "the river in" and "in Maine" there is only a max overlap percentage of 25% so
            both of these answers could be kept if this variable is set to 0.24 or lower.
            If None is provided then all answers are kept.
        :returns:
            List of answers sorted by (desc.) answer score.

        :raises RuntimeError:
            If the component was not warmed up by calling 'warm_up()' before.
        """
        if self.model is None:
            raise RuntimeError(
                "The component ExtractiveReader was not warmed up. Run 'warm_up()' before calling 'run()'."
            )

        if not documents:
            return {"answers": []}

        queries = [query]  # Temporary solution until we have decided what batching should look like in v2
        nested_documents = [documents]
        top_k = top_k or self.top_k
        score_threshold = score_threshold or self.score_threshold
        max_seq_length = max_seq_length or self.max_seq_length
        stride = stride or self.stride
        max_batch_size = max_batch_size or self.max_batch_size
        answers_per_seq = answers_per_seq or self.answers_per_seq or 20
        no_answer = no_answer if no_answer is not None else self.no_answer
        overlap_threshold = overlap_threshold or self.overlap_threshold

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

            with torch.inference_mode():
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
            start=start,
            end=end,
            probabilities=probabilities,
            flattened_documents=flattened_documents,
            queries=queries,
            answers_per_seq=answers_per_seq,
            top_k=top_k,
            score_threshold=score_threshold,
            query_ids=query_ids,
            document_ids=document_ids,
            no_answer=no_answer,
            overlap_threshold=overlap_threshold,
        )

        return {"answers": answers[0]}  # same temporary batching fix as above
