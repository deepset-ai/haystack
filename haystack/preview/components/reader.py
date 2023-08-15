from pathlib import Path
from typing import List, Optional, Tuple, Union
import os
from haystack.preview import component, Document, Answer, Pipeline
from haystack.lazy_imports import LazyImport
from haystack.nodes.prompt.invocation_layer.utils import get_task

with LazyImport(message="Run 'pip install farm-haystack[inference]'") as torch_and_transformers_import:
    from transformers import AutoModelForQuestionAnswering, AutoTokenizer
    from tokenizers import Encoding
    import torch


@component
class ExtractiveReader:
    def __init__(
        self,
        model: Union[Path, str],
        device: Optional[str] = None,
        max_seq_length: int = 384,
        top_k: int = 10,
        stride: int = 128,
    ) -> None:
        torch_and_transformers_import.check()
        self.model = str(model)
        self.device = device
        self.loaded = False
        self.max_seq_length = max_seq_length
        self.top_k = top_k
        self.stride = stride
        if os.path.exists(self.model):
            return
        try:
            task = get_task(self.model)
            if task != "question-answering":
                raise ValueError(f"The provided model does not support question answering. Its intended use is {task}.")
        except RuntimeError:
            raise ValueError(
                "The provided model either does not exist or it does not have a HF transformers compatible format."
            )

    def warm_up(self):
        if not self.loaded:
            if torch.cuda.is_available():
                self.device = "cuda:0"
            else:
                self.device = "cpu:0"
            self.model_ = AutoModelForQuestionAnswering.from_pretrained(self.model).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model)

    def _flatten(self, queries: List[str], documents: List[List[Document]]) -> Tuple[List[str], List[str], List[int]]:
        flattened_queries = [query for documents_, query in zip(documents, queries) for _ in documents_]
        flattened_documents = [document.content for documents_ in documents for document in documents_]
        query_ids = [i for i, documents_ in enumerate(documents) for _ in documents_]
        return flattened_queries, flattened_documents, query_ids

    def _preprocess(
        self, queries: List[str], documents: List[str], max_seq_length: int, query_ids: List[int], stride: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[Encoding], List[int], List[int]]:
        encodings = self.tokenizer(
            queries,
            documents,
            padding=True,
            truncation=True,
            max_length=max_seq_length,
            return_tensors="pt",
            return_overflowing_tokens=True,
            stride=stride,
        )

        input_ids = encodings.input_ids.to(self.device)
        attention_mask = encodings.attention_mask.to(self.device)

        query_ids = [query_ids[index] for index in encodings.overflow_to_sample_mapping]
        document_ids = encodings.overflow_to_sample_mapping

        encodings = encodings.encodings
        sequence_ids = torch.tensor(
            [[id_ if id_ is not None else -1 for id_ in encoding.sequence_ids] for encoding in encodings]
        ).to(self.device)

        return input_ids, attention_mask, sequence_ids, encodings, query_ids, document_ids

    def _postprocess(
        self,
        output,
        sequence_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        top_k: int,
        encodings: List[Encoding],
        query_ids: List[int],
    ) -> Tuple[List[List[int]], List[List[int]], List[List[float]]]:
        start = output.start_logits
        end = output.end_logits
        mask = sequence_ids == 1
        mask[..., 0] = True  # type: ignore # mypy doesn't understand broadcasting
        mask = torch.logical_and(mask, attention_mask == 1)  # type: ignore # see above
        start = torch.where(mask, start, -torch.inf)  # type: ignore # see above
        end = torch.where(mask, end, -torch.inf)  # type: ignore # see above
        start = start.unsqueeze(-1)
        end = end.unsqueeze(-2)

        logits = start + end  # shape: (batch_size, seq_length (start), seq_length (end))
        mask = torch.ones(logits.shape[-2:], dtype=bool)
        mask = torch.triu(mask)  # End shouldn't be before start
        mask[0, :1] = False
        masked_logits = torch.where(mask, logits, -torch.inf)

        last_docs = []
        not_last_docs = []
        i = 0
        for j in range(query_ids[-1]):
            while i < len(query_ids) and query_ids[i] == j:
                if i != 0:
                    not_last_docs.append(i - 1)
                i += 1
            last_docs.append(i - 1)

        no_answer_logits = masked_logits[..., 0, 0]
        no_answer_logits_sum = torch.cumsum(no_answer_logits, -1)
        no_answer_logits_sum[last_docs[1:]] -= no_answer_logits_sum[last_docs[:-1]]
        no_answer_logits_sum[not_last_docs] = -torch.inf
        masked_logits[..., 0, 0] = no_answer_logits_sum
        exp_logits = torch.exp(masked_logits)
        exp_logits_sum = torch.sum(exp_logits, -1)
        exp_logits_sum = torch.sum(exp_logits_sum, -1)
        exp_logits_sum = torch.cumsum(exp_logits_sum, -1)
        exp_logits_sum[not_last_docs] = 0
        exp_logits_sum[last_docs[1:]] -= exp_logits_sum[last_docs[:-1]]
        rev_sum = torch.flip(exp_logits_sum, [-1])
        rev_cumsum = torch.cumsum(rev_sum, -1)
        exp_logits_sum[last_docs[:-1]] = exp_logits_sum[last_docs[1:]]
        exp_logits_sum[-1] = 0
        rev_sum_shifted = torch.flip(exp_logits_sum, [-1])
        rev_sum_shifted = torch.cumsum(rev_sum_shifted, -1)
        rev_cumsum -= rev_sum_shifted
        exp_sum_per_query = torch.flip(rev_cumsum, [-1])
        exp_logits /= exp_sum_per_query.unsqueeze(-1).unsqueeze(-1)
        flat_logits = masked_logits.flatten(-2, -1)  # necessary for topk
        candidates = torch.topk(flat_logits, top_k)
        seq_length = logits.shape[-1]
        start_candidates = candidates.indices // seq_length  # Recover indices from flattening
        end_candidates = candidates.indices % seq_length
        start_candidates = start_candidates.cpu()
        end_candidates = end_candidates.cpu()

        start_candidates = [
            [encoding.token_to_chars(start)[0] if start != 0 else 0 for start in candidates]
            for candidates, encoding in zip(start_candidates, encodings)
        ]
        end_candidates = [
            [encoding.token_to_chars(end)[1] if end != 0 else 0 for end in candidates]
            for candidates, encoding in zip(end_candidates, encodings)
        ]
        probabilities = candidates.values.cpu().tolist()

        return start_candidates, end_candidates, probabilities

    def _unflatten(
        self,
        start: List[List[int]],
        end: List[List[int]],
        probabilities: List[List[float]],
        flattened_documents: List[str],
        documents: List[List[Document]],
        top_k: int,
        query_ids: List[int],
        document_ids: List[int],
    ) -> List[List[Answer]]:
        answers = [
            (flattened_documents[document_id][start:end], probability)
            for document_id, start_candidates_, end_candidates_, probabilities_ in zip(
                document_ids, start, end, probabilities
            )
            for start, end, probability in zip(start_candidates_, end_candidates_, probabilities_)
        ]

        i = 0
        nested_answers = []
        for query_id in range(query_ids[-1] + 1):
            current_answers = []
            while i < len(answers) and query_ids[i // top_k] == query_id:
                current_answers.append(answers[i])
                i += 1
            current_answers = sorted(current_answers, key=lambda answer: answer[1], reverse=True)[:top_k]
            nested_answers.append(current_answers)

        return nested_answers

    @component.output_types(answers=List[List[Answer]])
    def run(
        self,
        queries: List[str],
        documents: List[List[Document]],
        top_k: Optional[int] = None,
        max_seq_length: Optional[int] = None,
        stride: Optional[int] = None,
    ):
        top_k = top_k or self.top_k
        max_seq_length = max_seq_length or self.max_seq_length
        stride = stride or self.stride

        flattened_queries, flattened_documents, query_ids = self._flatten(queries, documents)
        input_ids, attention_mask, sequence_ids, encodings, query_ids, document_ids = self._preprocess(
            flattened_queries, flattened_documents, max_seq_length, query_ids, stride
        )

        output = self.model_(input_ids=input_ids, attention_mask=attention_mask)

        start, end, probabilities = self._postprocess(output, sequence_ids, attention_mask, top_k, encodings, query_ids)

        answers = self._unflatten(
            start, end, probabilities, flattened_documents, documents, top_k, query_ids, document_ids
        )

        return {"answers": answers}


if __name__ == "__main__":
    docs = [
        [
            Document(content="Angela Merkel is the chancellor of Germany."),
            Document(content="Olaf Scholz is the chancellor of Germany"),
        ],
        [Document(content="Jerry is the head of the department.")],
    ]
    queries = ["Who is the chancellor of Germany?", "What is Jerry's role?"]
    reader = ExtractiveReader("deepset/roberta-base-squad2")
    p = Pipeline()
    p.add_component("reader", reader)

    print(p.run({"reader": {"documents": docs, "queries": queries}}))
