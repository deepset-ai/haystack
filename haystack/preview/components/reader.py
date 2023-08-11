from pathlib import Path
from typing import List, Optional, Tuple, Union
from haystack.preview import component, Document, Answer
from haystack.lazy_imports import LazyImport

with LazyImport(message="Run 'pip install farm-haystack[inference]'") as torch_and_transformers_import:
    from transformers import AutoModelForQuestionAnswering
    from tokenizers import Tokenizer, Encoding
    import torch


@component
class ExtractiveReader:
    def __init__(
        self, reader: Union[Path, str], device: Optional[str] = None, max_seq_length: int = 384, top_k: int = 10
    ) -> None:
        self.reader = reader
        self.device = device
        self.loaded = False
        self.max_seq_length = max_seq_length
        self.top_k = top_k

    def warm_up(self):
        if not self.loaded:
            torch_and_transformers_import.check()
            if torch.cuda.is_available():
                self.device = "cuda:0"
            else:
                self.device = "cpu:0"
            self.model = AutoModelForQuestionAnswering.from_pretrained(self.reader).to(self.device)
            self.tokenizer = Tokenizer.from_pretrained(self.reader)

    def _flatten(self, queries: List[str], documents: List[List[Document]]) -> Tuple[List[str], List[str]]:
        flattened_queries = [query for documents_, query in zip(documents, queries) for _ in documents_]
        flattened_documents = [document.content for documents_ in documents for document in documents_]
        return flattened_queries, flattened_documents

    def _preprocess(
        self, queries: List[str], documents: List[str], max_seq_length: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[Encoding]]:
        self.tokenizer.enable_padding(length=max_seq_length)
        self.tokenizer.enable_truncation(max_seq_length)

        encodings = self.tokenizer.encode_batch(list(zip(queries, documents)))

        input_ids = torch.tensor([encoding.ids for encoding in encodings]).to(self.device)
        attention_mask = torch.tensor([encoding.attention_mask for encoding in encodings]).to(self.device)
        sequence_ids = torch.tensor(
            [[id_ if id_ is not None else -1 for id_ in encoding.sequence_ids] for encoding in encodings]
        ).to(self.device)

        return input_ids, attention_mask, sequence_ids, encodings

    def _postprocess(
        self,
        output,
        sequence_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        top_k: int,
        encodings: List[Encoding],
        max_seq_length: int,
    ) -> Tuple[List[List[int]], List[List[int]], List[List[float]]]:
        start = output.start_logits
        end = output.end_logits
        mask = sequence_ids == 1
        mask[..., 0] = True  # type: ignore # mypy doesn't understand broadcasting
        mask = torch.logical_and(mask, attention_mask == 1)  # type: ignore # see above
        start = torch.where(mask, start, -torch.inf)  # type: ignore # see above
        end = torch.where(mask, end, -torch.inf)  # type: ignore # see above
        start = torch.softmax(start, -1)
        end = torch.softmax(end, -1)
        start = start.unsqueeze(-1)
        end = end.unsqueeze(-2)

        probabilities = start * end  # shape: (batch_size, seq_length (start), seq_length (end))
        masked_probabilities = torch.triu(probabilities, diagonal=1)
        masked_probabilities[..., 0, 0] = probabilities[..., 0, 0]
        masked_probabilities[..., 0, 1:] = 0
        flat_probabilities = masked_probabilities.flatten(-2, -1)  # necessary for topk
        candidates = torch.topk(flat_probabilities, top_k)
        start_candidates = candidates.indices // max_seq_length
        end_candidates = candidates.indices % max_seq_length
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
    ) -> List[List[Answer]]:
        answers = [
            [document[start:end] for start, end in zip(start_candidates_, end_candidates_)]
            for document, start_candidates_, end_candidates_ in zip(flattened_documents, start, end)
        ]

        i = 0
        nested_answers = []
        for documents_ in documents:
            current_answers = []
            j = i + len(documents_)
            for answers_, probabilities_ in zip(answers[i:j], probabilities[i:j]):
                for answer, probability in zip(answers_, probabilities_):
                    current_answers.append(Answer(answer=answer, probability=probability))
            i = j
            current_answers = sorted(current_answers, key=lambda answer: answer.probability, reverse=True)[:top_k]
            nested_answers.append(current_answers)

        return nested_answers

    @component.output_types(answers=List[List[Answer]])
    def run(
        self,
        queries: List[str],
        documents: List[List[Document]],
        top_k: Optional[int] = None,
        max_seq_length: Optional[int] = None,
    ):
        top_k = top_k or self.top_k
        max_seq_length = max_seq_length or self.max_seq_length

        flattened_queries, flattened_documents = self._flatten(queries, documents)
        input_ids, attention_mask, sequence_ids, encodings = self._preprocess(
            flattened_queries, flattened_documents, max_seq_length
        )

        output = self.model(input_ids=input_ids, attention_mask=attention_mask)

        start, end, probabilities = self._postprocess(
            output, sequence_ids, attention_mask, top_k, encodings, max_seq_length
        )

        answers = self._unflatten(start, end, probabilities, flattened_documents, documents, top_k)

        return {"answers": answers}
