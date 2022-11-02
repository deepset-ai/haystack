from typing import List, Union, Dict

from tqdm.auto import tqdm

from haystack.errors import HaystackError
from haystack.schema import Document, Answer, Span
from haystack.nodes.base import BaseComponent


class Docs2Answers(BaseComponent):
    """
    This Node is used to convert retrieved documents into predicted answers format.
    It is useful for situations where you are calling a Retriever only pipeline via REST API.
    This ensures that your output is in a compatible format.

    :param progress_bar: Whether to show a progress bar
    """

    outgoing_edges = 1

    def __init__(self, progress_bar: bool = True):
        super().__init__()
        self.progress_bar = progress_bar

    def run(self, query: str, documents: List[Document]):  # type: ignore
        # conversion from Document -> Answer
        answers: List[Answer] = []
        for doc in documents:
            cur_answer = self._convert_doc_to_answer(doc)
            answers.append(cur_answer)

        output = {"query": query, "answers": answers}

        return output, "output_1"

    def run_batch(self, queries: List[str], documents: Union[List[Document], List[List[Document]]]):  # type: ignore
        output: Dict = {"queries": queries, "answers": []}

        # Docs case 1: single list of Documents
        if len(documents) > 0 and isinstance(documents[0], Document):
            for doc in tqdm(documents, disable=not self.progress_bar, desc="Converting to answers"):
                if not isinstance(doc, Document):
                    raise HaystackError(f"doc was of type {type(doc)}, but expected a Document.")
                answers = [self._convert_doc_to_answer(doc)]
                output["answers"].append(answers)

        # Docs case 2: list of lists of Documents
        elif len(documents) > 0 and isinstance(documents[0], list):
            for docs in tqdm(documents, disable=not self.progress_bar, desc="Converting to answers"):
                if not isinstance(docs, list):
                    raise HaystackError(f"docs was of type {type(docs)}, but expected a list of Documents.")
                answers = []
                for doc in docs:
                    cur_answer = self._convert_doc_to_answer(doc)
                    answers.append(cur_answer)
                output["answers"].append(answers)

        return output, "output_1"

    @staticmethod
    def _convert_doc_to_answer(doc: Document) -> Answer:
        # For FAQ style QA use cases
        if "answer" in doc.meta:
            doc.meta["query"] = doc.content  # question from the existing FAQ
            answer = Answer(
                answer=doc.meta["answer"],
                type="other",
                score=doc.score,
                context=doc.meta["answer"],
                offsets_in_context=[Span(start=0, end=len(doc.meta["answer"]))],
                document_id=doc.id,
                meta=doc.meta,
            )
        else:
            # Regular docs
            answer = Answer(
                answer="", type="other", score=doc.score, context=doc.content, document_id=doc.id, meta=doc.meta
            )

        return answer
