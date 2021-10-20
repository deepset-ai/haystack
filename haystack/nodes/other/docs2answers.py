from typing import List

from haystack.schema import Document, Answer, Span
from haystack.nodes.base import BaseComponent


class Docs2Answers(BaseComponent):
    """
    This Node is used to convert retrieved documents into predicted answers format.
    It is useful for situations where you are calling a Retriever only pipeline via REST API.
    This ensures that your output is in a compatible format.
    """
    outgoing_edges = 1

    def __init__(self):
        self.set_config()

    def run(self, query: str, documents: List[Document]):  # type: ignore
        # conversion from Document -> Answer
        answers: List[Answer] = []
        for doc in documents:
            # For FAQ style QA use cases
            if "answer" in doc.meta:
                doc.meta["query"] = doc.content # question from the existing FAQ
                cur_answer = Answer(answer=doc.meta["answer"],
                                    type="other",
                                    score=doc.score,
                                    context=doc.meta["answer"],
                                    offsets_in_context=[Span(start=0, end=len(doc.meta["answer"]))],
                                    document_id=doc.id,
                                    meta=doc.meta,
                                    )
            else:
                # Regular docs
                cur_answer = Answer(answer="",
                                    type="other",
                                    score=doc.score,
                                    context=doc.content,
                                    document_id=doc.id,
                                    meta=doc.meta,
                                    )
            answers.append(cur_answer)

        output = {"query": query, "answers": answers}

        return output, "output_1"
