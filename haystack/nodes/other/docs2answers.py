from typing import List, Union, Dict

from haystack.errors import HaystackError
from haystack.schema import Document, Answer, Span
from haystack.nodes.base import BaseComponent


class Docs2Answers(BaseComponent):
    """
    This Node is used to convert retrieved documents into predicted answers format.
    It is useful for situations where you are calling a Retriever only pipeline via REST API.
    This ensures that your output is in a compatible format.
    """

    outgoing_edges = 1

    def run(self, query: str, documents: List[Document]):  # type: ignore
        # conversion from Document -> Answer
        answers: List[Answer] = []
        for doc in documents:
            cur_answer = self._convert_doc_to_answer(doc)
            answers.append(cur_answer)

        output = {"query": query, "answers": answers}

        return output, "output_1"

    def run_batch(self, queries: Union[str, List[str]], documents: Union[List[Document], List[List[Document]]]):  # type: ignore
        output: Dict = {"queries": queries, "answers": []}

        # Query case 1: single query
        if isinstance(queries, str):
            query = queries
            # Docs case 1: single list of Documents
            if len(documents) > 0 and isinstance(documents[0], Document):
                for doc in documents:
                    if not isinstance(doc, Document):
                        raise HaystackError(f"doc was of type {type(doc)}, but expected a Document.")
                    answers = [self._convert_doc_to_answer(doc)]
                    output["answers"].append(answers)
            # Docs case 2: list of lists of Documents
            elif len(documents) > 0 and isinstance(documents[0], list):
                for docs in documents:
                    if not isinstance(docs, list):
                        raise HaystackError(f"docs was of type {type(docs)}, but expected a list of Documents.")
                    answers = []
                    for doc in docs:
                        cur_answer = self._convert_doc_to_answer(doc)
                        answers.append(cur_answer)
                    output["answers"].append(answers)

        # Query case 2: list of queries
        elif isinstance(queries, list) and len(queries) > 0 and isinstance(queries[0], str):
            # Docs case 1: single list of Documents -> apply each query to all Documents
            if len(documents) > 0 and isinstance(documents[0], Document):
                for query in queries:
                    for doc in documents:
                        if not isinstance(doc, Document):
                            raise HaystackError(f"doc was of type {type(doc)}, but expected a Document.")
                        answers = [self._convert_doc_to_answer(doc)]
                        output["answers"].append(answers)
            # Docs case 2: list of lists of Documents
            elif len(documents) > 0 and isinstance(documents[0], list):
                if len(queries) != len(documents):
                    raise HaystackError("Number of queries must be equal to number of provided Document lists.")
                for query, docs_ in zip(queries, documents):
                    answers = []
                    if not isinstance(docs_, list):
                        raise HaystackError(f"docs_ was of type {type(docs_)}, but expected a list of Documents.")
                    for doc in docs_:
                        if not isinstance(doc, Document):
                            raise HaystackError(f"doc was of type {type(doc)}, but expected a Document.")
                        cur_answer = self._convert_doc_to_answer(doc)
                        answers.append(cur_answer)
                    output["answers"].append(answers)

        else:
            raise HaystackError(f"'queries' was of type {type(queries)} but must be of type str or List[str].")

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
