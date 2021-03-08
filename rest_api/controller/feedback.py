from typing import Optional
import time

from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import Dict, Union, List

router = APIRouter()

from .search import PIPELINE

#TODO make this generic for other pipelines with different naming
retriever = PIPELINE.get_node(name="ESRetriever")
document_store = retriever.document_store

DB_INDEX_FEEDBACK = "label"

class FAQQAFeedback(BaseModel):
    question: str = Field(..., description="The question input by the user, i.e., the query.")
    is_correct_answer: bool = Field(..., description="Whether the answer is correct or not.")
    document_id: str = Field(..., description="The document in the query result for which feedback is given.")
    model_id: Optional[int] = Field(None, description="The model used for the query.")


class DocQAFeedback(FAQQAFeedback):
    is_correct_document: bool = Field(
        ...,
        description="In case of negative feedback, there could be two cases; incorrect answer but correct "
        "document & incorrect document. This flag denotes if the returned document was correct.",
    )
    answer: str = Field(..., description="The answer string.")
    offset_start_in_doc: int = Field(
        ..., description="The answer start offset in the original doc. Only required for doc-qa feedback."
    )

class FilterRequest(BaseModel):
    filters: Optional[Dict[str, Optional[Union[str, List[str]]]]] = None

@router.post("/feedback")
def doc_qa_feedback(feedback: DocQAFeedback):
    document_store.write_labels([{"origin": "user-feedback", **feedback.dict()}])

@router.post("/eval-feedback")
def eval_doc_qa_feedback(filters: FilterRequest = None):
    """
    Return basic accuracy metrics based on the user feedback.
    Which ratio of answers was correct? Which ratio of documents was correct?
    You can supply filters in the request to only use a certain subset of labels.

    **Example:**

            ```
                | curl --location --request POST 'http://127.0.0.1:8000/eval-doc-qa-feedback' \
                | --header 'Content-Type: application/json' \
                | --data-raw '{ "filters": {"document_id": ["XRR3xnEBCYVTkbTystOB"]} }'
    """

    if filters:
        filters = filters.filters
        filters["origin"] = ["user-feedback"]
    else:
        filters = {"origin": ["user-feedback"]}

    labels = document_store.get_all_labels(
        index=DB_INDEX_FEEDBACK,
        filters=filters
    )

    if len(labels) > 0:
        answer_feedback = [1 if l.is_correct_answer else 0 for l in labels]
        doc_feedback = [1 if l.is_correct_document else 0 for l in labels]

        answer_accuracy = sum(answer_feedback)/len(answer_feedback)
        doc_accuracy = sum(doc_feedback)/len(doc_feedback)

        res = {"answer_accuracy": answer_accuracy,
               "document_accuracy": doc_accuracy,
               "n_feedback": len(labels)}
    else:
        res = {"answer_accuracy": None,
               "document_accuracy": None,
               "n_feedback": 0}
    return res

@router.get("/export-feedback")
def export_doc_qa_feedback(context_size: int = 2_000):
    """
    SQuAD format JSON export for question/answer pairs that were marked as "relevant".

    The context_size param can be used to limit response size for large documents.
    """
    labels = document_store.get_all_labels(
        index=DB_INDEX_FEEDBACK, filters={"is_correct_answer": [True], "origin": ["user-feedback"]}
    )

    export_data = []
    for label in labels:
        document = document_store.get_document_by_id(label.document_id)
        text = document.text

        # the final length of context(including the answer string) is 'context_size'.
        # we try to add equal characters for context before and after the answer string.
        # if either beginning or end of text is reached, we correspondingly
        # append more context characters at the other end of answer string.
        context_to_add = int((context_size - len(label.answer)) / 2)

        start_pos = max(label.offset_start_in_doc - context_to_add, 0)
        additional_context_at_end = max(context_to_add - label.offset_start_in_doc, 0)

        end_pos = min(label.offset_start_in_doc + len(label.answer) + context_to_add, len(text) - 1)
        additional_context_at_start = max(label.offset_start_in_doc + len(label.answer) + context_to_add - len(text), 0)

        start_pos = max(0, start_pos - additional_context_at_start)
        end_pos = min(len(text) - 1, end_pos + additional_context_at_end)

        context_to_export = text[start_pos:end_pos]

        export_data.append({"paragraphs": [{"qas": label, "context": context_to_export}]})

    export = {"data": export_data}

    return export

