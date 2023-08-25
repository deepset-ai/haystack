from typing import Any, Dict, List, Union, Optional

import json
import logging

from fastapi import FastAPI, APIRouter
from haystack.schema import Label, Span
from haystack.document_stores import BaseDocumentStore
from rest_api.schema import FilterRequest, CreateLabelSerialized
from rest_api.utils import get_app, get_pipelines


logger = logging.getLogger(__name__)

router = APIRouter()
app: FastAPI = get_app()
document_store: BaseDocumentStore = get_pipelines().get("document_store", None)


@router.post("/feedback")
def post_feedback(feedback: CreateLabelSerialized, index: Optional[str] = None):
    """
    With this endpoint, the API user can submit their feedback on an answer for a particular query. This feedback is then written to the label_index of the DocumentStore.

    For example, the user can send feedback on whether the answer was correct and whether the right snippet was identified as the answer.

    Information submitted through this endpoint is used to train the underlying QA model.
    """

    if feedback.origin is None:
        feedback.origin = "user-feedback"

    label = Label(**feedback.dict())
    document_store.write_labels([label], index=index)


@router.get("/feedback", response_model=List[Label])
def get_feedback(index: Optional[str] = None):
    """
    This endpoint allows the API user to retrieve all the feedback that has been submitted through the `POST /feedback` endpoint.
    """
    labels = document_store.get_all_labels(index=index)
    return labels


@router.delete("/feedback")
def delete_feedback(index: Optional[str] = None):
    """
    This endpoint allows the API user to delete all the feedback that has been sumbitted through the
    `POST /feedback` endpoint.
    """
    all_labels = document_store.get_all_labels(index=index)
    user_label_ids = [label.id for label in all_labels if label.origin == "user-feedback"]
    document_store.delete_labels(ids=user_label_ids, index=index)


@router.post("/eval-feedback")
def get_feedback_metrics(filters: Optional[FilterRequest] = None, index: Optional[str] = None):
    """
    This endpoint returns basic accuracy metrics based on user feedback, for example, the ratio of correct answers or correctly identified documents.
    You can filter the output by document or label.

    Example:

    `curl --location --request POST 'http://127.0.0.1:8000/eval-doc-qa-feedback' \
     --header 'Content-Type: application/json' \
     --data-raw '{ "filters": {"document_id": ["XRR3xnEBCYVTkbTystOB"]} }'`
    """

    if filters:
        filters_content = filters.filters or {}
        filters_content["origin"] = ["user-feedback"]
    else:
        filters_content = {"origin": ["user-feedback"]}

    labels = document_store.get_all_labels(filters=filters_content, index=index)

    res: Dict[str, Optional[Union[float, int]]]
    if len(labels) > 0:
        answer_feedback = [1 if l.is_correct_answer else 0 for l in labels]
        doc_feedback = [1 if l.is_correct_document else 0 for l in labels]

        answer_accuracy = sum(answer_feedback) / len(answer_feedback)
        doc_accuracy = sum(doc_feedback) / len(doc_feedback)

        res = {"answer_accuracy": answer_accuracy, "document_accuracy": doc_accuracy, "n_feedback": len(labels)}
    else:
        res = {"answer_accuracy": None, "document_accuracy": None, "n_feedback": 0}
    return res


@router.get("/export-feedback")
def export_feedback(
    context_size: int = 100_000,
    full_document_context: bool = True,
    only_positive_labels: bool = False,
    index: Optional[str] = None,
):
    """
    This endpoint returns JSON output in the SQuAD format for question/answer pairs that were marked as "relevant" by user feedback through the `POST /feedback` endpoint.

    The context_size param can be used to limit response size for large documents.
    """
    if only_positive_labels:
        labels = document_store.get_all_labels(
            filters={"is_correct_answer": [True], "origin": ["user-feedback"]}, index=index
        )
    else:
        labels = document_store.get_all_labels(filters={"origin": ["user-feedback"]}, index=index)
        # Filter out the labels where the passage is correct but answer is wrong (in SQuAD this matches
        # neither a "positive example" nor a negative "is_impossible" one)
        labels = [l for l in labels if not (l.is_correct_document is True and l.is_correct_answer is False)]

    export_data = []

    for label in labels:
        answer_text = label.answer.answer if label and label.answer else ""

        offset_start_in_document = 0
        if label.answer and label.answer.offsets_in_document:
            if isinstance(label.answer.offsets_in_document[0], Span):
                offset_start_in_document = label.answer.offsets_in_document[0].start
            else:
                offset_start_in_document = label.answer.offsets_in_document[0].row

        if full_document_context:
            context = label.document.content
            answer_start = offset_start_in_document
        else:
            text = label.document.content
            # the final length of context(including the answer string) is 'context_size'.
            # we try to add equal characters for context before and after the answer string.
            # if either beginning or end of text is reached, we correspondingly
            # append more context characters at the other end of answer string.
            context_to_add = int((context_size - len(answer_text)) / 2)
            start_pos = max(offset_start_in_document - context_to_add, 0)
            additional_context_at_end = max(context_to_add - offset_start_in_document, 0)
            end_pos = min(offset_start_in_document + len(answer_text) + context_to_add, len(text) - 1)
            additional_context_at_start = max(
                offset_start_in_document + len(answer_text) + context_to_add - len(text), 0
            )
            start_pos = max(0, start_pos - additional_context_at_start)
            end_pos = min(len(text) - 1, end_pos + additional_context_at_end)
            context = text[start_pos:end_pos]
            answer_start = offset_start_in_document - start_pos

        squad_label: Dict[str, Any]
        if label.is_correct_answer is False and label.is_correct_document is False:  # No answer
            squad_label = {
                "paragraphs": [
                    {
                        "context": context,
                        "id": label.document.id,
                        "qas": [{"question": label.query, "id": label.id, "is_impossible": True, "answers": []}],
                    }
                ]
            }
        else:
            squad_label = {
                "paragraphs": [
                    {
                        "context": context,
                        "id": label.document.id,
                        "qas": [
                            {
                                "question": label.query,
                                "id": label.id,
                                "is_impossible": False,
                                "answers": [{"text": answer_text, "answer_start": answer_start}],
                            }
                        ],
                    }
                ]
            }

            # quality check
            start = squad_label["paragraphs"][0]["qas"][0]["answers"][0]["answer_start"]
            answer = squad_label["paragraphs"][0]["qas"][0]["answers"][0]["text"]
            context = squad_label["paragraphs"][0]["context"]
            if not context[start : start + len(answer)] == answer:
                logger.error(
                    "Skipping invalid squad label as string via offsets ('%s') does not match answer string ('%s') ",
                    context[start : start + len(answer)],
                    answer,
                )
        export_data.append(squad_label)

    export = {"data": export_data}

    feedback_file = "feedback_squad_direct.json"
    try:
        with open(feedback_file, "w", encoding="utf8") as f:
            json.dump(export_data, f, ensure_ascii=False, sort_keys=True, indent=4)
    except Exception as e:
        # might fail in some docker container environment.
        logger.error("Can't write feedback file. filename=%s", feedback_file)
        logger.exception(e)

    return export
