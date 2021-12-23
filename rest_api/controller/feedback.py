import json
import logging

from fastapi import APIRouter
from rest_api.schema import FilterRequest, LabelSerialized
from rest_api.controller.search import DOCUMENT_STORE


router = APIRouter()

logger = logging.getLogger(__name__)


@router.post("/feedback")
def post_feedback(feedback: LabelSerialized):
    if feedback.origin is None:
        feedback.origin = "user-feedback"
    DOCUMENT_STORE.write_labels([feedback])


@router.get("/feedback")
def get_feedback():
    labels = DOCUMENT_STORE.get_all_labels()
    return labels


@router.post("/eval-feedback")
def get_feedback_metrics(filters: FilterRequest = None):
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

    labels = DOCUMENT_STORE.get_all_labels(filters=filters)

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
    context_size: int = 100_000, full_document_context: bool = True, only_positive_labels: bool = False
):
    """
    SQuAD format JSON export for question/answer pairs that were marked as "relevant".

    The context_size param can be used to limit response size for large documents.
    """
    if only_positive_labels:
        labels = DOCUMENT_STORE.get_all_labels(filters={"is_correct_answer": [True], "origin": ["user-feedback"]})
    else:
        labels = DOCUMENT_STORE.get_all_labels(filters={"origin": ["user-feedback"]})
        # Filter out the labels where the passage is correct but answer is wrong (in SQuAD this matches
        # neither a "positive example" nor a negative "is_impossible" one)
        labels = [l for l in labels if not (l.is_correct_document is True and l.is_correct_answer is False)]

    export_data = []

    for label in labels:
        if full_document_context:
            context = label.document.content

            answer_start = label.answer.offsets_in_document[0].start
        else:
            text = label.document.content
            # the final length of context(including the answer string) is 'context_size'.
            # we try to add equal characters for context before and after the answer string.
            # if either beginning or end of text is reached, we correspondingly
            # append more context characters at the other end of answer string.
            context_to_add = int((context_size - len(label.answer.answer)) / 2)
            start_pos = max(label.answer.offsets_in_document[0].start - context_to_add, 0)
            additional_context_at_end = max(context_to_add - label.answer.offsets_in_document[0].start, 0)
            end_pos = min(label.answer.offsets_in_document[0].start + len(label.answer.answer) + context_to_add, len(text) - 1)
            additional_context_at_start = max(
                label.answer.offsets_in_document[0].start + len(label.answer.answer) + context_to_add - len(text), 0
            )
            start_pos = max(0, start_pos - additional_context_at_start)
            end_pos = min(len(text) - 1, end_pos + additional_context_at_end)
            context = text[start_pos:end_pos]
            answer_start = label.answer.offsets_in_document[0].start - start_pos

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
                                "answers": [{"text": label.answer.answer, "answer_start": answer_start}],
                            }
                        ],
                    }
                ]
            }

            # quality check
            start = squad_label["paragraphs"][0]["qas"][0]["answers"][0]["answer_start"]
            answer = squad_label["paragraphs"][0]["qas"][0]["answers"][0]["text"]
            context = squad_label["paragraphs"][0]["context"]
            if not context[start: start + len(answer)] == answer:
                logger.error(
                    f"Skipping invalid squad label as string via offsets "
                    f"('{context[start:start + len(answer)]}') does not match answer string ('{answer}') "
                )
        export_data.append(squad_label)

    export = {"data": export_data}

    with open("feedback_squad_direct.json", "w", encoding="utf8") as f:
        json.dump(export_data, f, ensure_ascii=False, sort_keys=True, indent=4)
    return export
