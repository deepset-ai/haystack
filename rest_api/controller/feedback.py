from collections import defaultdict
from typing import Optional

from elasticsearch.helpers import scan
from fastapi import APIRouter, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from rest_api.config import (
    DB_HOST,
    DB_PORT,
    DB_USER,
    DB_PW,
    DB_INDEX,
    ES_CONN_SCHEME,
    TEXT_FIELD_NAME,
    SEARCH_FIELD_NAME,
    EMBEDDING_DIM,
    EMBEDDING_FIELD_NAME,
    EXCLUDE_META_DATA_FIELDS,
)
from rest_api.config import DB_INDEX_FEEDBACK
from rest_api.elasticsearch_client import elasticsearch_client
from haystack.database.elasticsearch import ElasticsearchDocumentStore

router = APIRouter()

document_store = ElasticsearchDocumentStore(
    host=DB_HOST,
    port=DB_PORT,
    username=DB_USER,
    password=DB_PW,
    index=DB_INDEX,
    scheme=ES_CONN_SCHEME,
    ca_certs=False,
    verify_certs=False,
    text_field=TEXT_FIELD_NAME,
    search_fields=SEARCH_FIELD_NAME,
    embedding_dim=EMBEDDING_DIM,
    embedding_field=EMBEDDING_FIELD_NAME,
    excluded_meta_data=EXCLUDE_META_DATA_FIELDS,  # type: ignore
)


class Feedback(BaseModel):
    question: str = Field(..., description="The question input by the user, i.e., the query.")
    positive_sample: bool = Field(..., description="Whether the feedback is positive or negative.")
    document_id: str = Field(..., description="The document in the query result for which feedback is given.")
    answer: Optional[str] = Field(None, description="The answer string. Only required for doc-qa feedback.")
    offset_start_in_doc: Optional[int] = Field(None, description="The answer start offset in the original doc. Only required for doc-qa feedback.")
    model_id: Optional[int] = Field(None, description="The model used for the query.")


@router.post("/doc-qa-feedback")
def doc_qa_feedback(feedback: Feedback):
    if feedback.answer and feedback.offset_start_in_doc:
        document_store.write_labels([{"origin": "api", **feedback.dict()}])
    else:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content="doc-qa feedback must contain 'answer' and 'answer_doc_start' fields.",
        )


@router.post("/faq-qa-feedback")
def faq_qa_feedback(feedback: Feedback):
    elasticsearch_client.index(index=DB_INDEX_FEEDBACK, body=feedback.dict())


@router.get("/export-doc-qa-feedback")
def export_doc_qa_feedback(context_size: int = 10_000):
    """
    SQuAD format JSON export for question/answer pairs that were marked as "relevant".
    """
    # TODO filter out faq-qa feedback.
    labels = document_store.get_all_labels(index=DB_INDEX_FEEDBACK, filters={"positive_sample": [True], "origin": ["api"]})

    export_data = []
    for label in labels:
        document = document_store.get_document_by_id(label.document_id)
        context = document.text

        context_to_add = int((context_size - len(label.answer)) / 2)
        start_pos = max(label.offset_start_in_doc - context_to_add, 0)
        if context_to_add > label.offset_start_in_doc:
            append_at_end = context_to_add - label.offset_start_in_doc
        else:
            append_at_end = 0
        end_pos = min(context_to_add + label.offset_start_in_doc + len(label.answer) + append_at_end, len(context))

        context_to_export = context[start_pos:end_pos]

        export_data.append({"paragraphs": [{"qas": label, "context": context_to_export}]})

    export = {"data": export_data}

    return export


@router.get("/export-faq-qa-feedback")
def export_faq_feedback():
    """
    Export feedback for faq-qa in JSON format.
    """
    result = scan(elasticsearch_client, index=DB_INDEX_FEEDBACK)

    per_document_feedback = defaultdict(list)
    for feedback in result:
        document_id = feedback["_source"]["document_id"]
        question = feedback["_source"]["question"]
        feedback_id = feedback["_id"]
        feedback_label = feedback["_source"]["label"]
        per_document_feedback[document_id].append(
            {"question": question, "id": feedback_id, "feedback_label": feedback_label}
        )

    export_data = []
    for document_id, feedback in per_document_feedback.items():
        document = document_store.get_document_by_id(document_id)
        export_data.append(
            {"target_question": document.question, "target_answer": document.text, "queries": feedback}
        )

    export = {"data": export_data}

    return export
