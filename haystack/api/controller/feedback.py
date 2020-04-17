from collections import defaultdict
from typing import Optional

from elasticsearch.helpers import scan
from fastapi import APIRouter, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from haystack.api.config import (
    DB_HOST,
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
from haystack.api.config import DB_INDEX_FEEDBACK
from haystack.api.elasticsearch_client import elasticsearch_client
from haystack.database.elasticsearch import ElasticsearchDocumentStore

router = APIRouter()

document_store = ElasticsearchDocumentStore(
    host=DB_HOST,
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
    excluded_meta_data=EXCLUDE_META_DATA_FIELDS,
)


class Feedback(BaseModel):
    question: str = Field(..., description="The question input by the user, i.e., the query.")
    label: str = Field(..., description="The Label for the feedback, eg, relevant or irrelevant.")
    document_id: str = Field(..., description="The document in the query result for which feedback is given.")
    answer: Optional[str] = Field(None, description="The answer string. Only required for doc-qa feedback.")
    offset_start_in_doc: Optional[int] = Field(None, description="The answer start offset in the original doc. Only required for doc-qa feedback.")
    model_id: Optional[int] = Field(None, description="The model used for the query.")


@router.post("/doc-qa-feedback")
def feedback(feedback: Feedback):
    if feedback.answer and feedback.offset_start_in_doc:
        elasticsearch_client.index(index=DB_INDEX_FEEDBACK, body=feedback.dict())
    else:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content="doc-qa feedback must contain 'answer' and 'answer_doc_start' fields.",
        )


@router.post("/faq-qa-feedback")
def feedback(feedback: Feedback):
    elasticsearch_client.index(index=DB_INDEX_FEEDBACK, body=feedback.dict())


@router.get("/export-doc-qa-feedback")
def export_doc_qa_feedback():
    """
    SQuAD format JSON export for question/answer pairs that were marked as "relevant".

    #TODO filter out faq-qa feedback.
    """
    relevant_feedback_query = {"query": {"bool": {"must": [{"term": {"label": "relevant"}}]}}}
    result = scan(elasticsearch_client, index=DB_INDEX_FEEDBACK, query=relevant_feedback_query)

    per_document_feedback = defaultdict(list)
    for feedback in result:
        document_id = feedback["_source"]["document_id"]
        per_document_feedback[document_id].append(
            {
                "question": feedback["_source"]["question"],
                "id": feedback["_id"],
                "answers": [
                    {"text": feedback["_source"]["answer"], "answer_start": feedback["_source"]["offset_start_in_doc"]}
                ],
            }
        )

    export_data = []
    for document_id, feedback in per_document_feedback.items():
        document = document_store.get_document_by_id(document_id)
        context = document.text
        export_data.append({"paragraphs": [{"qas": feedback}], "context": context})

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
