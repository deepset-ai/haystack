from collections import defaultdict
from typing import Optional

from fastapi import APIRouter, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from haystack.api import application
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
from haystack.database.elasticsearch import ElasticsearchDocumentStore
from elasticsearch.helpers import scan

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
    # Note: the question here is the user's question (=query) and not the matched one from our FAQs (=response)
    question: str
    answer: Optional[str]
    feedback: str
    document_id: int


@router.post("/models/{model_id}/feedback")
def feedback(model_id: int, request: Feedback):
    feedback_payload = request.__dict__
    if feedback_payload["feedback"] not in ("relevant", "fake", "outdated", "irrelevant"):
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content="Invalid 'feedback'. It must be one of relevant, fake, outdated or irrelevant",
        )
    feedback_payload["model_id"] = model_id
    application.elasticsearch_client.index(index=DB_INDEX_FEEDBACK, body=feedback_payload)


@router.get("/models/{model_id}/export-faq-feedback")
def export_faq_feedback(model_id: int):
    """
    SQuAD-like JSON export for question/answer pairs that were marked as "relevant".
    """
    relevant_feedback_query = {
        "query": {"bool": {"must": [{"term": {"feedback": "relevant"}}, {"term": {"model_id": model_id}}]}}
    }
    result = scan(application.elasticsearch_client, index=DB_INDEX_FEEDBACK, query=relevant_feedback_query)

    per_document_feedback = defaultdict(list)
    for feedback in result:
        document_id = feedback["_source"]["document_id"]
        question = feedback["_source"]["question"]
        feedback_id = feedback["_id"]
        per_document_feedback[document_id].append({"question": question, "id": feedback_id})

    export_data = []
    for document_id, feedback in per_document_feedback.items():
        document = document_store.get_document_by_id(document_id)
        context = document["text"]
        export_data.append({"paragraphs": [{"qas": feedback}], "context": context})

    export = {"data": export_data}

    return export
