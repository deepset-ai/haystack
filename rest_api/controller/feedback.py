from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from haystack.document_store.elasticsearch import ElasticsearchDocumentStore
from rest_api.config import (
    DB_HOST,
    DB_PORT,
    DB_USER,
    DB_PW,
    DB_INDEX,
    DB_INDEX_FEEDBACK,
    ES_CONN_SCHEME,
    TEXT_FIELD_NAME,
    SEARCH_FIELD_NAME,
    EMBEDDING_DIM,
    EMBEDDING_FIELD_NAME,
    EXCLUDE_META_DATA_FIELDS,
    FAQ_QUESTION_FIELD_NAME,
    CREATE_INDEX,
    VECTOR_SIMILARITY_METRIC,
    UPDATE_EXISTING_DOCUMENTS
)

router = APIRouter()

document_store = ElasticsearchDocumentStore(
    host=DB_HOST,
    port=DB_PORT,
    username=DB_USER,
    password=DB_PW,
    index=DB_INDEX,
    label_index=DB_INDEX_FEEDBACK,
    scheme=ES_CONN_SCHEME,
    ca_certs=False,
    verify_certs=False,
    text_field=TEXT_FIELD_NAME,
    search_fields=SEARCH_FIELD_NAME,
    faq_question_field=FAQ_QUESTION_FIELD_NAME,
    embedding_dim=EMBEDDING_DIM,
    embedding_field=EMBEDDING_FIELD_NAME,
    excluded_meta_data=EXCLUDE_META_DATA_FIELDS,  # type: ignore
    create_index=CREATE_INDEX,
    update_existing_documents=UPDATE_EXISTING_DOCUMENTS,
    similarity=VECTOR_SIMILARITY_METRIC
)


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


@router.post("/doc-qa-feedback")
def doc_qa_feedback(feedback: DocQAFeedback):
    document_store.write_labels([{"origin": "user-feedback", **feedback.dict()}])


@router.post("/faq-qa-feedback")
def faq_qa_feedback(feedback: FAQQAFeedback):
    feedback_payload = {"is_correct_document": feedback.is_correct_answer, "answer": None, **feedback.dict()}
    document_store.write_labels([{"origin": "user-feedback-faq", **feedback_payload}])


@router.get("/export-doc-qa-feedback")
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


@router.get("/export-faq-qa-feedback")
def export_faq_feedback():
    """
    Export feedback for faq-qa in JSON format.
    """

    labels = document_store.get_all_labels(index=DB_INDEX_FEEDBACK, filters={"origin": ["user-feedback-faq"]})

    export_data = []
    for label in labels:
        document = document_store.get_document_by_id(label.document_id)
        feedback = {
            "question": document.question,
            "query": label.question,
            "is_correct_answer": label.is_correct_answer,
            "is_correct_document": label.is_correct_answer,
        }
        export_data.append(feedback)

    export = {"data": export_data}

    return export
