import os

import logging
import requests
import streamlit as st

API_ENDPOINT = os.getenv("API_ENDPOINT", "http://localhost:8000")
STATUS = "initialized"
DOC_REQUEST = "query"
DOC_FEEDBACK = "feedback"
DOC_UPLOAD = "file-upload"


def haystack_is_ready():
    url = f"{API_ENDPOINT}/{STATUS}"
    try:
        if requests.get(url).json():
            return True
    except Exception as e:
        logging.exception(e)
    return False


@st.cache(show_spinner=False)
def retrieve_doc(query, filters=None, top_k_reader=5, top_k_retriever=5):
    # Query Haystack API
    url = f"{API_ENDPOINT}/{DOC_REQUEST}"
    params = {"filters": filters, "ESRetriever": {"top_k": top_k_retriever}, "Reader": {"top_k": top_k_reader}}
    req = {"query": query, "params": params}
    response_raw = requests.post(url, json=req).json()

    # Format response
    result = []
    answers = response_raw["answers"]
    for i in range(len(answers)):
        answer = answers[i]["answer"]
        if answer:
            context = "..." + answers[i]["context"] + "..."
            meta_name = answers[i]["meta"]["name"]
            relevance = round(answers[i]["score"] * 100, 2)
            document_id = answers[i]["document_id"]
            offset_start_in_doc = answers[i]["offset_start_in_doc"]
            result.append(
                {
                    "context": context,
                    "answer": answer,
                    "source": meta_name,
                    "relevance": relevance,
                    "document_id": document_id,
                    "offset_start_in_doc": offset_start_in_doc,
                }
            )
    return result, response_raw


def feedback_doc(question, is_correct_answer, document_id, model_id, is_correct_document, answer, offset_start_in_doc):
    # Feedback Haystack API
    url = f"{API_ENDPOINT}/{DOC_FEEDBACK}"
    req = {
        "question": question,
        "is_correct_answer": is_correct_answer,
        "document_id": document_id,
        "model_id": model_id,
        "is_correct_document": is_correct_document,
        "answer": answer,
        "offset_start_in_doc": offset_start_in_doc,
    }
    response_raw = requests.post(url, json=req).json()
    return response_raw


def upload_doc(file):
    url = f"{API_ENDPOINT}/{DOC_UPLOAD}"
    files = [("files", file)]
    response_raw = requests.post(url, files=files).json()
    return response_raw
