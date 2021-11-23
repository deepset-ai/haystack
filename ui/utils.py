import os

import logging
import requests
import streamlit as st

API_ENDPOINT = os.getenv("API_ENDPOINT", "http://localhost:8000")
STATUS = "initialized"
HS_VERSION = "hs_version"
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

@st.cache
def haystack_version():
    url = f"{API_ENDPOINT}/{HS_VERSION}"
    return requests.get(url, timeout=0.1).json()["hs_version"]

def retrieve_doc(query, filters={}, top_k_reader=5, top_k_retriever=5):
    # Query Haystack API
    url = f"{API_ENDPOINT}/{DOC_REQUEST}"
    params = {"filters": filters, "Retriever": {"top_k": top_k_retriever}, "Reader": {"top_k": top_k_reader}}
    req = {"query": query, "params": params}
    response_raw = requests.post(url, json=req).json()

    # Format response
    result = []

    if "errors" in response_raw:
        raise Exception(", ".join(response_raw["errors"]))
        
    answers = response_raw["answers"]
    for i in range(len(answers)):
        answer = answers[i]
        answer_text = answer.get("answer", None)
        if answer_text:
            result.append(
                {
                    "context": "..." + answer["context"] + "...",
                    "answer": answer_text,
                    "source": answer["meta"]["name"],
                    "relevance": round(answer["score"] * 100, 2),
                    "document_id": answer["document_id"],
                    "offset_start_in_doc": answer["offsets_in_document"][0]["start"],
                }
            )
        else:
            result.append(
                {
                    "context": None,
                    "answer": None,
                    "relevance": round(answer["score"] * 100, 2),
                }
            )
    return result, response_raw


def feedback_doc(question, is_correct_answer, document_id, model_id, is_correct_document, answer, offset_start_in_doc):
    # Feedback Haystack API
    try:
        url = f"{API_ENDPOINT}/{DOC_FEEDBACK}"
        #TODO adjust after Label refactoring
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
    except Exception as e:
        logging.exception(e)


def upload_doc(file):
    url = f"{API_ENDPOINT}/{DOC_UPLOAD}"
    files = [("files", file)]
    response_raw = requests.post(url, files=files).json()
    return response_raw
