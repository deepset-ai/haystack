import os

import requests
import streamlit as st

API_ENDPOINT = os.getenv("API_ENDPOINT", "http://localhost:8000")
DOC_REQUEST = "query"


@st.cache(show_spinner=False)
def haystack_query(query, filters=None, top_k_reader=5, top_k_retriever=5):
    url = f"{API_ENDPOINT}/{DOC_REQUEST}"
    req = {"query": query, "filters": filters, "top_k_retriever": top_k_retriever, "top_k_reader": top_k_reader}
    response_raw = requests.post(url, json=req).json()

    result = []
    answers = response_raw["answers"]
    for i in range(len(answers)):
        answer = answers[i]["answer"]
        if answer:
            context = "..." + answers[i]["context"] + "..."
            meta_name = answers[i]["meta"].get("name")
            relevance = round(answers[i]["probability"] * 100, 2)
            result.append({"context": context, "answer": answer, "source": meta_name, "relevance": relevance})
    return result, response_raw
