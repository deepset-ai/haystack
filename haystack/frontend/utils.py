import pandas as pd
import requests
import streamlit as st

API_ENDPOINT = "http://localhost:8000/models/"
MODEL_ID = "1"
FAQ_REQUEST = "faq-qa"
DOC_REQUEST = "doc-qa"
COLUMNS = ["Answer","Context","Meta Name"]
def get_name(s):
    return s["name"]

def format_request(question,filters=None,top_k_retriever=5,top_k_reader=5):
    print("Filters",filters)
    if filters == None:
        return {
       "questions": [question],
       "top_k_retriever": top_k_retriever,
       "top_k_reader": top_k_reader
       }
    return {
        "questions": [question],
        "filters": {
            "option1":[filters]
        },
        "top_k_retriever": top_k_retriever,
        "top_k_reader": top_k_reader
    }    

@st.cache
def retrieve_doc(question,filters=None,top_k_reader=5):
   url = API_ENDPOINT + MODEL_ID +"/" + DOC_REQUEST
   req = format_request(question,filters,top_k_reader=top_k_reader)
   response = requests.post(url,json=req).json()["results"][0]["answers"]
   df = pd.DataFrame(response,columns=["context","answer","meta"])
   df['Meta Name'] = df['meta'].apply(get_name)
   df = df.rename(columns={"context": "Context", "answer": "Answer"})     
   return df[COLUMNS].to_html(escape=False)

@st.cache
def retrieve_faq(question,filters=None,top_k_reader=5):
   url = API_ENDPOINT + MODEL_ID +"/" + FAQ_REQUEST
   req = format_request(question,filters,top_k_retriever=max(10,top_k_reader+5),top_k_reader=top_k_reader)
   response = requests.post(url,json=req).json()["results"][0]["answers"]
   df = pd.DataFrame(response,columns=["context","answer","meta"])
   df['Meta Name'] = df['meta'].apply(get_name)
   df = df.rename(columns={"context": "Context", "answer": "Answer"})     
   return df[COLUMNS].to_html(escape=False)