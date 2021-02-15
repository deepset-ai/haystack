import requests
import streamlit as st
import os

API_ENDPOINT = os.getenv("API_ENDPOINT", "http://localhost:8000")
MODEL_ID = "1"
DOC_REQUEST = "doc-qa"

def format_request(question,filters=None,top_k_reader=5,top_k_retriever=5):
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
 
@st.cache(show_spinner=False)
def retrieve_doc(question,filters=None,top_k_reader=5,top_k_retriever=5):
   # Query Haystack API
   url = API_ENDPOINT +'/models/' + MODEL_ID + "/" + DOC_REQUEST
   req = format_request(question,filters,top_k_reader=top_k_reader,top_k_retriever=top_k_retriever)
   response_raw = requests.post(url,json=req).json()

   # Format response
   result = []
   answers = response_raw['results'][0]['answers']
   for i in range(len(answers)):
       answer = answers[i]['answer']
       if answer:
           context = '...' + answers[i]['context'] + '...'
           meta_name = answers[i]['meta']['name']
           relevance = round(answers[i]['probability']*100,2)
           result.append({'context':context,'answer':answer,'source':meta_name,'relevance':relevance})
   return result, response_raw