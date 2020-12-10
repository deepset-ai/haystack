import streamlit as st
from utils import retrieve_doc,retrieve_faq

def _max_width_():
    max_width_str = f"max-width: 2000px;"
    st.markdown(
        f"""
    <style>
    .reportview-container .main .block-container{{
        {max_width_str}
    }}
    </style>    
    """,
        unsafe_allow_html=True,
    )
    
_max_width_()   
st.write("# Haystack Q&A Demo")
st.sidebar.header("Options")
qtype = st.sidebar.radio("Question Type",options=["Document","FAQ"])
top_k_reader = st.sidebar.slider("How many answers ?",min_value=1,max_value=10,value=5,step=1)
filters = st.sidebar.selectbox("Filters",options=["","Option1","Option2"])
if filters == "":
    filters = None
question = st.text_input("Please provide your query:",value="Who is the father of Arya Starck?")
if question != "":
    if qtype == "Document":
        st.write("## Retrieved answers:")
        st.write(retrieve_doc(question,filters,top_k_reader=top_k_reader),unsafe_allow_html=True)
    if qtype == "FAQ":
        st.write("## Retrieved FAQ:")
        st.write(retrieve_faq(question,filters,top_k_reader=top_k_reader),unsafe_allow_html=True)
