import streamlit as st
from utils import retrieve_doc
from annotated_text import annotated_text

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

def annotate_answer(answer,context):
    start_idx = context.find(answer)
    end_idx = start_idx+len(answer)
    return annotated_text(context[:start_idx],(answer,"ANSWER","#8ef"),context[end_idx:],)
    
_max_width_()   
st.write("# Haystack Q&A Demo")
st.sidebar.header("Options")
top_k_reader = st.sidebar.slider("Number of answers",min_value=1,max_value=10,value=5,step=1)
top_k_retriever = st.sidebar.slider("Number of documents from retriever",min_value=1,max_value=15,value=5,step=1)
question = st.text_input("Please provide your query:",value="Who is the father of Arya Starck?")
if top_k_reader > top_k_retriever:
    st.error("'Number of answers' cannot be greater than 'Number of documents'")
else:
    run_query = st.button("Answer")
    debug = st.sidebar.checkbox("Show debug info")
    if run_query:
        with st.spinner("Performing neural search on documents... 🧠"):
            results,raw_json = retrieve_doc(question,top_k_reader=top_k_reader,top_k_retriever=top_k_retriever)
        st.write("## Retrieved answers:")
        for result in results:
            annotate_answer(result['answer'],result['context'])
            '**Relevance:** ', result['relevance'] , '**source:** ' , result['source']
        if debug:
            st.subheader('REST API JSON response')
            st.write(raw_json)