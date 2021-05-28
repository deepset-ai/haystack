import os
import sys

import pandas as pd
import streamlit as st
from annotated_text import annotated_text

# streamlit does not support any states out of the box. On every button click, streamlit reload the whole page
# and every value gets lost. To keep track of our feedback state we use the official streamlit gist mentioned
# here https://gist.github.com/tvst/036da038ab3e999a64497f42de966a92
import SessionState
from utils import feedback_doc
from utils import retrieve_doc
from utils import upload_doc


def annotate_answer(answer, context):
    start_idx = context.find(answer)
    end_idx = start_idx + len(answer)
    annotated_text(context[:start_idx], (answer, "ANSWER", "#8ef"), context[end_idx:])


def random_questions(df):
    random_row = df.sample(1)
    random_question = random_row["Question Text"].values[0]
    random_answer = random_row["Answer"].values[0]
    return random_question, random_answer


# Define state
state_question = SessionState.get(
    random_question="Who is the father of Arya Starck?", random_answer="", next_question="false", run_query="false"
)

# Initialize variables
eval_mode = False
random_question = "Who is the father of Arya Starck?"
eval_labels = os.getenv("EVAL_FILE", "eval_labels_example.csv")

# UI search bar and sidebar
st.write("# Haystack Demo")
st.sidebar.header("Options")
top_k_reader = st.sidebar.slider("Max. number of answers", min_value=1, max_value=10, value=3, step=1)
top_k_retriever = st.sidebar.slider(
    "Max. number of documents from retriever", min_value=1, max_value=10, value=3, step=1
)
eval_mode = st.sidebar.checkbox("Evaluation mode")
debug = st.sidebar.checkbox("Show debug info")

st.sidebar.write("## File Upload:")
data_file = st.sidebar.file_uploader("", type=["pdf", "txt", "docx"])
# Upload file
if data_file:
    raw_json = upload_doc(data_file)
    st.sidebar.write(raw_json)
    if debug:
        st.subheader("REST API JSON response")
        st.sidebar.write(raw_json)

# load csv into pandas dataframe
if eval_mode:
    try:
        df = pd.read_csv(eval_labels, sep=";")
    except Exception:
        sys.exit("The eval file was not found. Please check the README for more information.")
    if (
        state_question
        and hasattr(state_question, "next_question")
        and hasattr(state_question, "random_question")
        and state_question.next_question
    ):
        random_question = state_question.random_question
        random_answer = state_question.random_answer
    else:
        random_question, random_answer = random_questions(df)
        state_question.random_question = random_question
        state_question.random_answer = random_answer

# Generate new random question
if eval_mode:
    next_question = st.button("Load new question")
    if next_question:
        random_question, random_answer = random_questions(df)
        state_question.random_question = random_question
        state_question.random_answer = random_answer
        state_question.next_question = "true"
        state_question.run_query = "false"
    else:
        state_question.next_question = "false"

# Search bar
question = st.text_input("Please provide your query:", value=random_question)
if state_question and state_question.run_query:
    run_query = state_question.run_query
    st.button("Run")
else:
    run_query = st.button("Run")
    state_question.run_query = run_query

raw_json_feedback = ""

# Get results for query
if run_query:
    with st.spinner(
        "Performing neural search on documents... 🧠 \n "
        "Do you want to optimize speed or accuracy? \n"
        "Check out the docs: https://haystack.deepset.ai/docs/latest/optimizationmd "
    ):
        results, raw_json = retrieve_doc(question, top_k_reader=top_k_reader, top_k_retriever=top_k_retriever)

    # Show if we use a question of the given set
    if question == random_question and eval_mode:
        st.write("## Correct answers:")
        random_answer

    st.write("## Retrieved answers:")

    # Make every button key unique
    count = 0

    for result in results:
        annotate_answer(result["answer"], result["context"])
        "**Relevance:** ", result["relevance"], "**Source:** ", result["source"]
        if eval_mode:
            # Define columns for buttons
            button_col1, button_col2, button_col3, button_col4 = st.beta_columns([1, 1, 1, 6])
            if button_col1.button("👍", key=(result["answer"] + str(count)), help="Correct answer"):
                raw_json_feedback = feedback_doc(
                    question, "true", result["document_id"], 1, "true", result["answer"], result["offset_start_in_doc"]
                )
                st.success("Thanks for your feedback")
            if button_col2.button("👎", key=(result["answer"] + str(count)), help="Wrong answer and wrong passage"):
                raw_json_feedback = feedback_doc(
                    question,
                    "false",
                    result["document_id"],
                    1,
                    "false",
                    result["answer"],
                    result["offset_start_in_doc"],
                )
                st.success("Thanks for your feedback!")
            if button_col3.button("👎👍", key=(result["answer"] + str(count)), help="Wrong answer, but correct passage"):
                raw_json_feedback = feedback_doc(
                    question, "false", result["document_id"], 1, "true", result["answer"], result["offset_start_in_doc"]
                )
                st.success("Thanks for your feedback!")
            count += 1
        st.write("___")
    if debug:
        st.subheader("REST API JSON response")
        st.write(raw_json)
