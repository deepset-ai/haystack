import os
import sys

import logging
import pandas as pd
from pathlib import Path
import streamlit as st
from annotated_text import annotated_text   # pip install st-annotated-text

# streamlit does not support any states out of the box. On every button click, streamlit reload the whole page
# and every value gets lost. To keep track of our feedback state we use the official streamlit gist mentioned
# here https://gist.github.com/tvst/036da038ab3e999a64497f42de966a92
import SessionState
from utils import feedback_doc, haystack_is_ready, retrieve_doc, upload_doc


# Adjust to a question that you would like users to see in the search bar when they load the UI:
DEFAULT_QUESTION_AT_STARTUP = "Who is the father of Arya Stark?"

# Labels for the evaluation
EVAL_LABELS = os.getenv("EVAL_FILE", Path(__file__).parent / "eval_labels_example.csv")

# Whether the file upload should be enabled or not
DISABLE_FILE_UPLOAD = os.getenv("HAYSTACK_UI_DISABLE_FILE_UPLOAD")


def annotate_answer(answer, context):
    """ If we are using an extractive QA pipeline, we'll get answers
    from the API that we highlight in the given context"""
    start_idx = context.find(answer)
    end_idx = start_idx + len(answer)
    annotated_text(context[:start_idx], (answer, "ANSWER", "#8ef"), context[end_idx:])


def show_plain_documents(text):
    """ If we are using a plain document search pipeline, i.e. only retriever, we'll get plain documents
    from the API that we just show without any highlighting"""
    st.markdown(text)


def random_questions(df):
    """
    Helper to get one random question + gold random_answer from the user's CSV 'EVAL_LABELS_example'.
    This can then be shown in the UI when the evaluation mode is selected. Users can easily give feedback on the
    model's results and "enrich" the eval dataset with more acceptable labels
    """
    random_row = df.sample(1)
    random_question = random_row["Question Text"].values[0]
    random_answer = random_row["Answer"].values[0]
    return random_question, random_answer


def main():

    # Persistent state
    state = SessionState.get(
        random_question=DEFAULT_QUESTION_AT_STARTUP, 
        random_answer="",
        run_query=False,  # Needs to be preserved in evaluation mode, where needs to be True across reloads
        get_next_question=True
    )

    # Small callback to reset the value of state.run_query in case the text of the question changes
    def reset_run_query_value(*args):
        state.run_query = False

    # Title
    st.write("# Haystack Demo")

    # Sidebar
    st.sidebar.header("Options")
    top_k_reader = st.sidebar.slider("Max. number of answers", min_value=1, max_value=10, value=3, step=1)
    top_k_retriever = st.sidebar.slider("Max. number of documents from retriever", min_value=1, max_value=10, value=3, step=1)
    eval_mode = st.sidebar.checkbox("Evaluation mode")
    debug = st.sidebar.checkbox("Show debug info")

    # File upload block
    if not DISABLE_FILE_UPLOAD:
        st.sidebar.write("## File Upload:")
        data_files = st.sidebar.file_uploader("", type=["pdf", "txt", "docx"], accept_multiple_files=True)
        for data_file in data_files:
            # Upload file
            if data_file:
                raw_json = upload_doc(data_file)
                st.sidebar.write(raw_json)
                if debug:
                    st.subheader("REST API JSON response")
                    st.sidebar.write(raw_json)


    # Load csv into pandas dataframe
    if eval_mode:
        try:
            df = pd.read_csv(EVAL_LABELS, sep=";")
        except Exception:
            sys.exit(f"The eval file was not found under `{EVAL_LABELS}`. Please check the README for more information.")

        # Get next random question from the CSV
        state.get_next_question = st.button("Load new question")
        if state.get_next_question:
            state.run_query = False   # It probably was True because we're in evaluation mode. 
                                      # Set to False as we're changing the question and we need the user to press Run again.
            # Avoid picking the same question twice (the change is not visible on the UI)
            new_question = state.random_question
            while new_question == state.random_question:
                new_question, state.random_answer = random_questions(df)
            state.random_question = new_question

    # Search bar
    question = st.text_input(
        "Please provide your query:", 
        value=state.random_question, 
        max_chars=100, 
        on_change=reset_run_query_value
    )
    if not question:
        st.error("🚫 &nbsp;&nbsp; Please write a question")
        st.button("Run")
        return
    
    # In evaluation mode I should not re-assign the value of the Run button click
    run_state = st.button("Run")
    if not state.run_query:
        state.run_query = run_state

    # Check the connection
    with st.spinner("⌛️ &nbsp;&nbsp; Haystack is starting..."):
        if not haystack_is_ready():
            st.error("🚫 &nbsp;&nbsp; Connection Error. Is Haystack running?")
            state.run_query = False

    # Get results for query
    if state.run_query:
        with st.spinner(
            "🧠 &nbsp;&nbsp; Performing neural search on documents... \n "
            "Do you want to optimize speed or accuracy? \n"
            "Check out the docs: https://haystack.deepset.ai/usage/optimization "
        ):
            try:
                results, raw_json = retrieve_doc(question, top_k_reader=top_k_reader, top_k_retriever=top_k_retriever)
            except Exception as e:
                logging.exception(e)
                st.error("🐞 &nbsp;&nbsp; An error occurred during the request. Check the logs in the console to know more.")
                return

        # Show the gold answer if we use a question of the given set
        if question == state.random_question and eval_mode:
            st.write("## Correct answers:")
            st.write(state.random_answer)

        st.write("## Results:")
        count = 0  # Make every button key unique

        for result in results:
            if result["answer"]:
                annotate_answer(result["answer"], result["context"])
            else:
                show_plain_documents(result["context"])

            st.write("**Relevance:** ", result["relevance"], "**Source:** ", result["source"])
            if eval_mode:
                # Define columns for buttons
                button_col1, button_col2, button_col3, _ = st.columns([1, 1, 1, 6])
                if button_col1.button("👍", key=f"{result['context']}{count}1", help="Correct answer"):
                    feedback_doc(
                        question=question, 
                        is_correct_answer="true", 
                        document_id=result["document_id"], 
                        model_id=1, 
                        is_correct_document="true",
                        answer=result["answer"], 
                        offset_start_in_doc=result["offset_start_in_doc"]
                    )
                    st.success("Thanks for your feedback")

                if button_col2.button("👎", key=f"{result['context']}{count}2", help="Wrong answer and wrong passage"):
                    feedback_doc(
                        question=question, 
                        is_correct_answer="false", 
                        document_id=result["document_id"], 
                        model_id=1, 
                        is_correct_document="false",
                        answer=result["answer"], 
                        offset_start_in_doc=result["offset_start_in_doc"]
                    )
                    st.success("Thanks for your feedback!")

                if button_col3.button("👎👍", key=f"{result['context']}{count}3", help="Wrong answer, but correct passage"):
                    feedback_doc(
                        question=question, 
                        is_correct_answer="false", 
                        document_id=result["document_id"], 
                        model_id=1, 
                        is_correct_document="true",
                        answer=result["answer"], 
                        offset_start_in_doc=result["offset_start_in_doc"]
                    )
                    st.success("Thanks for your feedback!")
                count += 1
            st.write("___")

        if debug:
            st.subheader("REST API JSON response")
            st.write(raw_json)

main()
