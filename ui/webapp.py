import os
import sys

import logging
import pandas as pd
from pathlib import Path
import streamlit as st
from annotated_text import annotated_text

# streamlit does not support any states out of the box. On every button click, streamlit reload the whole page
# and every value gets lost. To keep track of our feedback state we use the official streamlit gist mentioned
# here https://gist.github.com/tvst/036da038ab3e999a64497f42de966a92
import SessionState
from utils import HS_VERSION, feedback_doc, haystack_is_ready, retrieve_doc, upload_doc, haystack_version


# Adjust to a question that you would like users to see in the search bar when they load the UI:
DEFAULT_QUESTION_AT_STARTUP = "Who's the father of Arya Stark?"

# Labels for the evaluation
EVAL_LABELS = os.getenv("EVAL_FILE", Path(__file__).parent / "eval_labels_example.csv")

# Whether the file upload should be enabled or not
DISABLE_FILE_UPLOAD = os.getenv("HAYSTACK_UI_DISABLE_FILE_UPLOAD")

# Retrieve Haystack version from the REST API
HS_VERSION = haystack_version()


def main():

    # Persistent state
    state = SessionState.get(
        random_question=DEFAULT_QUESTION_AT_STARTUP, 
        random_answer="",
        results=None,
        raw_json=None,
        get_next_question=True
    )

    # Small callback to reset the interface in case the text of the question changes
    def reset_results(*args):
        state.results = None
        state.raw_json = None

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
                st.sidebar.write(str(data_file.name) + " &nbsp;&nbsp; ✅ ")
                if debug:
                    st.subheader("REST API JSON response")
                    st.sidebar.write(raw_json)

    st.sidebar.markdown(f"""
    <style>
        a {{
            text-decoration: none;
        }}
        .haystack-footer {{
            text-align: center;
        }}
        .haystack-footer h4 {{
            margin: 0.1rem; 
            padding:0;
        }}
        footer {{
            opacity: 0;
        }}
    </style>
    <div class="haystack-footer">
        <hr />
        <h4>Built with <a href="https://www.deepset.ai/haystack">Haystack</a> <small>(v{HS_VERSION})</small></h4>
        <p>Get it on <a href="https://github.com/deepset-ai/haystack/">GitHub</a> &nbsp;&nbsp; - &nbsp;&nbsp; Read the <a href="https://haystack.deepset.ai/overview/intro">Docs</a></p>
        <small>Data crawled from <a href="https://en.wikipedia.org/wiki/Category:Lists_of_countries_by_continent">Wikipedia</a> in November 2021.<br />See the <a href="https://creativecommons.org/licenses/by-sa/3.0/">License</a> (CC BY-SA 3.0).</small>
    </div>
    """, unsafe_allow_html=True)

    # Load csv into pandas dataframe
    if eval_mode:
        try:
            df = pd.read_csv(EVAL_LABELS, sep=";")
        except Exception:
            st.error(f"The eval file was not found. Please check the demo's [README](https://github.com/deepset-ai/haystack/tree/master/ui/README.md) for more information.")
            sys.exit(f"The eval file was not found under `{EVAL_LABELS}`. Please check the README (https://github.com/deepset-ai/haystack/tree/master/ui/README.md) for more information.")

        # Get next random question from the CSV
        state.get_next_question = st.button("Load new question")
        if state.get_next_question:
            reset_results()
            new_row = df.sample(1)   
            while new_row["Question Text"].values[0] == state.random_question:  # Avoid picking the same question twice (the change is not visible on the UI)
                new_row = df.sample(1)
            state.random_question = new_row["Question Text"].values[0]
            state.random_answer = new_row["Answer"].values[0]

    # Search bar
    question = st.text_input(
        "Please provide your query:", 
        value=state.random_question, 
        max_chars=100, 
        on_change=reset_results
    )
    run_query = st.button("Run")

    # Check the connection
    with st.spinner("⌛️ &nbsp;&nbsp; Haystack is starting..."):
        if not haystack_is_ready():
            st.error("🚫 &nbsp;&nbsp; Connection Error. Is Haystack running?")
            run_query = False
            reset_results()

    # Get results for query
    if run_query and question:
        reset_results()
        with st.spinner(
            "🧠 &nbsp;&nbsp; Performing neural search on documents... \n "
            "Do you want to optimize speed or accuracy? \n"
            "Check out the docs: https://haystack.deepset.ai/usage/optimization "
        ):
            try:
                state.results, state.raw_json = retrieve_doc(question, top_k_reader=top_k_reader, top_k_retriever=top_k_retriever)
            except Exception as e:
                logging.exception(e)
                if "The server is busy processing requests" in str(e):
                    st.error("🧑‍🌾 &nbsp;&nbsp; All our workers are busy! Try again later.")
                else:
                    st.error("🐞 &nbsp;&nbsp; An error occurred during the request. Check the logs in the console to know more.")
                return

    if state.results:

        # Show the gold answer if we use a question of the given set
        if question == state.random_question and eval_mode:
            st.write("## Correct answers:")
            st.write(state.random_answer)

        st.write("## Results:")
        count = 0  # Make every button key unique

        for result in state.results:
            if result["answer"]:
                answer, context = result["answer"], result["context"]
                start_idx = context.find(answer)
                end_idx = start_idx + len(answer)
                annotated_text(context[:start_idx], (answer, "ANSWER", "#8ef"), context[end_idx:])
            else:
                st.markdown(result["context"])

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
                    st.success("✨ &nbsp;&nbsp; Thanks for your feedback! &nbsp;&nbsp; ✨")

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
                    st.success("✨ &nbsp;&nbsp; Thanks for your feedback! &nbsp;&nbsp; ✨")

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
                    st.success("✨ &nbsp;&nbsp; Thanks for your feedback! &nbsp;&nbsp; ✨")
                count += 1
            st.write("___")

        if debug:
            st.subheader("REST API JSON response")
            st.write(state.raw_json)


main()
