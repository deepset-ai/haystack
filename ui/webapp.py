import os
import sys

import logging
import pandas as pd
from json import JSONDecodeError
from pathlib import Path
import streamlit as st
from annotated_text import annotation
from markdown import markdown

# streamlit does not support any states out of the box. On every button click, streamlit reload the whole page
# and every value gets lost. To keep track of our feedback state we use the official streamlit gist mentioned
# here https://gist.github.com/tvst/036da038ab3e999a64497f42de966a92
import SessionState
from utils import HS_VERSION, haystack_is_ready, query, send_feedback, upload_doc, haystack_version


# Adjust to a question that you would like users to see in the search bar when they load the UI:
DEFAULT_QUESTION_AT_STARTUP = os.getenv("DEFAULT_QUESTION_AT_STARTUP", "What's the capital of France?")

# Sliders
DEFAULT_DOCS_FROM_RETRIEVER = int(os.getenv("DEFAULT_DOCS_FROM_RETRIEVER", 3))
DEFAULT_NUMBER_OF_ANSWERS = int(os.getenv("DEFAULT_NUMBER_OF_ANSWERS", 3))

# Labels for the evaluation
EVAL_LABELS = os.getenv("EVAL_FILE", Path(__file__).parent / "eval_labels_example.csv")

# Whether the file upload should be enabled or not
DISABLE_FILE_UPLOAD = bool(os.getenv("DISABLE_FILE_UPLOAD"))


def main():

    st.set_page_config(page_title='Haystack Demo', page_icon="https://haystack.deepset.ai/img/HaystackIcon.png")

    # Persistent state
    state = SessionState.get(
        random_question=DEFAULT_QUESTION_AT_STARTUP, 
        random_answer="",
        last_question=DEFAULT_QUESTION_AT_STARTUP,
        results=None,
        raw_json=None,
    )

    # Small callback to reset the interface in case the text of the question changes
    def reset_results(*args):
        state.results = None
        state.raw_json = None

    # Title
    st.write("# Haystack Demo - Explore the world")
    st.markdown("""
This demo takes its data from a selection of Wikipedia pages crawled in November 2021 on the topic of 

<h3 style='text-align:center;padding: 0 0 1rem;'>Countries and capital cities</h3>

Ask any question on this topic and see if Haystack can find the correct answer to your query!

*Note: do not use keywords, but full-fledged questions.* The demo is not optimized to deal with keyword queries and might misunderstand you.
""", unsafe_allow_html=True)

    # Sidebar
    st.sidebar.header("Options")
    top_k_reader = st.sidebar.slider(
        "Max. number of answers", 
        min_value=1, 
        max_value=10, 
        value=DEFAULT_NUMBER_OF_ANSWERS, 
        step=1, 
        on_change=reset_results)
    top_k_retriever = st.sidebar.slider(
        "Max. number of documents from retriever", 
        min_value=1, 
        max_value=10, 
        value=DEFAULT_DOCS_FROM_RETRIEVER, 
        step=1, 
        on_change=reset_results)
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

    hs_version = ""
    try:
        hs_version = f" <small>(v{haystack_version()})</small>"
    except Exception:
        pass

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
        <h4>Built with <a href="https://www.deepset.ai/haystack">Haystack</a>{hs_version}</h4>
        <p>Get it on <a href="https://github.com/deepset-ai/haystack/">GitHub</a> &nbsp;&nbsp; - &nbsp;&nbsp; Read the <a href="https://haystack.deepset.ai/overview/intro">Docs</a></p>
        <small>Data crawled from <a href="https://en.wikipedia.org/wiki/Category:Lists_of_countries_by_continent">Wikipedia</a> in November 2021.<br />See the <a href="https://creativecommons.org/licenses/by-sa/3.0/">License</a> (CC BY-SA 3.0).</small>
    </div>
    """, unsafe_allow_html=True)

    # Load csv into pandas dataframe
    try:
        df = pd.read_csv(EVAL_LABELS, sep=";")
    except Exception:
        st.error(f"The eval file was not found. Please check the demo's [README](https://github.com/deepset-ai/haystack/tree/master/ui/README.md) for more information.")
        sys.exit(f"The eval file was not found under `{EVAL_LABELS}`. Please check the README (https://github.com/deepset-ai/haystack/tree/master/ui/README.md) for more information.")

    # Search bar
    question = st.text_input("",
        value=state.random_question,
        max_chars=100, 
        on_change=reset_results
    )
    col1, col2 = st.columns(2)
    col1.markdown("<style>.stButton button {width:100%;}</style>", unsafe_allow_html=True)
    col2.markdown("<style>.stButton button {width:100%;}</style>", unsafe_allow_html=True)

    # Run button
    run_pressed = col1.button("Run")
    run_query = run_pressed or question != state.last_question

    # Get next random question from the CSV
    #state.get_next_question = col2.button("Random question")
    if col2.button("Random question"):
        reset_results()
        new_row = df.sample(1)   
        while new_row["Question Text"].values[0] == state.random_question:  # Avoid picking the same question twice (the change is not visible on the UI)
            new_row = df.sample(1)
        state.random_question = new_row["Question Text"].values[0]
        state.random_answer = new_row["Answer"].values[0]

        # Re-runs the script setting the random question as the textbox value
        # Unfortunately necessary as the Random Question button is _below_ the textbox
        raise st.script_runner.RerunException(st.script_request_queue.RerunData(None))

    # Check the connection
    with st.spinner("⌛️ &nbsp;&nbsp; Haystack is starting..."):
        if not haystack_is_ready():
            st.error("🚫 &nbsp;&nbsp; Connection Error. Is Haystack running?")
            run_query = False
            reset_results()

    # Get results for query
    if run_query and question:
        reset_results()
        state.last_question = question
        with st.spinner(
            "🧠 &nbsp;&nbsp; Performing neural search on documents... \n "
            "Do you want to optimize speed or accuracy? \n"
            "Check out the docs: https://haystack.deepset.ai/usage/optimization "
        ):
            try:
                state.results, state.raw_json = query(question, top_k_reader=top_k_reader, top_k_retriever=top_k_retriever)
            except JSONDecodeError as je:
                st.error("👓 &nbsp;&nbsp; An error occurred reading the results. Is the document store working?")
                return
            except Exception as e:
                logging.exception(e)
                if "The server is busy processing requests" in str(e) or "503" in str(e):
                    st.error("🧑‍🌾 &nbsp;&nbsp; All our workers are busy! Try again later.")
                else:
                    st.error("🐞 &nbsp;&nbsp; An error occurred during the request.")
                return

    if state.results:

        # Show the gold answer if we use a question of the given set
        if question == state.random_question and eval_mode and state.random_answer:
            st.write("## Correct answers:")
            st.write(state.random_answer)

        st.write("## Results:")

        for count, result in enumerate(state.results):
            if result["answer"]:
                answer, context = result["answer"], result["context"]
                start_idx = context.find(answer)
                end_idx = start_idx + len(answer)
                # Hack due to this bug: https://github.com/streamlit/streamlit/issues/3190 
                st.write(markdown(context[:start_idx] + str(annotation(answer, "ANSWER", "#8ef")) + context[end_idx:]), unsafe_allow_html=True)
                st.write("**Relevance:** ", result["relevance"], "**Source:** ", result["source"])

            else:
                st.info("🤔 &nbsp;&nbsp; Haystack is unsure whether any of the documents contain an answer to your question. Try to reformulate it!")
                st.write("**Relevance:** ", result["relevance"])
                
            if eval_mode and result["answer"]:
                # Define columns for buttons
                is_correct_answer = None
                is_correct_document = None

                button_col1, button_col2, button_col3, _ = st.columns([1, 1, 1, 6])
                if button_col1.button("👍", key=f"{result['context']}{count}1", help="Correct answer"):
                    is_correct_answer=True
                    is_correct_document=True

                if button_col2.button("👎", key=f"{result['context']}{count}2", help="Wrong answer and wrong passage"):
                    is_correct_answer=False
                    is_correct_document=False

                if button_col3.button("👎👍", key=f"{result['context']}{count}3", help="Wrong answer, but correct passage"):
                    is_correct_answer=False
                    is_correct_document=True

                if is_correct_answer is not None and is_correct_document is not None:
                    try:
                        send_feedback(
                            query=question,
                            answer_obj=result["_raw"],
                            is_correct_answer=is_correct_answer,
                            is_correct_document=is_correct_document,
                            document=result["document"]
                        )
                        st.success("✨ &nbsp;&nbsp; Thanks for your feedback! &nbsp;&nbsp; ✨")
                    except Exception as e:
                        logging.exception(e)
                        st.error("🐞 &nbsp;&nbsp; An error occurred while submitting your feedback!")

            st.write("___")

        if debug:
            st.subheader("REST API JSON response")
            st.write(state.raw_json)

main()
