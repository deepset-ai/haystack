import os
import sys
import logging
from pathlib import Path
from json import JSONDecodeError

import pandas as pd
import streamlit as st
from annotated_text import annotation
from markdown import markdown

from ui.utils import (
    haystack_is_ready,
    query,
    send_feedback as send_feedback_rulebook,
    upload_doc,
    haystack_version,
    get_backlink,
    send_feedback_faq,
)


# Adjust to a question that you would like users to see in the search bar when they load the UI:
DEFAULT_QUESTION_AT_STARTUP = "What is the objective of the game?"
DEFAULT_ANSWER_AT_STARTUP = None

# Sliders
DEFAULT_DOCS_RULEBOOK = 5
DEFAULT_N_ANS_RULEBOOK = 1
DEFAULT_N_ANS_FAQ = 5

# Whether the file upload should be enabled or not
DISABLE_FILE_UPLOAD = True


def set_state_if_absent(key, value):
    if key not in st.session_state:
        st.session_state[key] = value


def main():
    icon_url = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSUp4mzekWcCkn3aSZQ2vrWB7Os8CYCtiMLlA&usqp=CAU"
    st.set_page_config(page_title="Board Game Rules Explainer", page_icon=icon_url)

    # Persistent state
    set_state_if_absent("question", DEFAULT_QUESTION_AT_STARTUP)
    set_state_if_absent("answer", DEFAULT_ANSWER_AT_STARTUP)
    set_state_if_absent("results", None)
    set_state_if_absent("raw_json", None)
    set_state_if_absent("random_question_requested", False)

    # Small callback to reset the interface in case the text of the question changes
    def reset_results(*args):
        st.session_state.answer = None
        st.session_state.results = None
        st.session_state.raw_json = None

    # Title
    st.write("# Board game rules explainer 🎲")
    st.write("---")
    st.markdown(
        """
    Ask any question about Monopoly's boardgame rules, and see our AI in action! 🤖

    We support two types of answers:
    - searching through our predefined set of FAQ 📁
    - searching through the original rulebook 📖

    Additionally, feedbacks from you are more than welcome! 🙌🏾
    - If you check the 'evaluate' box on left panel, you'll be able to provide feedbacks on our AI's answers - this way, with every feedback it will get even better! 🦾

    """,
        unsafe_allow_html=True,
    )
    st.write("---")

    # Sidebar
    st.sidebar.header("Options")
    top_k_reader_rulebook = st.sidebar.slider(
        "Max. number of answers from rulebook",
        min_value=1,
        max_value=10,
        value=DEFAULT_N_ANS_RULEBOOK,
        step=1,
        on_change=reset_results,
    )

    top_k_reader_faq = st.sidebar.slider(
        "Max. number of answers from faq",
        min_value=1,
        max_value=10,
        value=DEFAULT_N_ANS_RULEBOOK,
        step=1,
        on_change=reset_results,
    )

    top_k_retriever_rulebook = st.sidebar.slider(
        "Max. number of documents from retriever",
        min_value=1,
        max_value=10,
        value=DEFAULT_DOCS_RULEBOOK,
        step=1,
        on_change=reset_results,
    )

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

    st.sidebar.markdown(
        f"""
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
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Search bar
    question = st.text_input("", value=st.session_state.question, max_chars=100, on_change=reset_results)
    col1, col2 = st.columns(2)
    col1.markdown("<style>.stButton button {width:100%;}</style>", unsafe_allow_html=True)
    col2.markdown("<style>.stButton button {width:100%;}</style>", unsafe_allow_html=True)

    # Run button
    run_pressed_faq = col1.button("Search in FAQ")
    run_pressed_rulebook = col2.button("Search in the rulebook")

    run_query = run_pressed_faq or run_pressed_rulebook or question != st.session_state.question

    # Check the connection
    with st.spinner("⌛️ &nbsp;&nbsp; Haystack is starting..."):
        if not haystack_is_ready():
            st.error("🚫 &nbsp;&nbsp; Connection Error. Is Haystack running?")
            run_query = False
            reset_results()

    # Get results for query
    if run_query and question:
        reset_results()
        st.session_state.question = question
        st.session_state.index_option = "faq" if run_pressed_faq else "rulebook"

        with st.spinner(
            "🧠 &nbsp;&nbsp; Performing neural search on documents... \n "
            "Do you want to optimize speed or accuracy? \n"
            "Check out the docs: https://haystack.deepset.ai/usage/optimization "
        ):
            try:
                st.session_state.results, st.session_state.raw_json = query(
                    question,
                    index_option=st.session_state.index_option,
                    top_k_reader_rulebook=top_k_reader_rulebook,
                    top_k_reader_faq=top_k_reader_faq,
                    top_k_retriever_rulebook=top_k_retriever_rulebook,
                )
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

    if st.session_state.results:

        # Show the gold answer if we use a question of the given set
        if eval_mode and st.session_state.answer:
            st.write("## Correct answer:")
            st.write(st.session_state.answer)

        st.write("## Results:")

        for count, result in enumerate(st.session_state.results):
            if result["answer"]:
                answer, context = result["answer"], result["context"]
                start_idx = context.find(answer)
                end_idx = start_idx + len(answer)
                # Hack due to this bug: https://github.com/streamlit/streamlit/issues/3190
                st.write(
                    markdown(context[:start_idx] + str(annotation(answer, "ANSWER", "#8ef")) + context[end_idx:]),
                    unsafe_allow_html=True,
                )
                source = ""
                url, title = get_backlink(result)
                if url and title:
                    source = f"[{result['document']['meta']['title']}]({result['document']['meta']['url']})"
                else:
                    source = f"{result['source']}"
                st.markdown(f"**Relevance:** {result['relevance']} -  **Source:** {source}")

            else:
                st.info(
                    "🤔 &nbsp;&nbsp; Haystack is unsure whether any of the documents contain an answer to your question. Try to reformulate it!"
                )
                st.write("**Relevance:** ", result["relevance"])

            if eval_mode and result["answer"]:
                # Define columns for buttons
                is_correct_answer = None
                is_correct_document = None

                button_col1, button_col2, button_col3, _ = st.columns([1, 1, 1, 6])
                if button_col1.button("👍", key=f"{result['context']}{count}1", help="Correct answer"):
                    is_correct_answer = True
                    is_correct_document = True

                if button_col2.button("👎", key=f"{result['context']}{count}2", help="Wrong answer and wrong passage"):
                    is_correct_answer = False
                    is_correct_document = False

                if button_col3.button(
                    "👎👍", key=f"{result['context']}{count}3", help="Wrong answer, but correct passage"
                ):
                    is_correct_answer = False
                    is_correct_document = True

                if is_correct_answer is not None and is_correct_document is not None:
                    if st.session_state.index_option == "faq":
                        feedback_fn = send_feedback_faq
                    elif st.session_state.index_option == "rulebook":
                        feedback_fn = send_feedback_rulebook

                    raise NotImplementedError("Still work in progress...")

                    # try:
                    #     feedback_fn(
                    #         query=question,
                    #         answer_obj=result["_raw"],
                    #         is_correct_answer=is_correct_answer,
                    #         is_correct_document=is_correct_document,
                    #         document=result["document"],
                    #         index = st.session_state.index_option,
                    #     )
                    #     st.success("✨ &nbsp;&nbsp; Thanks for your feedback! &nbsp;&nbsp; ✨")
                    # except Exception as e:
                    #     logging.exception(e)
                    #     st.error("🐞 &nbsp;&nbsp; An error occurred while submitting your feedback!")

            st.write("___")

        if debug:
            st.subheader("REST API JSON response")
            st.write(st.session_state.raw_json)


main()
