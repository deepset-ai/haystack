import os
import sys
import logging
from pathlib import Path
from json import JSONDecodeError

import pandas as pd
import streamlit as st
from annotated_text import annotation
from markdown import markdown

from ui.utils import send_feedback, haystack_is_ready, query, upload_doc, haystack_version, get_backlink, get_feedbacks

from st_aggrid.shared import GridUpdateMode
from st_aggrid import AgGrid, GridOptionsBuilder

# Adjust to a question that you would like users to see in the search bar when they load the UI:
DEFAULT_QUESTION_AT_STARTUP = "What is the objective of the game?"
DEFAULT_ANSWER_AT_STARTUP = None

# Sliders
DEFAULT_DOCS_RULEBOOK = 10
DEFAULT_N_ANS_RULEBOOK = 1
DEFAULT_N_ANS_FAQ = 5

# Whether the file upload should be enabled or not
DISABLE_FILE_UPLOAD = True

icon_url = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSUp4mzekWcCkn3aSZQ2vrWB7Os8CYCtiMLlA&usqp=CAU"
st.set_page_config(page_title="Board Game Rules Explainer", page_icon=icon_url, layout="wide")


def set_state_if_absent(key, value):
    if key not in st.session_state:
        st.session_state[key] = value


def load_labels():
    faq_labels = get_feedbacks("label_faq")
    extractive_labels = get_feedbacks("label_rulebook")

    faq_labels_prep = [
        (f["id"], f["query"], f["answer"]["answer"], f["is_correct_answer"], f["created_at"]) for f in faq_labels
    ]
    faq_labels_df = pd.DataFrame(faq_labels_prep, columns=["id", "query", "answer", "label", "date"])

    rulebook_labels_prep = [
        (
            f["id"],
            f["query"],
            f["answer"]["answer"],
            f["answer"]["context"],
            f["is_correct_answer"],
            f["answer"]["offsets_in_document"][0],
            f["answer"]["offsets_in_context"][0],
            f["created_at"],
        )
        for f in extractive_labels
    ]

    rulebook_labels_df = pd.DataFrame(
        rulebook_labels_prep,
        columns=["id", "query", "answer", "context", "label", "span_document", "span_context", "date"],
    )

    return faq_labels_df, rulebook_labels_df


def aggrid_interactive_table(df: pd.DataFrame):
    """Creates an st-aggrid interactive table based on a dataframe.

    Args:
        df (pd.DataFrame]): Source dataframe

    Returns:
        dict: The selected row
    """
    options = GridOptionsBuilder.from_dataframe(df, enableRowGroup=True, enableValue=True, enablePivot=True)
    # options.configure_selection(use_checkbox=True)

    options.configure_side_bar()
    options.configure_selection("single")
    selection = AgGrid(
        df,
        enable_enterprise_modules=True,
        gridOptions=options.build(),
        update_mode=GridUpdateMode.MODEL_CHANGED,
        allow_unsafe_jscode=True,
    )

    return selection


# Small callback to reset the interface in case the text of the question changes
def reset_results(*args):
    st.session_state.answer = None
    st.session_state.results = None
    st.session_state.raw_json = None


def ui_page():
    # Persistent state
    set_state_if_absent("question", DEFAULT_QUESTION_AT_STARTUP)
    set_state_if_absent("answer", DEFAULT_ANSWER_AT_STARTUP)
    set_state_if_absent("results", None)
    set_state_if_absent("raw_json", None)
    set_state_if_absent("random_question_requested", False)

    # Title
    st.write("# Board game rules explainer 🎲")
    st.write("---")
    st.markdown(
        """
    Ask any question about Monopoly's boardgame rules, and see our AI in action! 🤖

    We support two types of answers:
    - searching through our predefined set of FAQ 📁
    - searching through the original rulebook 📖

    Additionally, feedbacks from you are more than welcome - below each answer from our system, you should see thumbs up 👍 and down 👎 buttons. By using them, and providing feedbacks you'll make our AI even better! 🦾

    """,
        unsafe_allow_html=True,
    )
    st.write("---")

    # File upload block
    if not DISABLE_FILE_UPLOAD:
        st.sidebar.write("## File Upload:")
        data_files = st.sidebar.file_uploader("", type=["pdf", "txt", "docx"], accept_multiple_files=True)
        for data_file in data_files:
            # Upload file
            if data_file:
                raw_json = upload_doc(data_file)
                st.sidebar.write(str(data_file.name) + " &nbsp;&nbsp; ✅ ")

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
        st.session_state.index_option_query = "faq" if run_pressed_faq else "rulebook"
        st.session_state.index_option_feedback = "label_faq" if run_pressed_faq else "label_rulebook"

        with st.spinner(
            "🧠 &nbsp;&nbsp; Performing neural search on documents... \n "
            "Do you want to optimize speed or accuracy? \n"
            "Check out the docs: https://haystack.deepset.ai/usage/optimization "
        ):
            try:
                st.session_state.results, st.session_state.raw_json = query(
                    question,
                    index_option=st.session_state.index_option_query,
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
        st.write("## Results:")
        for count, result in enumerate(st.session_state.results):
            if result["answer"]:
                answer, context = result["answer"], result["context"]
                # Hack due to this bug: https://github.com/streamlit/streamlit/issues/3190
                if st.session_state.index_option_query == "rulebook":
                    start_idx = context.find(answer)
                    end_idx = start_idx + len(answer)
                    st.write(
                        markdown(context[:start_idx] + str(annotation(answer, "ANSWER", "#8ef")) + context[end_idx:]),
                        unsafe_allow_html=True,
                    )
                elif st.session_state.index_option_query == "faq":
                    st.write(markdown(str(annotation(answer, "", "#8ef"))), unsafe_allow_html=True)
                else:
                    pass

                source = ""
                url, title = get_backlink(result)
                if url and title:
                    source = f"[{result['document']['meta']['title']}]({result['document']['meta']['url']})"
                else:
                    source = f"{result['source']}"

                source = st.session_state.index_option_query

                st.markdown(f"**Relevance:** {result['relevance']} -  **Source:** {source}")

                # Define columns for buttons
                is_correct_answer = None
                is_correct_document = None

                if st.session_state.index_option_feedback == "label_rulebook":
                    button_col1, button_col2, button_col3, _ = st.columns([1, 1, 1, 6])
                    if button_col1.button("👍", key=f"{result['context']}{count}1", help="Correct answer"):
                        is_correct_answer = True
                        is_correct_document = True

                    if button_col2.button(
                        "👎", key=f"{result['context']}{count}2", help="Wrong answer and wrong passage"
                    ):
                        is_correct_answer = False
                        is_correct_document = False

                    if button_col3.button(
                        "👎👍", key=f"{result['context']}{count}3", help="Wrong answer, but correct passage"
                    ):
                        is_correct_answer = False
                        is_correct_document = True
                elif st.session_state.index_option_feedback == "label_faq":
                    button_col1, button_col2, _ = st.columns([1, 1, 7])
                    if button_col1.button("👍", key=f"{result['context']}{count}1", help="Correct answer"):
                        is_correct_answer = True
                        is_correct_document = True

                    if button_col2.button("👎", key=f"{result['context']}{count}2", help="Incorrect answer"):
                        is_correct_answer = False
                        is_correct_document = False

                if is_correct_answer is not None and is_correct_document is not None:
                    try:
                        send_feedback(
                            query=question,
                            answer_obj=result["_raw"],
                            is_correct_answer=is_correct_answer,
                            is_correct_document=is_correct_document,
                            document=result["document"],
                            index=st.session_state.index_option_feedback,
                        )
                        st.success("✨ &nbsp;&nbsp; Thanks for your feedback! &nbsp;&nbsp; ✨")
                    except Exception as e:
                        logging.exception(e)
                        st.error("🐞 &nbsp;&nbsp; An error occurred while submitting your feedback!")

            else:
                st.info(
                    "🤔 &nbsp;&nbsp; Haystack is unsure whether any of the documents contain an answer to your question. Try to reformulate it!"
                )
                st.write("**Relevance:** ", result["relevance"])

            st.write("___")


def feedbacks_page():
    st.write("# Board game rules explainer 🎲")
    st.markdown("#### Feedbacks exploration page")
    st.write("---")

    faq_labels_df, rulebook_labels_df = load_labels()

    st.write("##### FAQ annotations")
    aggrid_interactive_table(df=faq_labels_df)

    st.write("##### Rulebook extraction annotations")
    aggrid_interactive_table(df=rulebook_labels_df)


if __name__ == "__main__":
    page_names_to_funcs = {"User interface": ui_page, "Exploring feedbacks": feedbacks_page}
    option_name = st.sidebar.selectbox("Navigate through the app 📖", page_names_to_funcs.keys())

    if option_name == "User interface":
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

        # unnessecary for the user to control it
        top_k_retriever_rulebook = DEFAULT_DOCS_RULEBOOK

    page_names_to_funcs[option_name]()

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
