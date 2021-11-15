from typing import Dict, Any, List, Optional

import json
import pprint
import logging
import pandas as pd
from collections import defaultdict

from haystack.schema import Document, Answer
from haystack.document_stores.sql import DocumentORM


logger = logging.getLogger(__name__)


def print_answers(results: dict, details: str = "all", max_text_len: Optional[int] = None):
    """
    Utility function to print results of Haystack pipelines
    :param results: Results from a pipeline
    :param details: One of "minimum", "medium", "all". Defining the level of details to print.
    :param max_text_lenght: shorten lengthy text fields to the maximum allowed length. Set to
        None to not cut long text.
    :return: None
    """
    # Defines the fields to keep in the Answer for each detail level
    fields_to_keep_by_level = {
        "minimum": ["answer", "context"],
        "medium": ["answer", "context", "score"]
    }

    if not "answers" in results.keys():
        raise ValueError("The results object does not seem to come from a Reader: "
                         f"it does not contain the 'answers' key, but only: {results.keys()}.  "
                         "Try print_documents or print_questions.")

    if "query" in results.keys():
        print(f"\nQuery: {results['query']}\nAnswers:")

    answers = results["answers"]
    pp = pprint.PrettyPrinter(indent=4)

    # Filter the results by detail level
    filtered_answers = []
    if details in fields_to_keep_by_level.keys():
        for ans in answers:
            filtered_ans = {field: getattr(ans, field) for field in fields_to_keep_by_level[details] if getattr(ans, field) is not None}
            filtered_answers.append(filtered_ans)
    elif details == "all":  
        filtered_answers = answers
    else:
        logging.warn(f"print_answers received details='{details}', which was not understood. "
                     "Valid values are 'minimum', 'medium', and 'all'. Using 'all'.")
        filtered_answers = answers

    # Shorten long text fields
    if max_text_len is not None:
        for ans in answers:
            if getattr(ans, "context") and len(ans.context) > 50:
                ans.context = ans.context[:50] + "..."

    pp.pprint(filtered_answers) 


def print_documents(results: dict, max_text_len: Optional[int] = None, print_name: bool = True, print_meta: bool = False):
    """
    Utility that prints a compressed representation of the documents returned by a pipeline.
    :param max_text_lenght: shorten the document's content to a maximum number of chars. if None, does not cut.
    :param print_name: whether to print the document's name (from the metadata) or not.
    :param print_meta: whether to print the document's metadata or not.
    """
    print(f"\nQuery: {results['query']}\n")
    pp = pprint.PrettyPrinter(indent=4)
    
    # Verify that the input contains Documents under the `document` key
    if any(not isinstance(doc, Document) for doc in results["documents"]):
        raise ValueError("This results object does not contain `Document` objects under the `documents` key. "
                         "Please make sure the last node of your pipeline makes proper use of the "
                         "new Haystack primitive objects, and if you're using Haystack nodes/pipelines only, "
                         "please report this as a bug.")

    for doc in results["documents"]:
        content = doc.content
        if max_text_len:
            content = doc.content[:max_text_len] + ("..." if len(doc.content) > max_text_len else "")
        results = {"content": content}
        if print_name:
            results["name"] = doc.meta.get("name", None)
        if print_meta:
            results["meta"] = doc.meta
        pp.pprint(results)
        print()


def print_questions(results: dict):
    """
    Utility to print the output of a question generating pipeline in a readable format.
    """
    if "generated_questions" in results.keys():        
        print("\nGenerated questions:")
        for result in results["generated_questions"]:
            for question in result["questions"]:
                print(f" - {question}")

    elif "results" in results.keys():
        print("\nGenerated pairs:")
        for pair in results["results"]:
            print(f" - Q:{pair['query']}")
            for answer in pair["answers"]:

                # Verify that the pairs contains Answers under the `answer` key
                if not isinstance(answer, Answer):
                    raise ValueError("This results object does not contain `Answer` objects under the `answers` "
                                    "key of the generated question/answer pairs. "
                                    "Please make sure the last node of your pipeline makes proper use of the "
                                    "new Haystack primitive objects, and if you're using Haystack nodes/pipelines only, "
                                    "please report this as a bug.")
                print(f"      A: {answer.answer}")

    else:
        raise ValueError("This object does not seem to be the output " 
              "of a question generating pipeline: does not contain neither "
              f"'generated_questions' nor 'results', but only: {results.keys()}. "
              " Try `print_answers` or `print_documents`.")


def export_answers_to_csv(agg_results: list, output_file):
    """
    Exports answers coming from finder.get_answers() to a CSV file
    :param agg_results: list of predictions coming from finder.get_answers()
    :param output_file: filename of output file
    :return: None
    """
    if isinstance(agg_results, dict):
        agg_results = [agg_results]

    assert "query" in agg_results[0], f"Wrong format used for {agg_results[0]}"
    assert "answers" in agg_results[0], f"Wrong format used for {agg_results[0]}"

    data = {} # type: Dict[str, List[Any]]
    data["query"] = []
    data["prediction"] = []
    data["prediction_rank"] = []
    data["prediction_context"] = []

    for res in agg_results:
        for i in range(len(res["answers"])):
            temp = res["answers"][i]
            data["query"].append(res["query"])
            data["prediction"].append(temp["answer"])
            data["prediction_rank"].append(i + 1)
            data["prediction_context"].append(temp["context"])

    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)


def convert_labels_to_squad(labels_file: str):
    """
    Convert the export from the labeling UI to SQuAD format for training.

    :param labels_file: path for export file from the labeling tool
    :return:
    """
    with open(labels_file, encoding='utf-8') as label_file:
        labels = json.load(label_file)

    labels_grouped_by_documents = defaultdict(list)
    for label in labels:
        labels_grouped_by_documents[label["document_id"]].append(label)

    labels_in_squad_format = {"data": []}  # type: Dict[str, Any]
    for document_id, labels in labels_grouped_by_documents.items():
        qas = []
        for label in labels:
            doc = DocumentORM.query.get(label["document_id"])

            assert (
                    doc.content[label["start_offset"]: label["end_offset"]]
                == label["selected_text"]
            )

            qas.append(
                {
                    "question": label["question"],
                    "id": label["id"],
                    "question_id": label["question_id"],
                    "answers": [
                        {
                            "text": label["selected_text"],
                            "answer_start": label["start_offset"],
                            "labeller_id": label["labeler_id"],
                        }
                    ],
                    "is_impossible": False,
                }
            )

        squad_format_label = {
            "paragraphs": [
                {"qas": qas, "context": doc.content, "document_id": document_id}
            ]
        }

        labels_in_squad_format["data"].append(squad_format_label)

    with open("labels_in_squad_format.json", "w+", encoding='utf-8') as outfile:
        json.dump(labels_in_squad_format, outfile)
