from typing import Dict, Any, List, Optional

import io
import re
import time
import json
import pprint
import logging
import subprocess
import pandas as pd
from collections import defaultdict

from haystack.document_stores.sql import DocumentORM


logger = logging.getLogger(__name__)



def print_answers(results: dict, details: str = "all"):
    """
    Utilitiy function to print results of Haystack pipelines
    :param results: Results from a pipeline
    :param details: One of ["minimum", "medium", "all]. Defining the level of details to print.
    :return: None
    """
    # TODO: unify the output format of Generator and Reader so that this function doesn't have the try/except
    #  Or implement a class method like PredReader.print() and PredGenerator.print() that handles all this functionality.
    # This default case is when the answers come from a Reader
    try:
        answers = results["answers"]
        pp = pprint.PrettyPrinter(indent=4)
        if details in ("minimal", "medium"):
            if details == "minimal":
                keys_to_keep = set(["answer", "context"])
            elif details == "medium":
                keys_to_keep = set(["answer", "context", "score"])

            # filter the results
            filtered_answers = []
            for ans in answers:
                filtered_answers.append({k: getattr(ans, k) for k in keys_to_keep})
            pp.pprint(filtered_answers)
        else:
            pp.pprint(results)
    # This fall back case is when the answers come from a Generator
    except:
        if details == "minimal":
            print(f"Query: {results['query']}")
            for a in results["answers"]:
                print(f"Answer: {a['answer']}")
        else:
            pp.pprint(results)


def print_documents(results: dict, max_text_len: Optional[int] = None, print_meta: bool = False):
    print(f"Query: {results['query']}")
    pp = pprint.PrettyPrinter(indent=4)
    for d in results["documents"]:
        print()
        new_text = d.content[:max_text_len]
        if len(new_text) != len(d.content):
            new_text += "..."
        results = {
            "name": d.meta.get("name", None),
            "content": new_text
        }
        if print_meta:
            results["meta"] = d.meta
        pp.pprint(results)


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
