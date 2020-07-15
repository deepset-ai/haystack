import json
from collections import defaultdict
import logging
import pprint
from typing import Dict, Any
from haystack.database.sql import Document

logger = logging.getLogger(__name__)


def print_answers(results: dict, details: str = "all"):
    answers = results["answers"]
    pp = pprint.PrettyPrinter(indent=4)
    if details != "all":
        if details == "minimal":
            keys_to_keep = set(["answer", "context"])
        elif details == "medium":
            keys_to_keep = set(["answer", "context", "score"])
        else:
            keys_to_keep = answers.keys()

        # filter the results
        filtered_answers = []
        for ans in answers:
            filtered_answers.append({k: ans[k] for k in keys_to_keep})
        pp.pprint(filtered_answers)
    else:
        pp.pprint(results)


def convert_labels_to_squad(labels_file: str):
    """
    Convert the export from the labeling UI to SQuAD format for training.

    :param labels_file: path for export file from the labeling tool
    :return:
    """
    with open(labels_file) as label_file:
        labels = json.load(label_file)

    labels_grouped_by_documents = defaultdict(list)
    for label in labels:
        labels_grouped_by_documents[label["document_id"]].append(label)

    labels_in_squad_format = {"data": []}  # type: Dict[str, Any]
    for document_id, labels in labels_grouped_by_documents.items():
        qas = []
        for label in labels:
            doc = Document.query.get(label["document_id"])

            assert (
                doc.text[label["start_offset"] : label["end_offset"]]
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
                {"qas": qas, "context": doc.text, "document_id": document_id}
            ]
        }

        labels_in_squad_format["data"].append(squad_format_label)

    with open("labels_in_squad_format.json", "w+") as outfile:
        json.dump(labels_in_squad_format, outfile)
