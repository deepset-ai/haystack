import json
from collections import defaultdict
import logging
import pprint

from haystack.database.sql import Document

logger = logging.getLogger(__name__)


def print_answers(results, details="all"):
    answers = results["answers"]
    pp = pprint.PrettyPrinter(indent=4)
    if details != "all":
        if details == "minimal":
            keys_to_keep = set(["answer", "context"])
        elif details == "medium":
            keys_to_keep = set(["answer", "context", "score"])
        # filter the results
        keys_to_drop = set(answers[0].keys()) - keys_to_keep
        for a in answers:
            for key in keys_to_drop:
                if key in a:
                    del a[key]

        pp.pprint(answers)
    else:
        pp.pprint(results)


def convert_labels_to_squad(labels_file):
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

    labels_in_squad_format = {"data": []}
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
