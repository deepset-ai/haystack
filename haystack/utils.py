import json
from collections import defaultdict
from itertools import islice
import logging
import pprint
import pandas as pd
from typing import Dict, Any, List
from haystack.document_store.sql import DocumentORM
import subprocess
import time
import torch


logger = logging.getLogger(__name__)


def launch_es():
    # Start an Elasticsearch server
    # You can start Elasticsearch on your local machine instance using Docker. If Docker is not readily available in
    # your environment (eg., in Colab notebooks), then you can manually download and execute Elasticsearch from source.

    logger.info("Starting Elasticsearch ...")
    status = subprocess.run(
        ['docker run -d -p 9200:9200 -e "discovery.type=single-node" elasticsearch:7.9.2'], shell=True
    )
    if status.returncode:
        logger.warning("Tried to start Elasticsearch through Docker but this failed. "
                       "It is likely that there is already an existing Elasticsearch instance running. ")
    else:
        time.sleep(15)


def launch_milvus():
    # Start a Milvus server
    # You can start Milvus on your local machine instance using Docker. If Docker is not readily available in
    # your environment (eg., in Colab notebooks)
    logger.info("Starting Milvus ...")
    logger.warning("Automatic Milvus config creation not yet implemented. "
                   "If you are starting Milvus using launch_milvus(), "
                   "make sure you have a properly populated milvus/conf folder. "
                   "See (https://milvus.io/docs/v1.0.0/milvus_docker-cpu.md) for more details.")
    status = subprocess.run(
        ['sudo docker run -d --name milvus_cpu_1.0.0 \
          -p 19530:19530 \
          -p 19121:19121 \
          -v /home/$USER/milvus/db:/var/lib/milvus/db \
          -v /home/$USER/milvus/conf:/var/lib/milvus/conf \
          -v /home/$USER/milvus/logs:/var/lib/milvus/logs \
          -v /home/$USER/milvus/wal:/var/lib/milvus/wal \
          milvusdb/milvus:1.0.0-cpu-d030521-1ea92e'
        ],
        shell=True
    )
    if status.returncode:
        logger.warning("Tried to start Milvus through Docker but this failed. "
                       "It is likely that there is already an existing Milvus instance running. ")
    else:
        time.sleep(15)


def print_answers(results: dict, details: str = "all"):
    # TODO: unify the output format of Generator and Reader so that this function doesn't have the try/except
    #  Or implement a class method like PredReader.print() and PredGenerator.print() that handles all this functionality.
    # This default case is when the answers come from a Reader
    try:
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
    # This fall back case is when the answers come from a Generator
    except:
        if details == "minimal":
            print(f"Query: {results['query']}")
            for a in results["answers"]:
                print(f"Answer: {a['answer']}")
        else:
            pp.pprint(results)



def print_documents(results: dict, max_text_len: int=None):
    print(f"Query: {results['query']}")
    pp = pprint.PrettyPrinter(indent=4)
    for d in results["documents"]:
        print()
        new_text = d["text"][:max_text_len]
        if len(new_text) != len(d["text"]):
            new_text += "..."
        results = {
            "name": d["meta"]["name"],
            "text": new_text
        }
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

    with open("labels_in_squad_format.json", "w+", encoding='utf-8') as outfile:
        json.dump(labels_in_squad_format, outfile)


def get_batches_from_generator(iterable, n):
    """
    Batch elements of an iterable into fixed-length chunks or blocks.
    """
    it = iter(iterable)
    x = tuple(islice(it, n))
    while x:
        yield x
        x = tuple(islice(it, n))


def get_device(use_gpu: bool = True) -> str:
    if use_gpu and torch.cuda.is_available():
        return "cuda"
    return "cpu"
