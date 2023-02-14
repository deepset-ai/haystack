"""
Script to convert a SQuAD-like QA-dataset format JSON file to DPR Dense Retriever training format

Usage:
    squad_to_dpr.py --squad_input_filename <squad_input_filename> --dpr_output_filename <dpr_output_filename> [options]
Arguments:
    <squad_file_path>                   SQuAD file path
    <dpr_output_path>                   DPR output folder path
    --num_hard_negative_ctxs HNEG       Number of hard negative contexts [default: 30:int]
    --split_dataset                     Whether to split the created dataset or not [default: False]

SQuAD format
{
    version: "Version du dataset"
    data:[
            {
                title: "Titre de l'article Wikipedia"
                paragraphs:[
                    {
                        context: "Paragraph de l'article"
                        qas:[
                            {
                                id: "Id du pair question-réponse"
                                question: "Question"
                                answers:[
                                    {
                                        "answer_start": "Position de la réponse"
                                        "text": "Réponse"
                                    }
                                ],
                                is_impossible: (not in v1)
                            }
                        ]
                    }
                ]
            }
    ]
}


DPR format
[
    {
        "question": "....",
        "answers": ["...", "...", "..."],
        "positive_ctxs": [{
            "title": "...",
            "text": "...."
        }],
        "negative_ctxs": ["..."],
        "hard_negative_ctxs": ["..."]
    },
    ...
]
"""

from typing import Dict, Iterator, Tuple, List, Union

import json
import logging
import argparse
import subprocess
from time import sleep
from pathlib import Path
from itertools import islice

from tqdm.auto import tqdm
from elasticsearch import Elasticsearch

from haystack.document_stores.base import BaseDocumentStore
from haystack.document_stores.elasticsearch import ElasticsearchDocumentStore  # keep it here !
from haystack.document_stores.faiss import FAISSDocumentStore  # keep it here !
from haystack.nodes.retriever.sparse import BM25Retriever  # keep it here !  # pylint: disable=unused-import
from haystack.nodes.retriever.dense import DensePassageRetriever  # keep it here !  # pylint: disable=unused-import
from haystack.nodes.preprocessor import PreProcessor
from haystack.nodes.retriever.base import BaseRetriever


logger = logging.getLogger(__name__)


class HaystackDocumentStore:
    def __init__(self, store_type: str = "ElasticsearchDocumentStore", **kwargs):
        if store_type not in ["ElasticsearchDocumentStore", "FAISSDocumentStore"]:
            raise Exception(
                "At the moment we only deal with one of these types: ElasticsearchDocumentStore, FAISSDocumentStore"
            )

        self._store_type = store_type
        self._kwargs = kwargs
        self._preparation = {
            "ElasticsearchDocumentStore": self.__prepare_ElasticsearchDocumentStore,
            "FAISSDocumentStore": self.__prepare_FAISSDocumentStore,
        }

    def get_document_store(self):
        self._preparation[self._store_type]()
        return globals()[self._store_type](**self._kwargs)

    @staticmethod
    def __prepare_ElasticsearchDocumentStore():
        es = Elasticsearch(["http://localhost:9200/"], verify_certs=True)
        if not es.ping():
            logger.info("Starting Elasticsearch ...")
            status = subprocess.run(
                ['docker run -d -p 9200:9200 -e "discovery.type=single-node" elasticsearch:7.9.2'], shell=True
            )
            if status.returncode:
                raise Exception("Failed to launch Elasticsearch.")
            sleep(30)

        es.indices.delete(index="document", ignore=[400, 404])

    def __prepare_FAISSDocumentStore(self):
        pass


class HaystackRetriever:
    def __init__(self, document_store: BaseDocumentStore, retriever_type: str, **kwargs):
        if retriever_type not in ["BM25Retriever", "DensePassageRetriever", "EmbeddingRetriever"]:
            raise Exception("Use one of these types: BM25Retriever", "DensePassageRetriever", "EmbeddingRetriever")
        self._retriever_type = retriever_type
        self._document_store = document_store
        self._kwargs = kwargs

    def get_retriever(self):
        return globals()[self._retriever_type](document_store=self._document_store, **self._kwargs)


def add_is_impossible(squad_data: dict, json_file_path: Path):
    new_path = json_file_path.parent / Path(f"{json_file_path.stem}_impossible.json")
    squad_articles = list(squad_data["data"])  # create new list with this list although lists are inmutable :/
    for article in squad_articles:
        for paragraph in article["paragraphs"]:
            for question in paragraph["qas"]:
                question["is_impossible"] = False

    squad_data["data"] = squad_articles
    with open(new_path, "w", encoding="utf-8") as filo:
        json.dump(squad_data, filo, indent=4, ensure_ascii=False)

    return new_path, squad_data


def get_number_of_questions(squad_data: dict):
    nb_questions = 0
    for article in squad_data:
        for paragraph in article["paragraphs"]:
            nb_questions += len(paragraph["qas"])
    return nb_questions


def has_is_impossible(squad_data: dict):
    for article in squad_data["data"]:
        for paragraph in article["paragraphs"]:
            for question in paragraph["qas"]:
                if "is_impossible" in question:
                    return True
    return False


def create_dpr_training_dataset(squad_data: dict, retriever: BaseRetriever, num_hard_negative_ctxs: int = 30):
    n_non_added_questions = 0
    n_questions = 0
    for article in tqdm(squad_data, unit="article"):
        article_title = article.get("title", "")
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]
            for question in paragraph["qas"]:
                if "is_impossible" in question and question["is_impossible"]:
                    continue
                answers = [a["text"] for a in question["answers"]]
                hard_negative_ctxs = get_hard_negative_contexts(
                    retriever=retriever, question=question["question"], answers=answers, n_ctxs=num_hard_negative_ctxs
                )
                positive_ctxs = [{"title": article_title, "text": context, "passage_id": ""}]

                if not hard_negative_ctxs or not positive_ctxs:
                    logging.error(
                        "No retrieved candidates for article %s, with question %s", article_title, question["question"]
                    )
                    n_non_added_questions += 1
                    continue
                dict_DPR = {
                    "question": question["question"],
                    "answers": answers,
                    "positive_ctxs": positive_ctxs,
                    "negative_ctxs": [],
                    "hard_negative_ctxs": hard_negative_ctxs,
                }
                n_questions += 1
                yield dict_DPR

    logger.info("Number of skipped questions: %s", n_non_added_questions)
    logger.info("Number of added questions: %s", n_questions)


def save_dataset(iter_dpr: Iterator, dpr_output_filename: Path, total_nb_questions: int, split_dataset: bool):
    if split_dataset:
        nb_train_examples = int(total_nb_questions * 0.8)
        nb_dev_examples = int(total_nb_questions * 0.1)

        train_iter = islice(iter_dpr, nb_train_examples)
        dev_iter = islice(iter_dpr, nb_dev_examples)

        dataset_splits = {
            dpr_output_filename.parent / f"{dpr_output_filename.stem}.train.json": train_iter,
            dpr_output_filename.parent / f"{dpr_output_filename.stem}.dev.json": dev_iter,
            dpr_output_filename.parent / f"{dpr_output_filename.stem}.test.json": iter_dpr,
        }
    else:
        dataset_splits = {dpr_output_filename: iter_dpr}
    for path, set_iter in dataset_splits.items():
        with open(path, "w", encoding="utf-8") as json_ds:
            json.dump(list(set_iter), json_ds, indent=4, ensure_ascii=False)


def get_hard_negative_contexts(retriever: BaseRetriever, question: str, answers: List[str], n_ctxs: int = 30):
    list_hard_neg_ctxs = []
    retrieved_docs = retriever.retrieve(query=question, top_k=n_ctxs, index="document")
    for retrieved_doc in retrieved_docs:
        retrieved_doc_id = retrieved_doc.meta.get("name", "")
        retrieved_doc_text = retrieved_doc.content
        if any(answer.lower() in retrieved_doc_text.lower() for answer in answers):
            continue
        list_hard_neg_ctxs.append({"title": retrieved_doc_id, "text": retrieved_doc_text, "passage_id": ""})

    return list_hard_neg_ctxs


def load_squad_file(squad_file_path: Path):
    if not squad_file_path.exists():
        raise FileNotFoundError

    with open(squad_file_path, encoding="utf-8") as squad_file:
        squad_data = json.load(squad_file)

    # squad_data["data"] = squad_data["data"][:10]  # sample

    # check it has the is_impossible field
    if not has_is_impossible(squad_data=squad_data):
        squad_file_path, squad_data = add_is_impossible(squad_data, squad_file_path)

    return squad_file_path, squad_data["data"]


def main(
    squad_input_filename: Path,
    dpr_output_filename: Path,
    preprocessor,
    document_store_type_config: Tuple[str, Dict] = ("ElasticsearchDocumentStore", {}),
    retriever_type_config: Tuple[str, Dict] = ("BM25Retriever", {}),
    num_hard_negative_ctxs: int = 30,
    split_dataset: bool = False,
):
    tqdm.write(f"Using SQuAD-like file {squad_input_filename}")

    # 1. Load squad file data
    squad_file_path, squad_data = load_squad_file(squad_file_path=squad_input_filename)

    # 2. Prepare document store
    store_factory = HaystackDocumentStore(store_type=document_store_type_config[0], **document_store_type_config[1])
    document_store: Union[ElasticsearchDocumentStore, FAISSDocumentStore] = store_factory.get_document_store()

    # 3. Load data into the document store
    document_store.add_eval_data(squad_file_path.as_posix(), doc_index="document", preprocessor=preprocessor)

    # 4. Prepare retriever
    retriever_factory = HaystackRetriever(
        document_store=document_store, retriever_type=retriever_type_config[0], **retriever_type_config[1]
    )
    retriever = retriever_factory.get_retriever()

    # 5. Get embeddings if needed
    if retriever_type_config[0] in ["DensePassageRetriever", "EmbeddingRetriever"]:
        document_store.update_embeddings(retriever)

    # 6. Find positive and negative contexts and create new dataset
    iter_DPR = create_dpr_training_dataset(
        squad_data=squad_data, retriever=retriever, num_hard_negative_ctxs=num_hard_negative_ctxs
    )

    # 7. Split (maybe) and save dataset
    total_nb_questions = get_number_of_questions(squad_data)
    save_dataset(
        iter_dpr=iter_DPR,
        dpr_output_filename=dpr_output_filename,
        total_nb_questions=total_nb_questions,
        split_dataset=split_dataset,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a SQuAD JSON format dataset to DPR format.")
    parser.add_argument(
        "--squad_input_filename",
        dest="squad_input_filename",
        help="A dataset with a SQuAD JSON format.",
        metavar="SQUAD_in",
        required=True,
    )
    parser.add_argument(
        "--dpr_output_filename",
        dest="dpr_output_filename",
        help="The name of the DPR JSON formatted output file",
        metavar="DPR_out",
        required=True,
    )
    parser.add_argument(
        "--num_hard_negative_ctxs",
        dest="num_hard_negative_ctxs",
        help="Number of hard negative contexts to use",
        metavar="num_hard_negative_ctxs",
        default=30,
    )
    parser.add_argument(
        "--split_dataset",
        dest="split_dataset",
        action="store_true",
        help="Whether to split the created dataset or not (default: False)",
    )

    args = parser.parse_args()

    preprocessor = PreProcessor(
        split_length=100,
        split_overlap=0,
        clean_empty_lines=False,
        split_respect_sentence_boundary=False,
        clean_whitespace=False,
    )
    squad_input_filename = Path(args.squad_input_filename)
    dpr_output_filename = Path(args.dpr_output_filename)
    num_hard_negative_ctxs = args.num_hard_negative_ctxs
    split_dataset = args.split_dataset

    retriever_dpr_config = {"use_gpu": True}
    store_dpr_config = {"embedding_field": "embedding", "embedding_dim": 768}

    retriever_bm25_config: dict = {}

    main(
        squad_input_filename=squad_input_filename,
        dpr_output_filename=dpr_output_filename,
        preprocessor=preprocessor,
        document_store_type_config=("ElasticsearchDocumentStore", store_dpr_config),
        # retriever_type_config=("DensePassageRetriever", retriever_dpr_config),  # dpr
        retriever_type_config=("BM25Retriever", retriever_bm25_config),  # bm25
        num_hard_negative_ctxs=num_hard_negative_ctxs,
        split_dataset=split_dataset,
    )
