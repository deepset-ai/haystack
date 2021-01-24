"""
Script to convert a SQuAD-like QA-dataset format JSON file to DPR Dense Retriever training format

Usage:
    squad_to_dpr.py <squad_file_path> <dpr_output_path> [options]

Arguments:
    <squad_file_path>                   SQuAD file path
    <dpr_output_path>                   DPR output folder path
    --num_hard_negative_ctxs HNEG       Number of hard negative contexts [default: 30:int]
"""
import argparse
import logging
import subprocess
from itertools import islice
from pathlib import Path
from time import sleep
from typing import Dict, Iterator, Tuple

from elasticsearch import Elasticsearch
from haystack.document_store.base import BaseDocumentStore
from haystack.document_store.elasticsearch import ElasticsearchDocumentStore  # keep it here !
from haystack.document_store.faiss import FAISSDocumentStore  # keep it here !
from haystack.retriever.sparse import ElasticsearchRetriever  # keep it here !
from haystack.retriever.dense import DensePassageRetriever  # keep it here !
from haystack.preprocessor.preprocessor import PreProcessor
from haystack.retriever.base import BaseRetriever
from tqdm import tqdm
import json
import random

random.seed(42)

"""
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


class IteratorAsList(list):
    def __init__(self, it):
        self.it = it

    def __iter__(self):
        return self.it

    def __len__(self):
        return 1


class HaystackDocumentStore:
    def __init__(self,
                 store_type: str = "ElasticsearchDocumentStore",
                 **kwargs):
        if store_type not in ["ElasticsearchDocumentStore", "FAISSDocumentStore"]:
            raise Exception("At the moment we only deal with one of these types:"
                            "ElasticsearchDocumentStore",
                            "FAISSDocumentStore")

        self._store_type = store_type
        self._kwargs = kwargs
        self._preparation = {"ElasticsearchDocumentStore": self.__prepare_ElasticsearchDocumentStore,
                             "FAISSDocumentStore": self.__prepare_FAISSDocumentStore}

    def get_document_store(self):
        self._preparation[self._store_type]()
        return globals()[self._store_type](**self._kwargs)

    def __prepare_ElasticsearchDocumentStore(self):
        es = Elasticsearch(['http://localhost:9200/'], verify_certs=True)
        if not es.ping():
            logging.info("Starting Elasticsearch ...")
            status = subprocess.run(
                ['docker run -d -p 9200:9200 -e "discovery.type=single-node" elasticsearch:7.6.2'], shell=True
            )
            if status.returncode:
                raise Exception(
                    "Failed to launch Elasticsearch. If you want to connect to an existing Elasticsearch instance"
                    "then set LAUNCH_ELASTICSEARCH in the script to False.")
            sleep(30)

        es.indices.delete(index='document', ignore=[400, 404])

    def __prepare_FAISSDocumentStore(self):
        pass


class HaystackRetriever:
    def __init__(self,
                 document_store: BaseDocumentStore,
                 retriever_type: str,
                 **kwargs
                 ):
        if retriever_type not in ["ElasticsearchRetriever", "DensePassageRetriever", "EmbeddingRetriever"]:
            raise Exception("Use one of these types: ElasticsearchRetriever",
                            "DensePassageRetriever", "EmbeddingRetriever")
        self._retriever_type = retriever_type
        self._document_store = document_store
        self._kwargs = kwargs

    def get_retriever(self):
        return globals()[self._retriever_type](document_store=self._document_store, **self._kwargs)


def add_is_impossible(squad_data: dict, json_file_path: Path):
    new_path = json_file_path.parent / Path(f"{json_file_path.stem}_impossible.json")
    squad_articles = list(squad_data["data"])
    for article in squad_articles:
        for para_idx, paragraph in enumerate(article["paragraphs"]):
            for question in paragraph["qas"]:
                question["is_impossible"] = False

    squad_data["data"] = squad_articles
    with open(new_path, "w") as filo:
        json.dump(squad_data, filo, indent=4)

    return new_path, squad_data["data"]


def get_number_of_questions(squad_data: dict):
    nb_questions = 0
    for article in squad_data:
        for paragraph in article["paragraphs"]:
            nb_questions += len(paragraph["qas"])
    return nb_questions


def create_dpr_training_dataset(squad_data: dict, retriever: BaseRetriever,
                                num_hard_negative_ctxs: int = 30):
    n_non_added_questions = 0
    n_questions = 0
    for idx_article, article in enumerate(tqdm(squad_data, unit="article")):
        article_title = article["title"]
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]
            for question in paragraph["qas"]:
                answers = [a["text"] for a in question["answers"]]
                hard_negative_ctxs = get_hard_negative_context(retriever=retriever,
                                                               question=question["question"],
                                                               answer=answers[0],
                                                               n_ctxs=num_hard_negative_ctxs)
                positive_ctxs = [{
                    "title": f"{article_title}_{i}",
                    "text": c
                } for i, c in enumerate([context for _ in question["answers"]])]

                if not hard_negative_ctxs or not positive_ctxs:
                    logging.error(
                        f"No retrieved candidates for article {article_title}, with question {question['question']}")
                    n_non_added_questions += 1
                    continue
                dict_DPR = {
                    "question": question["question"],
                    "answers": answers,
                    "positive_ctxs": positive_ctxs,
                    "negative_ctxs": [],
                    "hard_negative_ctxs": hard_negative_ctxs
                }
                n_questions += 1
                yield dict_DPR

    print(f"Number of not added questions : {n_non_added_questions} / {n_questions}")


def split_and_save_dataset(iter_dpr: Iterator, dpr_output_path: Path,
                           total_nb_questions: int):
    nb_train_examples = int(total_nb_questions * 0.8)
    nb_dev_examples = int(total_nb_questions * 0.1)

    train_iter = islice(iter_dpr, nb_train_examples)
    dev_iter = islice(iter_dpr, nb_dev_examples)

    dataset_splits = {
        dpr_output_path / Path("DPR_train.json"): train_iter,
        dpr_output_path / Path("DPR_dev.json"): dev_iter,
        dpr_output_path / Path("DPR_test.json"): iter_dpr,
    }

    for path, set_iter in dataset_splits.items():
        with open(path, "w") as json_ds:
            json.dump(IteratorAsList(set_iter), json_ds, indent=4)


def get_hard_negative_context(retriever: BaseRetriever, question: str, answer: str,
                              n_ctxs: int = 30):
    list_hard_neg_ctxs = []
    retrieved_docs = retriever.retrieve(query=question, top_k=n_ctxs, index="document")
    for retrieved_doc in retrieved_docs:
        retrieved_doc_id = retrieved_doc.meta["name"]
        retrieved_doc_text = retrieved_doc.text
        if answer.lower() in retrieved_doc_text.lower():
            continue
        list_hard_neg_ctxs.append({"title": retrieved_doc_id, "text": retrieved_doc_text})

    return list_hard_neg_ctxs


def load_squad_file(squad_file_path: Path):
    squad_data = json.load(open(squad_file_path.as_posix()))
    return squad_data


def main(squad_file_path: Path, dpr_output_path: Path,
         document_store_type_config: Tuple[str, Dict] = ("ElasticsearchDocumentStore", {}),
         retriever_type_config: Tuple[str, Dict] = ("ElasticsearchRetriever", {}),
         num_hard_negative_ctxs: int = 30):
    tqdm.write(f"Using SQuAD-like file {squad_file_path}")

    # 1. Load squad file data
    squad_impossible_path, squad_data = add_is_impossible(load_squad_file(squad_file_path=squad_file_path),
                                                          json_file_path=squad_file_path)

    # 2. Prepare document store
    store_factory = HaystackDocumentStore(store_type=document_store_type_config[0],
                                          **document_store_type_config[1])
    document_store: BaseDocumentStore = store_factory.get_document_store()

    # 3. Load data into the document store
    document_store.add_eval_data(squad_impossible_path.as_posix(), doc_index="document",
                                 preprocessor=preprocessor)

    # 4. Prepare retriever
    retriever_factory = HaystackRetriever(document_store=document_store,
                                          retriever_type=retriever_type_config[0],
                                          **retriever_type_config[1])
    retriever = retriever_factory.get_retriever()

    # 5. Get embeddings if needed
    if retriever_type_config[0] in ["DensePassageRetriever", "EmbeddingRetriever"]:
        document_store.update_embeddings(retriever)

    # 6. Find positive and negative contexts and create new dataset
    iter_DPR = create_dpr_training_dataset(squad_data=squad_data,
                                           retriever=retriever,
                                           num_hard_negative_ctxs=num_hard_negative_ctxs)

    # 7. Split (train, dev, test) and save dataset
    total_nb_questions = get_number_of_questions(squad_data)
    split_and_save_dataset(iter_dpr=iter_DPR,
                           dpr_output_path=dpr_output_path,
                           total_nb_questions=total_nb_questions)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Convert a SQuAD JSON format dataset to DPR format.")
    parser.add_argument("--squad_file_path", dest="squad_file_path",
                        help="A dataset with a SQuAD JSON format.",
                        metavar="SQUAD_in", required=True)
    parser.add_argument("--dpr_output_path", dest="dpr_output_path",
                        help="The name of the DPR JSON formatted output file",
                        metavar="DPR_out", required=True)
    parser.add_argument("--num_hard_negative_ctxs", dest="nb_hard_neg_ctxs",
                        help="Number of hard negative contexts to use",
                        metavar="num_hard_negative_ctxs",
                        default=30)
    args = parser.parse_args()

    preprocessor = PreProcessor(split_length=100, split_overlap=0, clean_empty_lines=False,
                                clean_whitespace=False)
    squad_file_path = Path(args.squad_file_path)
    dpr_output_path = Path(args.dpr_output_path)
    num_hard_negative_ctxs = args.num_hard_negative_ctxs

    retriever_dpr_config = {
        "use_gpu": True,
    }
    store_dpr_config = {
        "embedding_field": "embedding",
        "embedding_dim": 768,
    }

    retriever_bm25_config = {}

    main(squad_file_path=squad_file_path, dpr_output_path=dpr_output_path,
         document_store_type_config=("ElasticsearchDocumentStore", store_dpr_config),
         # retriever_type_config=("ElasticsearchRetriever", retriever_bm25_config), # dpr
         retriever_type_config=("ElasticsearchRetriever", retriever_bm25_config),  # bm25
         num_hard_negative_ctxs=num_hard_negative_ctxs)
