import os
import tarfile
import tempfile

import pandas as pd

from haystack import Label, Document, Answer
from haystack.document_stores import eval_data_from_json
from haystack.utils import launch_es, launch_opensearch, launch_weaviate
from haystack.modeling.data_handler.processor import http_get

import logging
from typing import Dict, Union
from pathlib import Path

logger = logging.getLogger(__name__)


def prepare_environment(pipeline_config: Dict, benchmark_config: Dict):
    """
    Prepare the environment for running a benchmark.
    """
    # Download data if specified in benchmark config
    if "data_url" in benchmark_config:
        download_from_url(url=benchmark_config["data_url"], target_dir="data/")

    n_docs = 0
    if "documents_directory" in benchmark_config:
        documents_dir = Path(benchmark_config["documents_directory"])
        n_docs = len(
            [
                file_path
                for file_path in documents_dir.iterdir()
                if file_path.is_file() and not file_path.name.startswith(".")
            ]
        )

    # Launch DocumentStore Docker container if needed
    for comp in pipeline_config["components"]:
        if comp["type"].endswith("DocumentStore"):
            launch_document_store(comp["type"], n_docs=n_docs)
            break


def launch_document_store(document_store: str, n_docs: int = 0):
    """
    Launch a DocumentStore Docker container.
    """
    java_opts = None if n_docs < 500000 else "-Xms4096m -Xmx4096m"
    if document_store == "ElasticsearchDocumentStore":
        launch_es(sleep=30, delete_existing=True, java_opts=java_opts)
    elif document_store == "OpenSearchDocumentStore":
        launch_opensearch(sleep=30, delete_existing=True, java_opts=java_opts)
    elif document_store == "WeaviateDocumentStore":
        launch_weaviate(sleep=30, delete_existing=True)


def download_from_url(url: str, target_dir: Union[str, Path]):
    """
    Download from a URL to a local file.

    :param url: URL
    :param target_dir: Local directory where the URL content will be saved.
    """

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    url_path = Path(url)
    logger.info("Downloading %s to %s", url_path.name, target_dir)
    with tempfile.NamedTemporaryFile() as temp_file:
        http_get(url=url, temp_file=temp_file)
        temp_file.flush()
        temp_file.seek(0)
        if tarfile.is_tarfile(temp_file.name):
            with tarfile.open(temp_file.name) as tar:
                tar.extractall(target_dir)
        else:
            with open(Path(target_dir) / url_path.name, "wb") as file:
                file.write(temp_file.read())


def load_eval_data(eval_set_file: Path):
    """
    Load evaluation data from a file.
    :param eval_set_file: Path to the evaluation data file.
    """
    if eval_set_file.suffix == ".json":
        _, labels = eval_data_from_json(str(eval_set_file))
        queries = [label.query for label in labels]
    elif eval_set_file.suffix == ".csv":
        eval_data = pd.read_csv(eval_set_file)

        labels = []
        queries = []
        for idx, row in eval_data.iterrows():
            query = row["question"]
            context = row["context"]
            answer = Answer(answer=row["text"]) if "text" in row else None
            label = Label(
                query=query,
                document=Document(context),
                answer=answer,
                is_correct_answer=True,
                is_correct_document=True,
                origin="gold-label",
            )
            labels.append(label)
            queries.append(query)
    else:
        raise ValueError(
            f"Unsupported file format: {eval_set_file.suffix}. Provide a SQuAD-style .json or a .csv file containing "
            f"the columns 'question' and 'context' for Retriever evaluations and additionally 'text' (containing the "
            f"answer string) for Reader evaluations."
        )

    return labels, queries
