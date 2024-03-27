import os
from pathlib import Path
import tarfile
from typing import List, Union
import logging
import tempfile
import httpx
import csv

logger = logging.getLogger(__name__)


def get_docs(dataset: str) -> List[str]:
    """
    Prepare the environment for running a benchmark.
    """
    # Download data if specified in benchmark config
    _download(dataset=dataset, target_dir="data/")

    pathlist = Path(f"data/{dataset}/txt").rglob("*.txt")
    return [str(path) for path in pathlist]


def get_queries(dataset: str, evalset: str = "msmarco_evalset_25") -> List[str]:
    # Download data if specified in benchmark config
    _download(dataset=dataset, target_dir="data/")

    csv_path = Path(f"data/{dataset}/evalsets/{evalset}.csv")

    with open(csv_path) as csvfile:
        csvreader = csv.reader(csvfile)
        queries = [row[0] for row in csvreader]
    return queries[1:]


def _download(dataset: str, target_dir: Union[str, Path]) -> None:
    """
    Download from a URL to a local file.

    :param url: URL
    :param target_dir: Local directory where the URL content will be saved.
    """
    url = f"https://deepset-test-datasets.s3.eu-central-1.amazonaws.com/{dataset}.tar.bz2"
    url_path = Path(url)
    if _file_previously_downloaded(url_path, target_dir):
        logger.info(f"Skipping download of {dataset}-dataset, as a previous copy exists")
        return
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    logger.info("Downloading %s to %s", url_path.name, target_dir)
    with tempfile.NamedTemporaryFile() as temp_file:
        response = httpx.get(url=url)
        temp_file.write(response.content)
        temp_file.flush()
        temp_file.seek(0)
        if tarfile.is_tarfile(temp_file.name):
            with tarfile.open(temp_file.name) as tar:
                tar.extractall(target_dir)
        else:
            with open(Path(target_dir) / url_path.name, "wb") as file:
                file.write(temp_file.read())


def _file_previously_downloaded(url_path: Path, target_dir: Union[str, Path]) -> bool:
    if ".tar" in url_path.suffixes:
        return Path(target_dir, url_path.parent).exists()
    return Path(target_dir, url_path.name).exists()


# docs = get_docs("msmarco.1000")

# print(get_queries("msmarco.1000")[:100])
