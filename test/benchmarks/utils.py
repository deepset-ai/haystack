import os
from pathlib import Path
import tarfile
from typing import Dict, Union
import logging
import tempfile
import httpx

logger = logging.getLogger(__name__)

def get_docs(dataset: str):
    """
    Prepare the environment for running a benchmark.
    """
    # Download data if specified in benchmark config
    _download(dataset=dataset, target_dir="data/")
    n_docs = 0

    documents_dir = Path(documents_dir)



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


get_docs("https://deepset-test-datasets.s3.eu-central-1.amazonaws.com/msmarco.1000.tar.bz2", "/Users/rohan/repos/deepset/haystack/test/benchmarks/msmarco1000")