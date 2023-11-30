import os
from pathlib import Path
import tarfile
from typing import Dict, Union
import logging
import tempfile
import httpx

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


def download_from_url(url: str, target_dir: Union[str, Path]) -> None:
    """
    Download from a URL to a local file.

    :param url: URL
    :param target_dir: Local directory where the URL content will be saved.
    """
    url_path = Path(url)

    if file_previously_downloaded(url_path, target_dir):
        logger.info(f"Skipping download of {url}, as a previous copy exists")
        return

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    logger.info("Downloading %s to %s", url_path.name, target_dir)
    with tempfile.NamedTemporaryFile() as temp_file:
        httpx.get(url=url, temp_file=temp_file)
        temp_file.flush()
        temp_file.seek(0)
        if tarfile.is_tarfile(temp_file.name):
            with tarfile.open(temp_file.name) as tar:
                tar.extractall(target_dir)
        else:
            with open(Path(target_dir) / url_path.name, "wb") as file:
                file.write(temp_file.read())

def file_previously_downloaded(url_path: Path, target_dir: Union[str, Path]) -> bool:
    if ".tar" in url_path.suffixes:
        return Path(target_dir, url_path.parent).exists()
    return Path(target_dir, url_path.name).exists()