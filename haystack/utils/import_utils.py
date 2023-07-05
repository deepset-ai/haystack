import io
import gzip
import tarfile
import zipfile
import logging
import importlib
import importlib.util
from pathlib import Path
from typing import Optional, Dict, Union, Tuple, List
from urllib.parse import urlparse, unquote
from os.path import splitext, basename

import requests

from haystack.errors import DatasetsError
from haystack.schema import Document


logger = logging.getLogger(__name__)


def load_documents_from_hf_datasets(dataset_name: str, split: Optional[str] = "train") -> List[Document]:
    """
    Load a list of Haystack Documents from a remote Hugging Face dataset.

    :param dataset_name: A Hugging Face dataset containing Haystack Documents
    :param split: The split of the Hugging Face dataset to load from. By default, this is set to "train".
    :return: a List of Haystack Documents
    """
    try:
        from datasets import load_dataset, load_dataset_builder
    except ImportError:
        raise ImportError(
            "Failed to import `datasets`, Run 'pip install datasets>=2.6.0' "
            "to install the datasets library to use this function."
        )

    dataset = load_dataset_builder(dataset_name)
    if "content" not in dataset.info.features.keys():
        raise DatasetsError(
            f"{dataset_name} does not contain a `content` field which is required by Haystack to "
            f"create `Document` objects."
        )

    remote_dataset = load_dataset(dataset_name, split=split)
    documents = [Document.from_dict(document) for document in remote_dataset]

    return documents


def get_filename_extension_from_url(url: str) -> Tuple[str, str]:
    """
    Extracts the filename and file extension from an url.

    :param url: http address
    :return: Tuple (filename, file extension) of the file at the url.
    """
    parsed = urlparse(url)
    root, extension = splitext(parsed.path)
    archive_extension = extension[1:]
    file_name = unquote(basename(root[1:]))
    return file_name, archive_extension


def fetch_archive_from_http(
    url: str,
    output_dir: str,
    proxies: Optional[Dict[str, str]] = None,
    timeout: Union[float, Tuple[float, float]] = 10.0,
) -> bool:
    """
    Fetch an archive (zip, gz or tar.gz) from a url via http and extract content to an output directory.

    :param url: http address
    :param output_dir: local path
    :param proxies: proxies details as required by requests library
    :param timeout: How many seconds to wait for the server to send data before giving up,
        as a float, or a :ref:`(connect timeout, read timeout) <timeouts>` tuple.
        Defaults to 10 seconds.
    :return: if anything got fetched
    """
    # verify & prepare local directory
    path = Path(output_dir)
    if not path.exists():
        path.mkdir(parents=True)

    is_not_empty = len(list(Path(path).rglob("*"))) > 0
    if is_not_empty:
        logger.info("Found data stored in '%s'. Delete this first if you really want to fetch new data.", output_dir)
        return False
    else:
        logger.info("Fetching from %s to '%s'", url, output_dir)

        file_name, archive_extension = get_filename_extension_from_url(url)
        request_data = requests.get(url, proxies=proxies, timeout=timeout)

        if archive_extension == "zip":
            zip_archive = zipfile.ZipFile(io.BytesIO(request_data.content))
            zip_archive.extractall(output_dir)
        elif archive_extension == "gz" and not "tar.gz" in url:
            gzip_archive = gzip.GzipFile(fileobj=io.BytesIO(request_data.content))
            file_content = gzip_archive.read()
            with open(f"{output_dir}/{file_name}", "wb") as file:
                file.write(file_content)
        elif archive_extension in ["gz", "bz2", "xz"]:
            tar_archive = tarfile.open(fileobj=io.BytesIO(request_data.content), mode="r|*")
            tar_archive.extractall(output_dir)
        else:
            logger.warning(
                "Skipped url %s as file type is not supported here. "
                "See haystack documentation for support of more file types",
                url,
            )

        return True


def is_whisper_available():
    return importlib.util.find_spec("whisper") is not None
