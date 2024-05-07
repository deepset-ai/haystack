import logging
import importlib
import importlib.util
from typing import Optional, Tuple, List
from urllib.parse import urlparse, unquote
from os.path import splitext, basename

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
            "Failed to import `datasets`, Run 'pip install \"datasets>=2.6.0\"' "
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


def is_whisper_available():
    return importlib.util.find_spec("whisper") is not None
