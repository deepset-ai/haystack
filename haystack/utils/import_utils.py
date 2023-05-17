import io
import gzip
import tarfile
import zipfile
import logging
import importlib
import importlib.util
from pathlib import Path
from typing import Optional, Dict, Union, Tuple, List

import requests

from haystack.errors import DatasetsError
from haystack.schema import Document


logger = logging.getLogger(__name__)


def safe_import(import_path: str, classname: str, dep_group: str):
    """
    Method that allows the import of nodes that depend on missing dependencies.
    These nodes can be installed one by one with project.optional-dependencies
    (see pyproject.toml) but they need to be all imported in their respective
    package's __init__()

    Therefore, in case of an ImportError, the class to import is replaced by
    a hollow MissingDependency function, which will throw an error when
    inizialized.
    """
    try:
        module = importlib.import_module(import_path)
        classs = vars(module).get(classname)
        if classs is None:
            raise ImportError(f"Failed to import '{classname}' from '{import_path}'")
    except ImportError as ie:
        classs = _missing_dependency_stub_factory(classname, dep_group, ie)
    return classs


def _missing_dependency_stub_factory(classname: str, dep_group: str, import_error: Exception):
    """
    Create custom versions of MissingDependency using the given parameters.
    See `safe_import()`
    """

    class MissingDependency:
        def __init__(self, *args, **kwargs):
            _optional_component_not_installed(classname, dep_group, import_error)

        def __getattr__(self, *a, **k):
            return None

    return MissingDependency


def _optional_component_not_installed(component: str, dep_group: str, source_error: Exception):
    raise ImportError(
        f"Failed to import '{component}', "
        "which is an optional component in Haystack.\n"
        f"Run 'pip install 'farm-haystack[{dep_group}]'' "
        "to install the required dependencies and make this component available.\n"
        f"(Original error: {str(source_error)})"
    ) from source_error


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

        _, _, archive_extension = url.rpartition(".")
        request_data = requests.get(url, proxies=proxies, timeout=timeout)

        if archive_extension == "zip":
            zip_archive = zipfile.ZipFile(io.BytesIO(request_data.content))
            zip_archive.extractall(output_dir)
        elif archive_extension == "gz" and not "tar.gz" in url:
            gzip_archive = gzip.GzipFile(fileobj=io.BytesIO(request_data.content))
            file_content = gzip_archive.read()
            file_name = url.split("/")[-1][: -(len(archive_extension) + 1)]
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
