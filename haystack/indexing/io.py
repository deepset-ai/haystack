from pathlib import Path
import logging
from farm.data_handler.utils import http_get
import tempfile
import tarfile
import zipfile

logger = logging.getLogger(__name__)


def write_documents_to_db(document_store, document_dir, clean_func=None, only_empty_db=False, split_paragraphs=False):
    """
    Write all text files(.txt) in the sub-directories of the given path to the connected database.

    :param document_dir: path for the documents to be written to the database
    :param clean_func: a custom cleaning function that gets applied to each doc (input: str, output:str)
    :param only_empty_db: If true, docs will only be written if db is completely empty.
                              Useful to avoid indexing the same initial docs again and again.
    :return: None
    """
    file_paths = Path(document_dir).glob("**/*.txt")

    # check if db has already docs
    if only_empty_db:
        n_docs = document_store.get_document_count()
        if n_docs > 0:
            logger.info(f"Skip writing documents since DB already contains {n_docs} docs ...  "
                        "(Disable `only_empty_db`, if you want to add docs anyway.)")
            return None

    # read and add docs
    docs_to_index = []
    doc_id = 1
    for path in file_paths:
        with open(path) as doc:
            text = doc.read()
            if clean_func:
                text = clean_func(text)

            if split_paragraphs:
                for para in text.split("\n\n"):
                    if not para.strip():  # skip empty paragraphs
                        continue
                    docs_to_index.append(
                        {
                            "name": path.name,
                            "text": para
                        }
                    )
                    doc_id += 1
            else:
                docs_to_index.append(
                    {
                        "name": path.name,
                        "text": text
                    }
                )
    document_store.write_documents(docs_to_index)
    logger.info(f"Wrote {len(docs_to_index)} docs to DB")


def fetch_archive_from_http(url, output_dir, proxies=None):
    """
    Fetch an archive (zip or tar.gz) from a url via http and extract content to an output directory.

    :param url: http address
    :type url: str
    :param output_dir: local path
    :type output_dir: str
    :param proxies: proxies details as required by requests library
    :type proxies: dict
    :return: bool if anything got fetched
    """
    # verify & prepare local directory
    path = Path(output_dir)
    if not path.exists():
        path.mkdir(parents=True)

    is_not_empty = len(list(Path(path).rglob("*"))) > 0
    if is_not_empty:
        logger.info(
            f"Found data stored in `{output_dir}`. Delete this first if you really want to fetch new data."
        )
        return False
    else:
        logger.info(f"Fetching from {url} to `{output_dir}`")

        # download & extract
        with tempfile.NamedTemporaryFile() as temp_file:
            http_get(url, temp_file, proxies=proxies)
            temp_file.flush()
            temp_file.seek(0)  # making tempfile accessible
            # extract
            if url[-4:] == ".zip":
                archive = zipfile.ZipFile(temp_file.name)
                archive.extractall(output_dir)
            elif url[-7:] == ".tar.gz":
                archive = tarfile.open(temp_file.name)
                archive.extractall(output_dir)
            # temp_file gets deleted here
        return True
