import logging
import tarfile
import tempfile
import zipfile
from pathlib import Path
from typing import Callable, List, Optional, Tuple
import json

from farm.data_handler.utils import http_get

from haystack.indexing.file_converters.pdf import PDFToTextConverter
from haystack.database.base import Document, Label

logger = logging.getLogger(__name__)


def eval_data_from_file(filename: str) -> Tuple[List[Document], List[Label]]:
    """
    Read Documents + Labels from a SQuAD-style file.
    Document and Labels can then be indexed to the DocumentStore and be used for evaluation.

    :param filename: Path to file in SQuAD format
    :return: (List of Documents, List of Labels)
    """
    docs = []
    labels = []

    with open(filename, "r") as file:
        data = json.load(file)
        for document in data["data"]:
            # get all extra fields from document level (e.g. title)
            meta_doc = {k: v for k, v in document.items() if k not in ("paragraphs", "title")}
            for paragraph in document["paragraphs"]:
                cur_meta = {"name": document["title"]}
                # all other fields from paragraph level
                meta_paragraph = {k: v for k, v in paragraph.items() if k not in ("qas", "context")}
                cur_meta.update(meta_paragraph)
                # meta from parent document
                cur_meta.update(meta_doc)
                # Create Document
                cur_doc = Document(text=paragraph["context"], meta=cur_meta)
                docs.append(cur_doc)

                # Get Labels
                for qa in paragraph["qas"]:
                    for answer in qa["answers"]:
                        label = Label(
                            question=qa["question"],
                            answer=answer["text"],
                            is_correct_answer=True,
                            is_correct_document=True,
                            document_id=cur_doc.id,
                            offset_start_in_doc=answer["answer_start"],
                            no_answer=qa["is_impossible"],
                            origin="gold_label",
                            )
                        labels.append(label)

        return docs, labels


def convert_files_to_dicts(dir_path: str, clean_func: Optional[Callable] = None, split_paragraphs: bool = False) -> List[dict]:
    """
    Convert all files(.txt, .pdf) in the sub-directories of the given path to Python dicts that can be written to a
    Document Store.

    :param dir_path: path for the documents to be written to the database
    :param clean_func: a custom cleaning function that gets applied to each doc (input: str, output:str)
    :param split_paragraphs: split text in paragraphs.

    :return: None
    """

    file_paths = [p for p in Path(dir_path).glob("**/*")]
    if ".pdf" in [p.suffix.lower() for p in file_paths]:
        pdf_converter = PDFToTextConverter()  # type: Optional[PDFToTextConverter]
    else:
        pdf_converter = None

    documents = []
    for path in file_paths:
        if path.suffix.lower() == ".txt":
            with open(path) as doc:
                text = doc.read()
        elif path.suffix.lower() == ".pdf" and pdf_converter:
            pages = pdf_converter.extract_pages(path)
            text = "\n".join(pages)
        else:
            raise Exception(f"Indexing of {path.suffix} files is not currently supported.")

        if clean_func:
            text = clean_func(text)

        if split_paragraphs:
            for para in text.split("\n\n"):
                if not para.strip():  # skip empty paragraphs
                    continue
                documents.append({"text": para, "meta": {"name": path.name}})
        else:
            documents.append({"text": text, "meta": {"name": path.name}})

    return documents


def fetch_archive_from_http(url: str, output_dir: str, proxies: Optional[dict] = None):
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
                zip_archive = zipfile.ZipFile(temp_file.name)
                zip_archive.extractall(output_dir)
            elif url[-7:] == ".tar.gz":
                tar_archive = tarfile.open(temp_file.name)
                tar_archive.extractall(output_dir)
            # temp_file gets deleted here
        return True

