from typing import Union, List, Optional, Any, Dict

import logging
from pathlib import Path

import pandas as pd

from haystack import Document
from haystack.nodes.file_converter import BaseConverter


logger = logging.getLogger(__name__)


class CsvTextConverter(BaseConverter):
    """
    Converts Question & Answers CSV files to text Documents.
    """

    outgoing_edges = 1

    def convert(
        self,
        file_path: Union[Path, List[Path], str, List[str], List[Union[Path, str]]],
        meta: Optional[Dict[str, Any]],
        remove_numeric_tables: Optional[bool] = None,
        valid_languages: Optional[List[str]] = None,
        encoding: Optional[str] = "UTF-8",
        id_hash_keys: Optional[List[str]] = None,
    ) -> List[Document]:
        """
        Load CVS file and convert it to documents.

        :param file_path: Path to a CSV file containing two columns.
            The first will be interpreted as a question, the second as content.
        :returns: List of document, 1 document per line in the CSV.
        """
        if not isinstance(file_path, list):
            file_path = [file_path]

        docs: List[Document] = []
        for path in file_path:
            df = pd.read_csv(path, encoding=encoding)

            if len(df.columns) != 2 or df.columns[0] != "question" or df.columns[1] != "answer":
                raise ValueError("The CSV must contain two columns named 'question' and 'answer'")

            df.fillna(value="", inplace=True)
            df["question"] = df["question"].apply(lambda x: x.strip())

            df = df.rename(columns={"question": "content"})
            docs_dicts = df.to_dict(orient="records")

            for dictionary in docs_dicts:
                if meta:
                    dictionary["meta"] = meta
                if id_hash_keys:
                    dictionary["id_hash_keys"] = id_hash_keys
                docs.append(Document.from_dict(dictionary))

        return docs
