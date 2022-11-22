import logging
from pathlib import Path
from typing import Union, List, Iterable

import pandas as pd

from haystack import Document
from haystack.nodes.base import BaseComponent

logger = logging.getLogger(__name__)


class CsvToDocuments(BaseComponent):
    """
    Converts Question & Answers CSV files to documents.
    """

    outgoing_edges = 1

    @staticmethod
    def csv_qa_to_documents(csv_path: Union[Path, str]) -> Iterable[Document]:
        """
        Load CVS file, convert it to documents (without embeding).

        :param csv_path: Path to a CSV file.
        :returns: List of document, 1 document per question (line in the CSV).
        """
        df = pd.read_csv(csv_path)

        # Minimal cleaning
        df.fillna(value="", inplace=True)
        df["question"] = df["question"].apply(lambda x: x.strip())

        df = df.rename(columns={"question": "content"})
        docs_dict = df.to_dict(orient="records")
        docs = map(Document.from_dict, docs_dict)

        return docs

    def run(self, file_paths: Union[Path, List[Path], str, List[str], List[Union[Path, str]]]):  # type: ignore
        """
        Sends out files on a different output edge depending on their extension.

        :param file_paths: paths to route on different edges.
        """
        if not isinstance(file_paths, list):
            file_paths = [file_paths]

        paths = [Path(path) for path in file_paths]

        docs = []
        for p in paths:
            docs.extend(self.csv_qa_to_documents(p))

        output = {"documents": docs}
        return output, "output_1"

    def run_batch(self, file_paths: Union[Path, List[Path], str, List[str], List[Union[Path, str]]]):  # type: ignore
        self.run(file_paths)
