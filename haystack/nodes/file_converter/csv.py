import logging
from pathlib import Path
from typing import Union, List, Iterable

import pandas as pd

from haystack import Document
from haystack.nodes.base import BaseComponent

logger = logging.getLogger(__name__)


class CsvTextConverter(BaseComponent):
    """
    Converts Question & Answers CSV files to text Documents.
    """

    outgoing_edges = 1

    @staticmethod
    def csv_qa_to_documents(csv_path: Union[Path, str]) -> Iterable[Document]:
        """
        Load CVS file and convert it to documents.

        :param csv_path: Path to a CSV file containing two columns.
            The first will be interpreted as a question, the second as content.
        :returns: List of document, 1 document per line in the CSV.
        """
        df = pd.read_csv(csv_path)

        df.fillna(value="", inplace=True)
        df["question"] = df["question"].apply(lambda x: x.strip())

        df = df.rename(columns={"question": "content"})
        docs_dicts = df.to_dict(orient="records")

        docs = [Document.from_dict(dictionary) for dictionary in docs_dicts]
        return docs

    def run(self, file_paths: Union[Path, List[Path], str, List[str], List[Union[Path, str]]]):  # type: ignore
        """
        Converts CSV files into text documents
        """
        if not isinstance(file_paths, list):
            file_paths = [file_paths]

        paths = [Path(path) for path in file_paths]

        documents: List[Document] = []
        for path in paths:
            documents.extend(self.csv_qa_to_documents(path))

        output = {"documents": documents}
        return output, "output_1"

    def run_batch(self, file_paths: Union[Path, List[Path], str, List[str], List[Union[Path, str]]]):  # type: ignore
        """
        Converts CSV files into text documents
        """
        if isinstance(file_paths, list):
            documents = []
            for files in file_paths:
                results, _ = self.run(files)
                documents.append(results["documents"])
            return {"documents": documents}, "output_1"
            
        return self.run(file_paths)
