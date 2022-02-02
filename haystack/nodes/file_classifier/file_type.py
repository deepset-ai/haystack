from multiprocessing.sharedctypes import Value
from typing import List, Union
from pathlib import Path
from haystack.nodes.base import BaseComponent


class FileTypeClassifier(BaseComponent):
    """
    Route files in an Indexing Pipeline to corresponding file converters.
    """
    outgoing_edges = 10

    def __init__(self):
        self.set_config()

    def _get_extension(self, file_paths: list) -> str:
        """
        Return the extension found in the given list of files.
        Also makes sure that all files have the same extension.
        If this is not true, it throws an exception.

        :param file_paths: the paths to extract the extension from
        :return: a set of strings with all the extensions (without duplicates)
        """
        extension = file_paths[0].suffix

        for path in file_paths:
            if path.suffix != extension:
                raise ValueError(f"Multiple files types are not allowed at once.")

        return extension.lstrip(".")

    def run(self, 
            file_paths: Union[Path, List[Path]], 
            supported_types: List[str] = ["txt", "pdf", "md", "docx", "html"]
        ):  # type: ignore
        """
        Sends out files on a different output edge depending on their extension.

        :param file_paths: paths to route on different edges.
        :param supported_types: the file types that this node can distinguish.
            Note that it's limited to a maximum of 10 outgoing edges, which 
            correspond each to a file extension. Such extension are, by default
            `txt`, `pdf`, `md`, `docx`, `html`. Lists containing more than 10
            elements will not be allowed. Lists with duplicate elements will 
            also be rejected.
        """
        if isinstance(file_paths, Path):
            file_paths = [file_paths]

        if len(supported_types) > 5:
            raise ValueError("supported_types can't have more than 5 values.")

        extension = self._get_extension(file_paths)
        output = {"file_paths": file_paths}

        try:
            index = supported_types.index(extension) + 1
            return output, f"output_{index}"
        except ValueError:
            raise Exception(f"Files with an extension '{extension}' are not supported.")
