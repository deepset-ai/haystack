from typing import List, Union
from pathlib import Path
from haystack.nodes.base import BaseComponent


class FileTypeClassifier(BaseComponent):
    """
    Route files in an Indexing Pipeline to corresponding file converters.
    """
    outgoing_edges = 5

    def __init__(self):
        self.set_config()

    def _get_files_extension(self, file_paths: list) -> set:
        """
        Return the file extensions
        :param file_paths:
        :return: set
        """
        return {file_path.suffix.lstrip(".") for file_path in file_paths}

    def run(self, file_paths: Union[Path, List[Path]]):  # type: ignore
        """
        Return the output based on file extension
        """
        if isinstance(file_paths, Path):
            file_paths = [file_paths]

        extension: set = self._get_files_extension(file_paths)
        if len(extension) > 1:
            raise ValueError(f"Multiple files types are not allowed at once.")

        output = {"file_paths": file_paths}
        ext: str = extension.pop()
        try:
            index = ["txt", "pdf", "md", "docx", "html"].index(ext) + 1
            return output, f"output_{index}"
        except ValueError:
            raise Exception(f"Files with an extension '{ext}' are not supported.")
