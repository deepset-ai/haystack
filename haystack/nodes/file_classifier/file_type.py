import mimetypes
from typing import Any, Dict, List, Union

import logging
from pathlib import Path

try:
    import magic
except ImportError as ie:
    logging.debug(
        "Failed to import 'magic' (from 'python-magic' and 'python-magic-bin' on Windows). "
        "FileTypeClassifier will not perform mimetype detection on extensionless files. "
        "Please make sure the necessary OS libraries are installed if you need this functionality."
    )

from haystack.nodes.base import BaseComponent


logger = logging.getLogger(__name__)


DEFAULT_TYPES = ["txt", "pdf", "md", "docx", "html"]


class FileTypeClassifier(BaseComponent):
    """
    Route files in an Indexing Pipeline to corresponding file converters.
    """

    outgoing_edges = len(DEFAULT_TYPES)

    def __init__(self, supported_types: List[str] = DEFAULT_TYPES):
        """
        Node that sends out files on a different output edge depending on their extension.

        :param supported_types: The file types that this node can distinguish between.
             The default values are: `txt`, `pdf`, `md`, `docx`, and `html`.
             Lists with duplicate elements are not allowed.
        """
        if len(set(supported_types)) != len(supported_types):
            duplicates = supported_types
            for item in set(supported_types):
                duplicates.remove(item)
            raise ValueError(f"supported_types can't contain duplicate values ({duplicates}).")

        super().__init__()

        self.supported_types = supported_types

    @classmethod
    def _calculate_outgoing_edges(cls, component_params: Dict[str, Any]) -> int:
        supported_types = component_params.get("supported_types", DEFAULT_TYPES)
        return len(supported_types)

    def _estimate_extension(self, file_path: Path) -> str:
        """
        Return the extension found based on the contents of the given file

        :param file_path: the path to extract the extension from
        """
        try:
            extension = magic.from_file(str(file_path), mime=True)
            return mimetypes.guess_extension(extension) or ""
        except NameError as ne:
            logger.error(
                f"The type of '{file_path}' could not be guessed, probably because 'python-magic' is not installed. Ignoring this error."
                "Please make sure the necessary OS libraries are installed if you need this functionality ('python-magic' or 'python-magic-bin' on Windows)."
            )
            return ""

    def _get_extension(self, file_paths: List[Path]) -> str:
        """
        Return the extension found in the given list of files.
        Also makes sure that all files have the same extension.
        If this is not true, it throws an exception.

        :param file_paths: the paths to extract the extension from
        :return: a set of strings with all the extensions (without duplicates), the extension will be guessed if the file has none
        """
        extension = file_paths[0].suffix.lower()
        if extension == "":
            extension = self._estimate_extension(file_paths[0])

        for path in file_paths:
            path_suffix = path.suffix.lower()
            if path_suffix == "":
                path_suffix = self._estimate_extension(path)
            if path_suffix != extension:
                raise ValueError(f"Multiple file types are not allowed at once.")

        return extension.lstrip(".")

    def run(self, file_paths: Union[Path, List[Path], str, List[str], List[Union[Path, str]]]):  # type: ignore
        """
        Sends out files on a different output edge depending on their extension.

        :param file_paths: paths to route on different edges.
        """
        if not isinstance(file_paths, list):
            file_paths = [file_paths]

        paths = [Path(path) for path in file_paths]

        output = {"file_paths": paths}
        extension = self._get_extension(paths)
        try:
            index = self.supported_types.index(extension) + 1
        except ValueError:
            raise ValueError(
                f"Files of type '{extension}' ({paths[0]}) are not supported. "
                f"The supported types are: {self.supported_types}. "
                "Consider using the 'supported_types' parameter to "
                "change the types accepted by this node."
            )
        return output, f"output_{index}"

    def run_batch(self, file_paths: Union[Path, List[Path], str, List[str], List[Union[Path, str]]]):  # type: ignore
        return self.run(file_paths=file_paths)
