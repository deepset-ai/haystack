import logging
from collections import defaultdict
from pathlib import Path
from typing import List, Union, Optional
from dataclasses import make_dataclass

from haystack.preview import component

logger = logging.getLogger(__name__)


@component
class FileTypeClassifier:
    """
    A component that classifies files based on their file extensions.

    The FileTypeClassifier takes a list of file paths and groups them by their file extensions.
    The list of extensions to consider is provided during the initialization of the component.

    This component is particularly useful when working with a large number of files, and you
    want to categorize them based on their file types.
    """

    @component.input
    def input(self):
        """
        The input data for the FileTypeClassifier.

        It expects a list of file paths. The file paths can be of type str or pathlib.Path.
        """

        class Input:
            paths: List[Union[str, Path]]

        return Input

    @component.output
    def output(self):
        """
        The output data of the FileTypeClassifier.

        It returns a dictionary where the keys are the file extensions, and the values are lists
        of pathlib.Path objects, representing the file paths that match the corresponding extension.
        """
        return make_dataclass(
            "Output", fields=[(extension, Optional[List[Union[str, Path]]], None) for extension in self.extensions]
        )

    def __init__(self, extensions: List[str]):
        """
        Initialize the FileTypeClassifier.

        :param extensions: A list of file extensions to consider when classifying files (e.g. ["txt", "wav", "jpg"]).
        """
        self.defaults = {"extensions": extensions}
        self.extensions = extensions

    def run(self, data):
        """
        Run the FileTypeClassifier.

        This method takes the input data, iterates through the provided file paths, checks the file
        extension of each file, and groups the file paths by their extensions.

        :param data: The input data containing the file paths to classify.
        :return: The output data containing the classified file paths.
        """
        extensions = defaultdict(list)
        paths: List[Union[str, Path]] = data.paths
        for path in paths:
            if isinstance(path, str):
                path = Path(path)
            file_extension = path.suffix[1:]  # skip the dot
            if file_extension in self.extensions:
                extensions[file_extension].append(path)
        return self.output(**extensions)
