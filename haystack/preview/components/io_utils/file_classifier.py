import logging
from collections import defaultdict
from pathlib import Path
from typing import List, Union, Dict

from haystack.preview import component

logger = logging.getLogger(__name__)


@component
class FileTypeClassifier:
    @component.input
    def input(self):
        class Input:
            paths: List[Union[str, Path]]

        return Input

    @component.output
    def output(self):
        class Output:
            extensions: Dict[str, List[Path]]

        return Output

    def __init__(self, extensions: List[str]):
        self.defaults = {"extensions": extensions}

    def run(self, data):
        extensions = defaultdict(list)
        paths: List[Union[str, Path]] = data.paths
        for path in paths:
            if isinstance(path, str):
                path = Path(path)
            if path.suffix in self.defaults["extensions"]:
                extensions[path.suffix].append(path)
        return self.output(extensions=extensions)
