import logging
from pathlib import Path
from typing import Dict, Optional, Any

import docx

from haystack.file_converter.base import BaseConverter

logger = logging.getLogger(__name__)


class DocxToTextConverter(BaseConverter):
    def convert(self, file_path: Path, meta: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Extract text from a .docx file.
        Note: As docx doesn't contain "page" information, we actually extract and return a list of paragraphs here.
        For compliance with other converters we nevertheless opted for keeping the methods name.

        :param file_path: Path to the .docx file you want to convert
        """

        file = docx.Document(file_path)  # Creating word reader object.
        paragraphs = [para.text for para in file.paragraphs]
        text = "".join(paragraphs)
        document = {"text": text, "meta": meta}
        return document
