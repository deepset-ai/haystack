from haystack.file_converter.base import BaseConverter
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
import docx

logger = logging.getLogger(__name__)


class DocxToTextConverter(BaseConverter):
    def extract_pages(self, file_path: Path) -> Tuple[List[str], Optional[Dict[str, Any]]]:
        """
        Extract text from a .docx file.
        Note: As docx doesn't contain "page" information, we actually extract and return a list of paragraphs here.
        For compliance with other converters we nevertheless opted for keeping the methods name.

        :param file_path: Path to the .docx file you want to convert
        """

        #TODO We might want to join small passages here (e.g. titles)
        #TODO Investigate if there's a workaround to extract on a page level rather than passage level
        #  (e.g. in the test sample it seemed that page breaks resulted in a paragraphs with only a "\n"

        doc = docx.Document(file_path)  # Creating word reader object.
        fullText = []
        for para in doc.paragraphs:
            if para.text.strip() != "":
                fullText.append(para.text)
        return fullText, None
