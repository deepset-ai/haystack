from haystack.indexing.file_converters.base import BaseConverter
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class docxToTextConverter(BaseConverter):
    def page_text(self, file_path: Path) -> str:
        import docx
        doc = docx.Document(file_path)  # Creating word reader object.
        text = ""
        fullText = []
        for para in doc.paragraphs:
          fullText.append(para.text)
        text = '\n'.join(fullText)
        return text
