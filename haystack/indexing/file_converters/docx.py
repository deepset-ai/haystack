from haystack.indexing.file_converters.base import BaseConverter
import logging

logger = logging.getLogger(__name__)

class docxToTextConverter(BaseConverter):
    def __init__(self):
        verify_installation = subprocess.run(["docx-python -v"], shell=True)
        if verify_installation.returncode == 127:
            raise Exception(
                """ python-docx is not installed.
                
                   Run !pip install python-docx
                   
                   Find more info at https://python-docx.readthedocs.io/en/latest/
                """
            )
            
            def page_text(self, file_path: Path) -> str:
                import docx
                doc = docx.Document(file_path)  # Creating word reader object.
                text = ""
                fullText = []
                for para in doc.paragraphs:
                  fullText.append(para.text)
                text = '\n'.join(fullText)
                return text
