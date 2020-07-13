from haystack.indexing.file_converters.base import BaseConverter
import logging

logger = logging.getLogger(__name__)

class PDFToTextConverter(BaseConverter):
    def __init__(self):
    
