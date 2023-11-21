from haystack.nodes.base import BaseComponent
from haystack.schema import Document
from typing import Tuple, List, Optional, Any, Dict
from pathlib import Path

class PptxConverter(BaseComponent):
  outgoing_edges = 1

  """
  Microsoft powerpoint converter.

  :param file_paths: Location of the file.
  :param meta: User defined meta data to use for the Document object.
  :usage pptx_converter = PptxConverter()
  p.add_node(component=pptx_converter, name='PptxConverter', inputs=['File'])
  p.run(file_paths=myfile.pptx, meta={'doc_name': 'myfile.pptx'})
  """

  def __init__(self):
    pass

  def run(self, file_paths: Path, meta: dict) -> tuple[dict[str, lst[Document]], str]:
    pptx_path = file_paths
    pres = Presentation(pptx_path)
    text = ""
    for slide_num, slide in enumerate(pres.slides):
      for shape in slide.shapes:
        if hasattr(shape, "text"):
          text += shape.text
          
    document = Document(content=text.page_content, meta=meta)
    output = {
      "documents": document
    }
    return output, "output_1"  
