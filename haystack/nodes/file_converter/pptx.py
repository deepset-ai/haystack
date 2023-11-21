from haystack.nodes.base import BaseComponent
from haystack.schema import Document
from typing import Tuple, List, Optional, Any, Dict
from pathlib import Path

class PptxConverter(BaseComponent):
  outgoing_edges = 1

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
