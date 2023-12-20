import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from tqdm import tqdm

from haystack import Document, component
from haystack.dataclasses import ByteStream
from haystack.lazy_imports import LazyImport
from haystack.components.converters.utils import get_bytestream_from_source

with LazyImport("Run 'pip install markdown-it-py mdit_plain'") as markdown_conversion_imports:
    from markdown_it import MarkdownIt
    from mdit_plain.renderer import RendererPlain


logger = logging.getLogger(__name__)


@component
class MarkdownToDocument:
    """
    Converts a Markdown file into a text Document.

    Usage example:
    ```python
    from haystack.components.converters.markdown import MarkdownToDocument

    converter = MarkdownToDocument()
    results = converter.run(sources=["sample.md"])
    documents = results["documents"]
    print(documents[0].content)
    # 'This is a text from the markdown file.'
    ```
    """

    def __init__(self, table_to_single_line: bool = False, progress_bar: bool = True):
        """
        :param table_to_single_line: Convert contents of the table into a single line. Defaults to False.
        :param progress_bar: Show a progress bar for the conversion. Defaults to True.
        """
        markdown_conversion_imports.check()

        self.table_to_single_line = table_to_single_line
        self.progress_bar = progress_bar

    @component.output_types(documents=List[Document])
    def run(self, sources: List[Union[str, Path, ByteStream]], meta: Optional[List[Dict[str, Any]]] = None):
        """
        Reads text from a markdown file and executes optional preprocessing steps.

        :param sources: A list of markdown data sources (file paths or binary objects)
        :param meta: Optional list of metadata to attach to the Documents.
          The length of the list must match the number of paths. Defaults to `None`.
        :return: A dictionary containing a list of Document objects under the 'documents' key.
        """
        parser = MarkdownIt(renderer_cls=RendererPlain)
        if self.table_to_single_line:
            parser.enable("table")

        documents = []

        if meta is None:
            meta = [{}] * len(sources)
        elif len(sources) != len(meta):
            raise ValueError("The length of the metadata list must match the number of sources.")

        for source, metadata in tqdm(
            zip(sources, meta),
            total=len(sources),
            desc="Converting markdown files to Documents",
            disable=not self.progress_bar,
        ):
            try:
                bytestream = get_bytestream_from_source(source)
            except Exception as e:
                logger.warning("Could not read %s. Skipping it. Error: %s", source, e)
                continue
            try:
                file_content = bytestream.data.decode("utf-8")
                text = parser.render(file_content)
            except Exception as conversion_e:
                logger.warning("Failed to extract text from %s. Skipping it. Error: %s", source, conversion_e)
                continue

            merged_metadata = {**bytestream.metadata, **metadata}
            document = Document(content=text, meta=merged_metadata)
            documents.append(document)

        return {"documents": documents}
