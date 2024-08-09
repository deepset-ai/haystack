import io
import json
from pathlib import Path
from typing import List, Union

from haystack import Document, component, logging
from haystack.lazy_imports import LazyImport

logger = logging.getLogger(__name__)

with LazyImport("Run 'pip install jq'") as jq_import:
    import jq


@component
class JSONToDocument:
    """
    Converts JSON files to Documents.

    Usage example:
    ```python
    from haystack.components.converters.json import JSONToDocument

    converter = JSONToDocument()
    results = converter.run(sources=["sample.json"]})
    documents = results["documents"]
    print(documents[0].content)
    # 'This is the text from the PPTX file.'
    ```
    """

    def __init__(self, jq_schema: str):
        """
        Create an JSONToDocument component.

        :param jq_schema:
            The jq schema for the data or text extraction from JSON object
        """

        jq_import.check()

        self.jq_schema = jq.compile(jq_schema)

    @component.output_types(documents=List[Document])
    def run(self, sources: List[Union[str, Path]]):
        """
        Converts JSON files to Documents.

        :param sources:
            List of file paths or ByteStream objects.

        :returns:
            A dictionary with the following keys:
            - `documents`: Created Documents
        """
        documents = []

        for source in sources:
            file_path = Path(source).resolve()

            try:
                file_content = file_path.read_text(encoding="utf-8")
            except Exception as e:
                logger.warning(
                    "Could not read file in {file_path}. Skipping it. Error: {error}", file_path=file_path, error=e
                )
                continue

            try:
                data = self.jq_schema.input(json.loads(file_content))
            except Exception as conversion_e:
                logger.warning(
                    "Failed to extract JSON content from {file_content}. Skipping it. Error: {error}",
                    file_content=file_content,
                    error=conversion_e,
                )
                continue

            for i, sample in enumerate(data, 1):
                text = sample
                metadata = {"source": file_path, "row_idx": i}
                documents.append(Document(content=text, meta=metadata))

        return {"documents": documents}
