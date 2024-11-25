# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from tqdm import tqdm

from haystack import Document, component, logging
from haystack.components.converters.utils import get_bytestream_from_source, normalize_metadata
from haystack.dataclasses import ByteStream
from haystack.lazy_imports import LazyImport

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
    from haystack.components.converters import MarkdownToDocument
    from datetime import datetime

    converter = MarkdownToDocument()
    results = converter.run(sources=["path/to/sample.md"], meta={"date_added": datetime.now().isoformat()})
    documents = results["documents"]
    print(documents[0].content)
    # 'This is a text from the markdown file.'
    ```
    """

    def __init__(self, table_to_single_line: bool = False, progress_bar: bool = True, store_full_path: bool = True):
        """
        Create a MarkdownToDocument component.

        :param table_to_single_line:
            If True converts table contents into a single line.
        :param progress_bar:
            If True shows a progress bar when running.
        :param store_full_path:
            If True, the full path of the file is stored in the metadata of the document.
            If False, only the file name is stored.
        """
        markdown_conversion_imports.check()

        self.table_to_single_line = table_to_single_line
        self.progress_bar = progress_bar
        self.store_full_path = store_full_path

    @component.output_types(documents=List[Document])
    def run(
        self,
        sources: List[Union[str, Path, ByteStream]],
        meta: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
    ):
        """
        Converts a list of Markdown files to Documents.

        :param sources:
            List of file paths or ByteStream objects.
        :param meta:
            Optional metadata to attach to the Documents.
            This value can be either a list of dictionaries or a single dictionary.
            If it's a single dictionary, its content is added to the metadata of all produced Documents.
            If it's a list, the length of the list must match the number of sources, because the two lists will
            be zipped.
            If `sources` contains ByteStream objects, their `meta` will be added to the output Documents.

        :returns:
            A dictionary with the following keys:
            - `documents`: List of created Documents
        """
        parser = MarkdownIt(renderer_cls=RendererPlain)
        if self.table_to_single_line:
            parser.enable("table")

        documents = []
        meta_list = normalize_metadata(meta=meta, sources_count=len(sources))

        for source, metadata in tqdm(
            zip(sources, meta_list),
            total=len(sources),
            desc="Converting markdown files to Documents",
            disable=not self.progress_bar,
        ):
            try:
                bytestream = get_bytestream_from_source(source)
            except Exception as e:
                logger.warning("Could not read {source}. Skipping it. Error: {error}", source=source, error=e)
                continue
            try:
                file_content = bytestream.data.decode("utf-8")
                text = parser.render(file_content)
            except Exception as conversion_e:
                logger.warning(
                    "Failed to extract text from {source}. Skipping it. Error: {error}",
                    source=source,
                    error=conversion_e,
                )
                continue

            merged_metadata = {**bytestream.meta, **metadata}

            warnings.warn(
                "The `store_full_path` parameter defaults to True, storing full file paths in metadata. "
                "In the 2.9.0 release, the default value for `store_full_path` will change to False, "
                "storing only file names to improve privacy.",
                DeprecationWarning,
            )

            if not self.store_full_path and (file_path := bytestream.meta.get("file_path")):
                merged_metadata["file_path"] = os.path.basename(file_path)

            document = Document(content=text, meta=merged_metadata)
            documents.append(document)

        return {"documents": documents}
