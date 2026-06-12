# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
import re
from pathlib import Path
from typing import Any

import yaml
from tqdm import tqdm

from haystack import Document, component, logging
from haystack.components.converters.utils import get_bytestream_from_source, normalize_metadata
from haystack.dataclasses import ByteStream
from haystack.lazy_imports import LazyImport

with LazyImport("Run 'pip install markdown-it-py mdit_plain'") as markdown_conversion_imports:
    from markdown_it import MarkdownIt
    from mdit_plain.renderer import RendererPlain


logger = logging.getLogger(__name__)

_FRONTMATTER_PATTERN = re.compile(
    r"\A---[ \t]*\r?\n(?P<frontmatter>.*?)(?:\r?\n)---[ \t]*(?:\r?\n|$)", re.DOTALL
)


class _FrontmatterLoader(yaml.SafeLoader):
    """Safe YAML loader that keeps date-like scalars as strings for JSON-serializable metadata."""


_FrontmatterLoader.yaml_implicit_resolvers = {
    key: [(tag, regexp) for tag, regexp in resolvers if tag != "tag:yaml.org,2002:timestamp"]
    for key, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()
}


@component
class MarkdownToDocument:
    """
    Converts a Markdown file into a text Document.

    Usage example:

    ```python
    from haystack.components.converters import MarkdownToDocument
    from datetime import datetime

    converter = MarkdownToDocument()
    results = converter.run(
        sources=["test/test_files/markdown/sample.md"], meta={"date_added": datetime.now().isoformat()}
    )
    documents = results["documents"]
    print(documents[0].content)
    # 'This is a text from the markdown file.'
    ```
    """

    def __init__(
        self,
        table_to_single_line: bool = False,
        progress_bar: bool = True,
        store_full_path: bool = False,
        extract_frontmatter: bool = False,
    ) -> None:
        """
        Create a MarkdownToDocument component.

        :param table_to_single_line:
            If True converts table contents into a single line.
        :param progress_bar:
            If True shows a progress bar when running.
        :param store_full_path:
            If True, the full path of the file is stored in the metadata of the document.
            If False, only the file name is stored.
        :param extract_frontmatter:
            If True, YAML frontmatter at the beginning of the Markdown file is
            removed from the document content and added to the document metadata.
        """
        markdown_conversion_imports.check()

        self.table_to_single_line = table_to_single_line
        self.progress_bar = progress_bar
        self.store_full_path = store_full_path
        self.extract_frontmatter = extract_frontmatter

    @component.output_types(documents=list[Document])
    def run(
        self, sources: list[str | Path | ByteStream], meta: dict[str, Any] | list[dict[str, Any]] | None = None
    ) -> dict[str, Any]:
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
            zip(sources, meta_list, strict=True),
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
                file_content, frontmatter = self._extract_frontmatter(file_content, source)
                text = parser.render(file_content)
            except Exception as conversion_e:
                logger.warning(
                    "Failed to extract text from {source}. Skipping it. Error: {error}",
                    source=source,
                    error=conversion_e,
                )
                continue

            merged_metadata = {**bytestream.meta, **frontmatter, **metadata}

            if not self.store_full_path and (file_path := bytestream.meta.get("file_path")):
                merged_metadata["file_path"] = os.path.basename(file_path)

            document = Document(content=text, meta=merged_metadata)
            documents.append(document)

        return {"documents": documents}

    def _extract_frontmatter(self, file_content: str, source: str | Path | ByteStream) -> tuple[str, dict[str, Any]]:
        if not self.extract_frontmatter:
            return file_content, {}

        match = _FRONTMATTER_PATTERN.match(file_content)
        if not match:
            return file_content, {}

        frontmatter_text = match.group("frontmatter")
        try:
            frontmatter = yaml.load(frontmatter_text, Loader=_FrontmatterLoader) or {}
        except yaml.YAMLError as error:
            logger.warning(
                "Could not parse YAML frontmatter in {source}. Keeping it as content. Error: {error}",
                source=source,
                error=error,
            )
            return file_content, {}

        if not isinstance(frontmatter, dict):
            logger.warning(
                "Ignoring YAML frontmatter in {source}: expected a mapping, got {kind}.",
                source=source,
                kind=type(frontmatter).__name__,
            )
            return file_content, {}

        return file_content[match.end() :], frontmatter
