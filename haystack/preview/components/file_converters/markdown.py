import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

from tqdm import tqdm

from haystack.preview import Document, component
from haystack.preview.lazy_imports import LazyImport

with LazyImport("Run 'pip install beautifulsoup4 markdown python-frontmatter'") as markdown_conversion_imports:
    import frontmatter
    from bs4 import BeautifulSoup, NavigableString
    from markdown import markdown


logger = logging.getLogger(__name__)


@component
class MarkdownToTextDocument:
    """
    Converts a Markdown file into a text Document.
    """

    def __init__(
        self,
        remove_code_snippets: bool = True,
        extract_headlines: bool = False,
        add_frontmatter_to_meta: bool = False,
        progress_bar: bool = True,
    ):
        """
        :param remove_code_snippets: Whether to remove snippets from the markdown file. Defaults to True.
        :param extract_headlines: Whether to extract headings from the markdown file. Defaults to False.
        :param add_frontmatter_to_meta: Whether to add the contents of the frontmatter to `meta`. Defaults to False.
        :param progress_bar: Show a progress bar for the conversion.
        """
        markdown_conversion_imports.check()

        self.remove_code_snippets = remove_code_snippets
        self.extract_headlines = extract_headlines
        self.add_frontmatter_to_meta = add_frontmatter_to_meta
        self.progress_bar = progress_bar

    @component.output_types(documents=List[Document])
    def run(self, paths: List[Union[str, Path]], metadata: Optional[List[Union[Dict, Any]]] = None):
        """
        Reads text from a markdown file and executes optional preprocessing steps.

        :param file_path: path of the file to convert
        :param metadata: Optional list of metadata to attach to the Documents.
        The length of the list must match the number of paths. Defaults to `None`.
        """

        if metadata is None:
            metadata = [None] * len(paths)

        documents = []

        for file_path, meta in tqdm(
            zip(paths, metadata),
            total=len(paths),
            desc="Converting markdown files to Documents",
            disable=not self.progress_bar,
        ):
            with open(file_path, errors="ignore") as f:
                file_metadata, markdown_text = frontmatter.parse(f.read())

            # md -> html -> text since BeautifulSoup can extract text cleanly
            html = markdown(markdown_text, extensions=["fenced_code"])

            # remove code snippets
            if self.remove_code_snippets:
                html = re.sub(r"<pre>(.*?)</pre>", " ", html, flags=re.DOTALL)
                html = re.sub(r"<code>(.*?)</code>", " ", html, flags=re.DOTALL)
            soup = BeautifulSoup(html, "html.parser")

            if self.add_frontmatter_to_meta:
                if meta is None:
                    meta = file_metadata
                else:
                    meta.update(file_metadata)

            if self.extract_headlines:
                text, headlines = self._extract_text_and_headlines(soup)
                if meta is None:
                    meta = {}
                meta["headlines"] = headlines
            else:
                text = soup.get_text()

            if meta is None:
                document = Document(text=text)
            else:
                document = Document(text=text, metadata=meta)
            documents.append(document)

        return {"documents": documents}

    @staticmethod
    def _extract_text_and_headlines(soup: "BeautifulSoup") -> Tuple[str, List[Dict]]:
        """
        Extracts text and headings from a soup object.
        """
        headline_tags = {"h1", "h2", "h3", "h4", "h5", "h6"}
        headlines = []
        text = ""
        for desc in soup.descendants:
            if desc.name in headline_tags:
                current_headline = desc.get_text()
                current_start_idx = len(text)
                current_level = int(desc.name[-1]) - 1
                headlines.append({"headline": current_headline, "start_idx": current_start_idx, "level": current_level})

            if isinstance(desc, NavigableString):
                text += desc.get_text()

        return text, headlines
