import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from haystack.nodes.file_converter import BaseConverter


logger = logging.getLogger(__name__)


class MarkdownConverter(BaseConverter):
    def convert(
            self,
            file_path: Path,
            meta: Optional[Dict[str, str]] = None,
            remove_numeric_tables: Optional[bool] = None,
            valid_languages: Optional[List[str]] = None,
            encoding: Optional[str] = "utf-8",
    ) -> List[Dict[str, Any]]:
        """
        Reads text from a txt file and executes optional preprocessing steps.

        :param file_path: path of the file to convert
        :param meta: dictionary of meta data key-value pairs to append in the returned document.
        :param encoding: Select the file encoding (default is `utf-8`)
        :param remove_numeric_tables: Not applicable
        :param valid_languages: Not applicable

        :return: Dict of format {"text": "The text from file", "meta": meta}}
        """
        with open(file_path, encoding=encoding, errors="ignore") as f:
            markdown_text = f.read()
        text = self.markdown_to_text(markdown_text)
        document = {"content": text, "content_type": "text", "meta": meta}
        return [document]

    # Following code snippet is copied from https://gist.github.com/lorey/eb15a7f3338f959a78cc3661fbc255fe
    @staticmethod
    def markdown_to_text(markdown_string: str) -> str:
        """
        Converts a markdown string to plaintext

        :param markdown_string: String in markdown format
        """
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            raise ImportError("Can't find package `beautifulsoup4` \n"
                              "You can install it via `pip install beautifulsoup4`")

        try:
            from markdown import markdown
        except ImportError:
            raise ImportError("Can't find package `markdown` \n"
                              "You can install it via `pip install markdown`")

        # md -> html -> text since BeautifulSoup can extract text cleanly
        html = markdown(markdown_string)

        # remove code snippets
        html = re.sub(r'<pre>(.*?)</pre>', ' ', html)
        html = re.sub(r'<code>(.*?)</code >', ' ', html)

        # extract text
        soup = BeautifulSoup(html, "html.parser")
        text = ''.join(soup.findAll(text=True))

        return text
