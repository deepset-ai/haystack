import logging
from html.parser import HTMLParser
from pathlib import Path
from typing import List, Optional, Dict, Any

import requests
from tika import parser as tikaparser

from haystack.file_converter.base import BaseConverter

logger = logging.getLogger(__name__)


# Use the built-in HTML parser with minimum dependencies
class TikaXHTMLParser(HTMLParser):
    def __init__(self):
        self.ingest = True
        self.page = ""
        self.pages: List[str] = []
        super(TikaXHTMLParser, self).__init__()

    def handle_starttag(self, tag, attrs):
        # find page div
        pagediv = [value for attr, value in attrs if attr == "class" and value == "page"]
        if tag == "div" and pagediv:
            self.ingest = True

    def handle_endtag(self, tag):
        # close page div, or a single page without page div, save page and open a new page
        if (tag == "div" or tag == "body") and self.ingest:
            self.ingest = False
            # restore words hyphened to the next line
            self.pages.append(self.page.replace("-\n", ""))
            self.page = ""

    def handle_data(self, data):
        if self.ingest:
            self.page += data


class TikaConverter(BaseConverter):
    def __init__(
        self,
        tika_url: str = "http://localhost:9998/tika",
        remove_numeric_tables: Optional[bool] = False,
        valid_languages: Optional[List[str]] = None
    ):
        """
        :param tika_url: URL of the Tika server
        :param remove_numeric_tables: This option uses heuristics to remove numeric rows from the tables.
                                      The tabular structures in documents might be noise for the reader model if it
                                      does not have table parsing capability for finding answers. However, tables
                                      may also have long strings that could possible candidate for searching answers.
                                      The rows containing strings are thus retained in this option.
        :param valid_languages: validate languages from a list of languages specified in the ISO 639-1
                                (https://en.wikipedia.org/wiki/ISO_639-1) format.
                                This option can be used to add test for encoding errors. If the extracted text is
                                not one of the valid languages, then it might likely be encoding error resulting
                                in garbled text.
        """
        ping = requests.get(tika_url)
        if ping.status_code != 200:
            raise Exception(f"Apache Tika server is not reachable at the URL '{tika_url}'. To run it locally"
                            f"with Docker, execute: 'docker run -p 9998:9998 apache/tika:1.24.1'")
        self.tika_url = tika_url
        super().__init__(remove_numeric_tables=remove_numeric_tables, valid_languages=valid_languages)

    def convert(self, file_path: Path, meta: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        :param file_path: Path of file to be converted.

        :return: a list of pages and the extracted meta data of the file.
        """
        parsed = tikaparser.from_file(file_path.as_posix(), self.tika_url, xmlContent=True)
        parser = TikaXHTMLParser()
        parser.feed(parsed["content"])

        cleaned_pages = []
        # TODO investigate title of document appearing in the first extracted page
        for page in parser.pages:
            lines = page.splitlines()
            cleaned_lines = []
            for line in lines:
                words = line.split()
                digits = [word for word in words if any(i.isdigit() for i in word)]

                # remove lines having > 40% of words as digits AND not ending with a period(.)
                if self.remove_numeric_tables:
                    if words and len(digits) / len(words) > 0.4 and not line.strip().endswith("."):
                        logger.debug(f"Removing line '{line}' from {file_path}")
                        continue

                cleaned_lines.append(line)

            page = "\n".join(cleaned_lines)
            cleaned_pages.append(page)

        if self.valid_languages:
            document_text = "".join(cleaned_pages)
            if not self.validate_language(document_text):
                logger.warning(
                    f"The language for {file_path} is not one of {self.valid_languages}. The file may not have "
                    f"been decoded in the correct text format."
                )

        text = "\f".join(cleaned_pages)
        document = {"text": text, "meta": {**parsed["metadata"], **(meta or {})}}
        return document
