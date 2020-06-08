import logging
import re
import subprocess
from functools import partial, reduce
from itertools import chain
from pathlib import Path
from typing import List

import fitz
import langdetect

from haystack.indexing.file_converters.base import BaseConverter

logger = logging.getLogger(__name__)


class PDFToTextConverter(BaseConverter):
    def __init__(
        self,
        remove_numeric_tables: bool = False,
        remove_whitespace: bool = None,
        remove_empty_lines: bool = None,
        remove_header_footer: bool = None,
        valid_languages: List[str] = None,
    ):
        """
        :param remove_numeric_tables: This option uses heuristics to remove numeric rows from the tables.
                                      The tabular structures in documents might be noise for the reader model if it
                                      does not have table parsing capability for finding answers. However, tables
                                      may also have long strings that could possible candidate for searching answers.
                                      The rows containing strings are thus retained in this option.
        :param remove_whitespace: strip whitespaces before or after each line in the text.
        :param remove_empty_lines: remove more than two empty lines in the text.
        :param remove_header_footer: use heuristic to remove footers and headers across different pages by searching
                                     for the longest common string. This heuristic uses exact matches and therefore
                                     works well for footers like "Copyright 2019 by XXX", but won't detect "Page 3 of 4"
                                     or similar.
        :param valid_languages: validate languages from a list of languages specified in the ISO 639-1
                                (https://en.wikipedia.org/wiki/ISO_639-1) format.
                                This option can be used to add test for encoding errors. If the extracted text is
                                not one of the valid languages, then it might likely be encoding error resulting
                                in garbled text.
        """
        verify_installation = subprocess.run(["pdftotext -v"], shell=True)
        if verify_installation.returncode == 127:
            raise Exception(
                """pdftotext is not installed. It is part of xpdf or poppler-utils software suite.
                
                   Installation on Linux:
                   wget --no-check-certificate https://dl.xpdfreader.com/xpdf-tools-linux-4.02.tar.gz &&
                   tar -xvf xpdf-tools-linux-4.02.tar.gz && sudo cp xpdf-tools-linux-4.02/bin64/pdftotext /usr/local/bin
                   
                   Installation on MacOS:
                   brew install xpdf
                   
                   You can find more details here: https://www.xpdfreader.com
                """
            )

        super().__init__(
            remove_numeric_tables=remove_numeric_tables,
            remove_whitespace=remove_whitespace,
            remove_empty_lines=remove_empty_lines,
            remove_header_footer=remove_header_footer,
            valid_languages=valid_languages,
        )

    def extract_pages(self, file_path: Path) -> List[str]:

        page_count = fitz.open(file_path).pageCount

        pages = []
        for page_number in range(1, page_count + 1):
            # pdftotext tool provides an option to retain the original physical layout of a PDF page. This behaviour
            # can be toggled by using the layout param.
            #  layout=True
            #      + table structures get retained better
            #      - multi-column pages(eg, research papers) gets extracted with text from multiple columns on same line
            #  layout=False
            #      + keeps strings in content stream order, hence multi column layout works well
            #      - cells of tables gets split across line
            #
            #  Here, as a "safe" default, layout is turned off.
            page = self._extract_page(file_path, page_number, layout=False)
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

                if self.remove_whitespace:
                    line = line.strip()

                cleaned_lines.append(line)

            page = "\n".join(cleaned_lines)

            if self.remove_empty_lines:
                page = re.sub(r"\n\n+", "\n\n", page)

            pages.append(page)
            page_number += 1

        if self.valid_languages:
            document_text = "".join(pages)
            if not self._validate_language(document_text):
                logger.warning(
                    f"The language for {file_path} is not one of {self.valid_languages}. The file may not have "
                    f"been decoded in the correct text format."
                )

        if self.remove_header_footer:
            pages, header, footer = self.find_and_remove_header_footer(
                pages, n_chars=300, n_first_pages_to_ignore=1, n_last_pages_to_ignore=1
            )
            logger.info(f"Removed header '{header}' and footer {footer} in {file_path}")

        return pages

    def _extract_page(self, file_path: Path, page_number: int, layout: bool):
        """
        Extract a page from the pdf file at file_path.

        :param file_path: path of the pdf file
        :param page_number: page number to extract(starting from 1)
        :param layout: whether to retain the original physical layout for a page. If disabled, PDF pages are read in
                       the content stream order.
        """
        if layout:
            command = ["pdftotext", "-layout", "-f", str(page_number), "-l", str(page_number), file_path, "-"]
        else:
            command = ["pdftotext", "-f", str(page_number), "-l", str(page_number), file_path, "-"]
        output_page = subprocess.run(command, capture_output=True, shell=False)
        page = output_page.stdout.decode(errors="ignore")
        return page

    def _validate_language(self, text: str):
        """
        Validate if the language of the text is one of valid languages.
        """
        try:
            lang = langdetect.detect(text)
        except langdetect.lang_detect_exception.LangDetectException:
            lang = None

        if lang in self.valid_languages:
            return True
        else:
            return False

    def _ngram(self, seq: str, n: int):
        """
        Return ngram (of tokens - currently splitted by whitespace)
        :param seq: str, string from which the ngram shall be created
        :param n: int, n of ngram
        :return: str, ngram as string
        """

        # In order to maintain the original whitespace, but still consider \n and \t for n-gram tokenization,
        # we add a space here and remove it after creation of the ngrams again (see below)
        seq = seq.replace("\n", " \n")
        seq = seq.replace("\t", " \t")

        seq = seq.split(" ")
        ngrams = (
            " ".join(seq[i : i + n]).replace(" \n", "\n").replace(" \t", "\t") for i in range(0, len(seq) - n + 1)
        )

        return ngrams

    def _allngram(self, seq: str, min_ngram: int, max_ngram: int):
        lengths = range(min_ngram, max_ngram) if max_ngram else range(min_ngram, len(seq))
        ngrams = map(partial(self._ngram, seq), lengths)
        res = set(chain.from_iterable(ngrams))
        return res

    def find_longest_common_ngram(self, sequences: List[str], max_ngram: int = 30, min_ngram: int = 3) -> str:
        """
        Find the longest common ngram across different text sequences (e.g. start of pages).
        Considering all ngrams between the specified range. Helpful for finding footers, headers etc.

        :param sequences: list[str], list of strings that shall be searched for common n_grams
        :param max_ngram: int, maximum length of ngram to consider
        :param min_ngram: minimum length of ngram to consider
        :return: str, common string of all sections
        """

        seqs_ngrams = map(partial(self._allngram, min_ngram=min_ngram, max_ngram=max_ngram), sequences)
        intersection = reduce(set.intersection, seqs_ngrams)

        try:
            longest = max(intersection, key=len)
        except ValueError:
            # no common sequence found
            longest = ""
        return longest if longest.strip() else None

    def find_and_remove_header_footer(
        self, pages: List[str], n_chars: int, n_first_pages_to_ignore: int, n_last_pages_to_ignore: int
    ) -> str:
        """
        Heuristic to find footers and headers across different pages by searching for the longest common string.
        For headers we only search in the first n_chars characters (for footer: last n_chars).
        Note: This heuristic uses exact matches and therefore works well for footers like "Copyright 2019 by XXX",
         but won't detect "Page 3 of 4" or similar.

        :param pages: list of strings, one string per page
        :param n_chars: number of first/last characters where the header/footer shall be searched in
        :param n_first_pages_to_ignore: number of first pages to ignore (e.g. TOCs often don't contain footer/header)
        :param n_last_pages_to_ignore: number of last pages to ignore
        :return: (cleaned pages, found_header_str, found_footer_str)
        """

        # header
        start_of_pages = [p[:n_chars] for p in pages[n_first_pages_to_ignore:-n_last_pages_to_ignore]]
        found_header = self.find_longest_common_ngram(start_of_pages)
        if found_header:
            pages = [page.replace(found_header, "") for page in pages]

        # footer
        end_of_pages = [p[-n_chars:] for p in pages[n_first_pages_to_ignore:-n_last_pages_to_ignore]]
        found_footer = self.find_longest_common_ngram(end_of_pages)
        if found_footer:
            pages = [page.replace(found_footer, "") for page in pages]
        return pages, found_header, found_footer
