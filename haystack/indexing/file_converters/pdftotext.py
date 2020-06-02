import logging
import re
import subprocess
from functools import partial, reduce
from itertools import chain
from pathlib import Path

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
        valid_languages: [str] = None,
    ):
        """
        :param remove_numeric_tables: This option uses heuristics to remove numeric rows from the tables.
                                      The tabular structures in documents might be noise for the reader model if it
                                      does not have table parsing capability for finding answers. However, tables
                                      may also have long strings that could possible candidate for searching answers.
                                      The rows containing strings are thus retained in this option.
        :param remove_whitespace: strip whitespaces before or after each line in the text.
        :param remove_empty_lines: remove more than two empty lines in the text.
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
                 Installation Instructions:
                 * Ubuntu/Debian: 'sudo apt-get update && sudo apt-get install -y xpdf'
                 * CentOS: sudo yum install poppler-utils
                 * MacOS: brew install xpdf"""
            )

        super().__init__(
            remove_numeric_tables=remove_numeric_tables,
            remove_whitespace=remove_whitespace,
            remove_empty_lines=remove_empty_lines,
            valid_languages=valid_languages,
        )

    def extract_pages(self, file_path: Path) -> [str]:

        page_count = fitz.open(file_path).pageCount

        pages = []
        for page_number in range(1, page_count + 1):
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

        return pages

    def _extract_page(self, file_path, page_number, layout=True):
        if layout:
            command = ["pdftotext", "-layout", "-f", str(page_number), "-l", str(page_number), file_path, "-"]
        else:
            command = ["pdftotext", "-f", str(page_number), "-l", str(page_number), file_path, "-"]
        output_page = subprocess.run(command, capture_output=True, shell=False)
        page = output_page.stdout.decode(errors="ignore")
        return page

    def _validate_language(self, text):
        try:
            lang = langdetect.detect(text)
        except langdetect.lang_detect_exception.LangDetectException:
            lang = None

        if lang in self.valid_languages:
            return True
        else:
            return False

    def _ngram(self, seq, n):
        """
        Return ngram (of tokens - currently splitted by whitespace)
        :param seq: str, string from which the ngram shall be created
        :param n: int, n of ngram
        :return: str, ngram as string
        """
        seq = seq.split(" ")
        ngrams = (" ".join(seq[i : i + n]) for i in range(0, len(seq) - n + 1))
        return ngrams

    def _allngram(self, seq, min_ngram, max_ngram):
        lengths = range(min_ngram, max_ngram) if max_ngram else range(min_ngram, len(seq))
        ngrams = map(partial(self._ngram, seq), lengths)
        return set(chain.from_iterable(ngrams))

    def find_footer(self, sequences, chars=500, max_ngram=200, min_ngram=4):
        """
        Find a footer by searching for the longest common ngram across different pages/sections in the pdf.
        The search considers only the last "chars" characters of the files.
        :param sequences: list[str], list of strings from documents
        :param chars: int, number of chars at the end of the string in which the footer shall be searched
        :param max_ngram: int, maximum length of ngram to consider
        :param min_ngram: minimum length of ngram to consider
        :return: str, common footer of all sections
        """
        seqs_ngrams = map(partial(self._allngram, min_ngram=min_ngram, max_ngram=max_ngram), sequences)
        intersection = reduce(set.intersection, seqs_ngrams)
        try:
            longest = max(intersection, key=len)
        except ValueError:
            # no common sequence found
            longest = ""
        return longest if longest.strip() else None
