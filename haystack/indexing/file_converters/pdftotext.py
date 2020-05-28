import logging
import re
import subprocess
from functools import partial, reduce
from itertools import chain
from pathlib import Path

import langdetect

from haystack.indexing.file_converters.base import BaseConverter

logger = logging.getLogger(__name__)


class PDFToText(BaseConverter):
    def __init__(self, *args, **kwargs):
        verify_installation = subprocess.run(["pdftotext -v"], shell=True)
        if verify_installation.returncode == 127:
            raise Exception("pdftotext is not installed.")

        super().__init__(*args, **kwargs)

    def extract_text(self, file_path: Path) -> str:

        pages = []
        page_number = 1
        while True:  # loop until EOF
            page = self._extract_page(file_path, page_number, layout=True)
            if page == "":
                break

            lines = page.splitlines()

            if self.remove_tables:
                lines = self._remove_tables(lines)

            multi_column = self._detect_multi_column(lines) if lines else False
            if multi_column:
                page = self._extract_page(file_path, page_number, layout=False)
                lines = page.splitlines()

            page = "\n".join(lines)
            pages.append(page)
            page_number += 1

        document_text = "\n\n\n\n".join(pages)

        if not self._validate_language(document_text):
            logger.warning(
                f"The language for {file_path} is not one of {self.validate_languages}. The file may not have been "
                "decoded in the correct text format."
            )

        return document_text

    def _extract_page(self, file_path, page_number, layout=True):
        if layout:
            command = ["pdftotext", "-layout", "-f", str(page_number), "-l", str(page_number), file_path, "-"]
        else:
            command = ["pdftotext", "-f", str(page_number), "-l", str(page_number), file_path, "-"]
        output_page = subprocess.run(command, capture_output=True, shell=False)
        page = output_page.stdout.decode(errors="ignore")
        return page

    def _remove_tables(self, lines):
        cleaned_lines = []
        for line in lines:
            if line.strip() != "":
                if (
                        len(re.findall("\s\s\s\s+", line.strip())) > 0
                        or line.strip().replace(".", "").replace(",", "").isdigit()
                ):
                    continue

            cleaned_lines.append(line)
        return cleaned_lines

    def _validate_language(self, text):
        try:
            lang = langdetect.detect(text)
        except langdetect.lang_detect_exception.LangDetectException:
            lang = None

        if lang in self.validate_languages:
            return True
        else:
            return False

    def _detect_multi_column(self, lines):
        lines_with_possible_column_split = 0
        for line in lines:
            space_regex = re.finditer("\s\s\s+", line)
            space_position = [match.span() for match in space_regex if match.span()[0]]
            if len(space_position) == 1:
                lines_with_possible_column_split += 1

        if lines_with_possible_column_split / len(lines) > 0.5:
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
