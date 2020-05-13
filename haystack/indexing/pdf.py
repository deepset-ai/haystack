from tika import parser
from pathlib import Path


def convert_to_text(pdf_file: Path, remove_word_breaks: bool = True, headers_to_remove: [str] = None):
    """
    Convert a PDF file to string.

    :param pdf_file: Path of the file
    :type pdf_file: Path
    :param remove_word_breaks: remove hyphenated work breaks where a part of the word is written in the next line.
    :type remove_word_breaks: bool
    :param headers_to_remove: a list of string to remove from the file. Often, PDFs have headers or footers that are
                              repeated on each page.
    :type headers_to_remove: list(str)
    """
    raw = parser.from_file(str(pdf_file))
    cleaned_text = raw["content"]
    if remove_word_breaks:
        cleaned_text = cleaned_text.replace("-\n", "")
    if headers_to_remove:
        for header in headers_to_remove:
            cleaned_text = cleaned_text.replace(header, "")
    return cleaned_text
