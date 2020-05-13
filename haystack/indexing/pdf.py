from tika import parser
from pathlib import Path


def convert_to_text(pdf_file: Path, remove_word_breaks: bool = True):
    raw = parser.from_file(str(pdf_file))
    cleaned_text = raw["content"]
    if remove_word_breaks:
        cleaned_text = cleaned_text.replace("-\n", "")
    return cleaned_text
