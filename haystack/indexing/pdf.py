import subprocess
from functools import partial, reduce
from itertools import chain
from pathlib import Path

from tika import parser


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


def convert_to_text_xpdf(pdf_file: str):
    output = subprocess.run(["pdftotext", "-layout", pdf_file, "-"], capture_output=True, shell=False)
    text = output.stdout.decode(errors='ignore')
    return text


def _ngram(seq, n):
    """
    Return ngram (of tokens - currently splitted by whitespace)
    :param seq: str, string from which the ngram shall be created
    :param n: int, n of ngram
    :return: str, ngram as string
    """
    seq = seq.strip()
    seq = seq.split(" ")
    ngrams = (" ".join(seq[i: i+n]) for i in range(0, len(seq)-n+1))
    ngrams = list(ngrams)
    x = []
    for i in ngrams:
        if i.strip() == "":
            continue
        x.append(i)
    return x


def _allngram(seq, min_ngram=1, max_ngram=None):
    lengths = range(min_ngram, max_ngram) if max_ngram else range(min_ngram, len(seq))
    ngrams = map(partial(_ngram, seq), lengths)
    return set(chain.from_iterable(ngrams))


def find_footer(sequences, chars=200, max_ngram=15, min_ngram=1):
    """
    Find a footer by searching for the longest common ngram across different pages/sections in the pdf.
    The search considers only the last "chars" characters of the files.
    :param sequences: list[str], list of strings from documents
    :param chars: int, number of chars at the end of the string in which the footer shall be searched
    :param max_ngram: int, maximum length of ngram to consider
    :param min_ngram: minimum length of ngram to consider
    :return: str, common footer of all sections
    """
    last = []
    for seq in sequences:
        last.append(seq[:chars])

    seqs_ngrams = map(partial(_allngram, min_ngram=min_ngram, max_ngram=max_ngram), last)
    intersection = reduce(set.intersection, seqs_ngrams)
    try:
        longest = max(intersection, key=len)
    except ValueError:
        #no common sequence found
        longest = None
    return longest
