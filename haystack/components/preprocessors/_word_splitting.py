# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import re

from haystack.utils.misc import _CJK_HAN_KANA_CHAR_CLASS

# Matches a single CJK character (Han, Hiragana, Katakana). Korean is not included: Korean separates
# words with spaces, so it does not need per-character splitting. The kana ranges intentionally
# exclude punctuation such as the katakana middle dot (・) so it does not become a word on its own.
_CJK_CHAR_RE = re.compile(rf"[{_CJK_HAN_KANA_CHAR_CLASS}]")

# Matches one word: a single CJK character, or a run of non-space characters that contains no
# CJK characters. For text without CJK characters this is exactly `[^ ]+`.
_WORD_RE = re.compile(rf"[{_CJK_HAN_KANA_CHAR_CLASS}]|[^ {_CJK_HAN_KANA_CHAR_CLASS}]+")

# Matches one token for fixed word chunking: a single CJK character, a run of non-whitespace
# characters without CJK characters, or a run of whitespace. For text without CJK characters
# this is exactly `\S+|\s+`.
_WORD_TOKEN_RE = re.compile(rf"[{_CJK_HAN_KANA_CHAR_CLASS}]|[^\s{_CJK_HAN_KANA_CHAR_CLASS}]+|\s+")


def _split_words(text: str) -> list[str]:
    """
    Split text into word units, as `str.split(" ")` does, but with each CJK character as its own unit.

    Like `str.split(" ")` with the delimiter added back to every unit but the last, each unit
    carries the space that followed it in the text, so `"".join(_split_words(text)) == text`.
    Non-CJK text splits exactly like `str.split(" ")` with the delimiter added back, including
    the empty units a run of spaces produces.
    """
    units: list[str] = []
    pieces = text.split(" ")
    for piece_index, piece in enumerate(pieces):
        if not piece:
            piece_units: list[str] = [piece]
        else:
            piece_units = []
            start = 0
            for match in _CJK_CHAR_RE.finditer(piece):
                if match.start() > start:
                    piece_units.append(piece[start : match.start()])
                piece_units.append(match.group())
                start = match.end()
            if start < len(piece):
                piece_units.append(piece[start:])
        if piece_index < len(pieces) - 1:
            piece_units[-1] += " "
        units.extend(piece_units)
    return units


def _count_words(text: str) -> int:
    """
    Count the words in text, as `len(text.split())` does, but with each CJK character as one word.

    Non-CJK text is split on any whitespace, so newlines and tabs separate words just like spaces
    do. Whitespace-only tokens, such as the space between words, are not counted.
    """
    return len([token for token in _WORD_TOKEN_RE.findall(text) if token.strip()])


def _count_space_separated_words(text: str) -> int:
    """
    Count the words in text, splitting on spaces only, with each CJK character as one word.

    This is `len([word for word in text.split(" ") if word.strip()])`, but with each CJK
    character as one word. Unlike `_count_words`, only spaces separate words here: a piece such
    as `"bb\\fcc"` counts as one word, because the form feed marks a page break rather than a
    word boundary. This is the counting the recursive splitter uses, so its chunks stay aligned
    with the plain word split.
    """
    count = 0
    for piece in text.split(" "):
        count += len(_CJK_CHAR_RE.findall(piece))
        if _CJK_CHAR_RE.sub("", piece).strip():
            count += 1
    return count
