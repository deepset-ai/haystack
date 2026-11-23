# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import re

# Unicode ranges for scripts written without spaces between words (Chinese, Japanese, Korean).
# Each character in these ranges counts as one word when splitting by word, following the same
# convention as the BM25 tokenizer in haystack/document_stores/in_memory/document_store.py.
# The kana ranges intentionally exclude punctuation such as the katakana middle dot (・) so it
# does not become a word on its own.
_CJK_CHAR_CLASS = (
    r"\u1100-\u11ff"  # Hangul Jamo
    r"\ua960-\ua97f"  # Hangul Jamo Extended-A
    r"\ud7b0-\ud7ff"  # Hangul Jamo Extended-B
    r"\u3130-\u318f"  # Hangul Compatibility Jamo
    r"\uac00-\ud7af"  # Hangul Syllables
    r"\u3041-\u3096"  # Hiragana letters (excludes combining marks and kana iteration marks)
    r"\u30a1-\u30fa\u30fc"  # Katakana letters + prolonged sound mark, excludes ・ middle dot
    r"\u31f0-\u31ff"  # Katakana Phonetic Extensions
    r"\u3400-\u4dbf"  # CJK Unified Ideographs Extension A
    r"\u4e00-\u9fff"  # CJK Unified Ideographs
    r"\uf900-\ufaff"  # CJK Compatibility Ideographs
    r"\uff66-\uff9f"  # Halfwidth Katakana
    r"\uffa0-\uffdc"  # Halfwidth Hangul Jamo
)

# Matches a single CJK character.
_CJK_CHAR_RE = re.compile(rf"[{_CJK_CHAR_CLASS}]")

# Matches one word: a single CJK character, or a run of non-space characters that contains no
# CJK characters. For text without CJK characters this is exactly `[^ ]+`.
_WORD_RE = re.compile(rf"[{_CJK_CHAR_CLASS}]|[^ {_CJK_CHAR_CLASS}]+")

# Matches one token for fixed word chunking: a single CJK character, a run of non-whitespace
# characters without CJK characters, or a run of whitespace. For text without CJK characters
# this is exactly `\S+|\s+`.
_WORD_TOKEN_RE = re.compile(rf"[{_CJK_CHAR_CLASS}]|[^\s{_CJK_CHAR_CLASS}]+|\s+")


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
            piece_units = [piece]
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
    Count the words in text, as `len(text.split(" "))` does, but with each CJK character as one word.

    Whitespace-only pieces, such as the empty units a run of spaces produces, are not counted.
    """
    return len([unit for unit in _split_words(text) if unit.strip()])
