# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import re

from haystack.utils.misc import _CJK_HAN_KANA_CHAR_CLASS

# CJK punctuation that stays attached to the CJK character before it instead of becoming
# a word on its own: the full stop in "生成。再把" stays with "成", so a chunk never ends
# with a lone "。". The ideographic space (U+3000) is whitespace, not punctuation, so it is
# excluded here.
_CJK_PUNCT_CHAR_CLASS = (
    r"\u3001-\u303f"  # CJK Symbols and Punctuation, minus the ideographic space
    r"\uff01-\uff0f"  # fullwidth ASCII punctuation
    r"\uff1a-\uff20"  # fullwidth : ; < = > ? @
    r"\uff3b-\uff40"  # fullwidth [ \ ] ^ _ `
    r"\uff5b-\uff60"  # fullwidth { | } ~
    r"\uff61-\uff65"  # halfwidth ideographic punctuation
)

# Matches one word unit: a single CJK character with any CJK punctuation attached to it, a
# run of CJK punctuation with no CJK character before it, or a run of non-space characters
# that contains no CJK characters or CJK punctuation. For text without CJK characters or
# CJK punctuation this is exactly `[^ ]+`.
_WORD_RE = re.compile(
    rf"[{_CJK_HAN_KANA_CHAR_CLASS}][{_CJK_PUNCT_CHAR_CLASS}]*"
    rf"|[{_CJK_PUNCT_CHAR_CLASS}]+"
    rf"|[^ {_CJK_HAN_KANA_CHAR_CLASS}{_CJK_PUNCT_CHAR_CLASS}]+"
)

# Matches one token for fixed word chunking: a single CJK character with any CJK punctuation
# attached to it, a run of CJK punctuation with no CJK character before it, a run of
# non-whitespace characters without CJK characters or CJK punctuation, or a run of
# whitespace. For text without CJK characters or CJK punctuation this is exactly `\S+|\s+`.
_WORD_TOKEN_RE = re.compile(
    rf"[{_CJK_HAN_KANA_CHAR_CLASS}][{_CJK_PUNCT_CHAR_CLASS}]*"
    rf"|[{_CJK_PUNCT_CHAR_CLASS}]+"
    rf"|[^\s{_CJK_HAN_KANA_CHAR_CLASS}{_CJK_PUNCT_CHAR_CLASS}]+"
    rf"|\s+"
)


def _split_words(text: str) -> list[str]:
    """
    Split text into word units, as `str.split(" ")` does, but with each CJK character as its own unit.

    CJK punctuation stays attached to the CJK character before it, so a full stop never
    becomes a unit on its own. Like `str.split(" ")` with the delimiter added back to every
    unit but the last, each unit carries the space that followed it in the text, so
    `"".join(_split_words(text)) == text`. Non-CJK text splits exactly like `str.split(" ")`
    with the delimiter added back, including the empty units a run of spaces produces.
    """
    units: list[str] = []
    pieces = text.split(" ")
    for piece_index, piece in enumerate(pieces):
        if not piece:
            piece_units: list[str] = [piece]
        else:
            piece_units = _WORD_RE.findall(piece)
        if piece_index < len(pieces) - 1:
            piece_units[-1] += " "
        units.extend(piece_units)
    return units


def _count_words(text: str) -> int:
    """
    Count the words in text, as `len(text.split())` does, but with each CJK character as one word.

    CJK punctuation stays attached to the CJK character before it, so it never counts as a
    word on its own. Non-CJK text is split on any whitespace, so newlines and tabs separate
    words just like spaces do. Whitespace-only tokens, such as the space between words, are not counted.
    """
    return len([token for token in _WORD_TOKEN_RE.findall(text) if token.strip()])


def _count_space_separated_words(text: str) -> int:
    """
    Count the words in text, splitting on spaces only, with each CJK character as one word.

    This counts the same word units as `_count_words`: a CJK character with any CJK
    punctuation attached to it is one word, and each run of non-CJK characters is one word,
    so the two splitters agree on mixed text such as `"使用Haystack构建RAG系统"`.
    Unlike `_count_words`, only spaces separate words here: a piece such as `"bb\\fcc"`
    counts as one word, because the form feed marks a page break rather than a word
    boundary. This is the counting the recursive splitter uses, so its chunks stay aligned
    with the plain word split.
    """
    count = 0
    for piece in text.split(" "):
        count += len([unit for unit in _WORD_RE.findall(piece) if unit.strip()])
    return count
