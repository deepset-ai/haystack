# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""
The `page_number` convention shared by the splitters.

Splitters record the page a chunk came from in its `page_number` metadata field, counting page breaks
(form feed, "\f") in the source document. The page reported is the one the chunk's first non-page-break
character lies on: a chunk that spans a page break belongs to the page it starts on, and page breaks a
chunk opens with belong to that chunk's own page rather than to the page it was carved out of.
"""


def _leading_page_breaks(text: str, page_break_character: str = "\f") -> int:
    """
    Count how many pages a chunk's text starts past the page its first character is on.

    This is the run of page breaks the chunk opens with, except for a chunk made up entirely of page
    breaks: that chunk is an empty page and stays on the page it starts on, rather than being pushed onto
    the following one. `split_by="page"` emits such chunks for every blank page.

    :param text: The chunk to inspect.
    :param page_break_character: The character sequence marking a page break.
    :returns: The number of pages to advance, 0 if the chunk starts with anything but a page break or
        consists only of page breaks.
    """
    if not page_break_character:
        return 0

    offset = 0
    while text.startswith(page_break_character, offset):
        offset += len(page_break_character)

    # Nothing but page breaks: an empty page, which belongs to the page it starts on.
    if offset >= len(text):
        return 0
    return offset // len(page_break_character)
