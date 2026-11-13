# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from itertools import pairwise

import pytest
import tiktoken

from haystack import Document
from haystack.components.preprocessors import RecursiveDocumentSplitter


@pytest.mark.parametrize(
    ("text", "limits", "expected"),
    [
        ("ab🍕cd", (4, 0), [("ab🍕", 0), ("cd", 3)]),
        ("🍕🍕", (1, 0), [("🍕", 0), ("🍕", 1)]),
        ("a🍕bcd", (6, 2), [("a🍕b", 0), ("bcd", 2)]),
        ("ab한cd", (4, 0), [("ab한", 0), ("cd", 3)]),
        ("ab.cd🍕", (6, 1), [("ab.", 0), (".cd🍕", 2)]),
    ],
)
def test_token_splitting_preserves_unicode_and_source_offsets(
    monkeypatch: pytest.MonkeyPatch, text: str, limits: tuple[int, int], expected: list[tuple[str, int]]
) -> None:
    byte_encoding = tiktoken.Encoding(
        name="test_bytes",
        pat_str=r"(?s).",
        mergeable_ranks={bytes([byte]): byte for byte in range(256)},
        special_tokens={},
    )
    monkeypatch.setattr(tiktoken, "get_encoding", lambda _: byte_encoding)
    split_length, split_overlap = limits
    splitter = RecursiveDocumentSplitter(
        split_length=split_length, split_overlap=split_overlap, split_unit="token", separators=["\n\n", "."]
    )

    chunks = splitter.run([Document(content=text)])["documents"]

    assert [(chunk.content, chunk.meta["split_idx_start"]) for chunk in chunks] == expected
    assert [chunk.meta["split_id"] for chunk in chunks] == list(range(len(expected)))
    if split_overlap > 0:
        for (previous, current), ((previous_text, previous_start), (_, current_start)) in zip(
            pairwise(chunks), pairwise(expected), strict=True
        ):
            overlap = previous_start + len(previous_text) - current_start
            assert previous.meta["_split_overlap"] == [{"doc_id": current.id, "range": (0, overlap)}]
            assert current.meta["_split_overlap"] == [
                {"doc_id": previous.id, "range": (len(previous_text) - overlap, len(previous_text))}
            ]
