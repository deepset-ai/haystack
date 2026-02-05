# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from haystack import Document
from haystack.utils.misc import _deduplicate_documents


def test_deduplicate_documents_keeps_highest_score():
    documents = [
        Document(id="duplicate", content="keep me", score=0.9),
        Document(id="duplicate", content="drop me", score=0.1),
        Document(id="unique", content="unique"),
    ]

    result = _deduplicate_documents(documents)

    assert len(result) == 2
    assert result[0].content == "keep me"
    assert result[1].content == "unique"


def test_deduplicate_documents_keeps_first_when_scores_missing():
    documents = [Document(id="duplicate", content="first"), Document(id="duplicate", content="second")]

    result = _deduplicate_documents(documents)

    assert len(result) == 1
    assert result[0].content == "first"
