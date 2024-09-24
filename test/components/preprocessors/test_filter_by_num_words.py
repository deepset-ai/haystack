# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import logging

import pytest

from haystack import Document
from haystack.components.preprocessors import FilterByNumWords


class TestFilterByNumWords:
    def test_init(self):
        cleaner = FilterByNumWords()
        assert cleaner.max_size == 40000

    def test_run(self):
        cleaner = FilterByNumWords(10)
        documents = [
            Document(content="This is a text with some words."),
            Document(content="This is another text with more words."),
        ]
        result = cleaner.run(documents=documents)
        assert len(result["documents"]) == 1
        assert result["documents"][0].content == "This is a text with some words."
