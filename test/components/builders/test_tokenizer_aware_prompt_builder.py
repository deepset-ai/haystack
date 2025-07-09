# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from unittest.mock import MagicMock

from haystack.components.builders.tokenizer_aware_prompt_builder import TokenizerAwarePromptBuilder
from haystack.dataclasses import Document

class TestTokenizerAwarePromptBuilder:
    def test_run_with_documents_truncation(self):
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.side_effect = lambda x: list(range(len(x.split())))
        mock_tokenizer.decode.side_effect = lambda x: " ".join([str(i) for i in x])

        template = "Question: {{query}}\nDocuments:\n{% for doc in documents %}{{ doc.content }}\n{% endfor %}"
        builder = TokenizerAwarePromptBuilder(template=template, tokenizer=mock_tokenizer, max_length=20)

        documents = [
            Document(content="This is a long document that should be truncated"),
            Document(content="This is another document"),
        ]
        query = "What is the answer?"

        result = builder.run(query=query, documents=documents)

        # Expected behavior: The first document should be truncated to fit within the max_length.
        # The prompt without documents: "Question: What is the answer?\nDocuments:\n"
        # Length of "Question: What is the answer?\nDocuments:\n" is 5 tokens (based on mock tokenizer)
        # Remaining tokens for documents = 20 - 5 = 15
        # First document: "This is a long document that should be truncated" (8 tokens)
        # It should be truncated to 15 tokens. Since the mock tokenizer returns tokens based on word count, it will be truncated to 15 words.
        # The mock tokenizer's decode will return a string of numbers, so we need to check for that.
        
        # Let's re-evaluate the expected truncation based on the mock tokenizer's behavior.
        # mock_tokenizer.encode("word1 word2") -> [0, 1]
        # mock_tokenizer.decode([0, 1]) -> "0 1"

        # Prompt without docs: "Question: What is the answer?\nDocuments:\n"
        # mock_tokenizer.encode("Question: What is the answer?\nDocuments:\n") -> [0, 1, 2, 3, 4, 5] (length 6)
        # Remaining for docs = 11 - 6 = 5
        # Document 1: "This is a long document that should be truncated" (8 words/tokens)
        # Document 2: "This is another document" (4 words/tokens)

        # If we include the first document fully, 6 + 8 = 14 tokens used. Remaining = 6.
        # If we include the second document fully, 14 + 4 = 18 tokens used. Remaining = 2.
        # This means both documents should fit.

        # Let's adjust the max_length to force truncation of the first document.
        # Max length = 11
        # Prompt without docs = 6 tokens (based on mock tokenizer)
        # Remaining for docs = 11 - 6 = 5
        # First document is 8 tokens, so it should be truncated to 5 tokens.
        # Second document should not be included.

        builder_truncated = TokenizerAwarePromptBuilder(template=template, tokenizer=mock_tokenizer, max_length=11)
        result_truncated = builder_truncated.run(query=query, documents=documents)

        expected_truncated_content = "0 1 2 3 4"
        assert expected_truncated_content in result_truncated["prompt"]
        assert "This is another document" not in result_truncated["prompt"]

    def test_run_no_documents(self):
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.side_effect = lambda x: list(range(len(x.split())))
        mock_tokenizer.decode.side_effect = lambda x: " ".join([str(i) for i in x])

        template = "Question: {{query}}"
        builder = TokenizerAwarePromptBuilder(template=template, tokenizer=mock_tokenizer, max_length=10)

        query = "What is the answer?"
        result = builder.run(query=query)

        assert result["prompt"] == "Question: What is the answer?"

    def test_run_documents_fit(self):
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.side_effect = lambda x: list(range(len(x.split())))
        mock_tokenizer.decode.side_effect = lambda x: " ".join([str(i) for i in x])

        template = "Question: {{query}}\nDocuments:\n{% for doc in documents %}{{ doc.content }}\n{% endfor %}"
        builder = TokenizerAwarePromptBuilder(template=template, tokenizer=mock_tokenizer, max_length=50)

        documents = [
            Document(content="This is a short document"),
            Document(content="This is another short document"),
        ]
        query = "What is the answer?"

        result = builder.run(query=query, documents=documents)

        assert "This is a short document" in result["prompt"]
        assert "This is another short document" in result["prompt"]
