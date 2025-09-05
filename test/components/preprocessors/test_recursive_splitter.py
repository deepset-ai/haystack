# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import re

import pytest
from pytest import LogCaptureFixture

from haystack import Document, Pipeline
from haystack.components.preprocessors.recursive_splitter import RecursiveDocumentSplitter
from haystack.components.preprocessors.sentence_tokenizer import SentenceSplitter


def test_get_custom_sentence_tokenizer_success():
    tokenizer = RecursiveDocumentSplitter._get_custom_sentence_tokenizer({})
    assert isinstance(tokenizer, SentenceSplitter)


def test_init_with_negative_overlap():
    with pytest.raises(ValueError):
        _ = RecursiveDocumentSplitter(split_length=20, split_overlap=-1, separators=["."])


def test_init_with_overlap_greater_than_chunk_size():
    with pytest.raises(ValueError):
        _ = RecursiveDocumentSplitter(split_length=10, split_overlap=15, separators=["."])


def test_init_with_invalid_separators():
    with pytest.raises(ValueError):
        _ = RecursiveDocumentSplitter(separators=[".", 2])


def test_init_with_negative_split_length():
    with pytest.raises(ValueError):
        _ = RecursiveDocumentSplitter(split_length=-1, separators=["."])


def test_apply_overlap_no_overlap():
    # Test the case where there is no overlap between chunks
    splitter = RecursiveDocumentSplitter(split_length=20, split_overlap=0, separators=["."], split_unit="char")
    chunks = ["chunk1", "chunk2", "chunk3"]
    result = splitter._apply_overlap(chunks)
    assert result == ["chunk1", "chunk2", "chunk3"]


def test_apply_overlap_with_overlap():
    # Test the case where there is overlap between chunks
    splitter = RecursiveDocumentSplitter(split_length=20, split_overlap=4, separators=["."], split_unit="char")
    chunks = ["chunk1", "chunk2", "chunk3"]
    result = splitter._apply_overlap(chunks)
    assert result == ["chunk1", "unk1chunk2", "unk2chunk3"]


def test_apply_overlap_with_overlap_capturing_completely_previous_chunk(caplog):
    splitter = RecursiveDocumentSplitter(split_length=20, split_overlap=6, separators=["."], split_unit="char")
    chunks = ["chunk1", "chunk2", "chunk3", "chunk4"]
    _ = splitter._apply_overlap(chunks)
    assert (
        "Overlap is the same as the previous chunk. Consider increasing the `split_length` parameter or decreasing "
        "the `split_overlap` parameter." in caplog.text
    )


def test_apply_overlap_single_chunk():
    # Test the case where there is only one chunk
    splitter = RecursiveDocumentSplitter(split_length=20, split_overlap=3, separators=["."], split_unit="char")
    chunks = ["chunk1"]
    result = splitter._apply_overlap(chunks)
    assert result == ["chunk1"]


def test_chunk_text_smaller_than_chunk_size():
    splitter = RecursiveDocumentSplitter(split_length=20, split_overlap=0, separators=["."])
    text = "small text"
    chunks = splitter._chunk_text(text)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_chunk_text_by_period():
    splitter = RecursiveDocumentSplitter(split_length=20, split_overlap=0, separators=["."], split_unit="char")
    text = "This is a test. Another sentence. And one more."
    chunks = splitter._chunk_text(text)
    assert len(chunks) == 3
    assert chunks[0] == "This is a test."
    assert chunks[1] == " Another sentence."
    assert chunks[2] == " And one more."


def test_run_multiple_new_lines_unit_char():
    splitter = RecursiveDocumentSplitter(split_length=18, separators=["\n\n", "\n"], split_unit="char")
    text = "This is a test.\n\n\nAnother test.\n\n\n\nFinal test."
    doc = Document(content=text)
    chunks = splitter.run([doc])["documents"]
    assert chunks[0].content == "This is a test.\n\n"
    assert chunks[1].content == "\nAnother test.\n\n\n\n"
    assert chunks[2].content == "Final test."


def test_run_empty_documents(caplog: LogCaptureFixture):
    splitter = RecursiveDocumentSplitter(split_length=20, split_overlap=0, separators=["."])
    empty_doc = Document(content="")
    doc_chunks = splitter.run([empty_doc])
    doc_chunks = doc_chunks["documents"]
    assert len(doc_chunks) == 0
    assert "has an empty content. Skipping this document." in caplog.text


def test_run_using_custom_sentence_tokenizer():
    """
    This test includes abbreviations that are not handled by the simple sentence tokenizer based on "." and requires a
    more sophisticated sentence tokenizer like the one provided by NLTK.
    """
    splitter = RecursiveDocumentSplitter(
        split_length=400,
        split_overlap=0,
        split_unit="char",
        separators=["\n\n", "\n", "sentence", " "],
        sentence_splitter_params={"language": "en", "use_split_rules": True, "keep_white_spaces": False},
    )
    splitter.warm_up()
    text = """Artificial intelligence (AI) - Introduction

AI, in its broadest sense, is intelligence exhibited by machines, particularly computer systems.
AI technology is widely used throughout industry, government, and science. Some high-profile applications include advanced web search engines (e.g., Google Search); recommendation systems (used by YouTube, Amazon, and Netflix); interacting via human speech (e.g., Google Assistant, Siri, and Alexa); autonomous vehicles (e.g., Waymo); generative and creative tools (e.g., ChatGPT and AI art); and superhuman play and analysis in strategy games (e.g., chess and Go)."""  # noqa: E501

    chunks = splitter.run([Document(content=text)])
    chunks = chunks["documents"]

    assert len(chunks) == 4
    assert chunks[0].content == "Artificial intelligence (AI) - Introduction\n\n"
    assert (
        chunks[1].content
        == "AI, in its broadest sense, is intelligence exhibited by machines, particularly computer systems.\n"
    )
    assert chunks[2].content == "AI technology is widely used throughout industry, government, and science."
    assert (
        chunks[3].content
        == "Some high-profile applications include advanced web search engines (e.g., Google Search); recommendation "
        "systems (used by YouTube, Amazon, and Netflix); interacting via human speech (e.g., Google Assistant, "
        "Siri, and Alexa); autonomous vehicles (e.g., Waymo); generative and creative tools (e.g., ChatGPT and "
        "AI art); and superhuman play and analysis in strategy games (e.g., chess and Go)."
    )


def test_run_split_by_dot_count_page_breaks_split_unit_char() -> None:
    document_splitter = RecursiveDocumentSplitter(separators=["."], split_length=30, split_overlap=0, split_unit="char")

    text = (
        "Sentence on page 1. Another on page 1.\fSentence on page 2. Another on page 2.\f"
        "Sentence on page 3. Another on page 3.\f\f Sentence on page 5."
    )

    documents = document_splitter.run(documents=[Document(content=text)])["documents"]

    assert len(documents) == 7
    assert documents[0].content == "Sentence on page 1."
    assert documents[0].meta["page_number"] == 1
    assert documents[0].meta["split_id"] == 0
    assert documents[0].meta["split_idx_start"] == text.index(documents[0].content)

    assert documents[1].content == " Another on page 1."
    assert documents[1].meta["page_number"] == 1
    assert documents[1].meta["split_id"] == 1
    assert documents[1].meta["split_idx_start"] == text.index(documents[1].content)

    assert documents[2].content == "\fSentence on page 2."
    assert documents[2].meta["page_number"] == 2
    assert documents[2].meta["split_id"] == 2
    assert documents[2].meta["split_idx_start"] == text.index(documents[2].content)

    assert documents[3].content == " Another on page 2."
    assert documents[3].meta["page_number"] == 2
    assert documents[3].meta["split_id"] == 3
    assert documents[3].meta["split_idx_start"] == text.index(documents[3].content)

    assert documents[4].content == "\fSentence on page 3."
    assert documents[4].meta["page_number"] == 3
    assert documents[4].meta["split_id"] == 4
    assert documents[4].meta["split_idx_start"] == text.index(documents[4].content)

    assert documents[5].content == " Another on page 3."
    assert documents[5].meta["page_number"] == 3
    assert documents[5].meta["split_id"] == 5
    assert documents[5].meta["split_idx_start"] == text.index(documents[5].content)

    assert documents[6].content == "\f\f Sentence on page 5."
    assert documents[6].meta["page_number"] == 5
    assert documents[6].meta["split_id"] == 6
    assert documents[6].meta["split_idx_start"] == text.index(documents[6].content)


def test_run_split_by_word_count_page_breaks_split_unit_char():
    splitter = RecursiveDocumentSplitter(split_length=19, split_overlap=0, separators=[" "], split_unit="char")
    text = "This is some text. \f This text is on another page. \f This is the last pag3."
    doc = Document(content=text)
    doc_chunks = splitter.run([doc])
    doc_chunks = doc_chunks["documents"]

    assert len(doc_chunks) == 5
    assert doc_chunks[0].content == "This is some text. "
    assert doc_chunks[0].meta["page_number"] == 1
    assert doc_chunks[0].meta["split_id"] == 0
    assert doc_chunks[0].meta["split_idx_start"] == text.index(doc_chunks[0].content)

    assert doc_chunks[1].content == "\f This text is on "
    assert doc_chunks[1].meta["page_number"] == 2
    assert doc_chunks[1].meta["split_id"] == 1
    assert doc_chunks[1].meta["split_idx_start"] == text.index(doc_chunks[1].content)

    assert doc_chunks[2].content == "another page. \f "
    assert doc_chunks[2].meta["page_number"] == 3
    assert doc_chunks[2].meta["split_id"] == 2
    assert doc_chunks[2].meta["split_idx_start"] == text.index(doc_chunks[2].content)

    assert doc_chunks[3].content == "This is the last "
    assert doc_chunks[3].meta["page_number"] == 3
    assert doc_chunks[3].meta["split_id"] == 3
    assert doc_chunks[3].meta["split_idx_start"] == text.index(doc_chunks[3].content)

    assert doc_chunks[4].content == "pag3."
    assert doc_chunks[4].meta["page_number"] == 3
    assert doc_chunks[4].meta["split_id"] == 4
    assert doc_chunks[4].meta["split_idx_start"] == text.index(doc_chunks[4].content)


def test_run_split_by_page_break_count_page_breaks() -> None:
    document_splitter = RecursiveDocumentSplitter(
        separators=["\f"], split_length=50, split_overlap=0, split_unit="char"
    )

    text = (
        "Sentence on page 1. Another on page 1.\fSentence on page 2. Another on page 2.\f"
        "Sentence on page 3. Another on page 3.\f\f Sentence on page 5."
    )

    documents = document_splitter.run(documents=[Document(content=text)])
    chunks_docs = documents["documents"]
    assert len(chunks_docs) == 4
    assert chunks_docs[0].content == "Sentence on page 1. Another on page 1.\f"
    assert chunks_docs[0].meta["page_number"] == 1
    assert chunks_docs[0].meta["split_id"] == 0
    assert chunks_docs[0].meta["split_idx_start"] == text.index(chunks_docs[0].content)

    assert chunks_docs[1].content == "Sentence on page 2. Another on page 2.\f"
    assert chunks_docs[1].meta["page_number"] == 2
    assert chunks_docs[1].meta["split_id"] == 1
    assert chunks_docs[1].meta["split_idx_start"] == text.index(chunks_docs[1].content)

    assert chunks_docs[2].content == "Sentence on page 3. Another on page 3.\f\f"
    assert chunks_docs[2].meta["page_number"] == 3
    assert chunks_docs[2].meta["split_id"] == 2
    assert chunks_docs[2].meta["split_idx_start"] == text.index(chunks_docs[2].content)

    assert chunks_docs[3].content == " Sentence on page 5."
    assert chunks_docs[3].meta["page_number"] == 5
    assert chunks_docs[3].meta["split_id"] == 3
    assert chunks_docs[3].meta["split_idx_start"] == text.index(chunks_docs[3].content)


def test_run_split_by_new_line_count_page_breaks_split_unit_char() -> None:
    document_splitter = RecursiveDocumentSplitter(
        separators=["\n"], split_length=21, split_overlap=0, split_unit="char"
    )

    text = (
        "Sentence on page 1.\nAnother on page 1.\n\f"
        "Sentence on page 2.\nAnother on page 2.\n\f"
        "Sentence on page 3.\nAnother on page 3.\n\f\f"
        "Sentence on page 5."
    )

    documents = document_splitter.run(documents=[Document(content=text)])
    chunks_docs = documents["documents"]

    assert len(chunks_docs) == 7

    assert chunks_docs[0].content == "Sentence on page 1.\n"
    assert chunks_docs[0].meta["page_number"] == 1
    assert chunks_docs[0].meta["split_id"] == 0
    assert chunks_docs[0].meta["split_idx_start"] == text.index(chunks_docs[0].content)

    assert chunks_docs[1].content == "Another on page 1.\n"
    assert chunks_docs[1].meta["page_number"] == 1
    assert chunks_docs[1].meta["split_id"] == 1
    assert chunks_docs[1].meta["split_idx_start"] == text.index(chunks_docs[1].content)

    assert chunks_docs[2].content == "\fSentence on page 2.\n"
    assert chunks_docs[2].meta["page_number"] == 2
    assert chunks_docs[2].meta["split_id"] == 2
    assert chunks_docs[2].meta["split_idx_start"] == text.index(chunks_docs[2].content)

    assert chunks_docs[3].content == "Another on page 2.\n"
    assert chunks_docs[3].meta["page_number"] == 2
    assert chunks_docs[3].meta["split_id"] == 3
    assert chunks_docs[3].meta["split_idx_start"] == text.index(chunks_docs[3].content)

    assert chunks_docs[4].content == "\fSentence on page 3.\n"
    assert chunks_docs[4].meta["page_number"] == 3
    assert chunks_docs[4].meta["split_id"] == 4
    assert chunks_docs[4].meta["split_idx_start"] == text.index(chunks_docs[4].content)

    assert chunks_docs[5].content == "Another on page 3.\n"
    assert chunks_docs[5].meta["page_number"] == 3
    assert chunks_docs[5].meta["split_id"] == 5
    assert chunks_docs[5].meta["split_idx_start"] == text.index(chunks_docs[5].content)

    assert chunks_docs[6].content == "\f\fSentence on page 5."
    assert chunks_docs[6].meta["page_number"] == 5
    assert chunks_docs[6].meta["split_id"] == 6
    assert chunks_docs[6].meta["split_idx_start"] == text.index(chunks_docs[6].content)


def test_run_split_by_sentence_count_page_breaks_split_unit_char() -> None:
    document_splitter = RecursiveDocumentSplitter(
        separators=["sentence"], split_length=28, split_overlap=0, split_unit="char"
    )
    document_splitter.warm_up()

    text = (
        "Sentence on page 1. Another on page 1.\fSentence on page 2. Another on page 2.\f"
        "Sentence on page 3. Another on page 3.\f\fSentence on page 5."
    )

    documents = document_splitter.run(documents=[Document(content=text)])
    chunks_docs = documents["documents"]
    assert len(chunks_docs) == 7

    assert chunks_docs[0].content == "Sentence on page 1. "
    assert chunks_docs[0].meta["page_number"] == 1
    assert chunks_docs[0].meta["split_id"] == 0
    assert chunks_docs[0].meta["split_idx_start"] == text.index(chunks_docs[0].content)

    assert chunks_docs[1].content == "Another on page 1.\f"
    assert chunks_docs[1].meta["page_number"] == 1
    assert chunks_docs[1].meta["split_id"] == 1
    assert chunks_docs[1].meta["split_idx_start"] == text.index(chunks_docs[1].content)

    assert chunks_docs[2].content == "Sentence on page 2. "
    assert chunks_docs[2].meta["page_number"] == 2
    assert chunks_docs[2].meta["split_id"] == 2
    assert chunks_docs[2].meta["split_idx_start"] == text.index(chunks_docs[2].content)

    assert chunks_docs[3].content == "Another on page 2.\f"
    assert chunks_docs[3].meta["page_number"] == 2
    assert chunks_docs[3].meta["split_id"] == 3
    assert chunks_docs[3].meta["split_idx_start"] == text.index(chunks_docs[3].content)

    assert chunks_docs[4].content == "Sentence on page 3. "
    assert chunks_docs[4].meta["page_number"] == 3
    assert chunks_docs[4].meta["split_id"] == 4
    assert chunks_docs[4].meta["split_idx_start"] == text.index(chunks_docs[4].content)

    assert chunks_docs[5].content == "Another on page 3.\f\f"
    assert chunks_docs[5].meta["page_number"] == 3
    assert chunks_docs[5].meta["split_id"] == 5
    assert chunks_docs[5].meta["split_idx_start"] == text.index(chunks_docs[5].content)

    assert chunks_docs[6].content == "Sentence on page 5."
    assert chunks_docs[6].meta["page_number"] == 5
    assert chunks_docs[6].meta["split_id"] == 6
    assert chunks_docs[6].meta["split_idx_start"] == text.index(chunks_docs[6].content)


def test_run_split_document_with_overlap_character_unit():
    splitter = RecursiveDocumentSplitter(split_length=20, split_overlap=10, separators=["."], split_unit="char")
    text = """A simple sentence1. A bright sentence2. A clever sentence3"""

    doc = Document(content=text)
    doc_chunks = splitter.run([doc])
    doc_chunks = doc_chunks["documents"]

    assert len(doc_chunks) == 5
    assert doc_chunks[0].content == "A simple sentence1."
    assert doc_chunks[0].meta["split_id"] == 0
    assert doc_chunks[0].meta["split_idx_start"] == text.index(doc_chunks[0].content)
    assert doc_chunks[0].meta["_split_overlap"] == [{"doc_id": doc_chunks[1].id, "range": (0, 10)}]

    assert doc_chunks[1].content == "sentence1. A bright "
    assert doc_chunks[1].meta["split_id"] == 1
    assert doc_chunks[1].meta["split_idx_start"] == text.index(doc_chunks[1].content)
    assert doc_chunks[1].meta["_split_overlap"] == [
        {"doc_id": doc_chunks[0].id, "range": (9, 19)},
        {"doc_id": doc_chunks[2].id, "range": (0, 10)},
    ]

    assert doc_chunks[2].content == " A bright sentence2."
    assert doc_chunks[2].meta["split_id"] == 2
    assert doc_chunks[2].meta["split_idx_start"] == text.index(doc_chunks[2].content)
    assert doc_chunks[2].meta["_split_overlap"] == [
        {"doc_id": doc_chunks[1].id, "range": (10, 20)},
        {"doc_id": doc_chunks[3].id, "range": (0, 10)},
    ]

    assert doc_chunks[3].content == "sentence2. A clever "
    assert doc_chunks[3].meta["split_id"] == 3
    assert doc_chunks[3].meta["split_idx_start"] == text.index(doc_chunks[3].content)
    assert doc_chunks[3].meta["_split_overlap"] == [
        {"doc_id": doc_chunks[2].id, "range": (10, 20)},
        {"doc_id": doc_chunks[4].id, "range": (0, 10)},
    ]

    assert doc_chunks[4].content == " A clever sentence3"
    assert doc_chunks[4].meta["split_id"] == 4
    assert doc_chunks[4].meta["split_idx_start"] == text.index(doc_chunks[4].content)
    assert doc_chunks[4].meta["_split_overlap"] == [{"doc_id": doc_chunks[3].id, "range": (10, 20)}]


def test_run_split_document_with_overlap_and_fallback_character_unit():
    splitter = RecursiveDocumentSplitter(split_length=8, split_overlap=4, separators=["."], split_unit="char")
    text = "A simple sentence1. Short. Short."

    doc = Document(content=text)
    doc_chunks = splitter.run([doc])
    doc_chunks = doc_chunks["documents"]

    assert len(doc_chunks) == 8
    assert doc_chunks[0].content == "A simple"
    assert doc_chunks[1].content == "mple sen"
    assert doc_chunks[2].content == " sentenc"
    assert doc_chunks[3].content == "tence1. "
    assert doc_chunks[4].content == "e1. Shor"
    assert doc_chunks[5].content == "Short. S"
    assert doc_chunks[6].content == "t. Short"
    assert doc_chunks[7].content == "hort."


def test_run_separator_exists_but_split_length_too_small_fall_back_to_character_chunking():
    splitter = RecursiveDocumentSplitter(separators=[" "], split_length=2, split_unit="char")
    doc = Document(content="This is some text")
    result = splitter.run(documents=[doc])
    assert len(result["documents"]) == 10
    for doc in result["documents"]:
        if re.escape(doc.content) not in ["\\ "]:
            assert len(doc.content) == 2


def test_run_fallback_to_character_chunking_by_default_length_too_short():
    text = "abczdefzghizjkl"
    separators = ["\n\n", "\n", "z"]
    splitter = RecursiveDocumentSplitter(split_length=2, separators=separators, split_unit="char")
    doc = Document(content=text)
    chunks = splitter.run([doc])["documents"]
    for chunk in chunks:
        assert len(chunk.content) <= 2


def test_run_fallback_to_word_chunking_by_default_length_too_short():
    text = "This is some text. This is some more text, and even more text."
    separators = ["\n\n", "\n", "."]
    splitter = RecursiveDocumentSplitter(split_length=2, separators=separators, split_unit="word")
    doc = Document(content=text)
    chunks = splitter.run([doc])["documents"]
    for chunk in chunks:
        assert splitter._chunk_length(chunk.content) <= 2


def test_run_custom_sentence_tokenizer_document_and_overlap_char_unit():
    """Test that RecursiveDocumentSplitter works correctly with custom sentence tokenizer and overlap"""
    splitter = RecursiveDocumentSplitter(split_length=25, split_overlap=10, separators=["sentence"], split_unit="char")
    text = "This is sentence one. This is sentence two. This is sentence three."

    splitter.warm_up()
    doc = Document(content=text)
    doc_chunks = splitter.run([doc])["documents"]

    assert len(doc_chunks) == 4
    assert doc_chunks[0].content == "This is sentence one. "
    assert doc_chunks[0].meta["split_id"] == 0
    assert doc_chunks[0].meta["split_idx_start"] == text.index(doc_chunks[0].content)
    assert doc_chunks[0].meta["_split_overlap"] == [{"doc_id": doc_chunks[1].id, "range": (0, 10)}]

    assert doc_chunks[1].content == "ence one. This is sentenc"
    assert doc_chunks[1].meta["split_id"] == 1
    assert doc_chunks[1].meta["split_idx_start"] == text.index(doc_chunks[1].content)
    assert doc_chunks[1].meta["_split_overlap"] == [
        {"doc_id": doc_chunks[0].id, "range": (12, 22)},
        {"doc_id": doc_chunks[2].id, "range": (0, 10)},
    ]

    assert doc_chunks[2].content == "is sentence two. This is "
    assert doc_chunks[2].meta["split_id"] == 2
    assert doc_chunks[2].meta["split_idx_start"] == text.index(doc_chunks[2].content)
    assert doc_chunks[2].meta["_split_overlap"] == [
        {"doc_id": doc_chunks[1].id, "range": (15, 25)},
        {"doc_id": doc_chunks[3].id, "range": (0, 10)},
    ]

    assert doc_chunks[3].content == ". This is sentence three."
    assert doc_chunks[3].meta["split_id"] == 3
    assert doc_chunks[3].meta["split_idx_start"] == text.index(doc_chunks[3].content)
    assert doc_chunks[3].meta["_split_overlap"] == [{"doc_id": doc_chunks[2].id, "range": (15, 25)}]


def test_run_split_by_dot_count_page_breaks_word_unit() -> None:
    document_splitter = RecursiveDocumentSplitter(separators=["."], split_length=4, split_overlap=0, split_unit="word")

    text = (
        "Sentence on page 1. Another on page 1.\fSentence on page 2. Another on page 2.\f"
        "Sentence on page 3. Another on page 3.\f\f Sentence on page 5."
    )

    documents = document_splitter.run(documents=[Document(content=text)])["documents"]

    assert len(documents) == 8
    assert documents[0].content == "Sentence on page 1."
    assert documents[0].meta["page_number"] == 1
    assert documents[0].meta["split_id"] == 0
    assert documents[0].meta["split_idx_start"] == text.index(documents[0].content)

    assert documents[1].content == " Another on page 1."
    assert documents[1].meta["page_number"] == 1
    assert documents[1].meta["split_id"] == 1
    assert documents[1].meta["split_idx_start"] == text.index(documents[1].content)

    assert documents[2].content == "\fSentence on page 2."
    assert documents[2].meta["page_number"] == 2
    assert documents[2].meta["split_id"] == 2
    assert documents[2].meta["split_idx_start"] == text.index(documents[2].content)

    assert documents[3].content == " Another on page 2."
    assert documents[3].meta["page_number"] == 2
    assert documents[3].meta["split_id"] == 3
    assert documents[3].meta["split_idx_start"] == text.index(documents[3].content)

    assert documents[4].content == "\fSentence on page 3."
    assert documents[4].meta["page_number"] == 3
    assert documents[4].meta["split_id"] == 4
    assert documents[4].meta["split_idx_start"] == text.index(documents[4].content)

    assert documents[5].content == " Another on page 3."
    assert documents[5].meta["page_number"] == 3
    assert documents[5].meta["split_id"] == 5
    assert documents[5].meta["split_idx_start"] == text.index(documents[5].content)

    assert documents[6].content == "\f\f Sentence on page"
    assert documents[6].meta["page_number"] == 5
    assert documents[6].meta["split_id"] == 6
    assert documents[6].meta["split_idx_start"] == text.index(documents[6].content)

    assert documents[7].content == " 5."
    assert documents[7].meta["page_number"] == 5
    assert documents[7].meta["split_id"] == 7
    assert documents[7].meta["split_idx_start"] == text.index(documents[7].content)


def test_run_split_by_word_count_page_breaks_word_unit():
    splitter = RecursiveDocumentSplitter(split_length=4, split_overlap=0, separators=[" "], split_unit="word")
    text = "This is some text. \f This text is on another page. \f This is the last pag3."
    doc = Document(content=text)
    doc_chunks = splitter.run([doc])
    doc_chunks = doc_chunks["documents"]

    assert len(doc_chunks) == 5
    assert doc_chunks[0].content == "This is some text. "
    assert doc_chunks[0].meta["page_number"] == 1
    assert doc_chunks[0].meta["split_id"] == 0
    assert doc_chunks[0].meta["split_idx_start"] == text.index(doc_chunks[0].content)

    assert doc_chunks[1].content == "\f This text is "
    assert doc_chunks[1].meta["page_number"] == 2
    assert doc_chunks[1].meta["split_id"] == 1
    assert doc_chunks[1].meta["split_idx_start"] == text.index(doc_chunks[1].content)

    assert doc_chunks[2].content == "on another page. \f "
    assert doc_chunks[2].meta["page_number"] == 3
    assert doc_chunks[2].meta["split_id"] == 2
    assert doc_chunks[2].meta["split_idx_start"] == text.index(doc_chunks[2].content)

    assert doc_chunks[3].content == "This is the last "
    assert doc_chunks[3].meta["page_number"] == 3
    assert doc_chunks[3].meta["split_id"] == 3
    assert doc_chunks[3].meta["split_idx_start"] == text.index(doc_chunks[3].content)

    assert doc_chunks[4].content == "pag3."
    assert doc_chunks[4].meta["page_number"] == 3
    assert doc_chunks[4].meta["split_id"] == 4
    assert doc_chunks[4].meta["split_idx_start"] == text.index(doc_chunks[4].content)


def test_run_split_by_page_break_count_page_breaks_word_unit() -> None:
    document_splitter = RecursiveDocumentSplitter(separators=["\f"], split_length=8, split_overlap=0, split_unit="word")

    text = (
        "Sentence on page 1. Another on page 1.\fSentence on page 2. Another on page 2.\f"
        "Sentence on page 3. Another on page 3.\f\f Sentence on page 5."
    )

    documents = document_splitter.run(documents=[Document(content=text)])
    chunks_docs = documents["documents"]

    assert len(chunks_docs) == 4
    assert chunks_docs[0].content == "Sentence on page 1. Another on page 1.\f"
    assert chunks_docs[0].meta["page_number"] == 1
    assert chunks_docs[0].meta["split_id"] == 0
    assert chunks_docs[0].meta["split_idx_start"] == text.index(chunks_docs[0].content)

    assert chunks_docs[1].content == "Sentence on page 2. Another on page 2.\f"
    assert chunks_docs[1].meta["page_number"] == 2
    assert chunks_docs[1].meta["split_id"] == 1
    assert chunks_docs[1].meta["split_idx_start"] == text.index(chunks_docs[1].content)

    assert chunks_docs[2].content == "Sentence on page 3. Another on page 3.\f"
    assert chunks_docs[2].meta["page_number"] == 3
    assert chunks_docs[2].meta["split_id"] == 2
    assert chunks_docs[2].meta["split_idx_start"] == text.index(chunks_docs[2].content)

    assert chunks_docs[3].content == "\f Sentence on page 5."
    assert chunks_docs[3].meta["page_number"] == 5
    assert chunks_docs[3].meta["split_id"] == 3
    assert chunks_docs[3].meta["split_idx_start"] == text.index(chunks_docs[3].content)


def test_run_split_by_new_line_count_page_breaks_word_unit() -> None:
    document_splitter = RecursiveDocumentSplitter(separators=["\n"], split_length=4, split_overlap=0, split_unit="word")

    text = (
        "Sentence on page 1.\nAnother on page 1.\n\f"
        "Sentence on page 2.\nAnother on page 2.\n\f"
        "Sentence on page 3.\nAnother on page 3.\n\f\f"
        "Sentence on page 5."
    )

    documents = document_splitter.run(documents=[Document(content=text)])
    chunks_docs = documents["documents"]

    assert len(chunks_docs) == 7

    assert chunks_docs[0].content == "Sentence on page 1.\n"
    assert chunks_docs[0].meta["page_number"] == 1
    assert chunks_docs[0].meta["split_id"] == 0
    assert chunks_docs[0].meta["split_idx_start"] == text.index(chunks_docs[0].content)

    assert chunks_docs[1].content == "Another on page 1.\n"
    assert chunks_docs[1].meta["page_number"] == 1
    assert chunks_docs[1].meta["split_id"] == 1
    assert chunks_docs[1].meta["split_idx_start"] == text.index(chunks_docs[1].content)

    assert chunks_docs[2].content == "\fSentence on page 2.\n"
    assert chunks_docs[2].meta["page_number"] == 2
    assert chunks_docs[2].meta["split_id"] == 2
    assert chunks_docs[2].meta["split_idx_start"] == text.index(chunks_docs[2].content)

    assert chunks_docs[3].content == "Another on page 2.\n"
    assert chunks_docs[3].meta["page_number"] == 2
    assert chunks_docs[3].meta["split_id"] == 3
    assert chunks_docs[3].meta["split_idx_start"] == text.index(chunks_docs[3].content)

    assert chunks_docs[4].content == "\fSentence on page 3.\n"
    assert chunks_docs[4].meta["page_number"] == 3
    assert chunks_docs[4].meta["split_id"] == 4
    assert chunks_docs[4].meta["split_idx_start"] == text.index(chunks_docs[4].content)

    assert chunks_docs[5].content == "Another on page 3.\n"
    assert chunks_docs[5].meta["page_number"] == 3
    assert chunks_docs[5].meta["split_id"] == 5
    assert chunks_docs[5].meta["split_idx_start"] == text.index(chunks_docs[5].content)

    assert chunks_docs[6].content == "\f\fSentence on page 5."
    assert chunks_docs[6].meta["page_number"] == 5
    assert chunks_docs[6].meta["split_id"] == 6
    assert chunks_docs[6].meta["split_idx_start"] == text.index(chunks_docs[6].content)


def test_run_split_by_sentence_count_page_breaks_word_unit() -> None:
    document_splitter = RecursiveDocumentSplitter(
        separators=["sentence"], split_length=7, split_overlap=0, split_unit="word"
    )
    document_splitter.warm_up()

    text = (
        "Sentence on page 1. Another on page 1.\fSentence on page 2. Another on page 2.\f"
        "Sentence on page 3. Another on page 3.\f\fSentence on page 5."
    )

    documents = document_splitter.run(documents=[Document(content=text)])
    chunks_docs = documents["documents"]
    assert len(chunks_docs) == 7

    assert chunks_docs[0].content == "Sentence on page 1. "
    assert chunks_docs[0].meta["page_number"] == 1
    assert chunks_docs[0].meta["split_id"] == 0
    assert chunks_docs[0].meta["split_idx_start"] == text.index(chunks_docs[0].content)

    assert chunks_docs[1].content == "Another on page 1.\f"
    assert chunks_docs[1].meta["page_number"] == 1
    assert chunks_docs[1].meta["split_id"] == 1
    assert chunks_docs[1].meta["split_idx_start"] == text.index(chunks_docs[1].content)

    assert chunks_docs[2].content == "Sentence on page 2. "
    assert chunks_docs[2].meta["page_number"] == 2
    assert chunks_docs[2].meta["split_id"] == 2
    assert chunks_docs[2].meta["split_idx_start"] == text.index(chunks_docs[2].content)

    assert chunks_docs[3].content == "Another on page 2.\f"
    assert chunks_docs[3].meta["page_number"] == 2
    assert chunks_docs[3].meta["split_id"] == 3
    assert chunks_docs[3].meta["split_idx_start"] == text.index(chunks_docs[3].content)

    assert chunks_docs[4].content == "Sentence on page 3. "
    assert chunks_docs[4].meta["page_number"] == 3
    assert chunks_docs[4].meta["split_id"] == 4
    assert chunks_docs[4].meta["split_idx_start"] == text.index(chunks_docs[4].content)

    assert chunks_docs[5].content == "Another on page 3.\f\f"
    assert chunks_docs[5].meta["page_number"] == 3
    assert chunks_docs[5].meta["split_id"] == 5
    assert chunks_docs[5].meta["split_idx_start"] == text.index(chunks_docs[5].content)

    assert chunks_docs[6].content == "Sentence on page 5."
    assert chunks_docs[6].meta["page_number"] == 5
    assert chunks_docs[6].meta["split_id"] == 6
    assert chunks_docs[6].meta["split_idx_start"] == text.index(chunks_docs[6].content)


def test_run_split_by_sentence_tokenizer_document_and_overlap_word_unit_no_overlap():
    splitter = RecursiveDocumentSplitter(split_length=4, split_overlap=0, separators=["."], split_unit="word")
    text = "This is sentence one. This is sentence two. This is sentence three."
    chunks = splitter.run([Document(content=text)])["documents"]
    assert len(chunks) == 3
    assert chunks[0].content == "This is sentence one."
    assert chunks[1].content == " This is sentence two."
    assert chunks[2].content == " This is sentence three."


def test_run_split_by_dot_and_overlap_1_word_unit():
    splitter = RecursiveDocumentSplitter(split_length=4, split_overlap=1, separators=["."], split_unit="word")
    text = "This is sentence one. This is sentence two. This is sentence three. This is sentence four."
    chunks = splitter.run([Document(content=text)])["documents"]
    assert len(chunks) == 5
    assert chunks[0].content == "This is sentence one."
    assert chunks[1].content == "one. This is sentence"
    assert chunks[2].content == "sentence two. This is"
    assert chunks[3].content == "is sentence three. This"
    assert chunks[4].content == "This is sentence four."


def test_run_trigger_dealing_with_remaining_word_larger_than_split_length():
    splitter = RecursiveDocumentSplitter(split_length=3, split_overlap=2, separators=["."], split_unit="word")
    text = """A simple sentence1. A bright sentence2. A clever sentence3"""
    doc = Document(content=text)
    chunks = splitter.run([doc])["documents"]
    assert len(chunks) == 7
    assert chunks[0].content == "A simple sentence1."
    assert chunks[1].content == "simple sentence1. A"
    assert chunks[2].content == "sentence1. A bright"
    assert chunks[3].content == "A bright sentence2."
    assert chunks[4].content == "bright sentence2. A"
    assert chunks[5].content == "sentence2. A clever"
    assert chunks[6].content == "A clever sentence3"


def test_run_trigger_dealing_with_remaining_char_larger_than_split_length():
    splitter = RecursiveDocumentSplitter(split_length=20, split_overlap=15, separators=["."], split_unit="char")
    text = """A simple sentence1. A bright sentence2. A clever sentence3"""
    doc = Document(content=text)
    chunks = splitter.run([doc])["documents"]

    assert len(chunks) == 9

    assert chunks[0].content == "A simple sentence1."
    assert chunks[0].meta["split_id"] == 0
    assert chunks[0].meta["split_idx_start"] == text.index(chunks[0].content)
    assert chunks[0].meta["_split_overlap"] == [{"doc_id": chunks[1].id, "range": (0, 15)}]

    assert chunks[1].content == "mple sentence1. A br"
    assert chunks[1].meta["split_id"] == 1
    assert chunks[1].meta["split_idx_start"] == text.index(chunks[1].content)
    assert chunks[1].meta["_split_overlap"] == [
        {"doc_id": chunks[0].id, "range": (4, 19)},
        {"doc_id": chunks[2].id, "range": (0, 15)},
    ]

    assert chunks[2].content == "sentence1. A bright "
    assert chunks[2].meta["split_id"] == 2
    assert chunks[2].meta["split_idx_start"] == text.index(chunks[2].content)
    assert chunks[2].meta["_split_overlap"] == [
        {"doc_id": chunks[1].id, "range": (5, 20)},
        {"doc_id": chunks[3].id, "range": (0, 15)},
    ]

    assert chunks[3].content == "nce1. A bright sente"
    assert chunks[3].meta["split_id"] == 3
    assert chunks[3].meta["split_idx_start"] == text.index(chunks[3].content)
    assert chunks[3].meta["_split_overlap"] == [
        {"doc_id": chunks[2].id, "range": (5, 20)},
        {"doc_id": chunks[4].id, "range": (0, 15)},
    ]

    assert chunks[4].content == " A bright sentence2."
    assert chunks[4].meta["split_id"] == 4
    assert chunks[4].meta["split_idx_start"] == text.index(chunks[4].content)
    assert chunks[4].meta["_split_overlap"] == [
        {"doc_id": chunks[3].id, "range": (5, 20)},
        {"doc_id": chunks[5].id, "range": (0, 15)},
    ]

    assert chunks[5].content == "ight sentence2. A cl"
    assert chunks[5].meta["split_id"] == 5
    assert chunks[5].meta["split_idx_start"] == text.index(chunks[5].content)
    assert chunks[5].meta["_split_overlap"] == [
        {"doc_id": chunks[4].id, "range": (5, 20)},
        {"doc_id": chunks[6].id, "range": (0, 15)},
    ]

    assert chunks[6].content == "sentence2. A clever "
    assert chunks[6].meta["split_id"] == 6
    assert chunks[6].meta["split_idx_start"] == text.index(chunks[6].content)
    assert chunks[6].meta["_split_overlap"] == [
        {"doc_id": chunks[5].id, "range": (5, 20)},
        {"doc_id": chunks[7].id, "range": (0, 15)},
    ]

    assert chunks[7].content == "nce2. A clever sente"
    assert chunks[7].meta["split_id"] == 7
    assert chunks[7].meta["split_idx_start"] == text.index(chunks[7].content)
    assert chunks[7].meta["_split_overlap"] == [
        {"doc_id": chunks[6].id, "range": (5, 20)},
        {"doc_id": chunks[8].id, "range": (0, 15)},
    ]

    assert chunks[8].content == " A clever sentence3"
    assert chunks[8].meta["split_id"] == 8
    assert chunks[8].meta["split_idx_start"] == text.index(chunks[8].content)
    assert chunks[8].meta["_split_overlap"] == [{"doc_id": chunks[7].id, "range": (5, 20)}]


def test_run_custom_split_by_dot_and_overlap_3_char_unit():
    document_splitter = RecursiveDocumentSplitter(separators=["."], split_length=4, split_overlap=0, split_unit="word")
    text = "\x0c\x0c Sentence on page 5."
    chunks = document_splitter._fall_back_to_fixed_chunking(text, split_units="word")
    assert len(chunks) == 2
    assert chunks[0] == "\x0c\x0c Sentence on page"
    assert chunks[1] == " 5."


def test_run_serialization_in_pipeline():
    pipeline = Pipeline()
    pipeline.add_component("chunker", RecursiveDocumentSplitter(split_length=20, split_overlap=5, separators=["."]))
    pipeline_dict = pipeline.dumps()
    new_pipeline = Pipeline.loads(pipeline_dict)
    assert pipeline_dict == new_pipeline.dumps()


@pytest.mark.integration
def test_run_split_by_token_count():
    splitter = RecursiveDocumentSplitter(split_length=5, separators=["."], split_unit="token")
    splitter.warm_up()

    text = "This is a test. This is another test. This is the final test."
    doc = Document(content=text)
    chunks = splitter.run([doc])["documents"]

    assert len(chunks) == 4
    assert chunks[0].content == "This is a test."
    assert chunks[1].content == " This is another test."
    assert chunks[2].content == " This is the final test"
    assert chunks[3].content == "."


@pytest.mark.integration
def test_run_split_by_token_count_with_html_tags():
    splitter = RecursiveDocumentSplitter(split_length=4, separators=["."], split_unit="token")
    splitter.warm_up()

    text = "This is a test. <h><a><y><s><t><a><c><k><c><o><r><e> This is the final test."
    doc = Document(content=text)
    chunks = splitter.run([doc])["documents"]

    assert len(chunks) == 10


@pytest.mark.integration
def test_run_split_by_token_with_sentence_tokenizer():
    splitter = RecursiveDocumentSplitter(split_length=4, separators=["sentence"], split_unit="token")
    splitter.warm_up()

    text = "This is sentence one. This is sentence two."
    doc = Document(content=text)
    chunks = splitter.run([doc])["documents"]

    assert len(chunks) == 4
    assert chunks[0].content == "This is sentence one"
    assert chunks[1].content == ". "
    assert chunks[2].content == "This is sentence two"
    assert chunks[3].content == "."


@pytest.mark.integration
def test_run_split_by_token_with_empty_document(caplog: LogCaptureFixture):
    splitter = RecursiveDocumentSplitter(split_length=4, separators=["."], split_unit="token")
    splitter.warm_up()

    empty_doc = Document(content="")
    doc_chunks = splitter.run([empty_doc])
    doc_chunks = doc_chunks["documents"]

    assert len(doc_chunks) == 0
    assert "has an empty content. Skipping this document." in caplog.text


@pytest.mark.integration
def test_run_split_by_token_with_fallback():
    splitter = RecursiveDocumentSplitter(split_length=2, separators=["."], split_unit="token")
    splitter.warm_up()

    text = "This is a very long sentence that will need to be split into smaller chunks."
    doc = Document(content=text)
    chunks = splitter.run([doc])["documents"]

    assert len(chunks) > 1
    for chunk in chunks:
        assert splitter._chunk_length(chunk.content) <= 2


@pytest.mark.integration
def test_run_split_by_token_with_overlap_and_fallback():
    splitter = RecursiveDocumentSplitter(split_length=4, split_overlap=2, separators=["."], split_unit="token")
    splitter.warm_up()

    text = "This is a test. This is another test. This is the final test."
    doc = Document(content=text)
    chunks = splitter.run([doc])["documents"]

    assert len(chunks) == 8
    assert chunks[0].content == "This is a test"
    assert chunks[1].content == " a test."
    assert chunks[2].content == " test. This is"
    assert chunks[3].content == " This is another test"
    assert chunks[4].content == " another test. This"
    assert chunks[5].content == ". This is the"
    assert chunks[6].content == " is the final test"
    assert chunks[7].content == " final test."


def test_run_without_warm_up_raises_error():
    docs = [Document(content="text")]

    splitter_token = RecursiveDocumentSplitter(split_unit="token")
    with pytest.raises(RuntimeError, match="warmed up"):
        splitter_token.run(docs)

    splitter_sentence = RecursiveDocumentSplitter(separators=["sentence"])
    with pytest.raises(RuntimeError, match="warmed up"):
        splitter_sentence.run(docs)

    # Test that no error is raised when warm_up is not required
    splitter_no_warmup = RecursiveDocumentSplitter(separators=["."])
    result = splitter_no_warmup.run(docs)
    assert len(result["documents"]) == 1
    assert result["documents"][0].content == "text"


def test_run_complex_text_with_multiple_separators():
    """
    Test that RecursiveDocumentSplitter correctly handles complex text with multiple separators and chunks that exceed
    the split_length.
    """
    # Create a complex text with multiple separators and chunks of different sizes
    long_text = (
        "A" * 150
        + "\n\n"  # triggers first-level split on \n\n
        + "B" * 100
        + "\n"
        + "B" * 105
        + "\n\n"  # this chunk exceeds split_length and goes through recursion
        + "C" * 100
        + "\n\n"  # short chunk1
        + "D" * 50  # short chunk2
    )

    doc = Document(content=long_text)
    splitter = RecursiveDocumentSplitter(
        split_length=200, split_overlap=0, split_unit="char", separators=["\n\n", "\n", " "]
    )
    splitter.warm_up()
    result = splitter.run([doc])
    chunks = result["documents"]

    assert len(chunks) == 4

    assert len(chunks[0].content) == 152
    assert chunks[0].content.startswith("A")

    assert len(chunks[1].content) == 101
    assert chunks[1].content.startswith("B")

    assert len(chunks[2].content) == 107
    assert chunks[2].content.startswith("B")

    assert len(chunks[3].content) == 152
    assert chunks[3].content.startswith("C")
    assert chunks[3].content.endswith("D" * 50)


def test_recursive_splitter_generates_unique_ids_and_correct_meta():
    text = "Haystack is awesome. " * 5
    source_doc = Document(content=text)

    splitter = RecursiveDocumentSplitter(split_length=3)
    splitter.warm_up()

    chunks = splitter.run([source_doc])["documents"]

    # IDs must be unique
    assert len({c.id for c in chunks}) == len(chunks)

    # parent_id and split_id checks
    for idx, chunk in enumerate(chunks):
        assert chunk.meta["parent_id"] == source_doc.id
        assert chunk.meta["split_id"] == idx
