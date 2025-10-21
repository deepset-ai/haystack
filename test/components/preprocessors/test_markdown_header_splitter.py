# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import ANY

import pytest

from haystack import Document
from haystack.components.preprocessors.markdown_header_splitter import MarkdownHeaderSplitter


# Fixtures
@pytest.fixture
def sample_text():
    return (
        "# Header 1\n"
        "Content under header 1.\n"
        "## Header 1.1\n"
        "### Subheader 1.1.1\n"
        "Content under sub-header 1.1.1\n"
        "## Header 1.2\n"
        "### Subheader 1.2.1\n"
        "Content under header 1.2.1.\n"
        "### Subheader 1.2.2\n"
        "Content under header 1.2.2.\n"
        "### Subheader 1.2.3\n"
        "Content under header 1.2.3."
    )


# Basic splitting and structure
def test_basic_split(sample_text):
    splitter = MarkdownHeaderSplitter(keep_headers=False)
    docs = [Document(content=sample_text)]
    result = splitter.run(documents=docs)
    split_docs = result["documents"]

    # Should split into all headers with content
    headers = [doc.meta["header"] for doc in split_docs]
    assert "Header 1" in headers
    assert "Subheader 1.1.1" in headers
    assert "Subheader 1.2.1" in headers
    assert "Subheader 1.2.2" in headers
    assert "Subheader 1.2.3" in headers

    # Check that content is present and correct
    header1_doc = next(doc for doc in split_docs if doc.meta["header"] == "Header 1")
    assert "Content under header 1." in header1_doc.content

    subheader111_doc = next(doc for doc in split_docs if doc.meta["header"] == "Subheader 1.1.1")
    assert "Content under sub-header 1.1.1" in subheader111_doc.content

    subheader121_doc = next(doc for doc in split_docs if doc.meta["header"] == "Subheader 1.2.1")
    assert "Content under header 1.2.1." in subheader121_doc.content

    subheader122_doc = next(doc for doc in split_docs if doc.meta["header"] == "Subheader 1.2.2")
    assert "Content under header 1.2.2." in subheader122_doc.content

    subheader123_doc = next(doc for doc in split_docs if doc.meta["header"] == "Subheader 1.2.3")
    assert "Content under header 1.2.3." in subheader123_doc.content


def test_split_parentheaders(sample_text):
    splitter = MarkdownHeaderSplitter(keep_headers=False)
    docs = [Document(content=sample_text), Document(content="# H1\n## H2\n### H3\nContent")]
    result = splitter.run(documents=docs)
    split_docs = result["documents"]
    # Check parentheaders for both a deep subheader and a simple one
    subheader_doc = next(doc for doc in split_docs if doc.meta["header"] == "Subheader 1.2.2")
    assert "Header 1" in subheader_doc.meta["parent_headers"]
    assert "Header 1.2" in subheader_doc.meta["parent_headers"]
    h3_doc = next((doc for doc in split_docs if doc.meta["header"] == "H3"), None)
    assert h3_doc.meta["parent_headers"] == ["H1", "H2"]


def test_split_no_headers():
    splitter = MarkdownHeaderSplitter()
    docs = [Document(content="No headers here."), Document(content="Just some text without headers.")]
    result = splitter.run(documents=docs)
    split_docs = result["documents"]
    # Should return one doc per input, and no header key in meta
    assert len(split_docs) == 2
    for doc in split_docs:
        assert "header" not in doc.meta
    # Sanity Checks
    assert split_docs[0].content == docs[0].content
    assert split_docs[1].content == docs[1].content


def test_split_multiple_documents(sample_text):
    splitter = MarkdownHeaderSplitter(keep_headers=False)
    docs = [
        Document(content=sample_text),
        Document(content="# Another Header\nSome content."),
        Document(content="# H1\nA"),
        Document(content="# H2\nB"),
    ]
    result = splitter.run(documents=docs)
    split_docs = result["documents"]

    assert len(split_docs) == 8

    headers = {doc.meta["header"] for doc in split_docs}
    assert {"Another Header", "H1", "H2"}.issubset(headers)

    # Verify that all documents have a split_id and they're sequential
    split_ids = [doc.meta.get("split_id") for doc in split_docs]
    assert all(split_id is not None for split_id in split_ids)
    assert split_ids == list(range(len(split_ids)))


def test_split_only_headers():
    text = "# H1\n# H2\n# H3"
    splitter = MarkdownHeaderSplitter()
    docs = [Document(content=text)]
    result = splitter.run(documents=docs)
    split_docs = result["documents"]
    # Return doc without content unchunked
    assert len(split_docs) == 1
    assert split_docs[0].content == text


# Metadata preservation
def test_preserve_document_metadata():
    """Test that document metadata is preserved through splitting."""
    splitter = MarkdownHeaderSplitter(keep_headers=False)
    docs = [Document(content="# Header\nContent", meta={"source": "test", "importance": "high", "custom_field": 123})]

    result = splitter.run(documents=docs)
    split_docs = result["documents"]

    # Original metadata should be preserved
    assert split_docs[0].meta["source"] == "test"
    assert split_docs[0].meta["importance"] == "high"
    assert split_docs[0].meta["custom_field"] == 123

    # New metadata should be added
    assert "header" in split_docs[0].meta
    assert split_docs[0].meta["header"] == "Header"
    assert "split_id" in split_docs[0].meta
    assert split_docs[0].meta["split_id"] == 0


# Error and edge case handling
def test_non_text_document(caplog):
    """Test that the component correctly handles non-text documents."""
    splitter = MarkdownHeaderSplitter()
    docs = [Document(content=None)]

    # Should raise ValueError about text documents
    with pytest.raises(ValueError, match="only works with text documents"):
        splitter.run(documents=docs)


def test_empty_document_list():
    """Test handling of an empty document list."""
    splitter = MarkdownHeaderSplitter()
    result = splitter.run(documents=[])
    assert result["documents"] == []


def test_invalid_secondary_split_at_init():
    """Test that an invalid secondary split type raises an error at initialization time."""
    with pytest.raises(ValueError, match="split_by must be one of"):
        MarkdownHeaderSplitter(secondary_split="invalid_split_type")


def test_invalid_split_parameters_at_init():
    """Test invalid split parameter validation at initialization time."""
    # Test split_length validation
    with pytest.raises(ValueError, match="split_length must be greater than 0"):
        MarkdownHeaderSplitter(secondary_split="word", split_length=0)

    # Test split_overlap validation
    with pytest.raises(ValueError, match="split_overlap must be greater than or equal to 0"):
        MarkdownHeaderSplitter(secondary_split="word", split_overlap=-1)


def test_empty_content_handling():
    """Test handling of documents with empty content."""
    splitter_skip = MarkdownHeaderSplitter()  # skip empty documents by default
    docs = [Document(content="")]
    result = splitter_skip.run(documents=docs)
    assert len(result["documents"]) == 0

    splitter_no_skip = MarkdownHeaderSplitter(skip_empty_documents=False)
    docs = [Document(content="")]
    result = splitter_no_skip.run(documents=docs)
    assert len(result["documents"]) == 1


def test_split_id_sequentiality_primary_and_secondary(sample_text):
    # Test primary splitting
    splitter = MarkdownHeaderSplitter(keep_headers=False)
    docs = [Document(content=sample_text)]
    result = splitter.run(documents=docs)
    split_docs = result["documents"]

    # Test number of documents
    assert len(split_docs) == 5

    # Check that split_ids are sequential
    split_ids = [doc.meta["split_id"] for doc in split_docs]
    assert split_ids == list(range(len(split_ids)))

    # Test secondary splitting
    splitter = MarkdownHeaderSplitter(secondary_split="word", split_length=3)
    docs = [Document(content=sample_text)]
    result = splitter.run(documents=docs)
    split_docs = result["documents"]

    # Test number of documents
    assert len(split_docs) == 12

    split_ids = [doc.meta["split_id"] for doc in split_docs]
    assert split_ids == list(range(len(split_ids)))

    # Test with multiple input documents
    docs = [Document(content=sample_text), Document(content="# Another Header\nSome more content here.")]
    result = splitter.run(documents=docs)
    split_docs = result["documents"]

    # Test number of documents
    assert len(split_docs) == 14

    split_ids = [doc.meta["split_id"] for doc in split_docs]
    assert split_ids == list(range(len(split_ids)))


def test_secondary_split_with_overlap():
    text = (
        "# Introduction\n"
        "This is the introduction section with some words for testing overlap splitting. "
        "It should be split into chunks with overlap.\n"
        "## Details\n"
        "Here are more details about the topic. "
        "Splitting should work across multiple headers and content blocks.\n"
        "### Subsection\n"
        "This subsection contains additional information and should also be split with overlap."
    )
    splitter = MarkdownHeaderSplitter(secondary_split="word", split_length=4, split_overlap=2, keep_headers=False)
    docs = [Document(content=text)]
    result = splitter.run(documents=docs)
    split_docs = result["documents"]
    assert len(split_docs) == 21

    for i in range(1, len(split_docs)):
        prev_doc = split_docs[i - 1]
        curr_doc = split_docs[i]
        if prev_doc.meta["header"] == curr_doc.meta["header"]:  # only check overlap within same header
            prev_words = prev_doc.content.split()
            curr_words = curr_doc.content.split()
            assert prev_words[-2:] == curr_words[:2]


def test_secondary_split_with_threshold():
    text = "# Header\n" + " ".join([f"word{i}" for i in range(1, 11)])
    splitter = MarkdownHeaderSplitter(secondary_split="word", split_length=3, split_threshold=2, keep_headers=False)
    docs = [Document(content=text)]
    result = splitter.run(documents=docs)
    split_docs = result["documents"]
    for doc in split_docs[:-1]:
        assert len(doc.content.split()) == 3
    # The last chunk should have at least 2 words (threshold)
    assert len(split_docs[-1].content.split()) >= 2


def test_page_break_handling_in_secondary_split():
    text = "# Header\nFirst page\fSecond page\fThird page"
    splitter = MarkdownHeaderSplitter(secondary_split="word", split_length=1)
    docs = [Document(content=text)]
    result = splitter.run(documents=docs)
    split_docs = result["documents"]
    page_numbers = [doc.meta.get("page_number") for doc in split_docs]
    # Should start at 1 and increment at each \f
    assert page_numbers[0] == 1
    assert max(page_numbers) == 3


def test_page_break_handling_with_multiple_headers():
    text = "# Header\nFirst page\f Second page\f Third page"
    splitter = MarkdownHeaderSplitter(secondary_split="word", split_length=1, keep_headers=True)
    docs = [Document(content=text)]
    result = splitter.run(documents=docs)
    split_docs = result["documents"]
    assert len(split_docs) == 7

    # Split 1
    assert split_docs[0].content == "# "
    assert split_docs[0].meta == {"source_id": ANY, "page_number": 1, "split_id": 0, "split_idx_start": 0}

    # Split 2
    assert split_docs[1].content == "Header\nFirst "
    assert split_docs[1].meta == {"source_id": ANY, "page_number": 1, "split_id": 1, "split_idx_start": 2}

    # Split 3
    assert split_docs[2].content == "page\f "
    assert split_docs[2].meta == {"source_id": ANY, "page_number": 1, "split_id": 2, "split_idx_start": 15}

    # Split 4
    assert split_docs[3].content == "Second "
    assert split_docs[3].meta == {"source_id": ANY, "page_number": 2, "split_id": 3, "split_idx_start": 21}

    # Split 5
    assert split_docs[4].content == "page\f "
    assert split_docs[4].meta == {"source_id": ANY, "page_number": 2, "split_id": 4, "split_idx_start": 28}

    # Split 6
    assert split_docs[5].content == "Third "
    assert split_docs[5].meta == {"source_id": ANY, "page_number": 3, "split_id": 5, "split_idx_start": 34}

    # Split 7
    assert split_docs[6].content == "page"
    assert split_docs[6].meta == {"source_id": ANY, "page_number": 3, "split_id": 6, "split_idx_start": 40}

    # Check reconstruction
    reconstructed_text = "".join(doc.content for doc in split_docs)
    assert reconstructed_text == text
