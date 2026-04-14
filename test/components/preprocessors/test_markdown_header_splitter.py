# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from collections import defaultdict
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


@pytest.fixture
def sample_text_with_page_breaks():
    return (
        "# Header 1\n"
        "Content under header 1.\n\f\n"
        "## Header 1.1\n"
        "### Subheader 1.1.1\n"
        "Content under sub-header 1.1.1\n\f\n"
        "## Header 1.2\n"
        "### Subheader 1.2.1\n"
        "Content under header 1.2.1.\n\f\n"
        "### Subheader 1.2.2\n"
        "Content under header 1.2.2.\n\f\n"
        "### Subheader 1.2.3\n"
        "Content under header 1.2.3."
    )


# Basic splitting and structure
def test_basic_split(sample_text):
    splitter = MarkdownHeaderSplitter()
    docs = [Document(content=sample_text)]
    result = splitter.run(documents=docs)
    split_docs = result["documents"]

    # Check that content is present and correct
    # Test first split
    header1_doc = split_docs[0]
    assert header1_doc.meta["split_id"] == 0
    assert header1_doc.meta["page_number"] == 1
    assert header1_doc.content == "# Header 1\nContent under header 1.\n"

    # Test second split
    subheader111_doc = split_docs[1]
    assert subheader111_doc.meta["split_id"] == 1
    assert subheader111_doc.meta["page_number"] == 1
    assert subheader111_doc.content == "## Header 1.1\n### Subheader 1.1.1\nContent under sub-header 1.1.1\n"

    # Test third split
    subheader121_doc = split_docs[2]
    assert subheader121_doc.meta["split_id"] == 2
    assert subheader121_doc.meta["page_number"] == 1
    assert subheader121_doc.content == "## Header 1.2\n### Subheader 1.2.1\nContent under header 1.2.1.\n"

    # Test fourth split
    subheader122_doc = split_docs[3]
    assert subheader122_doc.meta["split_id"] == 3
    assert subheader122_doc.meta["page_number"] == 1
    assert subheader122_doc.content == "### Subheader 1.2.2\nContent under header 1.2.2.\n"

    # Test fifth split
    subheader123_doc = split_docs[4]
    assert subheader123_doc.meta["split_id"] == 4
    assert subheader123_doc.meta["page_number"] == 1
    assert subheader123_doc.content == "### Subheader 1.2.3\nContent under header 1.2.3."

    # Reconstruct original text
    reconstructed_doc = "".join([doc.content for doc in split_docs])
    assert reconstructed_doc == sample_text


def test_keep_headers_preserves_parent_headers_for_first_child():
    text = (
        "# Header 1\n"
        "Intro text\n\n"
        "## Header 1.1\n"
        "Text 1\n\n"
        "## Header 1.2\n"
        "Text 2\n\n"
        "### Header 1.2.1\n"
        "Text 3\n\n"
        "### Header 1.2.2\n"
        "Text 4\n"
    )
    splitter = MarkdownHeaderSplitter(keep_headers=True)
    split_docs = splitter.run(documents=[Document(content=text)])["documents"]

    assert [(doc.meta["header"], doc.meta["parent_headers"]) for doc in split_docs] == [
        ("Header 1", []),
        ("Header 1.1", ["Header 1"]),
        ("Header 1.2", ["Header 1"]),
        ("Header 1.2.1", ["Header 1", "Header 1.2"]),
        ("Header 1.2.2", ["Header 1", "Header 1.2"]),
    ]
    # reconstruct original text
    reconstructed_text = "".join(doc.content for doc in split_docs)
    assert reconstructed_text == text


def test_split_without_headers(sample_text):
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
    # Test first split
    header1_doc = split_docs[0]
    assert header1_doc.meta["header"] == "Header 1"
    assert header1_doc.meta["split_id"] == 0
    assert header1_doc.meta["page_number"] == 1
    assert header1_doc.meta["parent_headers"] == []
    assert header1_doc.content == "\nContent under header 1.\n"

    # Test second split
    subheader111_doc = split_docs[1]
    assert subheader111_doc.meta["header"] == "Subheader 1.1.1"
    assert subheader111_doc.meta["split_id"] == 1
    assert subheader111_doc.meta["page_number"] == 1
    assert subheader111_doc.meta["parent_headers"] == ["Header 1", "Header 1.1"]
    assert subheader111_doc.content == "\nContent under sub-header 1.1.1\n"

    # Test third split
    subheader121_doc = split_docs[2]
    assert subheader121_doc.meta["header"] == "Subheader 1.2.1"
    assert subheader121_doc.meta["split_id"] == 2
    assert subheader121_doc.meta["page_number"] == 1
    assert subheader121_doc.meta["parent_headers"] == ["Header 1", "Header 1.2"]
    assert subheader121_doc.content == "\nContent under header 1.2.1.\n"

    # Test fourth split
    subheader122_doc = split_docs[3]
    assert subheader122_doc.meta["header"] == "Subheader 1.2.2"
    assert subheader122_doc.meta["split_id"] == 3
    assert subheader122_doc.meta["page_number"] == 1
    assert subheader122_doc.meta["parent_headers"] == ["Header 1", "Header 1.2"]
    assert subheader122_doc.content == "\nContent under header 1.2.2.\n"

    # Test fifth split
    subheader123_doc = split_docs[4]
    assert subheader123_doc.meta["header"] == "Subheader 1.2.3"
    assert subheader123_doc.meta["split_id"] == 4
    assert subheader123_doc.meta["page_number"] == 1
    assert subheader123_doc.meta["parent_headers"] == ["Header 1", "Header 1.2"]
    assert subheader123_doc.content == "\nContent under header 1.2.3."


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

    # First 5 splits are from sample_text
    assert split_docs[5].meta["header"] == "Another Header"
    assert split_docs[6].meta["header"] == "H1"
    assert split_docs[7].meta["header"] == "H2"

    # Verify that split_ids are per-parent-document
    splits_by_source = defaultdict(list)
    for doc in split_docs:
        splits_by_source[doc.meta["source_id"]].append(doc.meta["split_id"])

    # Each parent document should have split_ids starting from 0
    for split_ids in splits_by_source.values():
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
    splitter = MarkdownHeaderSplitter(keep_headers=False)  # keep_headers=True case is covered by this test too
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
    assert split_docs[0].content == "\nContent"


# Error and edge case handling
def test_non_text_document():
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


class TestHeaderSplitLevels:
    def test_default_splits_on_all_levels(self, sample_text):
        """Default behaviour: all six header levels create split boundaries.

        Note: empty headers (no content of their own) are folded into the next chunk via pending_headers rather
        than appearing as standalone entries in meta.
        """
        splitter = MarkdownHeaderSplitter()  # header_split_levels defaults to [1,2,3,4,5,6]
        docs = splitter.run(documents=[Document(content=sample_text)])["documents"]

        # sample_text has 5 chunks with content; "Header 1.1" and "Header 1.2" are empty headers prepended to their
        # first child and do not appear as their own split boundary
        assert len(docs) == 5
        assert docs[0].content == "# Header 1\nContent under header 1.\n"
        assert docs[1].content == "## Header 1.1\n### Subheader 1.1.1\nContent under sub-header 1.1.1\n"
        assert docs[2].content == "## Header 1.2\n### Subheader 1.2.1\nContent under header 1.2.1.\n"
        assert docs[3].content == "### Subheader 1.2.2\nContent under header 1.2.2.\n"
        assert docs[4].content == "### Subheader 1.2.3\nContent under header 1.2.3."

        headers = [doc.meta["header"] for doc in docs]
        assert headers == ["Header 1", "Subheader 1.1.1", "Subheader 1.2.1", "Subheader 1.2.2", "Subheader 1.2.3"]

    def test_h1_and_h2_only(self, sample_text):
        """Only h1/h2 headers create splits; h3+ content is absorbed into the preceding chunk."""
        splitter = MarkdownHeaderSplitter(header_split_levels=[1, 2])
        docs = splitter.run(documents=[Document(content=sample_text)])["documents"]

        assert len(docs) == 3
        assert docs[0].content == "# Header 1\nContent under header 1.\n"
        assert docs[1].content == "## Header 1.1\n### Subheader 1.1.1\nContent under sub-header 1.1.1\n"
        assert docs[2].content == (
            "## Header 1.2\n"
            "### Subheader 1.2.1\nContent under header 1.2.1.\n"
            "### Subheader 1.2.2\nContent under header 1.2.2.\n"
            "### Subheader 1.2.3\nContent under header 1.2.3."
        )

        headers = [doc.meta["header"] for doc in docs]
        assert headers == ["Header 1", "Header 1.1", "Header 1.2"]
        # h3 headers must not appear as split boundaries
        assert "Subheader 1.1.1" not in headers
        assert "Subheader 1.2.1" not in headers

    def test_single_level(self, sample_text):
        """Splitting on only h1 yields one chunk that is the full document."""
        splitter = MarkdownHeaderSplitter(header_split_levels=[1])
        docs = splitter.run(documents=[Document(content=sample_text)])["documents"]

        assert len(docs) == 1
        assert docs[0].meta["header"] == "Header 1"
        # entire document is in one chunk — h1 is first, so content equals the full source text
        assert docs[0].content == sample_text

    def test_deep_levels_only(self):
        """Splitting on h3 only; h1/h2 headers above the first h3 are not captured in any chunk."""
        text = (
            "# Top Level\n"
            "Ignored top content.\n"
            "## Mid Level\n"
            "Ignored mid content.\n"
            "### Deep Section A\n"
            "Content A.\n"
            "### Deep Section B\n"
            "Content B.\n"
        )
        splitter = MarkdownHeaderSplitter(header_split_levels=[3])
        docs = splitter.run(documents=[Document(content=text)])["documents"]

        assert len(docs) == 2
        assert docs[0].content == "### Deep Section A\nContent A.\n"
        assert docs[1].content == "### Deep Section B\nContent B.\n"

        headers = [doc.meta["header"] for doc in docs]
        assert "Top Level" not in headers
        assert "Mid Level" not in headers
        # text before the first h3 match is not absorbed — it is dropped entirely
        assert "Ignored top content." not in docs[0].content
        assert "Ignored mid content." not in docs[0].content

    def test_non_contiguous_levels(self):
        """Non-contiguous level selection (e.g. [1, 3]) splits on h1 and h3 but not h2."""
        text = "# H1 Title\n## H2 Ignored\nH2 content.\n### H3 Section\nH3 content.\n"
        splitter = MarkdownHeaderSplitter(header_split_levels=[1, 3])
        docs = splitter.run(documents=[Document(content=text)])["documents"]

        assert len(docs) == 2
        # h2 content sits between the h1 and h3 match boundaries, so it is absorbed into the h1 chunk
        assert docs[0].content == "# H1 Title\n## H2 Ignored\nH2 content.\n"
        assert docs[0].meta["header"] == "H1 Title"
        assert docs[1].content == "### H3 Section\nH3 content.\n"
        assert docs[1].meta["header"] == "H3 Section"
        assert "H2 Ignored" not in [doc.meta["header"] for doc in docs]

    def test_validation_empty_list(self):
        with pytest.raises(ValueError, match="non-empty list"):
            MarkdownHeaderSplitter(header_split_levels=[])

    def test_validation_level_zero(self):
        with pytest.raises(ValueError, match="invalid values"):
            MarkdownHeaderSplitter(header_split_levels=[0, 1, 2])

    def test_validation_level_seven(self):
        with pytest.raises(ValueError, match="invalid values"):
            MarkdownHeaderSplitter(header_split_levels=[1, 7])

    def test_validation_non_integer(self):
        with pytest.raises(ValueError, match="invalid values"):
            MarkdownHeaderSplitter(header_split_levels=[1, "2"])  # type: ignore[list-item]

    def test_validation_duplicate_levels(self):
        with pytest.raises(ValueError, match="duplicate"):
            MarkdownHeaderSplitter(header_split_levels=[1, 2, 2])


class TestCodeBlockExclusion:
    """Tests that hash lines inside fenced code blocks are not treated as headers."""

    def test_backtick_fence(self):
        """Hash lines inside triple-backtick fences are ignored."""
        text = (
            "# Real Header\n"
            "Some content.\n"
            "```python\n"
            "# this is a Python comment, not a header\n"
            "## also not a header\n"
            "x = 1\n"
            "```\n"
            "More content.\n"
            "## Real Subheader\n"
            "Subheader content.\n"
        )
        splitter = MarkdownHeaderSplitter()
        docs = splitter.run(documents=[Document(content=text)])["documents"]

        assert len(docs) == 2
        assert docs[0].content == (
            "# Real Header\n"
            "Some content.\n"
            "```python\n"
            "# this is a Python comment, not a header\n"
            "## also not a header\n"
            "x = 1\n"
            "```\n"
            "More content.\n"
        )
        assert docs[0].meta["header"] == "Real Header"
        assert docs[1].content == "## Real Subheader\nSubheader content.\n"
        assert docs[1].meta["header"] == "Real Subheader"

        assert "this is a Python comment, not a header" not in [doc.meta["header"] for doc in docs]
        assert "also not a header" not in [doc.meta["header"] for doc in docs]

    def test_tilde_fence(self):
        """Hash lines inside triple-tilde fences are ignored."""
        text = "# Real Header\n~~~bash\n# shell comment\necho hello\n~~~\n## Real Subheader\nContent.\n"
        splitter = MarkdownHeaderSplitter()
        docs = splitter.run(documents=[Document(content=text)])["documents"]

        assert len(docs) == 2
        assert docs[0].content == "# Real Header\n~~~bash\n# shell comment\necho hello\n~~~\n"
        assert docs[0].meta["header"] == "Real Header"
        assert docs[1].content == "## Real Subheader\nContent.\n"
        assert docs[1].meta["header"] == "Real Subheader"
        assert "shell comment" not in [doc.meta["header"] for doc in docs]

    def test_multiple_code_blocks(self):
        """Multiple fenced code blocks in one document are all excluded."""
        text = (
            "# Section One\n"
            "Intro text.\n"
            "```\n"
            "# fake header A\n"
            "```\n"
            "Middle text.\n"
            "```python\n"
            "# fake header B\n"
            "```\n"
            "## Section Two\n"
            "More content.\n"
        )
        splitter = MarkdownHeaderSplitter()
        docs = splitter.run(documents=[Document(content=text)])["documents"]

        assert len(docs) == 2
        assert docs[0].content == (
            "# Section One\nIntro text.\n```\n# fake header A\n```\nMiddle text.\n```python\n# fake header B\n```\n"
        )
        assert docs[0].meta["header"] == "Section One"
        assert docs[1].content == "## Section Two\nMore content.\n"
        assert docs[1].meta["header"] == "Section Two"
        assert "fake header A" not in [doc.meta["header"] for doc in docs]
        assert "fake header B" not in [doc.meta["header"] for doc in docs]

    def test_longer_fence_delimiters(self):
        """Fences with more than three backticks/tildes are also recognised."""
        text = "# Real Header\n````python\n# not a header\n````\n## Real Subheader\nContent.\n"
        splitter = MarkdownHeaderSplitter()
        docs = splitter.run(documents=[Document(content=text)])["documents"]

        assert len(docs) == 2
        assert docs[0].content == "# Real Header\n````python\n# not a header\n````\n"
        assert docs[0].meta["header"] == "Real Header"
        assert docs[1].content == "## Real Subheader\nContent.\n"
        assert docs[1].meta["header"] == "Real Subheader"
        assert "not a header" not in [doc.meta["header"] for doc in docs]

    def test_code_block_with_no_real_headers(self):
        """If the only hash lines are inside code blocks, the document is returned unchunked."""
        text = "Plain text before code.\n```\n# entirely fake\n```\nPlain text after code.\n"
        splitter = MarkdownHeaderSplitter()
        docs = splitter.run(documents=[Document(content=text)])["documents"]

        assert len(docs) == 1
        assert docs[0].content == text
        assert "header" not in docs[0].meta


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
    # Test primary splitting with single document
    splitter = MarkdownHeaderSplitter()
    docs = [Document(content=sample_text)]
    result = splitter.run(documents=docs)
    split_docs = result["documents"]

    # Test number of documents
    assert len(split_docs) == 5

    # Check that split_ids are sequential from 0 for this single parent document
    split_ids = [doc.meta["split_id"] for doc in split_docs]
    assert split_ids == list(range(len(split_ids)))

    # Test secondary splitting with single document
    splitter = MarkdownHeaderSplitter(secondary_split="word", split_length=3)
    docs = [Document(content=sample_text)]
    result = splitter.run(documents=docs)
    split_docs = result["documents"]

    # Test number of documents
    assert len(split_docs) == 12

    # Check that split_ids are sequential from 0 for this single parent document
    split_ids = [doc.meta["split_id"] for doc in split_docs]
    assert split_ids == list(range(len(split_ids)))

    # Test with multiple input documents; each should have its own split_id sequence
    splitter = MarkdownHeaderSplitter(secondary_split="word", split_length=3)  # Use fresh instance
    docs = [Document(content=sample_text), Document(content="# Another Header\nSome more content here.")]
    result = splitter.run(documents=docs)
    split_docs = result["documents"]

    # Test number of documents
    assert len(split_docs) == 14

    # Verify split_ids are per-parent-document
    splits_by_source = defaultdict(list)
    for doc in split_docs:
        splits_by_source[doc.meta["source_id"]].append(doc.meta["split_id"])

    # Each parent document should have split_ids starting from 0
    for split_ids in splits_by_source.values():
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
    # keep_headers=False
    splitter = MarkdownHeaderSplitter(secondary_split="word", split_length=8, split_overlap=3, keep_headers=False)
    docs = [Document(content=text)]
    result = splitter.run(documents=docs)
    split_docs = result["documents"]
    assert len(split_docs) == 9

    # Verify exact content and metadata of each split
    # intro (4 docs)
    assert split_docs[0].content == "\nThis is the introduction section with some words "
    assert split_docs[0].meta["header"] == "Introduction"
    assert split_docs[0].meta["split_id"] == 0
    assert split_docs[1].content == "with some words for testing overlap splitting. It "
    assert split_docs[1].meta["header"] == "Introduction"
    assert split_docs[1].meta["split_id"] == 1
    assert split_docs[2].content == "overlap splitting. It should be split into chunks "
    assert split_docs[2].meta["header"] == "Introduction"
    assert split_docs[2].meta["split_id"] == 2
    assert split_docs[3].content == "split into chunks with overlap.\n"
    assert split_docs[3].meta["header"] == "Introduction"
    assert split_docs[3].meta["split_id"] == 3

    # details (3 docs)
    assert split_docs[4].content == "\nHere are more details about the topic. Splitting "
    assert split_docs[4].meta["header"] == "Details"
    assert split_docs[4].meta["split_id"] == 4
    assert split_docs[5].content == "the topic. Splitting should work across multiple headers "
    assert split_docs[5].meta["header"] == "Details"
    assert split_docs[5].meta["split_id"] == 5
    assert split_docs[6].content == "across multiple headers and content blocks.\n"
    assert split_docs[6].meta["header"] == "Details"
    assert split_docs[6].meta["split_id"] == 6

    # subsection (2 docs)
    assert split_docs[7].content == "\nThis subsection contains additional information and should also "
    assert split_docs[7].meta["header"] == "Subsection"
    assert split_docs[7].meta["split_id"] == 7
    assert split_docs[8].content == "and should also be split with overlap."
    assert split_docs[8].meta["header"] == "Subsection"
    assert split_docs[8].meta["split_id"] == 8

    # verify 3-word overlap behavior (split_overlap=3)
    # consecutive pairs within a header should share the 3 words at their boundary
    # intro
    assert split_docs[0].content.split()[-3:] == split_docs[1].content.split()[:3]
    assert split_docs[1].content.split()[-3:] == split_docs[2].content.split()[:3]
    assert split_docs[2].content.split()[-3:] == split_docs[3].content.split()[:3]

    # details
    assert split_docs[4].content.split()[-3:] == split_docs[5].content.split()[:3]
    assert split_docs[5].content.split()[-3:] == split_docs[6].content.split()[:3]

    # subsection
    assert split_docs[7].content.split()[-3:] == split_docs[8].content.split()[:3]

    # re-run with keep_headers=True, change split_length and split_overlap
    splitter = MarkdownHeaderSplitter(secondary_split="word", split_length=4, split_overlap=2)
    docs = [Document(content=text)]
    result = splitter.run(documents=docs)
    split_docs = result["documents"]
    assert len(split_docs) == 24

    assert split_docs[0].content.startswith("# Introduction")
    assert all("header" in doc.meta for doc in split_docs)


def test_secondary_split_with_threshold():
    text = "# Header\n" + " ".join([f"word{i}" for i in range(1, 11)])
    # keep_headers=True
    splitter = MarkdownHeaderSplitter(secondary_split="word", split_length=3, split_threshold=2, keep_headers=True)
    docs = [Document(content=text)]
    result = splitter.run(documents=docs)
    split_docs = result["documents"]

    # Explicitly test each split
    assert len(split_docs) == 4
    assert split_docs[0].content == "# Header\nword1 word2 "
    assert split_docs[0].meta["split_id"] == 0
    assert split_docs[1].content == "word3 word4 word5 "
    assert split_docs[1].meta["split_id"] == 1
    assert split_docs[2].content == "word6 word7 word8 "
    assert split_docs[2].meta["split_id"] == 2
    assert split_docs[3].content == "word9 word10"
    assert split_docs[3].meta["split_id"] == 3

    # keep_headers=False
    splitter = MarkdownHeaderSplitter(secondary_split="word", split_length=3, split_threshold=2, keep_headers=False)
    docs = [Document(content=text)]
    result = splitter.run(documents=docs)
    split_docs = result["documents"]

    # Explicitly test each split
    assert len(split_docs) == 3
    assert split_docs[0].content == "\nword1 word2 word3 "
    assert split_docs[0].meta["split_id"] == 0
    assert split_docs[1].content == "word4 word5 word6 "
    assert split_docs[1].meta["split_id"] == 1
    assert split_docs[2].content == "word7 word8 word9 word10"  # 4 words (due to threshold, not possible to split 3-1)
    assert split_docs[2].meta["split_id"] == 2


def test_page_break_handling_in_secondary_split():
    text = "# Header\nFirst page\f Second page\f Third page"
    splitter = MarkdownHeaderSplitter(secondary_split="word", split_length=1)
    docs = [Document(content=text)]
    result = splitter.run(documents=docs)
    split_docs = result["documents"]

    expected_page_numbers = [1, 1, 1, 2, 2, 3, 3]
    actual_page_numbers = [doc.meta.get("page_number") for doc in split_docs]
    assert actual_page_numbers == expected_page_numbers


def test_page_break_handling_with_multiple_headers(sample_text_with_page_breaks):
    splitter = MarkdownHeaderSplitter(secondary_split="word", split_length=3)
    docs = [Document(content=sample_text_with_page_breaks)]
    result = splitter.run(documents=docs)
    split_docs = result["documents"]
    assert len(split_docs) == 12

    assert split_docs[0].content == "# Header 1\nContent "
    assert split_docs[0].meta == {
        "source_id": ANY,
        "split_id": 0,
        "page_number": 1,
        "split_idx_start": 0,
        "header": "Header 1",
        "parent_headers": [],
    }

    assert split_docs[1].content == "under header 1.\n\f\n"
    assert split_docs[1].meta == {
        "source_id": ANY,
        "split_id": 1,
        "page_number": 1,
        "split_idx_start": 19,
        "header": "Header 1",
        "parent_headers": [],
    }

    assert split_docs[2].content == "## Header 1.1\n### "
    assert split_docs[2].meta == {
        "source_id": ANY,
        "split_id": 2,
        "page_number": 2,
        "split_idx_start": 0,
        "header": "Subheader 1.1.1",
        "parent_headers": ["Header 1", "Header 1.1"],
    }

    assert split_docs[3].content == "Subheader 1.1.1\nContent under "
    assert split_docs[3].meta == {
        "source_id": ANY,
        "split_id": 3,
        "page_number": 2,
        "split_idx_start": 18,
        "header": "Subheader 1.1.1",
        "parent_headers": ["Header 1", "Header 1.1"],
    }

    assert split_docs[4].content == "sub-header 1.1.1\n\f\n"
    assert split_docs[4].meta == {
        "source_id": ANY,
        "split_id": 4,
        "page_number": 2,
        "split_idx_start": 48,
        "header": "Subheader 1.1.1",
        "parent_headers": ["Header 1", "Header 1.1"],
    }

    assert split_docs[5].content == "## Header 1.2\n### "
    assert split_docs[5].meta == {
        "source_id": ANY,
        "split_id": 5,
        "page_number": 3,
        "split_idx_start": 0,
        "header": "Subheader 1.2.1",
        "parent_headers": ["Header 1", "Header 1.2"],
    }

    assert split_docs[6].content == "Subheader 1.2.1\nContent under "
    assert split_docs[6].meta == {
        "source_id": ANY,
        "split_id": 6,
        "page_number": 3,
        "split_idx_start": 18,
        "header": "Subheader 1.2.1",
        "parent_headers": ["Header 1", "Header 1.2"],
    }

    assert split_docs[7].content == "header 1.2.1.\n\f\n"
    assert split_docs[7].meta == {
        "source_id": ANY,
        "split_id": 7,
        "page_number": 3,
        "split_idx_start": 48,
        "header": "Subheader 1.2.1",
        "parent_headers": ["Header 1", "Header 1.2"],
    }

    assert split_docs[8].content == "### Subheader 1.2.2\nContent "
    assert split_docs[8].meta == {
        "source_id": ANY,
        "split_id": 8,
        "page_number": 4,
        "split_idx_start": 0,
        "header": "Subheader 1.2.2",
        "parent_headers": ["Header 1", "Header 1.2"],
    }

    assert split_docs[9].content == "under header 1.2.2.\n\f\n"
    assert split_docs[9].meta == {
        "source_id": ANY,
        "split_id": 9,
        "page_number": 4,
        "split_idx_start": 28,
        "header": "Subheader 1.2.2",
        "parent_headers": ["Header 1", "Header 1.2"],
    }

    assert split_docs[10].content == "### Subheader 1.2.3\nContent "
    assert split_docs[10].meta == {
        "source_id": ANY,
        "split_id": 10,
        "page_number": 5,
        "split_idx_start": 0,
        "header": "Subheader 1.2.3",
        "parent_headers": ["Header 1", "Header 1.2"],
    }

    assert split_docs[11].content == "under header 1.2.3."
    assert split_docs[11].meta == {
        "source_id": ANY,
        "split_id": 11,
        "page_number": 5,
        "split_idx_start": 28,
        "header": "Subheader 1.2.3",
        "parent_headers": ["Header 1", "Header 1.2"],
    }

    # reconstruct original
    reconstructed_text = "".join(doc.content for doc in split_docs)
    assert reconstructed_text == sample_text_with_page_breaks
