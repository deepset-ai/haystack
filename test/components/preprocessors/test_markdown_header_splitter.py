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
    splitter = MarkdownHeaderSplitter()
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
    for doc in split_docs:
        assert doc.content.startswith("#") or doc.content.startswith("##") or doc.content.startswith("###")
        assert doc.meta.get("header") is not None


def test_split_parentheaders(sample_text):
    splitter = MarkdownHeaderSplitter()
    docs = [Document(content=sample_text), Document(content="# H1\n## H2\n### H3\nContent")]
    result = splitter.run(documents=docs)
    split_docs = result["documents"]
    # Check parentheaders for both a deep subheader and a simple one
    subheader_doc = next(doc for doc in split_docs if doc.meta["header"] == "Subheader 1.2.2")
    assert "Header 1" in subheader_doc.meta["parentheaders"]
    assert "Header 1.2" in subheader_doc.meta["parentheaders"]
    h3_doc = next((doc for doc in split_docs if doc.meta["header"] == "H3"), None)
    if h3_doc:
        assert h3_doc.meta["parentheaders"] == ["H1", "H2"]


def test_split_no_headers():
    splitter = MarkdownHeaderSplitter()
    docs = [Document(content="No headers here."), Document(content="Just some text without headers.")]
    result = splitter.run(documents=docs)
    split_docs = result["documents"]
    # Should return one doc per input, header is None
    assert len(split_docs) == 2
    for doc in split_docs:
        assert doc.meta["header"] is None


def test_split_multiple_documents(sample_text):
    splitter = MarkdownHeaderSplitter()
    docs = [
        Document(content=sample_text),
        Document(content="# Another Header\nSome content."),
        Document(content="# H1\nA"),
        Document(content="# H2\nB"),
    ]
    result = splitter.run(documents=docs)
    split_docs = result["documents"]
    headers = {doc.meta["header"] for doc in split_docs}
    assert {"Another Header", "H1", "H2"}.issubset(headers)


def test_split_only_headers():
    text = "# H1\n# H2\n# H3"
    splitter = MarkdownHeaderSplitter()
    docs = [Document(content=text)]
    result = splitter.run(documents=docs)
    split_docs = result["documents"]
    # Should not create chunks for headers with no content
    assert len(split_docs) == 0


# Header inference and overrides
def test_split_infer_header_levels():
    text = "## H1\n## H2\nContent"
    splitter = MarkdownHeaderSplitter(infer_header_levels=True)
    docs = [Document(content=text)]
    result = splitter.run(documents=docs)
    split_docs = result["documents"]
    # Should rewrite headers to # and ##
    assert split_docs[0].content.startswith("## H2") or split_docs[0].content.startswith("# H1")


def test_infer_header_levels_complex():
    """Test header level inference with a complex document structure."""
    text = (
        "## All Headers Same Level\n"
        "Some content\n"
        "## Second Header\n"
        "Some content\n"  # Added content to ensure headers are processed correctly
        "## Third Header With No Content\n"
        "## Fourth Header With No Content\n"
        "## Fifth Header\n"
        "More content"
    )

    splitter = MarkdownHeaderSplitter(infer_header_levels=True)
    docs = [Document(content=text)]
    result = splitter.run(documents=docs)
    split_docs = result["documents"]

    # Get docs by header content to avoid position assumptions
    first_doc = next((doc for doc in split_docs if "All Headers Same Level" in doc.content), None)
    second_doc = next((doc for doc in split_docs if "Second Header" in doc.content), None)

    # First header should be level 1
    assert first_doc and "# All Headers Same Level" in first_doc.content

    # Second header with content should stay at level 1
    assert second_doc and "# Second Header" in second_doc.content


def test_infer_header_levels_override_both_directions():
    text = "## H1\n## H2\nContent"
    docs = [Document(content=text)]

    # False at init, True at run
    splitter = MarkdownHeaderSplitter(infer_header_levels=False)
    result = splitter.run(documents=docs, infer_header_levels=True)
    assert "# " in result["documents"][0].content

    # True at init, False at run
    splitter = MarkdownHeaderSplitter(infer_header_levels=True)
    result = splitter.run(documents=docs, infer_header_levels=False)
    assert all("## " in doc.content for doc in result["documents"])


# Metadata preservation
def test_preserve_document_metadata():
    """Test that document metadata is preserved through splitting."""
    splitter = MarkdownHeaderSplitter()
    docs = [Document(content="# Header\nContent", meta={"source": "test", "importance": "high", "custom_field": 123})]

    result = splitter.run(documents=docs)
    split_docs = result["documents"]

    # Original metadata should be preserved
    assert split_docs[0].meta["source"] == "test"
    assert split_docs[0].meta["importance"] == "high"
    assert split_docs[0].meta["custom_field"] == 123

    # New metadata should be added
    assert "header" in split_docs[0].meta
    assert "split_id" in split_docs[0].meta


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
    splitter = MarkdownHeaderSplitter()
    docs = [Document(content="")]
    result = splitter.run(documents=docs)

    # DocumentSplitter skips empty documents by default
    assert len(result["documents"]) == 0


# Output format and split ID checks
def test_document_splitting_format():
    """Test that the format of split documents is correct."""
    splitter = MarkdownHeaderSplitter()
    docs = [Document(content="# Header\nContent")]
    result = splitter.run(documents=docs)

    # Basic validation of the output structure
    assert isinstance(result, dict)
    assert "documents" in result
    assert isinstance(result["documents"], list)


def test_split_id_sequentiality_primary_and_secondary():
    text = "# Header\n" + "Word " * 30
    # Test primary splitting
    splitter = MarkdownHeaderSplitter()
    docs = [Document(content=text)]
    result = splitter.run(documents=docs)
    split_ids = [doc.meta["split_id"] for doc in result["documents"]]
    assert split_ids == list(range(len(split_ids)))

    # Test secondary splitting
    splitter = MarkdownHeaderSplitter(secondary_split="word", split_length=5)
    result = splitter.run(documents=docs)
    split_ids = [doc.meta["split_id"] for doc in result["documents"]]
    assert split_ids == list(range(len(split_ids)))
    docs = [Document(content=text)]
    result = splitter.run(documents=docs)
    split_ids = [doc.meta["split_id"] for doc in result["documents"]]
    assert split_ids == list(range(len(split_ids)))

    # Test secondary splitting
    splitter = MarkdownHeaderSplitter(secondary_split="word", split_length=5)
    result = splitter.run(documents=docs)
    split_ids = [doc.meta["split_id"] for doc in result["documents"]]
    assert split_ids == list(range(len(split_ids)))
    assert split_ids == list(range(len(split_ids)))
