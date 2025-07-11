import pytest
from haystack import Document

from deepset_cloud_custom_nodes.splitters.markdown_header_splitter import (
    MarkdownHeaderSplitter,
)


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


def test_parentheaders(sample_text):
    splitter = MarkdownHeaderSplitter()
    docs = [Document(content=sample_text)]
    result = splitter.run(documents=docs)
    split_docs = result["documents"]

    # Find a subheader and check parentheaders
    subheader_doc = next(doc for doc in split_docs if doc.meta["header"] == "Subheader 1.2.2")
    assert "Header 1" in subheader_doc.meta["parentheaders"]
    assert "Header 1.2" in subheader_doc.meta["parentheaders"]


def test_enforce_first_header(sample_text):
    splitter = MarkdownHeaderSplitter(enforce_first_header=True)
    docs = [Document(content=sample_text)]
    result = splitter.run(documents=docs)
    split_docs = result["documents"]

    # All parentheaders should start with the first header
    first_header = "Header 1"
    for doc in split_docs:
        if doc.meta["parentheaders"]:
            assert doc.meta["parentheaders"][0] == first_header


def test_no_headers():
    splitter = MarkdownHeaderSplitter()
    docs = [Document(content="Just some text without headers.")]
    result = splitter.run(documents=docs)
    assert len(result["documents"]) == 1


def test_multiple_documents(sample_text):
    splitter = MarkdownHeaderSplitter()
    docs = [
        Document(content=sample_text),
        Document(content="# Another Header\nSome content."),
    ]
    result = splitter.run(documents=docs)
    split_docs = result["documents"]
    assert any(doc.meta["header"] == "Another Header" for doc in split_docs)
