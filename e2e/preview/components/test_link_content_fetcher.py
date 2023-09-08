from haystack.preview.components.fetchers.link_content import LinkContentFetcher


HTML_URL = "https://docs.haystack.deepset.ai/docs"
TEXT_URL = "https://raw.githubusercontent.com/deepset-ai/haystack/main/README.md"
PDF_URL = "https://raw.githubusercontent.com/deepset-ai/haystack/b5987a6d8d0714eb2f3011183ab40093d2e4a41a/e2e/samples/pipelines/sample_pdf_1.pdf"


def test_link_content_fetcher_html():
    fetcher = LinkContentFetcher()
    document = fetcher.run(HTML_URL)["document"]
    assert document.mime_type == "text/html"
    assert "Introduction to Haystack" in document.text
    assert document.metadata["url"] == HTML_URL


def test_link_content_fetcher_text():
    fetcher = LinkContentFetcher()
    document = fetcher.run(TEXT_URL)["document"]
    assert document.mime_type == "text/plain"
    assert "Haystack" in document.text
    assert document.metadata["url"] == TEXT_URL


def test_link_content_fetcher_pdf():
    fetcher = LinkContentFetcher()
    document = fetcher.run(PDF_URL)["document"]
    assert document.mime_type == "application/octet-stream"  # FIXME Should be "application/pdf"?
    assert document.text is None
    assert document.blob is not None
    assert document.metadata["url"] == PDF_URL
