# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import base64
import logging
import warnings
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest

from haystack.dataclasses.file_content import FileContent


@contextmanager
def mock_fetch_response(
    url: str, status_code: int = 200, content: bytes | None = None, headers: dict[str, str] | None = None
) -> Iterator[None]:
    """
    Not a test: makes the LinkContentFetcher used by `FileContent.from_url` replay a single, real httpx
    response without touching the network.
    """
    response = httpx.Response(
        status_code,
        content=content,
        headers=headers or {"Content-Type": "application/pdf"},
        request=httpx.Request("GET", url),
    )
    with (
        patch("haystack.components.fetchers.link_content.httpx.Client.stream") as mock_stream,
        patch("haystack.components.fetchers.link_content._resolve_host", return_value=["93.184.216.34"]),
    ):
        mock_stream.side_effect = lambda method, url, headers=None, **kwargs: nullcontext(response)
        yield


def test_file_content_init(base64_pdf_string):
    file_content = FileContent(
        base64_data=base64_pdf_string, mime_type="application/pdf", filename="test.pdf", extra={"key": "value"}
    )
    assert file_content.base64_data == base64_pdf_string
    assert file_content.mime_type == "application/pdf"
    assert file_content.filename == "test.pdf"
    assert file_content.extra == {"key": "value"}
    assert file_content.validation


def test_file_content_to_dict(base64_pdf_string):
    file_content = FileContent(
        base64_data=base64_pdf_string, mime_type="application/pdf", filename="test.pdf", extra={"key": "value"}
    )
    assert file_content.to_dict() == {
        "base64_data": base64_pdf_string,
        "mime_type": "application/pdf",
        "filename": "test.pdf",
        "extra": {"key": "value"},
        "validation": True,
    }


def test_file_content_from_dict(base64_pdf_string):
    file_content = FileContent.from_dict(
        {
            "base64_data": base64_pdf_string,
            "mime_type": "application/pdf",
            "filename": "test.pdf",
            "extra": {"key": "value"},
            "validation": False,
        }
    )
    assert file_content.base64_data == base64_pdf_string
    assert file_content.mime_type == "application/pdf"
    assert file_content.filename == "test.pdf"
    assert file_content.extra == {"key": "value"}
    assert not file_content.validation


def test_file_content_init_with_invalid_base64_string():
    with pytest.raises(ValueError):
        FileContent(base64_data="invalid_base64_string")


def test_file_content_init_with_invalid_base64_string_and_validation_false():
    file_content = FileContent(base64_data="invalid_base64_string", validation=False)
    assert file_content.base64_data == "invalid_base64_string"
    assert file_content.mime_type is None
    assert file_content.filename is None
    assert file_content.extra == {}
    assert not file_content.validation


def test_file_content_mime_type_guessing(test_files_path):
    with open(test_files_path / "pdf" / "sample_pdf_3.pdf", "rb") as f:
        base64_data = base64.b64encode(f.read()).decode("utf-8")
    file_content = FileContent(base64_data=base64_data)
    assert file_content.mime_type == "application/pdf"

    # do not guess mime type if mime type is provided
    file_content = FileContent(base64_data=base64_data, mime_type="application/octet-stream")
    assert file_content.mime_type == "application/octet-stream"


def test_file_content_mime_type_guessing_warning(caplog):
    # A valid base64 string but with content that filetype cannot identify
    plain_text = base64.b64encode(b"just some plain text content").decode("utf-8")
    with caplog.at_level(logging.WARNING):
        file_content = FileContent(base64_data=plain_text)
    assert file_content.mime_type is None
    assert "Failed to guess the MIME type" in caplog.text


def test_file_content_repr(base64_pdf_string):
    file_content = FileContent(base64_data=base64_pdf_string, mime_type="application/pdf", validation=False)
    repr_str = repr(file_content)
    assert "FileContent(" in repr_str
    assert "mime_type='application/pdf'" in repr_str
    # base64_data should be truncated
    assert "..." in repr_str


def test_file_content_from_file_path(test_files_path):
    str_path = Path(test_files_path / "pdf" / "sample_pdf_3.pdf").as_posix()
    file_content = FileContent.from_file_path(file_path=str_path, filename="custom.pdf", extra={"test": "test"})

    assert isinstance(file_content.base64_data, str)
    assert file_content.mime_type == "application/pdf"
    assert file_content.filename == "custom.pdf"
    assert file_content.extra == {"test": "test"}


def test_file_content_from_file_path_default_filename(test_files_path):
    file_content = FileContent.from_file_path(file_path=test_files_path / "pdf" / "sample_pdf_3.pdf")

    assert isinstance(file_content.base64_data, str)
    assert file_content.mime_type == "application/pdf"
    assert file_content.filename == "sample_pdf_3.pdf"
    assert file_content.extra == {}


def test_file_content_from_url(test_files_path):
    with open(test_files_path / "pdf" / "sample_pdf_3.pdf", "rb") as f:
        pdf_bytes = f.read()
    with mock_fetch_response("https://example.com/sample.pdf", content=pdf_bytes):
        file_content = FileContent.from_url(
            url="https://example.com/sample.pdf", filename="custom.pdf", extra={"test": "test"}
        )

    assert isinstance(file_content.base64_data, str)
    assert file_content.mime_type == "application/pdf"
    assert file_content.filename == "custom.pdf"
    assert file_content.extra == {"test": "test"}


def test_file_content_from_url_default_filename(test_files_path):
    with open(test_files_path / "pdf" / "sample_pdf_3.pdf", "rb") as f:
        pdf_bytes = f.read()
    with mock_fetch_response("https://example.com/documents/sample.pdf", content=pdf_bytes):
        file_content = FileContent.from_url(url="https://example.com/documents/sample.pdf")

    assert file_content.filename == "sample.pdf"


def test_file_content_from_url_bad_request():
    with mock_fetch_response("https://non_existent_website_dot.com/file.pdf", status_code=403):
        with pytest.raises(httpx.HTTPStatusError):
            FileContent.from_url(url="https://non_existent_website_dot.com/file.pdf", retry_attempts=0, timeout=1)


def test_file_content_no_warning_on_init(base64_pdf_string):
    with warnings.catch_warnings():
        warnings.simplefilter("error", Warning)
        FileContent(base64_data=base64_pdf_string, mime_type="application/pdf")


def test_file_content_warn_on_inplace_mutation():
    fc = FileContent(base64_data="dGVzdA==", mime_type="text/plain", validation=False)
    with pytest.warns(Warning, match="dataclasses.replace"):
        fc.mime_type = "application/pdf"


def test_file_content_to_trace_dict(base64_pdf_string):
    file_content = FileContent(
        base64_data=base64_pdf_string, mime_type="application/pdf", filename="test.pdf", extra={"key": "value"}
    )
    data = file_content._to_trace_dict()
    assert data["base64_data"] == f"Base64 string ({len(base64_pdf_string)} characters)"
    assert data["mime_type"] == "application/pdf"
    assert data["filename"] == "test.pdf"
    assert data["extra"] == {"key": "value"}
