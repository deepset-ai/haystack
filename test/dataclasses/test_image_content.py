# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import base64
import logging
from unittest.mock import Mock, patch

import httpx
import pytest
from PIL import Image

from haystack.dataclasses.image_content import ImageContent


def test_image_content_init(base64_image_string):
    image_content = ImageContent(
        base64_image=base64_image_string, mime_type="image/png", detail="auto", meta={"key": "value"}
    )
    assert image_content.base64_image == base64_image_string
    assert image_content.mime_type == "image/png"
    assert image_content.detail == "auto"
    assert image_content.meta == {"key": "value"}
    assert image_content.validation


def test_image_content_to_dict(base64_image_string):
    image_content = ImageContent(
        base64_image=base64_image_string, mime_type="image/png", detail="auto", meta={"key": "value"}
    )
    assert image_content.to_dict() == {
        "base64_image": base64_image_string,
        "mime_type": "image/png",
        "detail": "auto",
        "meta": {"key": "value"},
        "validation": True,
    }


def test_image_content_from_dict(base64_image_string):
    image_content = ImageContent.from_dict(
        {
            "base64_image": base64_image_string,
            "mime_type": "image/png",
            "detail": "auto",
            "meta": {"key": "value"},
            "validation": False,
        }
    )
    assert image_content.base64_image == base64_image_string
    assert image_content.mime_type == "image/png"
    assert image_content.detail == "auto"
    assert image_content.meta == {"key": "value"}
    assert not image_content.validation


def test_image_content_init_with_invalid_base64_string():
    with pytest.raises(ValueError):
        ImageContent(base64_image="invalid_base64_string")


def test_image_content_init_with_invalid_base64_string_and_validation_false():
    image_content = ImageContent(base64_image="invalid_base64_string", validation=False)
    assert image_content.base64_image == "invalid_base64_string"
    assert image_content.mime_type is None
    assert image_content.detail is None
    assert image_content.meta == {}
    assert not image_content.validation


def test_image_content_init_with_invalid_mime_type(test_files_path, base64_image_string):
    with pytest.raises(ValueError):
        ImageContent(base64_image=base64_image_string, mime_type="text/xml")

    with open(test_files_path / "docx" / "sample_docx.docx", "rb") as docx_file:
        docx_base64 = base64.b64encode(docx_file.read()).decode("utf-8")
    with pytest.raises(ValueError):
        ImageContent(base64_image=docx_base64)


def test_image_content_init_with_invalid_mime_type_and_validation_false(test_files_path, base64_image_string):
    image_content = ImageContent(base64_image=base64_image_string, mime_type="text/xml", validation=False)
    assert image_content.base64_image == base64_image_string
    assert image_content.mime_type == "text/xml"
    assert image_content.detail is None
    assert image_content.meta == {}
    assert not image_content.validation

    with open(test_files_path / "docx" / "sample_docx.docx", "rb") as docx_file:
        docx_base64 = base64.b64encode(docx_file.read()).decode("utf-8")
    image_content = ImageContent(base64_image=docx_base64, validation=False)
    assert image_content.base64_image == docx_base64
    assert image_content.mime_type is None
    assert image_content.detail is None
    assert image_content.meta == {}
    assert not image_content.validation


def test_image_content_mime_type_guessing(test_files_path):
    image_path = test_files_path / "images" / "apple.jpg"
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")
    image_content = ImageContent(base64_image=base64_image)
    assert image_content.mime_type == "image/jpeg"

    # do not guess mime type if mime type is provided
    image_content = ImageContent(base64_image=base64_image, mime_type="image/png")
    assert image_content.mime_type == "image/png"


def test_image_content_show_in_jupyter(test_files_path):
    image_path = test_files_path / "images" / "apple.jpg"
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")
    image_content = ImageContent(base64_image=base64_image)

    with (
        patch("haystack.dataclasses.image_content.is_in_jupyter", return_value=True),
        patch("IPython.display.display") as mock_display,
    ):
        image_content.show()

        mock_display.assert_called_once()
        displayed_image = mock_display.call_args[0][0]
        assert isinstance(displayed_image, Image.Image)


def test_image_content_show_outside_jupyter(test_files_path):
    image_path = test_files_path / "images" / "apple.jpg"
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")
    image_content = ImageContent(base64_image=base64_image)

    # mocking is_in_jupyter is not needed because we don't test in a Jupyter notebook
    with patch.object(Image.Image, "show") as mock_show:
        image_content.show()
        mock_show.assert_called_once()


def test_image_content_from_file_path(test_files_path):
    image_content = ImageContent.from_file_path(
        file_path=test_files_path / "images" / "apple.jpg", size=(100, 100), detail="high", meta={"test": "test"}
    )

    assert isinstance(image_content.base64_image, str)
    assert image_content.mime_type == "image/jpeg"
    assert image_content.detail == "high"
    assert image_content.meta == {"test": "test", "file_path": str(test_files_path / "images" / "apple.jpg")}


def test_image_content_from_file_path_pdf_unsupported(test_files_path, caplog):
    with pytest.raises(IndexError):
        ImageContent.from_file_path(
            file_path=test_files_path / "pdf" / "sample_pdf_1.pdf",
            size=(100, 100),
            detail="high",
            meta={"test": "test"},
        )

    assert "Could not convert file" in caplog.text
    assert "PDF" in caplog.text


def test_image_content_from_file_path_non_existing(test_files_path, caplog):
    caplog.set_level(logging.WARNING)

    with pytest.raises(IndexError):
        ImageContent.from_file_path(file_path=test_files_path / "images" / "non_existing.jpg")
    assert "No such file" in caplog.text


def test_image_content_from_url(test_files_path):
    with patch("haystack.components.fetchers.link_content.httpx.Client.get") as mock_get:
        with open(test_files_path / "images" / "apple.jpg", "rb") as image_file:
            image_bytes = image_file.read()
        mock_response = Mock(status_code=200, content=image_bytes, headers={"Content-Type": "image/jpeg"})
        mock_get.return_value = mock_response

        image_content = ImageContent.from_url(
            url="https://example.com/apple.jpg", size=(100, 100), detail="high", meta={"test": "test"}
        )

    assert isinstance(image_content.base64_image, str)
    assert image_content.mime_type == "image/jpeg"
    assert image_content.detail == "high"
    assert image_content.meta == {"test": "test", "url": "https://example.com/apple.jpg", "content_type": "image/jpeg"}


def test_image_content_from_url_bad_request():
    with patch("haystack.components.fetchers.link_content.httpx.Client.get") as mock_get:
        mock_get.side_effect = httpx.HTTPStatusError("403 Client Error", request=Mock(), response=Mock())

        with pytest.raises(httpx.HTTPStatusError):
            ImageContent.from_url(url="https://non_existent_website_dot.com/image.jpg", retry_attempts=0, timeout=1)


def test_image_content_from_url_wrong_mime_type_text():
    with patch("haystack.components.fetchers.link_content.httpx.Client.get") as mock_get:
        mock_response = Mock(status_code=200, text="a text", headers={"Content-Type": "text/plain"})
        mock_get.return_value = mock_response

        with pytest.raises(ValueError):
            ImageContent.from_url(
                url="https://example.com/text.txt", size=(100, 100), detail="high", meta={"test": "test"}
            )


def test_image_content_from_url_wrong_mime_type_pdf(test_files_path):
    with patch("haystack.components.fetchers.link_content.httpx.Client.get") as mock_get:
        with open(test_files_path / "pdf" / "sample_pdf_1.pdf", "rb") as pdf_file:
            pdf_bytes = pdf_file.read()
        mock_response = Mock(status_code=200, content=pdf_bytes, headers={"Content-Type": "application/pdf"})
        mock_get.return_value = mock_response

        with pytest.raises(ValueError):
            ImageContent.from_url(
                url="https://example.com/sample_pdf_1.pdf", size=(100, 100), detail="high", meta={"test": "test"}
            )


@pytest.mark.integration
def test_image_content_from_url_wrong_mime_type():
    with pytest.raises(ValueError):
        ImageContent.from_url(url="https://example.com", size=(100, 100), detail="high", meta={"test": "test"})
