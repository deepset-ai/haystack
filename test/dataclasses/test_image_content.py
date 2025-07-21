# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import base64
from unittest.mock import patch

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
