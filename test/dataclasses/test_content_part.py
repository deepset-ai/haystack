import pytest

import base64

from haystack.dataclasses import ContentPart, ContentType, ImageDetail, ByteStream


def test_from_text_with_valid_content():
    content = "This is some text."
    content_part = ContentPart.from_text(content)
    assert content_part.content == content
    assert content_part.type == ContentType.TEXT


def test_from_image_url_with_valid_content():
    content = "image.com/sample.jpg"
    content_part = ContentPart.from_image_url(content)
    assert content_part.content == content
    assert content_part.type == ContentType.IMAGE_URL

    content_part = ContentPart.from_image_url(url=content, image_detail=ImageDetail.LOW)
    assert content_part.content == content
    assert content_part.type == ContentType.IMAGE_URL
    assert content_part.image_detail == ImageDetail.LOW


def test_from_base64_with_valid_content(test_files_path):
    # Function to encode the image
    def encode_image(image_path) -> ByteStream:
        with open(image_path, "rb") as image_file:
            return ByteStream(base64.b64encode(image_file.read()))

    image_path = test_files_path / "images" / "apple.jpg"

    base64_image = encode_image(image_path=image_path)

    content_part = ContentPart.from_base64_image(base64_image)
    assert content_part.content == base64_image
    assert content_part.type == ContentType.IMAGE_BASE64

    content_part = ContentPart.from_base64_image(image=base64_image, image_detail=ImageDetail.HIGH)
    assert content_part.content == base64_image
    assert content_part.type == ContentType.IMAGE_BASE64
    assert content_part.image_detail == ImageDetail.HIGH


def test_content_type_to_openai_format():
    assert (ContentType.TEXT).to_openai_format() == "text"
    assert (ContentType.IMAGE_URL).to_openai_format() == "image_url"
    assert (ContentType.IMAGE_BASE64).to_openai_format() == "image_url"


def test_to_openai_format_with_text():
    assert ContentPart.from_text("Text").to_openai_format() == {"type": "text", "text": "Text"}


def test_to_openai_format_with_image_url():
    assert ContentPart.from_image_url("image.com/test.jpg").to_openai_format() == {
        "type": "image_url",
        "image_url": {"url": "image.com/test.jpg"},
    }

    assert ContentPart.from_image_url(url="image.com/test.jpg", image_detail=ImageDetail.HIGH).to_openai_format() == {
        "type": "image_url",
        "image_url": {"url": "image.com/test.jpg", "detail": "high"},
    }


def test_to_openai_format_with_base64_image(test_files_path):
    # Function to encode the image
    def encode_image(image_path) -> ByteStream:
        with open(image_path, "rb") as image_file:
            return ByteStream(base64.b64encode(image_file.read()))

    image_path = test_files_path / "images" / "apple.jpg"

    base64_image = encode_image(image_path=image_path)

    assert ContentPart.from_base64_image(base64_image).to_openai_format() == {
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{base64_image.to_string()}"},
    }

    assert ContentPart.from_base64_image(image=base64_image, image_detail=ImageDetail.LOW).to_openai_format() == {
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{base64_image.to_string()}", "detail": "low"},
    }


def test_to_dict_with_text():
    assert ContentPart.from_text("Text").to_dict() == {"type": "text", "content": "Text"}


def test_from_dict_with_text():
    assert ContentPart.from_dict({"type": "text", "content": "content"}) == ContentPart.from_text("content")


def test_to_dict_With_image_url():
    assert ContentPart.from_image_url("image.com/test.jpg").to_dict() == {
        "type": "image_url",
        "content": "image.com/test.jpg",
    }

    assert ContentPart.from_image_url(url="image.com/test.jpg", image_detail=ImageDetail.HIGH).to_dict() == {
        "type": "image_url",
        "content": "image.com/test.jpg",
        "image_detail": "high",
    }


def test_from_dict_with_image_url():
    assert ContentPart.from_dict({"type": "image_url", "content": "image.com/test.jpg"}) == ContentPart.from_image_url(
        "image.com/test.jpg"
    )

    assert ContentPart.from_dict(
        {"type": "image_url", "content": "image.com/test.jpg", "image_detail": "high"}
    ) == ContentPart.from_image_url(url="image.com/test.jpg", image_detail=ImageDetail.HIGH)


def test_to_dict_with_base64_image(test_files_path):
    # Function to encode the image
    def encode_image(image_path) -> ByteStream:
        with open(image_path, "rb") as image_file:
            return ByteStream(base64.b64encode(image_file.read()))

    image_path = test_files_path / "images" / "apple.jpg"

    base64_image = encode_image(image_path=image_path)

    assert ContentPart.from_base64_image(base64_image).to_dict() == {
        "type": "image_base64",
        "content": base64_image.to_string(),
    }

    assert ContentPart.from_base64_image(image=base64_image, image_detail=ImageDetail.LOW).to_dict() == {
        "type": "image_base64",
        "content": base64_image.to_string(),
        "image_detail": "low",
    }


def test_from_dict_with_base64_image(test_files_path):
    # Function to encode the image
    def encode_image(image_path) -> ByteStream:
        with open(image_path, "rb") as image_file:
            return ByteStream(base64.b64encode(image_file.read()))

    image_path = test_files_path / "images" / "apple.jpg"

    base64_image = encode_image(image_path=image_path)

    assert ContentPart.from_dict(
        {"type": "image_base64", "content": base64_image.to_string()}
    ) == ContentPart.from_base64_image(base64_image)

    assert ContentPart.from_dict(
        {"type": "image_base64", "content": base64_image.to_string(), "image_detail": "low"}
    ) == ContentPart.from_base64_image(image=base64_image, image_detail=ImageDetail.LOW)
