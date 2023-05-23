import pytest

from haystack import Document
from haystack.nodes.image_to_text.transformers import TransformersImageToText
from haystack.nodes.image_to_text.base import BaseImageToText
from haystack.errors import ImageToTextError


@pytest.fixture
def image_file_paths(samples_path):
    image_file_names = ["apple.jpg", "car.jpg", "cat.jpg", "galaxy.jpg", "paris.jpg"]
    return [str(samples_path / "images" / file_name) for file_name in image_file_names]


@pytest.fixture
def image_docs(image_file_paths):
    return [Document(content=image_path, content_type="image") for image_path in image_file_paths]


EXPECTED_CAPTIONS = [
    "a red apple is sitting on a pile of hay",
    "a white car parked in a parking lot",
    "a cat laying in the grass",
    "a blurry photo of a blurry shot of a black object",
    "a city with a large building and a clock tower",
]


@pytest.fixture
def image_to_text():
    return TransformersImageToText(
        model_name_or_path="nlpconnect/vit-gpt2-image-captioning",
        devices=["cpu"],
        generation_kwargs={"max_new_tokens": 50},
    )


@pytest.mark.integration
def test_image_to_text_from_files(image_to_text, image_file_paths):
    assert isinstance(image_to_text, BaseImageToText)

    results = image_to_text.run(file_paths=image_file_paths)
    image_paths = [doc.meta["image_path"] for doc in results[0]["documents"]]
    assert image_paths == image_file_paths
    generated_captions = [doc.content for doc in results[0]["documents"]]
    assert generated_captions == EXPECTED_CAPTIONS


@pytest.mark.integration
def test_image_to_text_from_documents(image_to_text, image_file_paths, image_docs):
    results = image_to_text.run(documents=image_docs)
    image_paths = [doc.meta["image_path"] for doc in results[0]["documents"]]
    assert image_paths == image_file_paths
    generated_captions = [doc.content for doc in results[0]["documents"]]
    assert generated_captions == EXPECTED_CAPTIONS


@pytest.mark.integration
def test_image_to_text_from_files_and_documents(image_to_text, image_file_paths, image_docs):
    results = image_to_text.run(file_paths=image_file_paths[:3], documents=image_docs[3:])
    image_paths = [doc.meta["image_path"] for doc in results[0]["documents"]]
    assert image_paths == image_file_paths
    generated_captions = [doc.content for doc in results[0]["documents"]]
    assert generated_captions == EXPECTED_CAPTIONS


@pytest.mark.integration
def test_image_to_text_invalid_image(image_to_text, samples_path):
    markdown_path = str(samples_path / "markdown" / "sample.md")
    with pytest.raises(ImageToTextError, match="cannot identify image file"):
        image_to_text.run(file_paths=[markdown_path])


@pytest.mark.integration
def test_image_to_text_incorrect_path(image_to_text):
    with pytest.raises(ImageToTextError, match="Incorrect path"):
        image_to_text.run(file_paths=["wrong_path.jpg"])


@pytest.mark.integration
def test_image_to_text_not_image_document(image_to_text):
    textual_document = Document(content="this document is textual", content_type="text")
    with pytest.raises(ValueError, match="The ImageToText node only supports image documents."):
        image_to_text.run(documents=[textual_document])


@pytest.mark.integration
def test_image_to_text_unsupported_model_after_loading():
    with pytest.raises(
        ValueError,
        match="The model 'deepset/minilm-uncased-squad2' \(class 'BertForQuestionAnswering'\) is not supported for ImageToText",
    ):
        _ = TransformersImageToText(model_name_or_path="deepset/minilm-uncased-squad2")
