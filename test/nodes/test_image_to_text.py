import os
import pytest

from PIL import UnidentifiedImageError

from haystack import Document
from haystack.nodes.image_to_text.transformers import TransformersImageToText
from haystack.nodes.image_to_text.base import BaseImageToText

from ..conftest import SAMPLES_PATH


IMAGE_FILE_NAMES = ["apple.jpg", "car.jpg", "cat.jpg", "galaxy.jpg", "paris.jpg"]
IMAGE_FILE_PATHS = [os.path.join(SAMPLES_PATH, "images", file_name) for file_name in IMAGE_FILE_NAMES]
IMAGE_DOCS = [Document(content=image_path, content_type="image") for image_path in IMAGE_FILE_PATHS]
INVALID_IMAGE_FILE_PATH = str(SAMPLES_PATH / "markdown" / "sample.md")

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
def test_image_to_text(image_to_text):
    assert isinstance(image_to_text, BaseImageToText)

    results_0 = image_to_text.run(file_paths=IMAGE_FILE_PATHS)
    image_paths_0 = [doc.meta["image_path"] for doc in results_0[0]["documents"]]
    assert image_paths_0 == IMAGE_FILE_PATHS
    generated_captions_0 = [doc.content for doc in results_0[0]["documents"]]
    assert generated_captions_0 == EXPECTED_CAPTIONS

    results_1 = image_to_text.run(documents=IMAGE_DOCS)
    image_paths_1 = [doc.meta["image_path"] for doc in results_1[0]["documents"]]
    assert image_paths_1 == IMAGE_FILE_PATHS
    generated_captions_1 = [doc.content for doc in results_1[0]["documents"]]
    assert generated_captions_1 == EXPECTED_CAPTIONS

    results_2 = image_to_text.run(file_paths=IMAGE_FILE_PATHS[:3], documents=IMAGE_DOCS[3:])
    image_paths_2 = [doc.meta["image_path"] for doc in results_2[0]["documents"]]
    assert image_paths_2 == IMAGE_FILE_PATHS
    generated_captions_2 = [doc.content for doc in results_2[0]["documents"]]
    assert generated_captions_2 == EXPECTED_CAPTIONS


@pytest.mark.integration
def test_image_to_text_invalid_image(image_to_text):
    with pytest.raises(UnidentifiedImageError, match="cannot identify image file"):
        image_to_text.run(file_paths=[INVALID_IMAGE_FILE_PATH])
