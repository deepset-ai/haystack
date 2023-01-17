import pytest
import os
from pathlib import Path

from haystack import Document
from haystack.nodes.image_to_text.transformers import TransformersImageToText
from haystack.nodes.image_to_text.base import BaseImageToText

from ..conftest import SAMPLES_PATH

IMAGE_FILE_PATHS = sorted([str(image_path) for image_path in Path(SAMPLES_PATH / "images").glob("*.jpg")])

IMAGE_DOCS = [Document(content=image_path, content_type="image") for image_path in IMAGE_FILE_PATHS]

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

    results = image_to_text.run(file_paths=IMAGE_FILE_PATHS)
    generated_captions = [doc.content for doc in results[0]["documents"]]

    assert generated_captions == EXPECTED_CAPTIONS


# improve!!!!

# no image!

#     docs = [
#         Document(
#             content="""That's good. I like it.""" * 700,  # extra long text to check truncation
#             meta={"name": "0"},
#             id="1",
#         ),
#         Document(content="""That's bad. I don't like it.""", meta={"name": "1"}, id="2"),
#     ]
#     results = document_classifier.predict(documents=docs)
#     expected_labels = ["joy", "sadness"]
#     for i, doc in enumerate(results):
#         assert doc.to_dict()["meta"]["classification"]["label"] == expected_labels[i]


# # test node
# ti2t = TransformersImageToText(model_name_or_path="nlpconnect/vit-gpt2-image-captioning", batch_size=1, generation_kwargs={'max_new_tokens':50})
# # print(ti2t.generate_captions(image_file_paths=glob.glob('/home/anakin87/apps/haystack/test/samples/images/*.jpg')))

# # # test in a pipeline, passing file_paths
# # from haystack.pipelines import Pipeline

# # p = Pipeline()
# # p.add_node(component=ti2t, name="ti2t", inputs=["File"])


# # print(p.run(file_paths=glob.glob('/home/anakin87/apps/haystack/test/samples/images/*.jpg')[:2]))

# # test in a pipeline, passing documents
# from haystack.pipelines import Pipeline
# # from haystack.document_stores import InMemoryDocumentStore
# from haystack import Document

# # ds = InMemoryDocumentStore()
# file_paths=glob.glob('/home/anakin87/apps/haystack/test/samples/images/*.jpg')

# docs= []
# for path in file_paths:
#     doc = Document(content=path, content_type="image")
#     docs.append(doc)

# print(ti2t.run(documents=docs))
