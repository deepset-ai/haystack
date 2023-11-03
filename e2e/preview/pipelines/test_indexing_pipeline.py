import os
import json

import pytest

from haystack.preview import Pipeline
from haystack.preview.components.audio import RemoteWhisperTranscriber, LocalWhisperTranscriber
from haystack.preview.components.routers import FileTypeRouter
from haystack.preview.components.writers import DocumentWriter
from haystack.preview.components.file_converters import (
    TextFileToDocument,
    HTMLToDocument,
    PyPDFToDocument,
    AzureOCRDocumentConverter,
    TikaDocumentConverter,
)

from haystack.preview.document_stores import InMemoryDocumentStore


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY", None) or not os.environ.get("AZURE_API_KEY", None),
    reason="Export an env var called OPENAI_API_KEY containing the OpenAI API key and an env var called AZURE_API_KEY containing the Azure API key to run this test.",
)
def test_indexing_pipeline(tmp_path, samples_path):
    # Create the pipeline and its components
    document_store = InMemoryDocumentStore()

    pipeline = Pipeline()
    pipeline.add_component(
        "router",
        FileTypeRouter(
            mime_types=[
                "text/plain",
                "text/html",
                "application/pdf",
                "audio/x-wav",
                "image/png",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ]
        ),
    )
    pipeline.add_component("text_converter", TextFileToDocument())
    pipeline.add_component("html_converter", HTMLToDocument())
    pipeline.add_component("pdf_converter", PyPDFToDocument())
    pipeline.add_component("ocr_converter", AzureOCRDocumentConverter())
    pipeline.add_component("tika_converter", TikaDocumentConverter())
    pipeline.add_component("remote_audio_converter", RemoteWhisperTranscriber(api_key=os.environ["OPENAI_API_KEY"]))
    pipeline.add_component("local_audio_converter", LocalWhisperTranscriber())

    # pipeline.add_component("join", DocumentsJoiner())
    pipeline.add_component("writer", DocumentWriter(document_store=document_store))

    pipeline.connect("router.text/plain", "text_converter")
    pipeline.connect("router.text/html", "html_converter")
    pipeline.connect("router.application/pdf", "pdf_converter")
    pipeline.connect("router.application/pdf", "ocr_converter")
    pipeline.connect("router.image/png", "ocr_converter")
    pipeline.connect("router.application/vnd.openxmlformats-officedocument.wordprocessingml.document", "ocr_converter")
    pipeline.connect("router.application/vnd.openxmlformats-officedocument.wordprocessingml.document", "tika_converter")
    pipeline.connect("router.audio/x-wav", "remote_audio_converter")
    pipeline.connect("router.audio/x-wav", "local_audio_converter")

    pipeline.connect("text_converter", "join.docs")
    pipeline.connect("html_converter", "join.docs")
    pipeline.connect("pdf_converter", "join.docs")
    pipeline.connect("ocr_converter", "join.docs")
    pipeline.connect("tika_converter", "join.docs")
    pipeline.connect("remote_audio_converter", "join.docs")
    pipeline.connect("local_audio_converter", "join.docs")
    pipeline.connect("join", "writer")

    # Draw the pipeline
    pipeline.draw(tmp_path / "test_indexing_pipeline.png")

    # Serialize the pipeline to JSON
    with open(tmp_path / "test_indexing_pipeline.json", "w") as f:
        print(json.dumps(pipeline.to_dict(), indent=4))
        json.dump(pipeline.to_dict(), f)

    # Load the pipeline back
    with open(tmp_path / "test_indexing_pipeline.json", "r") as f:
        pipeline = Pipeline.from_dict(json.load(f))

    # Run the pipeline
    outputs = pipeline.run(
        {
            "router": {
                "sources": [
                    samples_path / "preview" / "sample.pdf",
                    samples_path / "preview" / "sample.txt",
                    samples_path / "preview" / "sample.html",
                    samples_path / "preview" / "this is the content of the document.wav",
                    samples_path / "preview" / "sample.png",
                    samples_path / "preview" / "sample.docx",
                ]
            }
        }
    )

    print(outputs)
    assert False
