import json

from haystack.preview import Pipeline
from haystack.preview.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.preview.components.file_converters import TextFileToDocument
from haystack.preview.components.preprocessors import DocumentSplitter, DocumentCleaner
from haystack.preview.components.classifiers import DocumentLanguageClassifier
from haystack.preview.components.routers import FileTypeRouter, MetadataRouter
from haystack.preview.components.writers import DocumentWriter
from haystack.preview.document_stores import InMemoryDocumentStore


def test_preprocessing_pipeline(tmp_path):
    # Create the pipeline and its components
    document_store = InMemoryDocumentStore()
    preprocessing_pipeline = Pipeline()
    preprocessing_pipeline.add_component(instance=FileTypeRouter(mime_types=["text/plain"]), name="file_type_router")
    preprocessing_pipeline.add_component(instance=TextFileToDocument(), name="text_file_converter")
    preprocessing_pipeline.add_component(instance=DocumentLanguageClassifier(), name="language_classifier")
    preprocessing_pipeline.add_component(
        instance=MetadataRouter(rules={"en": {"language": {"$eq": "en"}}}), name="router"
    )
    preprocessing_pipeline.add_component(instance=DocumentCleaner(), name="cleaner")
    preprocessing_pipeline.add_component(
        instance=DocumentSplitter(split_by="sentence", split_length=1), name="splitter"
    )
    preprocessing_pipeline.add_component(
        instance=SentenceTransformersDocumentEmbedder(model_name_or_path="sentence-transformers/all-MiniLM-L6-v2"),
        name="embedder",
    )
    preprocessing_pipeline.add_component(instance=DocumentWriter(document_store=document_store), name="writer")
    preprocessing_pipeline.connect("file_type_router.text/plain", "text_file_converter.sources")
    preprocessing_pipeline.connect("text_file_converter.documents", "language_classifier.documents")
    preprocessing_pipeline.connect("language_classifier.documents", "router.documents")
    preprocessing_pipeline.connect("router.en", "cleaner.documents")
    preprocessing_pipeline.connect("cleaner.documents", "splitter.documents")
    preprocessing_pipeline.connect("splitter.documents", "embedder.documents")
    preprocessing_pipeline.connect("embedder.documents", "writer.documents")

    # Draw the pipeline
    preprocessing_pipeline.draw(tmp_path / "test_preprocessing_pipeline.png")

    # Serialize the pipeline to JSON
    with open(tmp_path / "test_preprocessing_pipeline.json", "w") as f:
        print(json.dumps(preprocessing_pipeline.to_dict(), indent=4))
        json.dump(preprocessing_pipeline.to_dict(), f)

    # Load the pipeline back
    with open(tmp_path / "test_preprocessing_pipeline.json", "r") as f:
        preprocessing_pipeline = Pipeline.from_dict(json.load(f))

    # Write a txt file
    with open(tmp_path / "test_file_english.txt", "w") as f:
        f.write(
            "This is an english sentence. There is more to it. It's a long text."
            "Spans multiple lines."
            ""
            "Even contains empty lines.  And extra whitespaces."
        )

    # Write a txt file
    with open(tmp_path / "test_file_german.txt", "w") as f:
        f.write("Ein deutscher Satz ohne Verb.")

    # Add two txt files and one non-txt file
    paths = [
        tmp_path / "test_file_english.txt",
        tmp_path / "test_file_german.txt",
        tmp_path / "test_preprocessing_pipeline.json",
    ]

    result = preprocessing_pipeline.run({"file_type_router": {"sources": paths}})

    assert result["writer"]["documents_written"] == 6
    filled_document_store = preprocessing_pipeline.get_component("writer").document_store
    assert filled_document_store.count_documents() == 6

    # Check preprocessed texts
    stored_documents = filled_document_store.filter_documents()
    expected_texts = [
        "This is an english sentence.",
        " There is more to it.",
        " It's a long text.",
        "Spans multiple lines.",
        "Even contains empty lines.",
        " And extra whitespaces.",
    ]
    assert expected_texts == [document.content for document in stored_documents]
    assert all(document.meta["language"] == "en" for document in stored_documents)
