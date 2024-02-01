import json

from haystack import Pipeline
from haystack.components.classifiers import DocumentLanguageClassifier
from haystack.components.converters import TextFileToDocument
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.routers import FileTypeRouter, MetadataRouter
from haystack.components.writers import DocumentWriter
from haystack.document_stores.in_memory import InMemoryDocumentStore


def test_preprocessing_pipeline(tmp_path):
    # Create the pipeline and its components
    document_store = InMemoryDocumentStore()
    preprocessing_pipeline = Pipeline()
    file_type_router = FileTypeRouter(mime_types=["text/plain"])
    text_file_converter = TextFileToDocument()
    language_classifier = DocumentLanguageClassifier()
    router = MetadataRouter(rules={"en": {"field": "language", "operator": "==", "value": "en"}})
    cleaner = DocumentCleaner()
    splitter = DocumentSplitter(split_by="sentence", split_length=1)
    embedder = SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
    writer = DocumentWriter(document_store=document_store)
    preprocessing_pipeline.add_component(instance=file_type_router, name="file_type_router")
    preprocessing_pipeline.add_component(instance=text_file_converter, name="text_file_converter")
    preprocessing_pipeline.add_component(instance=language_classifier, name="language_classifier")
    preprocessing_pipeline.add_component(instance=router, name="router")
    preprocessing_pipeline.add_component(instance=cleaner, name="cleaner")
    preprocessing_pipeline.add_component(instance=splitter, name="splitter")
    preprocessing_pipeline.add_component(instance=embedder, name="embedder")
    preprocessing_pipeline.add_component(instance=writer, name="writer")
    preprocessing_pipeline.connect("file_type_router.text/plain", text_file_converter.inputs.sources)
    preprocessing_pipeline.connect(text_file_converter.outputs.documents, language_classifier.inputs.documents)
    preprocessing_pipeline.connect(language_classifier.outputs.documents, router.inputs.documents)
    preprocessing_pipeline.connect(router.outputs.en, cleaner.inputs.documents)
    preprocessing_pipeline.connect(cleaner.outputs.documents, splitter.inputs.documents)
    preprocessing_pipeline.connect(splitter.outputs.documents, embedder.inputs.documents)
    preprocessing_pipeline.connect(embedder.outputs.documents, writer.inputs.documents)

    # Draw the pipeline
    preprocessing_pipeline.draw(tmp_path / "test_preprocessing_pipeline.png")

    # Serialize the pipeline to YAML
    with open(tmp_path / "test_preprocessing_pipeline.yaml", "w") as f:
        preprocessing_pipeline.dump(f)

    # Load the pipeline back
    with open(tmp_path / "test_preprocessing_pipeline.yaml", "r") as f:
        preprocessing_pipeline = Pipeline.load(f)

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
