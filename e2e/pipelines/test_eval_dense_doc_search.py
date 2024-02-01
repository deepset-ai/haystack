from haystack import Pipeline
from haystack.components.converters import PyPDFToDocument, TextFileToDocument
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack.components.joiners import DocumentJoiner
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.routers import FileTypeRouter
from haystack.components.writers import DocumentWriter
from haystack.dataclasses import Document
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.evaluation.eval import eval


def test_dense_doc_search_pipeline(samples_path):
    # Create the indexing pipeline
    indexing_pipeline = Pipeline()
    file_type_router = FileTypeRouter(mime_types=["text/plain", "application/pdf"])
    text_file_converter = TextFileToDocument()
    pdf_file_converter = PyPDFToDocument()
    joiner = DocumentJoiner()
    cleaner = DocumentCleaner()
    splitter = DocumentSplitter(split_by="sentence", split_length=250, split_overlap=30)
    embedder = SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
    writer = DocumentWriter(document_store=InMemoryDocumentStore())
    indexing_pipeline.add_component(instance=file_type_router, name="file_type_router")
    indexing_pipeline.add_component(instance=text_file_converter, name="text_file_converter")
    indexing_pipeline.add_component(instance=pdf_file_converter, name="pdf_file_converter")
    indexing_pipeline.add_component(instance=joiner, name="joiner")
    indexing_pipeline.add_component(instance=cleaner, name="cleaner")
    indexing_pipeline.add_component(instance=splitter, name="splitter")
    indexing_pipeline.add_component(instance=embedder, name="embedder")
    indexing_pipeline.add_component(instance=writer, name="writer")

    indexing_pipeline.connect("file_type_router.outputs.text/plain", text_file_converter.inputs.sources)
    indexing_pipeline.connect("file_type_router.outputs.application/pdf", pdf_file_converter.inputs.sources)
    indexing_pipeline.connect(text_file_converter.outputs.documents, joiner.inputs.documents)
    indexing_pipeline.connect(pdf_file_converter.outputs.documents, joiner.inputs.documents)
    indexing_pipeline.connect(joiner.outputs.documents, cleaner.inputs.documents)
    indexing_pipeline.connect(cleaner.outputs.documents, splitter.inputs.documents)
    indexing_pipeline.connect(splitter.outputs.documents, embedder.inputs.documents)
    indexing_pipeline.connect(embedder.outputs.documents, writer.inputs.documents)

    indexing_pipeline.run({"file_type_router": {"sources": list(samples_path.iterdir())}})
    filled_document_store = indexing_pipeline.get_component("writer").document_store

    # Create the querying pipeline
    query_pipeline = Pipeline()
    query_pipeline.add_component(
        instance=SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2"), name="text_embedder"
    )
    query_pipeline.add_component(
        instance=InMemoryEmbeddingRetriever(document_store=filled_document_store, top_k=20), name="embedding_retriever"
    )
    query_pipeline.connect("text_embedder", "embedding_retriever")

    inputs = [{"text_embedder": {"text": "Who lives in Rome?"}}]
    expected_outputs = [
        {
            "embedding_retriever": {
                "documents": [
                    Document(
                        id="d219162e5d0b8e5eab901e32ce0d9c12d24e5ea26a92780442fcfa560eb0b7d6",
                        content="My name is Giorgio and I live in Rome.",
                        meta={
                            "file_path": "/home/ashwin/data_science/0ashwin/opensource/haystack/e2e/samples/doc_1.txt",
                            "source_id": "0366ae1654f4573564e29184cd4a2232286a93f4f25d6790ce703ae7d4d7d63c",
                        },
                        score=0.627746287158654,
                    ),
                    Document(
                        id="2dcf2bc0307ba21fbb7e97a307d987a05297e577a44f170081acdbab9fc4b95f",
                        content="A sample PDF ﬁle History and standardizationFormat (PDF) Adobe Systems made the PDF speciﬁcation ava...",
                        meta={"source_id": "ec1ac6c430ecd0cc74ae56f3e2d84f93fef3f5393de6901fe8aa01e494ebcdbe"},
                        score=-0.060180130727963355,
                    ),
                ]
            }
        }
    ]

    eval_result = eval(query_pipeline, inputs=inputs, expected_outputs=expected_outputs)

    assert eval_result.inputs == inputs
    assert eval_result.expected_outputs == expected_outputs
    assert len(eval_result.outputs) == len(expected_outputs) == len(inputs)
    assert eval_result.runnable.to_dict() == query_pipeline.to_dict()
