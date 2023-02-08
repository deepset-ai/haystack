import pytest

from haystack.pipelines import TranslationWrapperPipeline, ExtractiveQAPipeline
from haystack.nodes import DensePassageRetriever, EmbeddingRetriever
from .test_summarizer import SPLIT_DOCS


# Keeping few (retriever,document_store,reader) combination to reduce test time
@pytest.mark.integration
@pytest.mark.elasticsearch
@pytest.mark.summarizer
@pytest.mark.parametrize("retriever,document_store,reader", [("embedding", "memory", "farm")], indirect=True)
def test_extractive_qa_pipeline_with_translator(
    document_store, retriever, reader, en_to_de_translator, de_to_en_translator
):
    document_store.write_documents(SPLIT_DOCS)

    if isinstance(retriever, EmbeddingRetriever) or isinstance(retriever, DensePassageRetriever):
        document_store.update_embeddings(retriever=retriever)

    query = "Wo steht der Eiffelturm?"
    base_pipeline = ExtractiveQAPipeline(retriever=retriever, reader=reader)
    pipeline = TranslationWrapperPipeline(
        input_translator=de_to_en_translator, output_translator=en_to_de_translator, pipeline=base_pipeline
    )
    output = pipeline.run(query=query, params={"Retriever": {"top_k": 2}})
    assert len(output["documents"]) == 2
    answers_texts = [el.answer for el in output["answers"]]

    assert "Frankreich" in answers_texts
