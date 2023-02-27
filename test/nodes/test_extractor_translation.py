import pytest

from haystack.pipelines import TranslationWrapperPipeline, ExtractiveQAPipeline
from haystack.nodes import DensePassageRetriever, EmbeddingRetriever
from haystack.schema import Document


SPLIT_DOCS = [
    Document(
        content="""The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building, and the tallest structure in Paris. Its base is square, measuring 125 metres (410 ft) on each side. During its construction, the Eiffel Tower surpassed the Washington Monument to become the tallest man-made structure in the world, a title it held for 41 years until the Chrysler Building in New York City was finished in 1930."""
    ),
    Document(
        content="""It was the first structure to reach a height of 300 metres. Due to the addition of a broadcasting aerial at the top of the tower in 1957, it is now taller than the Chrysler Building by 5.2 metres (17 ft). Excluding transmitters, the Eiffel Tower is the second tallest free-standing structure in France after the Millau Viaduct."""
    ),
]


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
