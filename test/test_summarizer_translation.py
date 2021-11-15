import pytest

from haystack.schema import Document
from haystack.pipelines import TranslationWrapperPipeline, SearchSummarizationPipeline
from haystack.nodes import DensePassageRetriever, EmbeddingRetriever

SPLIT_DOCS = [
    Document(
        content="""The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building, and the tallest structure in Paris. Its base is square, measuring 125 metres (410 ft) on each side. During its construction, the Eiffel Tower surpassed the Washington Monument to become the tallest man-made structure in the world, a title it held for 41 years until the Chrysler Building in New York City was finished in 1930."""
    ),
    Document(
        content="""It was the first structure to reach a height of 300 metres. Due to the addition of a broadcasting aerial at the top of the tower in 1957, it is now taller than the Chrysler Building by 5.2 metres (17 ft). Excluding transmitters, the Eiffel Tower is the second tallest free-standing structure in France after the Millau Viaduct."""
    )
]

# Keeping few (retriever,document_store) combination to reduce test time
@pytest.mark.slow
@pytest.mark.elasticsearch
@pytest.mark.summarizer
@pytest.mark.parametrize(
    "retriever,document_store",
    [("embedding", "memory"), ("elasticsearch", "elasticsearch")],
    indirect=True,
)
def test_summarization_pipeline_with_translator(
    document_store,
    retriever,
    summarizer,
    en_to_de_translator,
    de_to_en_translator
):
    document_store.write_documents(SPLIT_DOCS)

    if isinstance(retriever, EmbeddingRetriever) or isinstance(retriever, DensePassageRetriever):
        document_store.update_embeddings(retriever=retriever)

    query = "Wo steht der Eiffelturm?"
    base_pipeline = SearchSummarizationPipeline(retriever=retriever, summarizer=summarizer)
    pipeline = TranslationWrapperPipeline(
        input_translator=de_to_en_translator,
        output_translator=en_to_de_translator,
        pipeline=base_pipeline
    )
    output = pipeline.run(query=query, params={"Retriever": {"top_k": 2}, "Summarizer": {"generate_single_summary": True}})
    # SearchSummarizationPipeline return answers but Summarizer return documents
    documents = output["documents"]
    assert len(documents) == 1
    assert documents[0].content in [
        "Der Eiffelturm ist ein Wahrzeichen in Paris, Frankreich.",
        "Der Eiffelturm, der 1889 in Paris, Frankreich, erbaut wurde, ist das höchste freistehende Bauwerk der Welt."
    ]
