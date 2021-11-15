import pytest

from haystack.schema import Document
from haystack.pipelines import TranslationWrapperPipeline, SearchSummarizationPipeline
from haystack.nodes import DensePassageRetriever, EmbeddingRetriever
from test_summarizer import SPLIT_DOCS

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
