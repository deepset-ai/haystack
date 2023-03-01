from haystack import Document
from haystack.pipelines import TranslationWrapperPipeline, GenerativeQAPipeline
from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import DensePassageRetriever, RAGenerator, TransformersTranslator


def test_generative_pipeline_with_translator():
    docs = [
        Document(content="The capital of Germany is the city state of Berlin."),
        Document(content="Berlin is the capital and largest city of Germany by both area and population."),
    ]
    ds = InMemoryDocumentStore(use_bm25=True)
    ds.write_documents(docs)
    retriever = DensePassageRetriever(  # Needs DPR or RAGenerator will thrown an exception...
        document_store=ds,
        query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
        passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base",
        use_gpu=False,
        embed_title=True,
    )
    ds.update_embeddings(retriever=retriever)
    rag_generator = RAGenerator(
        model_name_or_path="facebook/rag-token-nq", generator_type="token", max_length=20, retriever=retriever
    )
    en_to_de_translator = TransformersTranslator(model_name_or_path="Helsinki-NLP/opus-mt-en-de")
    de_to_en_translator = TransformersTranslator(model_name_or_path="Helsinki-NLP/opus-mt-de-en")

    query = "Was ist die Hauptstadt der Bundesrepublik Deutschland?"
    base_pipeline = GenerativeQAPipeline(retriever=retriever, generator=rag_generator)
    pipeline = TranslationWrapperPipeline(
        input_translator=de_to_en_translator, output_translator=en_to_de_translator, pipeline=base_pipeline
    )
    output = pipeline.run(query=query, params={"Generator": {"top_k": 2}, "Retriever": {"top_k": 1}})
    answers = output["answers"]
    assert len(answers) == 2
    assert "berlin" in answers[0].answer.lower()
