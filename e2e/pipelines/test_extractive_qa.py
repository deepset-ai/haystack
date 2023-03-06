from haystack.nodes import TransformersTranslator, FARMReader, TfidfRetriever
from haystack.pipelines import ExtractiveQAPipeline, TranslationWrapperPipeline
from haystack.document_stores import InMemoryDocumentStore


def test_extractive_qa_answers_with_translator(docs):
    en_to_de_translator = TransformersTranslator(model_name_or_path="Helsinki-NLP/opus-mt-en-de")
    de_to_en_translator = TransformersTranslator(model_name_or_path="Helsinki-NLP/opus-mt-de-en")

    ds = InMemoryDocumentStore(use_bm25=False)
    retriever = TfidfRetriever(document_store=ds)
    reader = FARMReader(
        model_name_or_path="deepset/bert-medium-squad2-distilled", use_gpu=False, top_k_per_sample=5, num_processes=0
    )
    ds.write_documents(docs)

    base_pipeline = ExtractiveQAPipeline(reader=reader, retriever=retriever)
    pipeline = TranslationWrapperPipeline(
        input_translator=de_to_en_translator, output_translator=en_to_de_translator, pipeline=base_pipeline
    )

    prediction = pipeline.run(query="Wer lebt in Berlin?", params={"Reader": {"top_k": 3}})
    assert prediction is not None
    assert prediction["query"] == "Wer lebt in Berlin?"
    assert "Carla" in prediction["answers"][0].answer
    assert prediction["answers"][0].score <= 1
    assert prediction["answers"][0].score >= 0
    assert prediction["answers"][0].meta["meta_field"] == "test1"
    assert prediction["answers"][0].context == "My name is Carla and I live in Berlin"
