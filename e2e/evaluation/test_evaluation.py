from haystack.document_stores.memory import InMemoryDocumentStore
from haystack.pipelines import GenerativeQAPipeline
from haystack.nodes import DensePassageRetriever, RAGenerator
from haystack.schema import MultiLabel, Label, Answer, Document, Span


EVAL_LABELS = [
    MultiLabel(
        labels=[
            Label(
                query="Who lives in Berlin?",
                answer=Answer(answer="Carla", offsets_in_context=[Span(11, 16)]),
                document=Document(
                    id="a0747b83aea0b60c4b114b15476dd32d",
                    content_type="text",
                    content="My name is Carla and I live in Berlin",
                ),
                is_correct_answer=True,
                is_correct_document=True,
                origin="gold-label",
            )
        ]
    ),
    MultiLabel(
        labels=[
            Label(
                query="Who lives in Munich?",
                answer=Answer(answer="Carla", offsets_in_context=[Span(11, 16)]),
                document=Document(
                    id="something_else", content_type="text", content="My name is Carla and I live in Munich"
                ),
                is_correct_answer=True,
                is_correct_document=True,
                origin="gold-label",
            )
        ]
    ),
]


def test_eval_generative_qa_rag_generator(docs):
    ds = InMemoryDocumentStore()
    retriever = DensePassageRetriever(
        document_store=ds,
        query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
        passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base",
        embed_title=True,
    )
    ds.write_documents(docs)
    ds.update_embeddings(retriever=retriever)
    rag_generator = RAGenerator(model_name_or_path="facebook/rag-token-nq", generator_type="token", max_length=20)

    pipeline = GenerativeQAPipeline(generator=rag_generator, retriever=retriever)
    eval_result = pipeline.eval(labels=EVAL_LABELS, params={"Retriever": {"top_k": 5}})

    metrics = eval_result.calculate_metrics(document_scope="document_id")

    assert "Retriever" in eval_result
    assert "Generator" in eval_result
    assert len(eval_result) == 2

    assert metrics["Retriever"]["mrr"] == 0.5
    assert metrics["Retriever"]["map"] == 0.5
    assert metrics["Retriever"]["recall_multi_hit"] == 0.5
    assert metrics["Retriever"]["recall_single_hit"] == 0.5
    assert metrics["Retriever"]["precision"] == 0.1
    assert metrics["Retriever"]["ndcg"] == 0.5
    assert metrics["Generator"]["exact_match"] == 0.0
    assert metrics["Generator"]["f1"] == 1.0 / 3
