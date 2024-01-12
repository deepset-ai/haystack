from haystack import Document, Pipeline
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.rankers import TransformersSimilarityRanker
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever, InMemoryEmbeddingRetriever
from haystack.components.joiners.document_joiner import DocumentJoiner
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.evaluation.eval import eval


def test_hybrid_doc_search_pipeline():
    # Create the pipeline
    document_store = InMemoryDocumentStore()
    hybrid_pipeline = Pipeline()
    hybrid_pipeline.add_component(instance=InMemoryBM25Retriever(document_store=document_store), name="bm25_retriever")
    hybrid_pipeline.add_component(
        instance=SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2"), name="text_embedder"
    )
    hybrid_pipeline.add_component(
        instance=InMemoryEmbeddingRetriever(document_store=document_store), name="embedding_retriever"
    )
    hybrid_pipeline.add_component(instance=DocumentJoiner(), name="joiner")
    hybrid_pipeline.add_component(instance=TransformersSimilarityRanker(top_k=2), name="ranker")

    hybrid_pipeline.connect("bm25_retriever", "joiner")
    hybrid_pipeline.connect("text_embedder", "embedding_retriever")
    hybrid_pipeline.connect("embedding_retriever", "joiner")
    hybrid_pipeline.connect("joiner", "ranker")

    # Populate the document store
    documents = [
        Document(content="My name is Jean and I live in Paris."),
        Document(content="My name is Mark and I live in Berlin."),
        Document(content="My name is Mario and I live in the capital of Italy."),
        Document(content="My name is Giorgio and I live in Rome."),
    ]
    hybrid_pipeline.get_component("bm25_retriever").document_store.write_documents(documents)

    questions = ["Who lives in Paris?", "Who lives in Berlin?", "Who lives in Rome?"]
    inputs = [
        {"bm25_retriever": {"query": question}, "text_embedder": {"text": question}, "ranker": {"query": question}}
        for question in questions
    ]
    expected_outputs = [
        {
            "ranker": {
                "documents": [
                    Document(
                        id="6c90b78ad94e4e634e2a067b5fe2d26d4ce95405ec222cbaefaeb09ab4dce81e",
                        content="My name is Jean and I live in Paris.",
                        score=2.2277960777282715,
                    ),
                    Document(
                        id="10a183e965c2e107e20507c717f16559c58a8ba4bc7c577ea8dc32a8d6ca7a20",
                        content="My name is Mark and I live in Berlin.",
                        score=-7.304897308349609,
                    ),
                ]
            }
        },
        {
            "ranker": {
                "documents": [
                    Document(
                        id="10a183e965c2e107e20507c717f16559c58a8ba4bc7c577ea8dc32a8d6ca7a20",
                        content="My name is Mark and I live in Berlin.",
                        score=3.694173812866211,
                    ),
                    Document(
                        id="f7533b5c6c968680d0ef8e38f366d4e68b7ac0d7238f1b1b366d15cb9c33efd8",
                        content="My name is Mario and I live in the capital of Italy.",
                        score=-9.008655548095703,
                    ),
                ]
            }
        },
        {
            "ranker": {
                "documents": [
                    Document(
                        id="fb0f1efe94b3c78aa1c4e5a17a5ef8270f70e89d36a3665c8362675e8a769a27",
                        content="My name is Giorgio and I live in Rome.",
                        score=3.487802028656006,
                    ),
                    Document(
                        id="f7533b5c6c968680d0ef8e38f366d4e68b7ac0d7238f1b1b366d15cb9c33efd8",
                        content="My name is Mario and I live in the capital of Italy.",
                        score=-2.873128890991211,
                    ),
                ]
            }
        },
    ]

    eval_result = eval(hybrid_pipeline, inputs=inputs, expected_outputs=expected_outputs)

    assert eval_result.inputs == inputs
    assert eval_result.expected_outputs == expected_outputs
    assert len(eval_result.outputs) == len(expected_outputs) == len(inputs)
    assert eval_result.runnable.to_dict() == hybrid_pipeline.to_dict()
