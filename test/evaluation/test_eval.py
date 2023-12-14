import pytest

from haystack import Document, Pipeline
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack.components.generators import HuggingFaceLocalGenerator
from haystack.components.rankers import TransformersSimilarityRanker
from haystack.components.retrievers import InMemoryBM25Retriever, InMemoryEmbeddingRetriever
from haystack.components.routers.document_joiner import DocumentJoiner
from haystack.components.writers import DocumentWriter
from haystack.document_stores import InMemoryDocumentStore
from haystack.evaluation.eval import eval


@pytest.mark.integration
def test_rag_pipeline():
    """
    Test for evaluating a Retrieval-Augmented Generation (RAG) Pipeline without using a EmbeddingRetriever.
    This test creates a RAG pipeline using an InMemoryDocumentStore, BM25 Retriever, PromptBuilder, HuggingFaceLocalGenerator.

    The pipeline is evaluated on a set of input questions and their corresponding expected answers.
    """
    prompt_template = """Given these documents, answer the question.\n
    Documents:
    {% for doc in documents %}
        {{ doc.content }}
    {% endfor %}
    Question: {{question}}
    Answer:
    """

    rag_pipeline = Pipeline()

    rag_pipeline.add_component(instance=InMemoryBM25Retriever(document_store=InMemoryDocumentStore()), name="retriever")
    rag_pipeline.add_component(instance=PromptBuilder(template=prompt_template), name="prompt_builder")
    rag_pipeline.add_component(
        instance=HuggingFaceLocalGenerator(
            model_name_or_path="google/flan-t5-small",
            task="text2text-generation",
            generation_kwargs={"max_new_tokens": 100, "temperature": 0.5, "do_sample": True},
        ),
        name="llm",
    )

    rag_pipeline.connect("retriever", "prompt_builder.documents")
    rag_pipeline.connect("prompt_builder", "llm")

    documents = [
        Document(content="My name is Jean and I live in Paris."),
        Document(content="My name is Mark and I live in Berlin."),
        Document(content="My name is Giorgio and I live in Rome."),
    ]

    document_store = rag_pipeline.get_component("retriever").document_store
    document_store.write_documents(documents)

    questions = ["Who lives in Paris?", "Who lives in Berlin?", "Who lives in Rome?"]
    inputs = [{"retriever": {"query": question}, "prompt_builder": {"question": question}} for question in questions]

    expected_outputs = [
        {"llm": {"replies": ["Jean"]}},
        {"llm": {"replies": ["Mark"]}},
        {"llm": {"replies": ["Giorgio"]}},
    ]

    eval_result = eval(rag_pipeline, inputs=inputs, expected_outputs=expected_outputs)

    assert eval_result.runnable == rag_pipeline
    assert eval_result.inputs == inputs
    assert eval_result.expected_outputs == expected_outputs
    assert len(eval_result.outputs) == len(expected_outputs) == len(inputs)


@pytest.mark.integration
def test_embedding_retrieval_rag_pipeline():
    """
    Test for evaluating a Retrieval-Augmented Generation (RAG) Pipeline using a EmbeddingRetriever.
    This test creates a RAG pipeline using an InMemoryDocumentStore, SentenceTransformer Embedders, Embedding Retriever, PromptBuilder, HuggingFaceLocalGenerator.

    The pipeline is evaluated on a set of input questions and their corresponding expected answers.
    """
    prompt_template = """Given these documents, answer the question.\n
    Documents:
    {% for doc in documents %}
        {{ doc.content }}
    {% endfor %}
    Question: {{question}}
    Answer:
    """

    rag_pipeline = Pipeline()
    rag_pipeline.add_component(
        instance=SentenceTransformersTextEmbedder(model_name_or_path="sentence-transformers/all-MiniLM-L6-v2"),
        name="text_embedder",
    )
    rag_pipeline.add_component(
        instance=InMemoryEmbeddingRetriever(document_store=InMemoryDocumentStore()), name="retriever"
    )
    rag_pipeline.add_component(instance=PromptBuilder(template=prompt_template), name="prompt_builder")
    rag_pipeline.add_component(
        instance=HuggingFaceLocalGenerator(
            model_name_or_path="google/flan-t5-small",
            task="text2text-generation",
            generation_kwargs={"max_new_tokens": 100, "temperature": 0.5, "do_sample": True},
        ),
        name="llm",
    )

    rag_pipeline.connect("text_embedder", "retriever")
    rag_pipeline.connect("retriever", "prompt_builder.documents")
    rag_pipeline.connect("prompt_builder", "llm")

    documents = [
        Document(content="My name is Jean and I live in Paris."),
        Document(content="My name is Mark and I live in Berlin."),
        Document(content="My name is Giorgio and I live in Rome."),
    ]

    document_store = rag_pipeline.get_component("retriever").document_store
    indexing_pipeline = Pipeline()
    indexing_pipeline.add_component(
        instance=SentenceTransformersDocumentEmbedder(model_name_or_path="sentence-transformers/all-MiniLM-L6-v2"),
        name="document_embedder",
    )
    indexing_pipeline.add_component(instance=DocumentWriter(document_store=document_store), name="document_writer")
    indexing_pipeline.connect("document_embedder", "document_writer")
    indexing_pipeline.run({"document_embedder": {"documents": documents}})

    questions = ["Who lives in Paris?", "Who lives in Berlin?", "Who lives in Rome?"]
    inputs = [{"text_embedder": {"text": question}, "prompt_builder": {"question": question}} for question in questions]

    expected_outputs = [
        {"llm": {"replies": ["Jean"]}},
        {"llm": {"replies": ["Mark"]}},
        {"llm": {"replies": ["Giorgio"]}},
    ]

    eval_result = eval(rag_pipeline, inputs=inputs, expected_outputs=expected_outputs)

    assert eval_result.runnable == rag_pipeline
    assert eval_result.inputs == inputs
    assert eval_result.expected_outputs == expected_outputs
    assert len(eval_result.outputs) == len(expected_outputs) == len(inputs)


@pytest.mark.integration
def test_hybrid_doc_search():
    """
    Test for evaluating a Hybrid RAG Pipeline.

    The Hybrid RAG Pipeline pipeline is created using the InMemoryDocumentStore, BM25 Retriever, EmbeddingRetriever, SentenceTransformer Embedders, DocumentJoiner and TransformersSimilarityRanker components.

    The pipeline is evaluated on a set of input questions and their corresponding expected answers.
    """
    document_store = InMemoryDocumentStore()
    hybrid_pipeline = Pipeline()
    hybrid_pipeline.add_component(instance=InMemoryBM25Retriever(document_store=document_store), name="bm25_retriever")
    hybrid_pipeline.add_component(
        instance=SentenceTransformersTextEmbedder(model_name_or_path="sentence-transformers/all-MiniLM-L6-v2"),
        name="text_embedder",
    )
    hybrid_pipeline.add_component(
        instance=InMemoryEmbeddingRetriever(document_store=document_store), name="embedding_retriever"
    )
    hybrid_pipeline.add_component(instance=DocumentJoiner(), name="joiner")
    hybrid_pipeline.add_component(instance=TransformersSimilarityRanker(top_k=20), name="ranker")

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

    questions = [
        "Who lives in Paris?",
        "Who lives in Berlin?",
        "Who lives in Rome?",
        "Who lives in the capital of Italy?",
    ]
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
                    Document(
                        id="fb0f1efe94b3c78aa1c4e5a17a5ef8270f70e89d36a3665c8362675e8a769a27",
                        content="My name is Giorgio and I live in Rome.",
                        score=-7.6049394607543945,
                    ),
                    Document(
                        id="f7533b5c6c968680d0ef8e38f366d4e68b7ac0d7238f1b1b366d15cb9c33efd8",
                        content="My name is Mario and I live in the capital of Italy.",
                        score=-7.680310249328613,
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
                    Document(
                        id="6c90b78ad94e4e634e2a067b5fe2d26d4ce95405ec222cbaefaeb09ab4dce81e",
                        content="My name is Jean and I live in Paris.",
                        score=-9.615274429321289,
                    ),
                    Document(
                        id="fb0f1efe94b3c78aa1c4e5a17a5ef8270f70e89d36a3665c8362675e8a769a27",
                        content="My name is Giorgio and I live in Rome.",
                        score=-9.727143287658691,
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
                    Document(
                        id="10a183e965c2e107e20507c717f16559c58a8ba4bc7c577ea8dc32a8d6ca7a20",
                        content="My name is Mark and I live in Berlin.",
                        score=-8.914161682128906,
                    ),
                    Document(
                        id="6c90b78ad94e4e634e2a067b5fe2d26d4ce95405ec222cbaefaeb09ab4dce81e",
                        content="My name is Jean and I live in Paris.",
                        score=-9.272953033447266,
                    ),
                ]
            }
        },
        {
            "ranker": {
                "documents": [
                    Document(
                        id="f7533b5c6c968680d0ef8e38f366d4e68b7ac0d7238f1b1b366d15cb9c33efd8",
                        content="My name is Mario and I live in the capital of Italy.",
                        score=5.7425055503845215,
                    ),
                    Document(
                        id="fb0f1efe94b3c78aa1c4e5a17a5ef8270f70e89d36a3665c8362675e8a769a27",
                        content="My name is Giorgio and I live in Rome.",
                        score=-3.1027421951293945,
                    ),
                    Document(
                        id="10a183e965c2e107e20507c717f16559c58a8ba4bc7c577ea8dc32a8d6ca7a20",
                        content="My name is Mark and I live in Berlin.",
                        score=-9.590137481689453,
                    ),
                    Document(
                        id="6c90b78ad94e4e634e2a067b5fe2d26d4ce95405ec222cbaefaeb09ab4dce81e",
                        content="My name is Jean and I live in Paris.",
                        score=-10.2391357421875,
                    ),
                ]
            }
        },
    ]

    eval_result = eval(hybrid_pipeline, inputs=inputs, expected_outputs=expected_outputs)

    assert eval_result.runnable == hybrid_pipeline
    assert eval_result.inputs == inputs
    assert eval_result.expected_outputs == expected_outputs
    assert len(eval_result.outputs) == len(expected_outputs) == len(inputs)
