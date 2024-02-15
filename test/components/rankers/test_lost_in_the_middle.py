import pytest
from datasets import load_dataset
from haystack import Document, Pipeline
from haystack.components.builders import AnswerBuilder, PromptBuilder
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack.components.generators import HuggingFaceLocalGenerator
from haystack.components.rankers.lost_in_the_middle import LostInTheMiddleRanker
from haystack.components.retrievers import InMemoryBM25Retriever, InMemoryEmbeddingRetriever
from haystack.components.writers import DocumentWriter
from haystack.document_stores import InMemoryDocumentStore


@pytest.mark.unit
def test_lost_in_the_middle_order_odd():
    # tests that lost_in_the_middle order works with an odd number of documents
    docs = [Document(content=str(i)) for i in range(1, 10)]
    ranker = LostInTheMiddleRanker()
    result = ranker.run(query="", documents=docs)
    assert result["documents"]
    expected_order = "1 3 5 7 9 8 6 4 2".split()
    assert all(doc.content == expected_order[idx] for idx, doc in enumerate(result["documents"]))


@pytest.mark.unit
def test_lost_in_the_middle_order_even():
    # tests that lost_in_the_middle order works with an even number of documents
    docs = [Document(content=str(i)) for i in range(1, 11)]
    ranker = LostInTheMiddleRanker()
    # result, _ = ranker.run(query="", documents=docs)
    result = ranker.run(query="", documents=docs)
    expected_order = "1 3 5 7 9 10 8 6 4 2".split()
    assert all(doc.content == expected_order[idx] for idx, doc in enumerate(result["documents"]))


@pytest.mark.unit
def test_lost_in_the_middle_order_two_docs():
    # tests that lost_in_the_middle order works with two documents
    ranker = LostInTheMiddleRanker()
    # two docs
    docs = [Document(content="1"), Document(content="2")]
    # result, _ = ranker.run(query="", documents=docs)
    result = ranker.run(query="", documents=docs)
    assert result["documents"][0].content == "1"
    assert result["documents"][1].content == "2"


@pytest.mark.unit
def test_lost_in_the_middle_init():
    # tests that LostInTheMiddleRanker initializes with default values
    ranker = LostInTheMiddleRanker()
    assert ranker.word_count_threshold is None

    ranker = LostInTheMiddleRanker(word_count_threshold=10)
    assert ranker.word_count_threshold == 10


@pytest.mark.unit
def test_lost_in_the_middle_init_invalid_word_count_threshold():
    # tests that LostInTheMiddleRanker raises an error when word_count_threshold is <= 0
    with pytest.raises(ValueError, match="Invalid value for word_count_threshold"):
        LostInTheMiddleRanker(word_count_threshold=0)

    with pytest.raises(ValueError, match="Invalid value for word_count_threshold"):
        LostInTheMiddleRanker(word_count_threshold=-5)


@pytest.mark.unit
def test_lost_in_the_middle_with_word_count_threshold():
    # tests that lost_in_the_middle with word_count_threshold works as expected
    ranker = LostInTheMiddleRanker(word_count_threshold=6)
    docs = [Document(content="word" + str(i)) for i in range(1, 10)]
    # result, _ = ranker.run(query="", documents=docs)
    result = ranker.run(query="", documents=docs)
    expected_order = "word1 word3 word5 word6 word4 word2".split()
    assert all(doc.content == expected_order[idx] for idx, doc in enumerate(result["documents"]))

    ranker = LostInTheMiddleRanker(word_count_threshold=9)
    # result, _ = ranker.run(query="", documents=docs)
    result = ranker.run(query="", documents=docs)
    expected_order = "word1 word3 word5 word7 word9 word8 word6 word4 word2".split()
    assert all(doc.content == expected_order[idx] for idx, doc in enumerate(result["documents"]))


@pytest.mark.unit
def test_word_count_threshold_greater_than_total_number_of_words_returns_all_documents():
    ranker = LostInTheMiddleRanker(word_count_threshold=100)
    docs = [Document(content="word" + str(i)) for i in range(1, 10)]
    ordered_docs = ranker.run(query="test", documents=docs)
    # assert len(ordered_docs) == len(docs)
    expected_order = "word1 word3 word5 word7 word9 word8 word6 word4 word2".split()
    assert all(doc.content == expected_order[idx] for idx, doc in enumerate(ordered_docs["documents"]))


@pytest.mark.unit
def test_empty_documents_returns_empty_list():
    ranker = LostInTheMiddleRanker()
    result = ranker.run(query="test", documents=[])
    assert result == {"documents": []}


@pytest.mark.unit
def test_list_of_one_document_returns_same_document():
    ranker = LostInTheMiddleRanker()
    doc = Document(content="test", content_type="text")
    assert ranker.run(query="test", documents=[doc]) == {"documents": [doc]}


@pytest.mark.unit
@pytest.mark.parametrize("top_k", [1, 2, 3, 4, 5, 6, 7, 8, 12, 20])
def test_lost_in_the_middle_order_with_postive_top_k(top_k: int):
    # tests that lost_in_the_middle order works with an odd number of documents and a top_k parameter
    docs = [Document(content=str(i)) for i in range(1, 10)]
    ranker = LostInTheMiddleRanker()
    result = ranker.run(query="irrelevant", documents=docs, top_k=top_k)
    if top_k < len(docs):
        # top_k is less than the number of documents, so only the top_k documents should be returned in LITM order
        assert len(result["documents"]) == top_k
        expected_order = ranker.run(
            query="irrelevant", documents=[Document(content=str(i)) for i in range(1, top_k + 1)]
        )
        assert result == expected_order
    else:
        # top_k is greater than the number of documents, so all documents should be returned in LITM order
        assert len(result["documents"]) == len(docs)
        assert result == ranker.run(query="irrelevant", documents=docs)


@pytest.mark.unit
@pytest.mark.parametrize("top_k", [-20, -10, -5, -1])
def test_lost_in_the_middle_order_with_negative_top_k(top_k: int):
    # tests that lost_in_the_middle order works with an odd number of documents and an invalid top_k parameter
    docs = [Document(content=str(i)) for i in range(1, 10)]
    ranker = LostInTheMiddleRanker()
    result = ranker.run(query="irrelevant", documents=docs, top_k=top_k)
    if top_k < len(docs) * -1:
        assert len(result["documents"]) == 0  # top_k is too negative, so no documents should be returned
    else:
        # top_k is negative, subtract it from the total number of documents to get the expected number of documents
        expected_docs = ranker.run(query="irrelevant", documents=docs, top_k=len(docs) + top_k)
        assert result == expected_docs


@pytest.mark.unit
def to_dict(self):
    component = LostInTheMiddleRanker()
    data = component.to_dict()
    assert data == {
        "type": "haystack.components.rankers.lost_in_the_middle.LostInTheMiddleRanker",
        "init_parameters": {"word_count_threshold": "None", "top_k": "None"},
    }


@pytest.mark.integration
def test_ranker_retrieval_pipeline():
    docs = [
        Document(content="Paris is in France"),
        Document(content="Berlin is in Germany"),
        Document(content="Lyon is in France"),
    ]

    document_store = InMemoryDocumentStore()
    document_store.write_documents(docs)

    retriever = InMemoryBM25Retriever(document_store=document_store)
    ranker = LostInTheMiddleRanker()

    document_ranker_pipeline = Pipeline()
    document_ranker_pipeline.add_component(instance=retriever, name="retriever")
    document_ranker_pipeline.add_component(instance=ranker, name="ranker")

    document_ranker_pipeline.connect("retriever.documents", "ranker.documents")

    query = "Cities in France"
    result = document_ranker_pipeline.run(
        data={"retriever": {"query": query, "top_k": 3}, "ranker": {"query": query, "top_k": 2}}
    )
    assert len(result) == 1


@pytest.mark.integration
def test_rag_pipeline():
    prompt_template = """
    Given these documents, answer the question.\nDocuments:
    {% for doc in documents %}
        {{ doc.content }}
    {% endfor %}

    \nQuestion: {{question}}
    \nAnswer:
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
    rag_pipeline.add_component(instance=AnswerBuilder(), name="answer_builder")
    # Add the lost in the middle ranker
    rag_pipeline.add_component(instance=LostInTheMiddleRanker(), name="ranker")
    rag_pipeline.connect("retriever", "prompt_builder.documents")
    rag_pipeline.connect("prompt_builder", "llm")
    rag_pipeline.connect("llm.replies", "answer_builder.replies")
    rag_pipeline.connect("retriever", "answer_builder.documents")
    rag_pipeline.connect("retriever.documents", "ranker.documents")

    # Populate the document store
    documents = [
        Document(content="My name is Jean and I live in Paris."),
        Document(content="My name is Mark and I live in Berlin."),
        Document(content="My name is Giorgio and I live in Rome."),
    ]
    rag_pipeline.get_component("retriever").document_store.write_documents(documents)

    # Query and assert
    questions = ["Who lives in Paris?", "Who lives in Berlin?", "Who lives in Rome?"]
    answers_spywords = ["Jean", "Mark", "Giorgio"]

    for question, spyword in zip(questions, answers_spywords):
        result = rag_pipeline.run(
            {
                "retriever": {"query": question},
                "ranker": {"query": question, "top_k": 2},
                "prompt_builder": {"question": question},
                "answer_builder": {"query": question},
            }
        )

    assert len(result["answer_builder"]["answers"]) == 1
    generated_answer = result["answer_builder"]["answers"][0]
    assert spyword in generated_answer.data
    assert generated_answer.query == question
    assert hasattr(generated_answer, "documents")
    assert hasattr(generated_answer, "meta")
    assert len(result) == 2


@pytest.mark.integration
def test_embedding_retrieval_rag_pipeline(tmp_path):
    # Create the RAG pipeline
    prompt_template = """
    Given these documents, answer the question.\nDocuments:
    {% for doc in documents %}
        {{ doc.content }}
    {% endfor %}

    \nQuestion: {{question}}
    \nAnswer:
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
    rag_pipeline.add_component(instance=AnswerBuilder(), name="answer_builder")
    rag_pipeline.add_component(instance=LostInTheMiddleRanker(), name="ranker")
    rag_pipeline.connect("text_embedder", "retriever")
    rag_pipeline.connect("retriever", "prompt_builder.documents")
    rag_pipeline.connect("prompt_builder", "llm")
    rag_pipeline.connect("llm.replies", "answer_builder.replies")
    rag_pipeline.connect("retriever", "answer_builder.documents")
    rag_pipeline.connect("retriever.documents", "ranker.documents")

    # Populate the document store
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

    # Query and assert
    questions = ["Who lives in Paris?", "Who lives in Berlin?", "Who lives in Rome?"]
    answers_spywords = ["Jean", "Mark", "Giorgio"]

    for question, spyword in zip(questions, answers_spywords):
        result = rag_pipeline.run(
            {
                "ranker": {"query": question, "top_k": 2},
                "text_embedder": {"text": question},
                "prompt_builder": {"question": question},
                "answer_builder": {"query": question},
            }
        )
        assert len(result["answer_builder"]["answers"]) == 1
        generated_answer = result["answer_builder"]["answers"][0]
        assert spyword in generated_answer.data
        assert generated_answer.query == question
        assert hasattr(generated_answer, "documents")
        assert hasattr(generated_answer, "meta")
        assert len(result) == 2


@pytest.mark.integration
def test_rag_pipeline_wikipedia():
    document_store = InMemoryDocumentStore()
    # Load the wikipedia dataset (first 10 rows)
    data = load_dataset("wikipedia", "20220301.simple", split="train[:10]")
    documents = []
    # Store the content along with metadata in the haystack document object
    for record in data:
        content = record["text"]
        metadata = {"wiki-id": str(record["id"]), "source": record["url"], "title": record["title"]}

        ##Create a Haystack Document object with the content and metadata
        document = Document(content=content, meta=metadata)
        documents.append(document)
        print(documents)

    # Populate the document store
    indexing_pipeline = Pipeline()
    indexing_pipeline.add_component(
        instance=SentenceTransformersDocumentEmbedder(model_name_or_path="sentence-transformers/all-MiniLM-L6-v2"),
        name="document_embedder",
    )
    indexing_pipeline.add_component(instance=DocumentWriter(document_store=document_store), name="document_writer")
    indexing_pipeline.connect("document_embedder", "document_writer")
    indexing_pipeline.run({"document_embedder": {"documents": documents}})

    template = """
    Given these documents, answer the question.\nDocuments:
        {% for doc in documents %}
            {{ doc.content }}
        {% endfor %}
        \nQuestion: {{question}}
        \nAnswer:
    """

    rag_pipeline = Pipeline()
    rag_pipeline.add_component(
        instance=SentenceTransformersTextEmbedder(model_name_or_path="sentence-transformers/all-MiniLM-L6-v2"),
        name="text_embedder",
    )
    rag_pipeline.add_component(instance=InMemoryEmbeddingRetriever(document_store=document_store), name="retriever")
    rag_pipeline.add_component(instance=PromptBuilder(template=template), name="prompt_builder")
    rag_pipeline.add_component(
        instance=HuggingFaceLocalGenerator(
            model_name_or_path="google/flan-t5-small",
            task="text2text-generation",
            generation_kwargs={"max_new_tokens": 512, "temperature": 0.5, "do_sample": True},
        ),
        name="llm",
    )
    rag_pipeline.add_component(instance=AnswerBuilder(), name="answer_builder")
    rag_pipeline.add_component(instance=LostInTheMiddleRanker(top_k=3), name="ranker")
    rag_pipeline.connect("text_embedder", "retriever")
    rag_pipeline.connect("retriever", "prompt_builder.documents")
    rag_pipeline.connect("prompt_builder", "llm")
    rag_pipeline.connect("llm.replies", "answer_builder.replies")
    rag_pipeline.connect("retriever", "answer_builder.documents")
    rag_pipeline.connect("retriever.documents", "ranker.documents")
    question = "What is a computer application?"

    result = rag_pipeline.run(
        {
            "ranker": {"query": question, "top_k": 3},
            "text_embedder": {"text": question},
            "prompt_builder": {"question": question},
            "answer_builder": {"query": question},
        }
    )
    print(result)
    assert len(result["answer_builder"]["answers"]) == 1
    assert len(result) == 3
