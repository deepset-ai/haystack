import os
import pytest
from cohere_haystack.generator import CohereGenerator
from gradient_haystack.generator.base import GradientGenerator
from haystack import Pipeline
from haystack.components.builders import AnswerBuilder, PromptBuilder, MetadataBuilder
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack.components.generators import HuggingFaceLocalGenerator, HuggingFaceTGIGenerator
from haystack.components.retrievers import InMemoryBM25Retriever, InMemoryEmbeddingRetriever
from haystack.components.writers import DocumentWriter
from haystack.dataclasses import Document
from haystack.document_stores import InMemoryDocumentStore


class TestMetadataBuilder:
    def test_receives_list_of_replies_and_metadata(self):
        """
        The component receives a list of Documents, replies and metadata.
        """
        metadata_builder = MetadataBuilder()

        documents = [Document(content="document_0"), Document(content="document_1"), Document(content="document_1")]
        replies = ["reply_0", "reply_1", "reply_2"]
        metadata = [{"key_0": "value_0"}, {"key_1": "value_1"}, {"key_2": "value_2"}]

        result = metadata_builder.run(documents=documents, replies=replies, meta=metadata)

        assert isinstance(result["documents"], list)
        assert len(result["documents"]) == 3
        for idx, doc in enumerate(result["documents"]):
            assert doc.meta == {"reply": f"reply_{idx}", f"key_{idx}": f"value_{idx}"}

    def test_receives_list_of_replies_and_no_metadata(self):
        """
        The component receives only a list of Documents and replies and no metadata.
        """
        metadata_builder = MetadataBuilder()

        documents = [Document(content="document_0"), Document(content="document_1"), Document(content="document_2")]
        replies = ["reply_0", "reply_1", "reply_2"]

        # Invoke the run method without providing metadata
        result = metadata_builder.run(documents=documents, replies=replies)

        assert isinstance(result["documents"], list)
        assert len(result["documents"]) == 3
        for idx, doc in enumerate(result["documents"]):
            assert doc.meta == {"reply": f"reply_{idx}"}

    def test_mismatched_documents_replies_and_no_metadata(self):
        """
        If the length of the Document list and the replies list are different having no metadata, the component raises a ValueError.
        """
        metadata_builder = MetadataBuilder()

        documents = [Document(content="document_0"), Document(content="document_1")]
        replies = ["reply_0", "reply_1", "reply_2"]

        # Check that a ValueError is raised when invoking the run method
        with pytest.raises(ValueError):
            metadata_builder.run(documents=documents, replies=replies)

    def test_mismatched_documents_replies(self):
        """
        If the length of the Document list and the replies list are different, having metadata the component raises a ValueError.
        """
        metadata_builder = MetadataBuilder()

        documents = [Document(content="document_0"), Document(content="document_1")]
        replies = ["reply_0", "reply_1", "reply_2"]
        metadata = [{"key_0": "value_0"}, {"key_1": "value_1"}, {"key_2": "value_2"}]

        # Check that a ValueError is raised when invoking the run method
        with pytest.raises(ValueError):
            metadata_builder.run(documents=documents, replies=replies, meta=metadata)

    def test_mismatched_documents_metadata(self):
        """
        If the length of the Document list and the metadata list are different, the component raises a ValueError.
        """
        metadata_builder = MetadataBuilder()

        documents = [Document(content="document_0"), Document(content="document_1"), Document(content="document_2")]
        replies = ["reply_0", "reply_1", "reply_2"]
        metadata = [{"key_0": "value_0"}, {"key_1": "value_1"}]

        # Check that a ValueError is raised when invoking the run method
        with pytest.raises(ValueError):
            metadata_builder.run(documents=documents, replies=replies, meta=metadata)

    def test_mismatched_documents_replies_metadata(self):
        """
        If the length of the Document list, replies list and the metadata list are all different, the component raises a ValueError.
        """
        metadata_builder = MetadataBuilder()

        documents = [Document(content="document_0"), Document(content="document_1")]
        replies = ["reply_0"]
        metadata = [{"key_0": "value_0"}, {"key_1": "value_1"}, {"key_2": "value_2"}]

        # Check that a ValueError is raised when invoking the run method
        with pytest.raises(ValueError):
            metadata_builder.run(documents=documents, replies=replies, meta=metadata)

    def test_metadata_with_same_keys(self):
        """
        The component should correctly add the metadata if the Document metadata already has a reply.
        """
        metadata_builder = MetadataBuilder()

        documents = [Document(content="document content", meta={"reply": "original text"})]
        replies = ["generated text"]

        result = metadata_builder.run(documents=documents, replies=replies)

        assert isinstance(result["documents"], list)
        assert len(result["documents"]) == 1
        assert result["documents"][0].meta == {"reply": "generated text"}

    @pytest.mark.integration
    def test_summarization_pipeline(self):
        """
        Test function for a summarization pipeline that generates summaries for given documents.
        The function sets up a pipeline for document summarization using the components:
        PromptBuilder to generate prompts based on a template.
        HuggingFaceLocalGenerator for text generation.
        MetadataBuilder to add the metadata from the LLM output in the Documents.
        The test checks:
          - Three results are obtained from the RAG pipeline.
          - Each result contains extracted answers from the generated responses.
          - The LLM reply has been added to the Document metadata correctly by the  MetadataBuilder.
        """
        prompt_template = """
        Summarize the document text:
            {{ document.content }}
        \nSummary:
        """
        summarization_pipeline = Pipeline()
        summarization_pipeline.add_component(instance=PromptBuilder(template=prompt_template), name="prompt_builder")
        summarization_pipeline.add_component(
            instance=HuggingFaceLocalGenerator(
                model_name_or_path="google/flan-t5-small",
                task="text2text-generation",
                generation_kwargs={"max_new_tokens": 100, "temperature": 0.5, "do_sample": True},
            ),
            name="llm",
        )
        summarization_pipeline.add_component(instance=MetadataBuilder(), name="metadata_builder")
        summarization_pipeline.connect("prompt_builder", "llm")
        summarization_pipeline.connect("llm.replies", "metadata_builder.replies")

        documents = [
            Document(
                content="The Mariana Islands (also the Marianas; up to the early 20th century sometimes called Islas de los Ladrones meaning 'Islands of Thieves') are a group of islands made up by the summits of 15 volcanic mountains in the western Pacific Ocean. They are the southern part of a seamount range that goes on for 1,565 miles (2,519 km) from Guam to near Japan. The Marianas are the northernmost islands of a larger island group called Micronesia."
            ),
            Document(
                content="The Zimbabwe dollar was a currency for Zimbabwe from 1980 to 2009. Zimbabwe had the highest rate of inflation in the world. The rate of inflation grew to 231,150,888.87% in July 2008. Because of hyperinflation, or inflation that is out of control, the Reserve Bank of Zimbabwe had to print banknotes with higher values to cope with the rising cost of living. In January 2009, Zimbabwe released a banknote for one hundred trillion dollars, or $100,000,000,000,000."
            ),
            Document(
                content="PayPal is a website that allows transfer of money among people via web services and email. The money can be deposited into a bank account. PayPal was owned by eBay, from 2002 to 2015. It can be used in more than 200 countries. PayPal allows customers to send, receive, and hold funds in 25 currencies worldwide. However, PayPal is not a bank. Funds that are not used are converted into PayPal's profit."
            ),
        ]
        # Assumes number of LLM responses is set to 1
        inputs = [
            {"prompt_builder": {"document": doc.content}, "metadata_builder": {"documents": [doc]}} for doc in documents
        ]
        results = [summarization_pipeline.run(input) for input in inputs]

        assert len(results) == 3
        for result in results:
            updated_documents = result["metadata_builder"]["documents"]
            assert "reply" in updated_documents[0].meta.keys()

    @pytest.mark.integration
    def test_rag_pipeline_huggingface_local_generator(self):
        """
        Test function for a RAG pipeline with a HuggingFaceLocalGenerator.

        The function sets up a pipeline consisting of the components:
        SentenceTransformersTextEmbedder: Embeds text using the all-MiniLM-L6-v2 embedding model.
        InMemoryEmbeddingRetriever: Retrieves relevant documents based on embeddings.
        PromptBuilder: Creates prompts for the RAG model generation based on provided documents and a question.
        HuggingFaceLocalGenerator:  A LLM model from Huggingface for text generation.
        MetadataBuilder to add the metadata from the LLM output in the Documents .
        AnswerBuilder: Extracts answers from the generated responses.
        The test checks:
        - Three results are obtained from the RAG pipeline.
        - Each result contains extracted answers from the generated responses.
        - The LLM reply has been added to the Document metadata correctly by the  MetadataBuilder.
        """
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
            instance=InMemoryEmbeddingRetriever(document_store=InMemoryDocumentStore(), top_k=1), name="retriever"
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
        rag_pipeline.add_component(instance=MetadataBuilder(), name="metadata_builder")
        rag_pipeline.add_component(instance=AnswerBuilder(), name="answer_builder")

        rag_pipeline.connect("text_embedder", "retriever")
        rag_pipeline.connect("retriever", "prompt_builder.documents")
        rag_pipeline.connect("prompt_builder", "llm")
        rag_pipeline.connect("retriever", "metadata_builder.documents")
        rag_pipeline.connect("llm.replies", "metadata_builder.replies")
        rag_pipeline.connect("llm.replies", "answer_builder.replies")
        rag_pipeline.connect("metadata_builder", "answer_builder.documents")

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
        inputs = [
            {
                "prompt_builder": {"question": question},
                "text_embedder": {"text": question},
                "answer_builder": {"query": question},
            }
            for question in questions
        ]

        results = [rag_pipeline.run(input) for input in inputs]

        assert len(results) == 3
        for result in results:
            generated_answers = result["answer_builder"]["answers"]
            assert "reply" in generated_answers[0].documents[0].meta.keys()

    @pytest.mark.integration
    @pytest.mark.skipif(
        not os.environ.get("COHERE_API_KEY", None),
        reason="Export an env var called COHERE_API_KEY containing the Cohere api key to run this test.",
    )
    def test_rag_pipeline_cohere_generator(self):
        """
        Test function for a RAG pipeline with a CohereGenerator

        The function sets up a pipeline consisting of the components:
        SentenceTransformersTextEmbedder: Embeds text using the all-MiniLM-L6-v2 embedding model.
        InMemoryEmbeddingRetriever: Retrieves relevant documents based on embeddings.
        PromptBuilder: Creates prompts for the RAG model generation based on provided documents and a question.
        CohereGenerator: Cohere LLM for text generation.
        MetadataBuilder to add the metadata from the LLM output in the Documents .
        AnswerBuilder: Extracts answers from the generated responses.
        The test checks:
        - Three results are obtained from the RAG pipeline.
        - Each result contains extracted answers from the generated responses.
        - The LLM reply has been added to the Document metadata correctly by the  MetadataBuilder.
        """
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
            instance=InMemoryEmbeddingRetriever(document_store=InMemoryDocumentStore(), top_k=1), name="retriever"
        )
        rag_pipeline.add_component(instance=PromptBuilder(template=prompt_template), name="prompt_builder")
        rag_pipeline.add_component(instance=CohereGenerator(api_key="COHERE_API_KEY"), name="llm")
        rag_pipeline.add_component(instance=MetadataBuilder(), name="metadata_builder")
        rag_pipeline.add_component(instance=AnswerBuilder(), name="answer_builder")

        rag_pipeline.connect("text_embedder", "retriever")
        rag_pipeline.connect("retriever", "prompt_builder.documents")
        rag_pipeline.connect("prompt_builder", "llm")
        rag_pipeline.connect("retriever", "metadata_builder.documents")
        rag_pipeline.connect("llm.replies", "metadata_builder.replies")
        rag_pipeline.connect("llm.replies", "answer_builder.replies")
        rag_pipeline.connect("metadata_builder", "answer_builder.documents")

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
        inputs = [
            {
                "prompt_builder": {"question": question},
                "text_embedder": {"text": question},
                "answer_builder": {"query": question},
            }
            for question in questions
        ]

        results = [rag_pipeline.run(input) for input in inputs]

        assert len(results) == 3
        for result in results:
            generated_answers = result["answer_builder"]["answers"]
            assert "reply" in generated_answers[0].documents[0].meta.keys()

    @pytest.mark.integration
    @pytest.mark.skipif(
        not os.environ.get("HF_TOKEN", None),
        reason="Export an env var called HF_TOKEN containing the HuggingFaceTGI key to run this test.",
    )
    def rag_pipeline_huggingface_tgi_generator(self):
        """
        Test function for a RAG pipeline with a HuggingFaceTGIGenerator.

        The function sets up a pipeline consisting of the components:
        SentenceTransformersTextEmbedder: Embeds text using the all-MiniLM-L6-v2 embedding model.
        InMemoryEmbeddingRetriever: Retrieves relevant documents based on embeddings.
        PromptBuilder: Creates prompts for the RAG model generation based on provided documents and a question.
        HuggingFaceTGIGenerator: A component for LLMs hosted on Hugging Face Inference endpoints.
        MetadataBuilder to add the metadata from the LLM output in the Documents .
        AnswerBuilder: Extracts answers from the generated responses.
        The test checks:
        - Three results are obtained from the RAG pipeline.
        - Each result contains extracted answers from the generated responses.
        - The LLM reply has been added to the Document metadata correctly by the  MetadataBuilder.
        """
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
            instance=InMemoryEmbeddingRetriever(document_store=InMemoryDocumentStore(), top_k=1), name="retriever"
        )
        rag_pipeline.add_component(instance=PromptBuilder(template=prompt_template), name="prompt_builder")
        rag_pipeline.add_component(
            instance=HuggingFaceTGIGenerator(model="google/flan-t5-small", token="HF_TGI_TOKEN"), name="llm"
        )
        rag_pipeline.add_component(instance=MetadataBuilder(), name="metadata_builder")
        rag_pipeline.add_component(instance=AnswerBuilder(), name="answer_builder")

        rag_pipeline.connect("text_embedder", "retriever")
        rag_pipeline.connect("retriever", "prompt_builder.documents")
        rag_pipeline.connect("prompt_builder", "llm")
        rag_pipeline.connect("retriever", "metadata_builder.documents")
        rag_pipeline.connect("llm.replies", "metadata_builder.replies")
        rag_pipeline.connect("llm.replies", "answer_builder.replies")
        rag_pipeline.connect("metadata_builder", "answer_builder.documents")

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
        inputs = [
            {
                "prompt_builder": {"question": question},
                "text_embedder": {"text": question},
                "answer_builder": {"query": question},
            }
            for question in questions
        ]

        results = [rag_pipeline.run(input) for input in inputs]

        assert len(results) == 3
        for result in results:
            generated_answers = result["answer_builder"]["answers"]
            assert "reply" in generated_answers[0].documents[0].meta.keys()

    @pytest.mark.integration
    @pytest.mark.skipif(
        not os.environ.get("GRADIENT_ACCESS_TOKEN", None) or not os.environ.get("GRADIENT_WORKSPACE_ID", None),
        reason="Export env variables called GRADIENT_ACCESS_TOKEN and GRADIENT_WORKSPACE_ID \
            containing the Gradient configuration settings to run this test.",
    )
    def test_rag_pipeline_gradient_ai_generator(self):
        """
        Test function for a RAG pipeline with a GradientGenerator.

        The function sets up a pipeline consisting of the components:
        SentenceTransformersTextEmbedder: Embeds text using the all-MiniLM-L6-v2 embedding model.
        InMemoryEmbeddingRetriever: Retrieves relevant documents based on embeddings.
        PromptBuilder: Creates prompts for the RAG model generation based on provided documents and a question.
        HuggingFaceTGIGenerator: A component for text generation with LLMs deployed on the Gradient AI platform.
        MetadataBuilder to add the metadata from the LLM output in the Documents .
        AnswerBuilder: Extracts answers from the generated responses.
        The test checks:
        - Three results are obtained from the RAG pipeline.
        - Each result contains extracted answers from the generated responses.
        - The LLM reply has been added to the Document metadata correctly by the  MetadataBuilder.
        """
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
            instance=InMemoryEmbeddingRetriever(document_store=InMemoryDocumentStore(), top_k=1), name="retriever"
        )
        rag_pipeline.add_component(instance=PromptBuilder(template=prompt_template), name="prompt_builder")
        rag_pipeline.add_component(
            instance=GradientGenerator(
                access_token="GRADIENT_ACCESS_TOKEN",
                workspace_id="GRADIENT_WORKSPACE_ID",
                base_model_slug="llama2-7b-chat",
                max_generated_token_count=350,
            ),
            name="llm",
        )

        rag_pipeline.add_component(instance=MetadataBuilder(), name="metadata_builder")
        rag_pipeline.add_component(instance=AnswerBuilder(), name="answer_builder")

        rag_pipeline.connect("text_embedder", "retriever")
        rag_pipeline.connect("retriever", "prompt_builder.documents")
        rag_pipeline.connect("prompt_builder", "llm")
        rag_pipeline.connect("retriever", "metadata_builder.documents")
        rag_pipeline.connect("llm.replies", "metadata_builder.replies")
        rag_pipeline.connect("llm.replies", "answer_builder.replies")
        rag_pipeline.connect("metadata_builder", "answer_builder.documents")

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
        inputs = [
            {
                "prompt_builder": {"question": question},
                "text_embedder": {"text": question},
                "answer_builder": {"query": question},
            }
            for question in questions
        ]

        results = [rag_pipeline.run(input) for input in inputs]

        assert len(results) == 3
        for result in results:
            generated_answers = result["answer_builder"]["answers"]
            assert "reply" in generated_answers[0].documents[0].meta.keys()
