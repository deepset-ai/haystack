# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
from unittest.mock import Mock, patch

import pytest

from haystack import Document
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.preprocessors import EmbeddingBasedDocumentSplitter
from haystack.utils import ComponentDevice

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# disable tqdm entirely for tests
from tqdm import tqdm

tqdm.disable = True


class TestEmbeddingBasedDocumentSplitter:
    def test_init(self):
        mock_embedder = Mock()
        splitter = EmbeddingBasedDocumentSplitter(
            document_embedder=mock_embedder, sentences_per_group=2, percentile=0.9, min_length=50, max_length=1000
        )

        assert splitter.document_embedder == mock_embedder
        assert splitter.sentences_per_group == 2
        assert splitter.percentile == 0.9
        assert splitter.min_length == 50
        assert splitter.max_length == 1000

    def test_init_invalid_sentences_per_group(self):
        mock_embedder = Mock()
        with pytest.raises(ValueError, match="sentences_per_group must be greater than 0"):
            EmbeddingBasedDocumentSplitter(document_embedder=mock_embedder, sentences_per_group=0)

    def test_init_invalid_percentile(self):
        mock_embedder = Mock()
        with pytest.raises(ValueError, match="percentile must be between 0.0 and 1.0"):
            EmbeddingBasedDocumentSplitter(document_embedder=mock_embedder, percentile=1.5)

    def test_init_invalid_min_length(self):
        mock_embedder = Mock()
        with pytest.raises(ValueError, match="min_length must be greater than or equal to 0"):
            EmbeddingBasedDocumentSplitter(document_embedder=mock_embedder, min_length=-1)

    def test_init_invalid_max_length(self):
        mock_embedder = Mock()
        with pytest.raises(ValueError, match="max_length must be greater than min_length"):
            EmbeddingBasedDocumentSplitter(document_embedder=mock_embedder, min_length=100, max_length=50)

    def test_warm_up(self):
        mock_embedder = Mock()
        splitter = EmbeddingBasedDocumentSplitter(document_embedder=mock_embedder)

        with patch(
            "haystack.components.preprocessors.embedding_based_document_splitter.SentenceSplitter"
        ) as mock_splitter_class:
            mock_splitter = Mock()
            mock_splitter_class.return_value = mock_splitter

            splitter.warm_up()

            assert splitter.sentence_splitter == mock_splitter
            mock_splitter_class.assert_called_once()

    def test_run_not_warmed_up(self):
        mock_embedder = Mock()
        splitter = EmbeddingBasedDocumentSplitter(document_embedder=mock_embedder)

        with pytest.raises(RuntimeError, match="wasn't warmed up"):
            splitter.run(documents=[Document(content="test")])

    def test_run_invalid_input(self):
        mock_embedder = Mock()
        splitter = EmbeddingBasedDocumentSplitter(document_embedder=mock_embedder)
        splitter.sentence_splitter = Mock()
        splitter._is_warmed_up = True

        with pytest.raises(TypeError, match="expects a List of Documents"):
            splitter.run(documents="not a list")

    def test_run_document_with_none_content(self):
        mock_embedder = Mock()
        splitter = EmbeddingBasedDocumentSplitter(document_embedder=mock_embedder)
        splitter.sentence_splitter = Mock()
        splitter._is_warmed_up = True

        with pytest.raises(ValueError, match="content for document ID"):
            splitter.run(documents=[Document(content=None)])

    def test_run_empty_document(self):
        mock_embedder = Mock()
        splitter = EmbeddingBasedDocumentSplitter(document_embedder=mock_embedder)
        splitter.sentence_splitter = Mock()
        splitter._is_warmed_up = True

        result = splitter.run(documents=[Document(content="")])
        assert result["documents"] == []

    def test_group_sentences_single(self):
        mock_embedder = Mock()
        splitter = EmbeddingBasedDocumentSplitter(document_embedder=mock_embedder, sentences_per_group=1)

        sentences = ["Sentence 1.", "Sentence 2.", "Sentence 3."]
        groups = splitter._group_sentences(sentences)

        assert groups == sentences

    def test_group_sentences_multiple(self):
        mock_embedder = Mock()
        splitter = EmbeddingBasedDocumentSplitter(document_embedder=mock_embedder, sentences_per_group=2)

        sentences = ["Sentence 1. ", "Sentence 2. ", "Sentence 3. ", "Sentence 4."]
        groups = splitter._group_sentences(sentences)

        assert groups == ["Sentence 1. Sentence 2. ", "Sentence 3. Sentence 4."]

    def test_cosine_distance(self):
        mock_embedder = Mock()
        splitter = EmbeddingBasedDocumentSplitter(document_embedder=mock_embedder)

        # Test with identical vectors
        embedding1 = [1.0, 0.0, 0.0]
        embedding2 = [1.0, 0.0, 0.0]
        distance = splitter._cosine_distance(embedding1, embedding2)
        assert distance == 0.0

        # Test with orthogonal vectors
        embedding1 = [1.0, 0.0, 0.0]
        embedding2 = [0.0, 1.0, 0.0]
        distance = splitter._cosine_distance(embedding1, embedding2)
        assert distance == 1.0

        # Test with zero vectors
        embedding1 = [0.0, 0.0, 0.0]
        embedding2 = [1.0, 0.0, 0.0]
        distance = splitter._cosine_distance(embedding1, embedding2)
        assert distance == 1.0

    def test_find_split_points_empty(self):
        mock_embedder = Mock()
        splitter = EmbeddingBasedDocumentSplitter(document_embedder=mock_embedder)

        split_points = splitter._find_split_points([])
        assert split_points == []

        split_points = splitter._find_split_points([[1.0, 0.0]])
        assert split_points == []

    def test_find_split_points(self):
        mock_embedder = Mock()
        splitter = EmbeddingBasedDocumentSplitter(document_embedder=mock_embedder, percentile=0.5)

        # Create embeddings where the second pair has high distance
        embeddings = [
            [1.0, 0.0, 0.0],  # Similar to next
            [0.9, 0.1, 0.0],  # Similar to previous
            [0.0, 1.0, 0.0],  # Very different from next
            [0.1, 0.9, 0.0],  # Similar to previous
        ]

        split_points = splitter._find_split_points(embeddings)
        # Should find a split point after the second embedding (index 2)
        assert 2 in split_points

    def test_create_splits_from_points(self):
        mock_embedder = Mock()
        splitter = EmbeddingBasedDocumentSplitter(document_embedder=mock_embedder)

        sentence_groups = ["Group 1 ", "Group 2 ", "Group 3 ", "Group 4"]
        split_points = [2]  # Split after index 1

        splits = splitter._create_splits_from_points(sentence_groups, split_points)
        assert splits == ["Group 1 Group 2 ", "Group 3 Group 4"]

    def test_create_splits_from_points_no_points(self):
        mock_embedder = Mock()
        splitter = EmbeddingBasedDocumentSplitter(document_embedder=mock_embedder)

        sentence_groups = ["Group 1 ", "Group 2 ", "Group 3"]
        split_points = []

        splits = splitter._create_splits_from_points(sentence_groups, split_points)
        assert splits == ["Group 1 Group 2 Group 3"]

    def test_merge_small_splits(self):
        mock_embedder = Mock()
        splitter = EmbeddingBasedDocumentSplitter(document_embedder=mock_embedder, min_length=10)

        splits = ["Short ", "Also short ", "Long enough text ", "Another short"]
        merged = splitter._merge_small_splits(splits)

        assert len(merged) == 3
        assert merged[0] == "Short Also short "
        assert merged[1] == "Long enough text "
        assert merged[2] == "Another short"

    def test_merge_small_splits_respect_max_length(self):
        mock_embedder = Mock()
        splitter = EmbeddingBasedDocumentSplitter(document_embedder=mock_embedder, min_length=10, max_length=15)

        splits = ["123456", "123456789", "1234"]
        merged = splitter._merge_small_splits(splits=splits)

        assert len(merged) == 2
        # First split remains beneath min_length b/c next split is too long
        assert merged[0] == "123456"
        # Second split is merged with third split to get above min_length and still beneath max_length
        assert merged[1] == "1234567891234"

    def test_create_documents_from_splits(self):
        mock_embedder = Mock()
        splitter = EmbeddingBasedDocumentSplitter(document_embedder=mock_embedder)

        original_doc = Document(content="test", meta={"key": "value"})
        splits = ["Split 1", "Split 2"]

        documents = splitter._create_documents_from_splits(splits, original_doc)

        assert len(documents) == 2
        assert documents[0].content == "Split 1"
        assert documents[0].meta["source_id"] == original_doc.id
        assert documents[0].meta["split_id"] == 0
        assert documents[0].meta["key"] == "value"
        assert documents[1].content == "Split 2"
        assert documents[1].meta["split_id"] == 1

    def test_create_documents_from_splits_with_page_numbers(self):
        mock_embedder = Mock()
        splitter = EmbeddingBasedDocumentSplitter(document_embedder=mock_embedder)

        original_doc = Document(content="Page 1 content.\fPage 2 content.\f\fPage 4 content.", meta={"key": "value"})
        splits = ["Page 1 content.\f", "Page 2 content.\f\f", "Page 4 content."]

        documents = splitter._create_documents_from_splits(splits, original_doc)

        assert len(documents) == 3
        assert documents[0].content == "Page 1 content.\f"
        assert documents[0].meta["page_number"] == 1
        assert documents[1].content == "Page 2 content.\f\f"
        assert documents[1].meta["page_number"] == 2
        assert documents[2].content == "Page 4 content."
        assert documents[2].meta["page_number"] == 4

    def test_create_documents_from_splits_with_consecutive_page_breaks(self):
        mock_embedder = Mock()
        splitter = EmbeddingBasedDocumentSplitter(document_embedder=mock_embedder)

        # Test with consecutive page breaks at the end
        original_doc = Document(content="Page 1 content.\fPage 2 content.\f\f\f", meta={"key": "value"})
        splits = ["Page 1 content.\f", "Page 2 content.\f\f\f"]

        documents = splitter._create_documents_from_splits(splits, original_doc)

        assert len(documents) == 2
        assert documents[0].content == "Page 1 content.\f"
        assert documents[0].meta["page_number"] == 1
        assert documents[1].content == "Page 2 content.\f\f\f"
        # Should be page 2, not 4, because consecutive page breaks at the end are adjusted
        assert documents[1].meta["page_number"] == 2

    def test_calculate_embeddings(self):
        mock_embedder = Mock()

        # Mock the document embedder to return documents with embeddings
        def mock_run(documents):
            for doc in documents:
                doc.embedding = [1.0, 2.0, 3.0]  # Simple mock embedding
            return {"documents": documents}

        mock_embedder.run = Mock(side_effect=mock_run)
        splitter = EmbeddingBasedDocumentSplitter(document_embedder=mock_embedder)

        sentence_groups = ["Group 1", "Group 2", "Group 3"]
        embeddings = splitter._calculate_embeddings(sentence_groups)

        assert len(embeddings) == 3
        assert all(embedding == [1.0, 2.0, 3.0] for embedding in embeddings)
        mock_embedder.run.assert_called_once()

    def test_to_dict(self):
        mock_embedder = Mock()
        mock_embedder.to_dict.return_value = {"type": "MockEmbedder"}

        splitter = EmbeddingBasedDocumentSplitter(
            document_embedder=mock_embedder, sentences_per_group=2, percentile=0.9, min_length=50, max_length=1000
        )

        result = splitter.to_dict()

        assert "EmbeddingBasedDocumentSplitter" in result["type"]
        assert result["init_parameters"]["sentences_per_group"] == 2
        assert result["init_parameters"]["percentile"] == 0.9
        assert result["init_parameters"]["min_length"] == 50
        assert result["init_parameters"]["max_length"] == 1000
        assert "document_embedder" in result["init_parameters"]

    @pytest.mark.integration
    def test_split_document_with_multiple_topics(self):
        import os

        import torch

        # Force CPU usage to avoid MPS memory issues
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        torch.backends.mps.is_available = lambda: False

        embedder = SentenceTransformersDocumentEmbedder(
            model="sentence-transformers/all-MiniLM-L6-v2", device=ComponentDevice.from_str("cpu")
        )

        embedder.warm_up()

        splitter = EmbeddingBasedDocumentSplitter(
            document_embedder=embedder, sentences_per_group=2, percentile=0.9, min_length=30, max_length=300
        )
        splitter.warm_up()

        # A document with multiple topics
        text = (
            "The weather today is beautiful. The sun is shining brightly. The temperature is perfect for a walk. "
            "Machine learning has revolutionized many industries. Neural networks can process vast amounts of data. "
            "Deep learning models achieve remarkable accuracy on complex tasks. "
            "Cooking is both an art and a science. Fresh ingredients make all the difference. "
            "Proper seasoning enhances the natural flavors of food. "
            "The history of ancient civilizations fascinates researchers. Archaeological discoveries reveal new insights. "  # noqa: E501
            "Ancient texts provide valuable information about past societies."
        )
        doc = Document(content=text)

        result = splitter.run(documents=[doc])
        split_docs = result["documents"]

        # There should be more than one split
        assert len(split_docs) > 1
        # Each split should be non-empty and respect min_length
        for split_doc in split_docs:
            assert split_doc.content.strip() != ""
            assert len(split_doc.content) >= 30
        # The splits should cover the original text
        combined = "".join([d.content for d in split_docs])
        original = text
        assert combined in original or original in combined

    @pytest.mark.integration
    def test_trailing_whitespace_is_preserved(self):
        embedder = SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
        embedder.warm_up()

        splitter = EmbeddingBasedDocumentSplitter(document_embedder=embedder, sentences_per_group=1)
        splitter.warm_up()

        # Normal trailing whitespace
        text = "The weather today is beautiful.  "
        result = splitter.run(documents=[Document(content=text)])
        assert result["documents"][0].content == text

        # Newline at the end
        text = "The weather today is beautiful.\n"
        result = splitter.run(documents=[Document(content=text)])
        assert result["documents"][0].content == text

        # Page break at the end
        text = "The weather today is beautiful.\f"
        result = splitter.run(documents=[Document(content=text)])
        assert result["documents"][0].content == text

    @pytest.mark.integration
    def test_no_extra_whitespaces_between_sentences(self):
        embedder = SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
        embedder.warm_up()

        splitter = EmbeddingBasedDocumentSplitter(
            document_embedder=embedder, sentences_per_group=1, percentile=0.9, min_length=10, max_length=500
        )
        splitter.warm_up()

        text = (
            "The weather today is beautiful. The sun is shining brightly. The temperature is perfect for a walk. "
            "There are no clouds and no rain. Machine learning has revolutionized many industries. "
            "Neural networks can process vast amounts of data. Deep learning models achieve remarkable accuracy on complex tasks."  # noqa: E501
        )
        doc = Document(content=text)

        result = splitter.run(documents=[doc])
        split_docs = result["documents"]
        assert len(split_docs) == 2
        # Expect the original whitespace structure with trailing spaces where they exist
        assert (
            split_docs[0].content
            == "The weather today is beautiful. The sun is shining brightly. The temperature is perfect for a walk. There are no clouds and no rain. "  # noqa: E501
        )  # noqa: E501
        assert (
            split_docs[1].content
            == "Machine learning has revolutionized many industries. Neural networks can process vast amounts of data. Deep learning models achieve remarkable accuracy on complex tasks."  # noqa: E501
        )  # noqa: E501

    @pytest.mark.integration
    def test_split_large_splits_recursion(self):
        """
        Test that _split_large_splits() works correctly without infinite loops.
        This test uses a longer text that will trigger the recursive splitting logic.
        If the chunk cannot be split further, it is allowed to be larger than max_length.
        """
        embedder = SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-MiniLM-L6-v2", batch_size=32)
        semantic_chunker = EmbeddingBasedDocumentSplitter(
            document_embedder=embedder, sentences_per_group=5, percentile=0.95, min_length=50, max_length=1000
        )
        semantic_chunker.warm_up()

        text = """# Artificial intelligence and its Impact on Society
## Article from Wikipedia, the free encyclopedia
### Introduction to Artificial Intelligence
Artificial intelligence (AI) is the capability of computational systems to perform tasks typically associated with human intelligence, such as learning, reasoning, problem-solving, perception, and decision-making. It is a field of research in computer science that develops and studies methods and software that enable machines to perceive their environment and use learning and intelligence to take actions that maximize their chances of achieving defined goals.

### The History of Software
The history of software is closely tied to the development of digital computers in the mid-20th century. Early programs were written in the machine language specific to the hardware. The introduction of high-level programming languages in 1958 allowed for more human-readable instructions, making software development easier and more portable across different computer architectures. Software in a programming language is run through a compiler or interpreter to execute on the architecture's hardware. Over time, software has become complex, owing to developments in networking, operating systems, and databases."""  # noqa: E501

        doc = Document(content=text)
        result = semantic_chunker.run(documents=[doc])
        split_docs = result["documents"]

        assert len(split_docs) == 1

        # If the chunk cannot be split further, it is allowed to be larger than max_length
        # At least one split should be larger than max_length in this test case
        assert any(len(split_doc.content) > 1000 for split_doc in split_docs)

        # Verify that the splits cover the original content
        combined_content = "".join([d.content for d in split_docs])
        assert combined_content == text

        for i, split_doc in enumerate(split_docs):
            assert split_doc.meta["source_id"] == doc.id
            assert split_doc.meta["split_id"] == i
            assert "page_number" in split_doc.meta

    @pytest.mark.integration
    def test_split_large_splits_actually_splits(self):
        """
        Test that _split_large_splits() actually works and can split long texts into multiple chunks.
        This test uses a very long text that should be split into multiple chunks.
        """
        embedder = SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-MiniLM-L6-v2", batch_size=32)
        semantic_chunker = EmbeddingBasedDocumentSplitter(
            document_embedder=embedder,
            sentences_per_group=3,
            percentile=0.85,  # Lower percentile to create more splits
            min_length=100,
            max_length=500,  # Smaller max_length to force more splits
        )
        semantic_chunker.warm_up()

        # Create a very long text with multiple paragraphs and topics
        text = """# Comprehensive Guide to Machine Learning and Artificial Intelligence

## Introduction to Machine Learning
Machine learning is a subset of artificial intelligence that focuses on the development of computer programs that can access data and use it to learn for themselves. The process of learning begins with observations or data, such as examples, direct experience, or instruction, in order to look for patterns in data and make better decisions in the future based on the examples that we provide. The primary aim is to allow the computers learn automatically without human intervention or assistance and adjust actions accordingly.

## Types of Machine Learning
There are several types of machine learning algorithms, each with their own strengths and weaknesses. Supervised learning involves training a model on a labeled dataset, where the correct answers are provided. The model learns to map inputs to outputs based on these examples. Unsupervised learning, on the other hand, deals with unlabeled data and seeks to find hidden patterns or structures within the data. Reinforcement learning is a type of learning where an agent learns to behave in an environment by performing certain actions and receiving rewards or penalties.

## Deep Learning and Neural Networks
Deep learning is a subset of machine learning that uses neural networks with multiple layers to model and understand complex patterns. Neural networks are inspired by the human brain and consist of interconnected nodes or neurons. Each connection between neurons has a weight that is adjusted during training. The network learns by adjusting these weights based on the error between predicted and actual outputs. Deep learning has been particularly successful in areas such as computer vision, natural language processing, and speech recognition.

\f

## Natural Language Processing
Natural Language Processing (NLP) is a field of artificial intelligence that focuses on the interaction between computers and human language. It involves developing algorithms and models that can understand, interpret, and generate human language. NLP applications include machine translation, sentiment analysis, text summarization, and question answering systems. Recent advances in deep learning have significantly improved the performance of NLP systems, leading to more accurate and sophisticated language models.

## Computer Vision and Image Recognition
Computer vision is another important area of artificial intelligence that deals with how computers can gain high-level understanding from digital images or videos. It involves developing algorithms that can identify and understand visual information from the world. Applications include facial recognition, object detection, medical image analysis, and autonomous vehicle navigation. Deep learning models, particularly convolutional neural networks (CNNs), have revolutionized computer vision by achieving human-level performance on many tasks.

## The Future of Artificial Intelligence
The future of artificial intelligence holds immense potential for transforming various industries and aspects of human life. We can expect to see more sophisticated AI systems that can handle complex reasoning tasks, understand context better, and interact more naturally with humans. However, this rapid advancement also brings challenges related to ethics, privacy, and the impact on employment. It's crucial to develop AI systems that are not only powerful but also safe, fair, and beneficial to society as a whole.

\f

## Ethical Considerations in AI
As artificial intelligence becomes more prevalent, ethical considerations become increasingly important. Issues such as bias in AI systems, privacy concerns, and the potential for misuse need to be carefully addressed. AI systems can inherit biases from their training data, leading to unfair outcomes for certain groups. Privacy concerns arise from the vast amounts of data required to train AI systems. Additionally, there are concerns about the potential for AI to be used maliciously or to replace human workers in certain industries.

## Applications in Healthcare
Artificial intelligence has the potential to revolutionize healthcare by improving diagnosis, treatment planning, and patient care. Machine learning algorithms can analyze medical images to detect diseases earlier and more accurately than human doctors. AI systems can also help in drug discovery by predicting the effectiveness of potential treatments. In addition, AI-powered chatbots and virtual assistants can provide basic healthcare information and support to patients, reducing the burden on healthcare professionals.

## AI in Finance and Banking
The financial industry has been quick to adopt artificial intelligence for various applications. AI systems can analyze market data to make investment decisions, detect fraudulent transactions, and provide personalized financial advice. Machine learning algorithms can assess credit risk more accurately than traditional methods, leading to better lending decisions. Additionally, AI-powered chatbots can handle customer service inquiries, reducing costs and improving customer satisfaction.

\f

## Transportation and Autonomous Vehicles
Autonomous vehicles represent one of the most visible applications of artificial intelligence in transportation. Self-driving cars use a combination of sensors, cameras, and AI algorithms to navigate roads safely. These systems can detect obstacles, read traffic signs, and make decisions about speed and direction. Beyond autonomous cars, AI is also being used in logistics and supply chain management to optimize routes and reduce delivery times.

## Education and Personalized Learning
Artificial intelligence is transforming education by enabling personalized learning experiences. AI systems can adapt to individual student needs, providing customized content and pacing. Intelligent tutoring systems can provide immediate feedback and support to students, helping them learn more effectively. Additionally, AI can help educators by automating administrative tasks and providing insights into student performance and learning patterns."""  # noqa: E501

        doc = Document(content=text)
        result = semantic_chunker.run(documents=[doc])
        split_docs = result["documents"]

        assert len(split_docs) == 11

        # Verify that the splits cover the original content
        combined_content = "".join([d.content for d in split_docs])
        assert combined_content == text

        for i, split_doc in enumerate(split_docs):
            assert split_doc.meta["source_id"] == doc.id
            assert split_doc.meta["split_id"] == i
            assert "page_number" in split_doc.meta

            if i in [0, 1, 2, 3]:
                assert split_doc.meta["page_number"] == 1
            if i in [4, 5, 6]:
                assert split_doc.meta["page_number"] == 2
            if i in [7, 8]:
                assert split_doc.meta["page_number"] == 3
            if i in [9, 10]:
                assert split_doc.meta["page_number"] == 4
