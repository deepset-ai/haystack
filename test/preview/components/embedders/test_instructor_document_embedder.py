from unittest.mock import patch, MagicMock
import pytest
import numpy as np

from haystack.preview import Document
from haystack.preview.components.embedders.instructor_document_embedder import InstructorDocumentEmbedder


class TestInstructorDocumentEmbedder:
    @pytest.mark.unit
    def test_init_default(self):
        """
        Test default initialization parameters for InstructorDocumentEmbedder.
        """
        embedder = InstructorDocumentEmbedder(model_name_or_path="hkunlp/instructor-base")
        assert embedder.model_name_or_path == "hkunlp/instructor-base"
        assert embedder.device == "cpu"
        assert embedder.use_auth_token is None
        assert embedder.instruction == "Represent the 'domain' 'text_type' for 'task_objective'"
        assert embedder.batch_size == 32
        assert embedder.progress_bar is True
        assert embedder.normalize_embeddings is False
        assert embedder.metadata_fields_to_embed == []
        assert embedder.embedding_separator == "\n"

    @pytest.mark.unit
    def test_init_with_parameters(self):
        """
        Test custom initialization parameters for InstructorDocumentEmbedder.
        """
        embedder = InstructorDocumentEmbedder(
            model_name_or_path="hkunlp/instructor-base",
            device="cuda",
            use_auth_token=True,
            instruction="Represent the 'domain' 'text_type' for 'task_objective'",
            batch_size=64,
            progress_bar=False,
            normalize_embeddings=True,
            metadata_fields_to_embed=["test_field"],
            embedding_separator=" | ",
        )
        assert embedder.model_name_or_path == "hkunlp/instructor-base"
        assert embedder.device == "cuda"
        assert embedder.use_auth_token is True
        assert embedder.instruction == "Represent the 'domain' 'text_type' for 'task_objective'"
        assert embedder.batch_size == 64
        assert embedder.progress_bar is False
        assert embedder.normalize_embeddings is True
        assert embedder.metadata_fields_to_embed == ["test_field"]
        assert embedder.embedding_separator == " | "

    @pytest.mark.unit
    def test_to_dict(self):
        """
        Test serialization of InstructorDocumentEmbedder to a dictionary, using default initialization parameters.
        """
        embedder = InstructorDocumentEmbedder(model_name_or_path="hkunlp/instructor-base")
        embedder_dict = embedder.to_dict()
        assert embedder_dict == {
            "type": "InstructorDocumentEmbedder",
            "init_parameters": {
                "model_name_or_path": "hkunlp/instructor-base",
                "device": "cpu",
                "use_auth_token": None,
                "instruction": "Represent the 'domain' 'text_type' for 'task_objective'",
                "batch_size": 32,
                "progress_bar": True,
                "normalize_embeddings": False,
                "embedding_separator": "\n",
                "metadata_fields_to_embed": [],
            },
        }

    @pytest.mark.unit
    def test_to_dict_with_custom_init_parameters(self):
        """
        Test serialization of InstructorDocumentEmbedder to a dictionary, using custom initialization parameters.
        """
        embedder = InstructorDocumentEmbedder(
            model_name_or_path="hkunlp/instructor-base",
            device="cuda",
            use_auth_token=True,
            instruction="Represent the financial document for retrieval",
            batch_size=64,
            progress_bar=False,
            normalize_embeddings=True,
            metadata_fields_to_embed=["test_field"],
            embedding_separator=" | ",
        )
        embedder_dict = embedder.to_dict()
        assert embedder_dict == {
            "type": "InstructorDocumentEmbedder",
            "init_parameters": {
                "model_name_or_path": "hkunlp/instructor-base",
                "device": "cuda",
                "use_auth_token": True,
                "instruction": "Represent the financial document for retrieval",
                "batch_size": 64,
                "progress_bar": False,
                "normalize_embeddings": True,
                "metadata_fields_to_embed": ["test_field"],
                "embedding_separator": " | ",
            },
        }

    @pytest.mark.unit
    def test_from_dict(self):
        """
        Test deserialization of InstructorDocumentEmbedder from a dictionary, using default initialization parameters.
        """
        embedder_dict = {
            "type": "InstructorDocumentEmbedder",
            "init_parameters": {
                "model_name_or_path": "hkunlp/instructor-base",
                "device": "cpu",
                "use_auth_token": None,
                "instruction": "Represent the 'domain' 'text_type' for 'task_objective'",
                "batch_size": 32,
                "progress_bar": True,
                "normalize_embeddings": False,
                "metadata_fields_to_embed": [],
                "embedding_separator": "\n",
            },
        }
        embedder = InstructorDocumentEmbedder.from_dict(embedder_dict)
        assert embedder.model_name_or_path == "hkunlp/instructor-base"
        assert embedder.device == "cpu"
        assert embedder.use_auth_token is None
        assert embedder.instruction == "Represent the 'domain' 'text_type' for 'task_objective'"
        assert embedder.batch_size == 32
        assert embedder.progress_bar is True
        assert embedder.normalize_embeddings is False
        assert embedder.metadata_fields_to_embed == []
        assert embedder.embedding_separator == "\n"

    @pytest.mark.unit
    def test_from_dict_with_custom_init_parameters(self):
        """
        Test deserialization of InstructorDocumentEmbedder from a dictionary, using custom initialization parameters.
        """
        embedder_dict = {
            "type": "InstructorDocumentEmbedder",
            "init_parameters": {
                "model_name_or_path": "hkunlp/instructor-base",
                "device": "cuda",
                "use_auth_token": True,
                "instruction": "Represent the financial document for retrieval",
                "batch_size": 64,
                "progress_bar": False,
                "normalize_embeddings": True,
                "metadata_fields_to_embed": ["test_field"],
                "embedding_separator": " | ",
            },
        }
        embedder = InstructorDocumentEmbedder.from_dict(embedder_dict)
        assert embedder.model_name_or_path == "hkunlp/instructor-base"
        assert embedder.device == "cuda"
        assert embedder.use_auth_token is True
        assert embedder.instruction == "Represent the financial document for retrieval"
        assert embedder.batch_size == 64
        assert embedder.progress_bar is False
        assert embedder.normalize_embeddings is True
        assert embedder.metadata_fields_to_embed == ["test_field"]
        assert embedder.embedding_separator == " | "

    @pytest.mark.unit
    @patch("haystack.preview.components.embedders.instructor_document_embedder._InstructorEmbeddingBackendFactory")
    def test_warmup(self, mocked_factory):
        """
        Test for checking embedder instances after warm-up.
        """
        embedder = InstructorDocumentEmbedder(model_name_or_path="hkunlp/instructor-base")
        mocked_factory.get_embedding_backend.assert_not_called()
        embedder.warm_up()
        mocked_factory.get_embedding_backend.assert_called_once_with(
            model_name_or_path="hkunlp/instructor-base", device="cpu", use_auth_token=None
        )

    @pytest.mark.unit
    @patch("haystack.preview.components.embedders.instructor_document_embedder._InstructorEmbeddingBackendFactory")
    def test_warmup_does_not_reload(self, mocked_factory):
        """
        Test for checking backend instances after multiple warm-ups.
        """
        embedder = InstructorDocumentEmbedder(model_name_or_path="hkunlp/instructor-base")
        mocked_factory.get_embedding_backend.assert_not_called()
        embedder.warm_up()
        embedder.warm_up()
        mocked_factory.get_embedding_backend.assert_called_once()

    @pytest.mark.unit
    def test_embed(self):
        """
        Test for checking output dimensions and embedding dimensions.
        """
        embedder = InstructorDocumentEmbedder(model_name_or_path="hkunlp/instructor-base")
        embedder.embedding_backend = MagicMock()
        embedder.embedding_backend.embed = lambda x, **kwargs: np.random.rand(len(x), 16).tolist()

        documents = [Document(text=f"Sample-document text {i}") for i in range(5)]

        result = embedder.run(documents=documents)

        assert isinstance(result["documents"], list)
        assert len(result["documents"]) == len(documents)
        for doc in result["documents"]:
            assert isinstance(doc, Document)
            assert isinstance(doc.embedding, list)
            assert isinstance(doc.embedding[0], float)

    @pytest.mark.unit
    def test_embed_incorrect_input_format(self):
        """
        Test for checking incorrect input format when creating embedding.
        """
        embedder = InstructorDocumentEmbedder(model_name_or_path="hkunlp/instructor-base")

        string_input = "text"
        list_integers_input = [1, 2, 3]

        with pytest.raises(TypeError, match="InstructorDocumentEmbedder expects a list of Documents as input."):
            embedder.run(documents=string_input)

        with pytest.raises(TypeError, match="InstructorDocumentEmbedder expects a list of Documents as input."):
            embedder.run(documents=list_integers_input)

    @pytest.mark.unit
    def test_embed_metadata(self):
        """
        Test for checking output dimensions and embedding dimensions for documents with a custom instruction and metadata.
        """
        embedder = InstructorDocumentEmbedder(
            model_name_or_path="model",
            instruction="Represent the financial document for retrieval",
            metadata_fields_to_embed=["meta_field"],
            embedding_separator="\n",
        )
        embedder.embedding_backend = MagicMock()

        documents = [
            Document(text=f"document-number {i}", metadata={"meta_field": f"meta_value {i}"}) for i in range(5)
        ]

        embedder.run(documents=documents)

        embedder.embedding_backend.embed.assert_called_once_with(
            [
                ["Represent the financial document for retrieval", "meta_value 0\ndocument-number 0"],
                ["Represent the financial document for retrieval", "meta_value 1\ndocument-number 1"],
                ["Represent the financial document for retrieval", "meta_value 2\ndocument-number 2"],
                ["Represent the financial document for retrieval", "meta_value 3\ndocument-number 3"],
                ["Represent the financial document for retrieval", "meta_value 4\ndocument-number 4"],
            ],
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=False,
        )
