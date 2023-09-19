from unittest.mock import patch, MagicMock
import pytest

import numpy as np

from haystack.preview.components.embedders.instructor_text_embedder import InstructorTextEmbedder


class TestInstructorTextEmbedder:
    @pytest.mark.unit
    def test_init_default(self):
        """
        Test default initialization parameters for InstructorTextEmbedder.
        """
        embedder = InstructorTextEmbedder(model_name_or_path="hkunlp/instructor-base")
        assert embedder.model_name_or_path == "hkunlp/instructor-base"
        assert embedder.device == "cpu"
        assert embedder.use_auth_token is None
        assert embedder.instruction == "Represent the 'domain' 'text_type' for 'task_objective'"
        assert embedder.batch_size == 32
        assert embedder.progress_bar is True
        assert embedder.normalize_embeddings is False

    @pytest.mark.unit
    def test_init_with_parameters(self):
        """
        Test custom initialization parameters for InstructorTextEmbedder.
        """
        embedder = InstructorTextEmbedder(
            model_name_or_path="hkunlp/instructor-base",
            device="cuda",
            use_auth_token=True,
            instruction="Represent the 'domain' 'text_type' for 'task_objective'",
            batch_size=64,
            progress_bar=False,
            normalize_embeddings=True,
        )
        assert embedder.model_name_or_path == "hkunlp/instructor-base"
        assert embedder.device == "cuda"
        assert embedder.use_auth_token is True
        assert embedder.instruction == "Represent the 'domain' 'text_type' for 'task_objective'"
        assert embedder.batch_size == 64
        assert embedder.progress_bar is False
        assert embedder.normalize_embeddings is True

    @pytest.mark.unit
    def test_to_dict(self):
        """
        Test serialization of InstructorTextEmbedder to a dictionary, using default initialization parameters.
        """
        embedder = InstructorTextEmbedder(model_name_or_path="hkunlp/instructor-base")
        embedder_dict = embedder.to_dict()
        assert embedder_dict == {
            "type": "InstructorTextEmbedder",
            "init_parameters": {
                "model_name_or_path": "hkunlp/instructor-base",
                "device": "cpu",
                "use_auth_token": None,
                "instruction": "Represent the 'domain' 'text_type' for 'task_objective'",
                "batch_size": 32,
                "progress_bar": True,
                "normalize_embeddings": False,
            },
        }

    @pytest.mark.unit
    def test_to_dict_with_custom_init_parameters(self):
        """
        Test serialization of InstructorTextEmbedder to a dictionary, using custom initialization parameters.
        """
        embedder = InstructorTextEmbedder(
            model_name_or_path="hkunlp/instructor-base",
            device="cuda",
            use_auth_token=True,
            instruction="Represent the financial document for retrieval",
            batch_size=64,
            progress_bar=False,
            normalize_embeddings=True,
        )
        embedder_dict = embedder.to_dict()
        assert embedder_dict == {
            "type": "InstructorTextEmbedder",
            "init_parameters": {
                "model_name_or_path": "hkunlp/instructor-base",
                "device": "cuda",
                "use_auth_token": True,
                "instruction": "Represent the financial document for retrieval",
                "batch_size": 64,
                "progress_bar": False,
                "normalize_embeddings": True,
            },
        }

    @pytest.mark.unit
    def test_from_dict(self):
        """
        Test deserialization of InstructorTextEmbedder from a dictionary, using default initialization parameters.
        """
        embedder_dict = {
            "type": "InstructorTextEmbedder",
            "init_parameters": {
                "model_name_or_path": "hkunlp/instructor-base",
                "device": "cpu",
                "use_auth_token": None,
                "instruction": "Represent the 'domain' 'text_type' for 'task_objective'",
                "batch_size": 32,
                "progress_bar": True,
                "normalize_embeddings": False,
            },
        }
        embedder = InstructorTextEmbedder.from_dict(embedder_dict)
        assert embedder.model_name_or_path == "hkunlp/instructor-base"
        assert embedder.device == "cpu"
        assert embedder.use_auth_token is None
        assert embedder.instruction == "Represent the 'domain' 'text_type' for 'task_objective'"
        assert embedder.batch_size == 32
        assert embedder.progress_bar is True
        assert embedder.normalize_embeddings is False

    @pytest.mark.unit
    def test_from_dict_with_custom_init_parameters(self):
        """
        Test deserialization of InstructorTextEmbedder from a dictionary, using custom initialization parameters.
        """
        embedder_dict = {
            "type": "InstructorTextEmbedder",
            "init_parameters": {
                "model_name_or_path": "hkunlp/instructor-base",
                "device": "cuda",
                "use_auth_token": True,
                "instruction": "Represent the financial document for retrieval",
                "batch_size": 64,
                "progress_bar": False,
                "normalize_embeddings": True,
            },
        }
        embedder = InstructorTextEmbedder.from_dict(embedder_dict)
        assert embedder.model_name_or_path == "hkunlp/instructor-base"
        assert embedder.device == "cuda"
        assert embedder.use_auth_token is True
        assert embedder.instruction == "Represent the financial document for retrieval"
        assert embedder.batch_size == 64
        assert embedder.progress_bar is False
        assert embedder.normalize_embeddings is True

    @pytest.mark.unit
    @patch("haystack.preview.components.embedders.instructor_text_embedder._InstructorEmbeddingBackendFactory")
    def test_warmup(self, mocked_factory):
        """
        Test for checking embedder instances after warm-up.
        """
        embedder = InstructorTextEmbedder(model_name_or_path="hkunlp/instructor-base")
        mocked_factory.get_embedding_backend.assert_not_called()
        embedder.warm_up()
        mocked_factory.get_embedding_backend.assert_called_once_with(
            model_name_or_path="hkunlp/instructor-base", device="cpu", use_auth_token=None
        )

    @pytest.mark.unit
    @patch("haystack.preview.components.embedders.instructor_text_embedder._InstructorEmbeddingBackendFactory")
    def test_warmup_does_not_reload(self, mocked_factory):
        """
        Test for checking backend instances after multiple warm-ups.
        """
        embedder = InstructorTextEmbedder(model_name_or_path="hkunlp/instructor-base")
        mocked_factory.get_embedding_backend.assert_not_called()
        embedder.warm_up()
        embedder.warm_up()
        mocked_factory.get_embedding_backend.assert_called_once()

    @pytest.mark.unit
    def test_embed(self):
        """
        Test for checking output dimensions and embedding dimensions.
        """
        embedder = InstructorTextEmbedder(model_name_or_path="hkunlp/instructor-base")
        embedder.embedding_backend = MagicMock()
        embedder.embedding_backend.embed = lambda x, **kwargs: np.random.rand(len(x), 16).tolist()

        text = "Good text to embed"

        result = embedder.run(text=text)
        embedding = result["embedding"]

        assert isinstance(embedding, list)
        assert all(isinstance(emb, float) for emb in embedding)

    @pytest.mark.unit
    def test_run_wrong_incorrect_format(self):
        """
        Test for checking incorrect input format when creating embedding.
        """
        embedder = InstructorTextEmbedder(model_name_or_path="hkunlp/instructor-base")
        embedder.embedding_backend = MagicMock()

        list_integers_input = [1, 2, 3]

        with pytest.raises(TypeError, match="InstructorTextEmbedder expects a string as input"):
            embedder.run(text=list_integers_input)
