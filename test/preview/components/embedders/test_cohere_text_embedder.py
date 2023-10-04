from unittest.mock import patch
import pytest
from cohere.responses.embeddings import Embeddings
from haystack.preview.components.embedders.cohere_text_embedder import CohereTextEmbedder


class TestCohereTextEmbedder:
    @pytest.mark.unit
    def test_init_default(self):
        """
        Test default initialization parameters for CohereTextEmbedder.
        """
        embedder = CohereTextEmbedder(api_key="test-api-key")

        assert embedder.api_key == "test-api-key"
        assert embedder.model_name == "embed-english-v2.0"
        assert embedder.api_base_url == "https://api.cohere.ai/v1/embed"
        assert embedder.truncate == "END"
        assert embedder.use_async_client == False
        assert embedder.max_retries == 3
        assert embedder.timeout == 120

    @pytest.mark.unit
    def test_init_with_parameters(self):
        """
        Test custom initialization parameters for CohereTextEmbedder.
        """
        embedder = CohereTextEmbedder(
            api_key="test-api-key",
            model_name="embed-multilingual-v2.0",
            api_base_url="https://custom-api-base-url.com",
            truncate="START",
            use_async_client=True,
            max_retries=5,
            timeout=60,
        )
        assert embedder.api_key == "test-api-key"
        assert embedder.model_name == "embed-multilingual-v2.0"
        assert embedder.api_base_url == "https://custom-api-base-url.com"
        assert embedder.truncate == "START"
        assert embedder.use_async_client == True
        assert embedder.max_retries == 5
        assert embedder.timeout == 60

    @pytest.mark.unit
    def test_to_dict(self):
        """
        Test serialization of this component to a dictionary, using default initialization parameters.
        """
        embedder_component = CohereTextEmbedder(api_key="test-api-key")
        component_dict = embedder_component.to_dict()
        assert component_dict == {
            "type": "CohereTextEmbedder",
            "init_parameters": {
                "model_name": "embed-english-v2.0",
                "api_base_url": "https://api.cohere.ai/v1/embed",
                "truncate": "END",
                "use_async_client": False,
                "max_retries": 3,
                "timeout": 120,
            },
        }

    @pytest.mark.unit
    def test_to_dict_with_custom_init_parameters(self):
        """
        Test serialization of this component to a dictionary, using custom initialization parameters.
        """
        embedder_component = CohereTextEmbedder(
            api_key="test-api-key",
            model_name="embed-multilingual-v2.0",
            api_base_url="https://custom-api-base-url.com",
            truncate="START",
            use_async_client=True,
            max_retries=5,
            timeout=60,
        )
        component_dict = embedder_component.to_dict()
        assert component_dict == {
            "type": "CohereTextEmbedder",
            "init_parameters": {
                "model_name": "embed-multilingual-v2.0",
                "api_base_url": "https://custom-api-base-url.com",
                "truncate": "START",
                "use_async_client": True,
                "max_retries": 5,
                "timeout": 60,
            },
        }

    @pytest.mark.unit
    def test_from_dict(self):
        """
        Test deserialization of this component from a dictionary, using default initialization parameters.
        """
        embedder_component_dict = {
            "type": "CohereTextEmbedder",
            "init_parameters": {
                "api_key": "test-api-key",
                "model_name": "embed-english-v2.0",
                "api_base_url": "https://api.cohere.ai/v1/embed",
                "truncate": "END",
                "use_async_client": False,
                "max_retries": 3,
                "timeout": 120,
            },
        }
        embedder = CohereTextEmbedder.from_dict(embedder_component_dict)

        assert embedder.api_key == "test-api-key"
        assert embedder.model_name == "embed-english-v2.0"
        assert embedder.api_base_url == "https://api.cohere.ai/v1/embed"
        assert embedder.truncate == "END"
        assert embedder.use_async_client == False
        assert embedder.max_retries == 3
        assert embedder.timeout == 120

    @pytest.mark.unit
    def test_from_dict_with_custom_init_parameters(self):
        """
        Test deserialization of this component from a dictionary, using custom initialization parameters.
        """
        embedder_component_dict = {
            "type": "CohereTextEmbedder",
            "init_parameters": {
                "api_key": "test-api-key",
                "model_name": "embed-multilingual-v2.0",
                "api_base_url": "https://custom-api-base-url.com",
                "truncate": "START",
                "use_async_client": True,
                "max_retries": 5,
                "timeout": 60,
            },
        }
        embedder = CohereTextEmbedder.from_dict(embedder_component_dict)

        assert embedder.api_key == "test-api-key"
        assert embedder.model_name == "embed-multilingual-v2.0"
        assert embedder.api_base_url == "https://custom-api-base-url.com"
        assert embedder.truncate == "START"
        assert embedder.use_async_client == True
        assert embedder.max_retries == 5
        assert embedder.timeout == 60

    @pytest.mark.unit
    def test_run_wrong_input_format(self):
        """
        Test for checking incorrect input when creating embedding.
        """
        embedder = CohereTextEmbedder(api_key="test-api-key")

        list_integers_input = ["text_snippet_1", "text_snippet_2"]

        with pytest.raises(TypeError, match="CohereTextEmbedder expects a string as input"):
            embedder.run(text=list_integers_input)
