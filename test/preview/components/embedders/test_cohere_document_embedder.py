from unittest.mock import patch, MagicMock
import pytest
from cohere.responses.embeddings import Embeddings
import numpy as np
from haystack.preview import Document
from haystack.preview.components.embedders.cohere_document_embedder import CohereDocumentEmbedder


class TestCohereDocumentEmbedder:
    @pytest.mark.unit
    def test_init_default(self):
        embedder = CohereDocumentEmbedder(api_key="test-api-key")
        assert embedder.api_key == "test-api-key"
        assert embedder.model_name == "embed-english-v2.0"
        assert embedder.api_base_url == "https://api.cohere.ai/v1/embed"
        assert embedder.truncate == "END"
        assert embedder.use_async_client == False
        assert embedder.max_retries == 3
        assert embedder.timeout == 120
        assert embedder.batch_size == 32
        assert embedder.progress_bar == True
        assert embedder.metadata_fields_to_embed == []
        assert embedder.embedding_separator == "\n"

    @pytest.mark.unit
    def test_init_with_parameters(self):
        embedder = CohereDocumentEmbedder(
            api_key="test-api-key",
            model_name="embed-multilingual-v2.0",
            api_base_url="https://custom-api-base-url.com",
            truncate="START",
            use_async_client=True,
            max_retries=5,
            timeout=60,
            batch_size=64,
            progress_bar=False,
            metadata_fields_to_embed=["test_field"],
            embedding_separator="-",
        )
        assert embedder.api_key == "test-api-key"
        assert embedder.model_name == "embed-multilingual-v2.0"
        assert embedder.api_base_url == "https://custom-api-base-url.com"
        assert embedder.truncate == "START"
        assert embedder.use_async_client == True
        assert embedder.max_retries == 5
        assert embedder.timeout == 60
        assert embedder.batch_size == 64
        assert embedder.progress_bar == False
        assert embedder.metadata_fields_to_embed == ["test_field"]
        assert embedder.embedding_separator == "-"

    @pytest.mark.unit
    def test_to_dict(self):
        embedder_component = CohereDocumentEmbedder(api_key="test-api-key")
        component_dict = embedder_component.to_dict()
        assert component_dict == {
            "type": "CohereDocumentEmbedder",
            "init_parameters": {
                "api_key": "test-api-key",
                "model_name": "embed-english-v2.0",
                "api_base_url": "https://api.cohere.ai/v1/embed",
                "truncate": "END",
                "use_async_client": False,
                "max_retries": 3,
                "timeout": 120,
                "batch_size": 32,
                "progress_bar": True,
                "metadata_fields_to_embed": [],
                "embedding_separator": "\n",
            },
        }

    @pytest.mark.unit
    def test_to_dict_with_custom_init_parameters(self):
        embedder_component = CohereDocumentEmbedder(
            api_key="test-api-key",
            model_name="embed-multilingual-v2.0",
            api_base_url="https://custom-api-base-url.com",
            truncate="START",
            use_async_client=True,
            max_retries=5,
            timeout=60,
            batch_size=64,
            progress_bar=False,
            metadata_fields_to_embed=["text_field"],
            embedding_separator="-",
        )
        component_dict = embedder_component.to_dict()
        assert component_dict == {
            "type": "CohereDocumentEmbedder",
            "init_parameters": {
                "api_key": "test-api-key",
                "model_name": "embed-multilingual-v2.0",
                "api_base_url": "https://custom-api-base-url.com",
                "truncate": "START",
                "use_async_client": True,
                "max_retries": 5,
                "timeout": 60,
                "batch_size": 64,
                "progress_bar": False,
                "metadata_fields_to_embed": ["text_field"],
                "embedding_separator": "-",
            },
        }

    @pytest.mark.unit
    def test_from_dict(self):
        embedder_component_dict = {
            "type": "CohereDocumentEmbedder",
            "init_parameters": {
                "api_key": "test-api-key",
                "model_name": "embed-english-v2.0",
                "api_base_url": "https://api.cohere.ai/v1/embed",
                "truncate": "START",
                "use_async_client": True,
                "max_retries": 5,
                "timeout": 60,
                "batch_size": 32,
                "progress_bar": False,
                "metadata_fields_to_embed": ["test_field"],
                "embedding_separator": "-",
            },
        }
        embedder = CohereDocumentEmbedder.from_dict(embedder_component_dict)
        assert embedder.api_key == "test-api-key"
        assert embedder.model_name == "embed-english-v2.0"
        assert embedder.api_base_url == "https://api.cohere.ai/v1/embed"
        assert embedder.truncate == "START"
        assert embedder.use_async_client == True
        assert embedder.max_retries == 5
        assert embedder.timeout == 60
        assert embedder.batch_size == 32
        assert embedder.progress_bar == False
        assert embedder.metadata_fields_to_embed == ["test_field"]
        assert embedder.embedding_separator == "-"

    @pytest.mark.unit
    def test_run(self):
        embedder = CohereDocumentEmbedder(api_key="test-api-key")
        embedder.embedding_backend = MagicMock()
        embedder.embedding_backend.embed = lambda x, **kwargs: np.random.rand(len(x), 2).tolist()

        docs = [
            Document(text="I love cheese", metadata={"topic": "Cuisine"}),
            Document(text="A transformer is a deep learning architecture", metadata={"topic": "ML"}),
        ]

        result = embedder.run(docs)
        embeddings = result["documents"]

        assert isinstance(embeddings, list)
        assert len(embeddings) == len(docs)
        for embedding in embeddings:
            assert isinstance(embedding, list)
            assert isinstance(embedding[0], float)

    @pytest.mark.unit
    def test_run_wrong_input_format(self):
        embedder = CohereDocumentEmbedder(api_key="test-api-key")

        string_input = "text"
        list_integers_input = [1, 2, 3]

        with pytest.raises(TypeError, match="CohereDocumentEmbedder expects a list of Documents as input"):
            embedder.run(documents=string_input)
        with pytest.raises(TypeError, match="CohereDocumentEmbedder expects a list of Documents as input"):
            embedder.run(documents=list_integers_input)
