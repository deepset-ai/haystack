# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, patch

import pytest
import torch

from haystack import Document
from haystack.components.rankers.sentence_transformers_similarity import SentenceTransformersSimilarityRanker
from haystack.utils.auth import Secret
from haystack.utils.device import ComponentDevice


class TestSentenceTransformersSimilarityRanker:
    def test_init_invalid_top_k(self):
        with pytest.raises(ValueError):
            SentenceTransformersSimilarityRanker(top_k=-1)

    @patch("haystack.components.rankers.sentence_transformers_similarity.CrossEncoder")
    def test_init_warm_up_torch_backend(self, mocked_cross_encoder):
        ranker = SentenceTransformersSimilarityRanker(
            model="sentence-transformers/all-MiniLM-L6-v2",
            token=None,
            device=ComponentDevice.from_str("cpu"),
            backend="torch",
            trust_remote_code=True,
        )

        ranker.warm_up()
        mocked_cross_encoder.assert_called_once_with(
            model_name_or_path="sentence-transformers/all-MiniLM-L6-v2",
            device="cpu",
            token=None,
            trust_remote_code=True,
            model_kwargs=None,
            tokenizer_kwargs=None,
            config_kwargs=None,
            backend="torch",
        )

    @patch("haystack.components.rankers.sentence_transformers_similarity.CrossEncoder")
    def test_init_warm_up_onnx_backend(self, mocked_cross_encoder):
        onnx_ranker = SentenceTransformersSimilarityRanker(
            model="sentence-transformers/all-MiniLM-L6-v2",
            token=None,
            device=ComponentDevice.from_str("cpu"),
            backend="onnx",
        )
        onnx_ranker.warm_up()

        mocked_cross_encoder.assert_called_once_with(
            model_name_or_path="sentence-transformers/all-MiniLM-L6-v2",
            device="cpu",
            token=None,
            trust_remote_code=False,
            model_kwargs=None,
            tokenizer_kwargs=None,
            config_kwargs=None,
            backend="onnx",
        )

    @patch("haystack.components.rankers.sentence_transformers_similarity.CrossEncoder")
    def test_init_warm_up_openvino_backend(self, mocked_cross_encoder):
        openvino_ranker = SentenceTransformersSimilarityRanker(
            model="sentence-transformers/all-MiniLM-L6-v2",
            token=None,
            device=ComponentDevice.from_str("cpu"),
            backend="openvino",
        )
        openvino_ranker.warm_up()

        mocked_cross_encoder.assert_called_once_with(
            model_name_or_path="sentence-transformers/all-MiniLM-L6-v2",
            device="cpu",
            token=None,
            trust_remote_code=False,
            model_kwargs=None,
            tokenizer_kwargs=None,
            config_kwargs=None,
            backend="openvino",
        )

    def test_to_dict(self):
        component = SentenceTransformersSimilarityRanker()
        data = component.to_dict()
        assert data == {
            "type": "haystack.components.rankers.sentence_transformers_similarity.SentenceTransformersSimilarityRanker",
            "init_parameters": {
                "device": ComponentDevice.resolve_device(None).to_dict(),
                "top_k": 10,
                "token": {"env_vars": ["HF_API_TOKEN", "HF_TOKEN"], "strict": False, "type": "env_var"},
                "query_prefix": "",
                "document_prefix": "",
                "model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
                "meta_fields_to_embed": [],
                "embedding_separator": "\n",
                "scale_score": True,
                "score_threshold": None,
                "trust_remote_code": False,
                "model_kwargs": None,
                "tokenizer_kwargs": None,
                "config_kwargs": None,
                "backend": "torch",
                "batch_size": 16,
            },
        }

    def test_to_dict_with_custom_init_parameters(self):
        component = SentenceTransformersSimilarityRanker(
            model="my_model",
            device=ComponentDevice.from_str("cuda:0"),
            token=Secret.from_env_var("ENV_VAR", strict=False),
            top_k=5,
            query_prefix="query_instruction: ",
            document_prefix="document_instruction: ",
            scale_score=False,
            score_threshold=0.01,
            trust_remote_code=True,
            model_kwargs={"torch_dtype": torch.float16},
            tokenizer_kwargs={"model_max_length": 512},
            batch_size=32,
        )
        data = component.to_dict()
        assert data == {
            "type": "haystack.components.rankers.sentence_transformers_similarity.SentenceTransformersSimilarityRanker",
            "init_parameters": {
                "device": {"type": "single", "device": "cuda:0"},
                "model": "my_model",
                "token": {"env_vars": ["ENV_VAR"], "strict": False, "type": "env_var"},
                "top_k": 5,
                "query_prefix": "query_instruction: ",
                "document_prefix": "document_instruction: ",
                "meta_fields_to_embed": [],
                "embedding_separator": "\n",
                "scale_score": False,
                "score_threshold": 0.01,
                "trust_remote_code": True,
                "model_kwargs": {"torch_dtype": "torch.float16"},
                "tokenizer_kwargs": {"model_max_length": 512},
                "config_kwargs": None,
                "backend": "torch",
                "batch_size": 32,
            },
        }

    def test_to_dict_with_quantization_options(self):
        component = SentenceTransformersSimilarityRanker(
            model_kwargs={
                "load_in_4bit": True,
                "bnb_4bit_use_double_quant": True,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_compute_dtype": torch.bfloat16,
            }
        )
        data = component.to_dict()
        assert data == {
            "type": "haystack.components.rankers.sentence_transformers_similarity.SentenceTransformersSimilarityRanker",
            "init_parameters": {
                "device": ComponentDevice.resolve_device(None).to_dict(),
                "top_k": 10,
                "query_prefix": "",
                "document_prefix": "",
                "token": {"env_vars": ["HF_API_TOKEN", "HF_TOKEN"], "strict": False, "type": "env_var"},
                "model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
                "meta_fields_to_embed": [],
                "embedding_separator": "\n",
                "scale_score": True,
                "score_threshold": None,
                "trust_remote_code": False,
                "model_kwargs": {
                    "load_in_4bit": True,
                    "bnb_4bit_use_double_quant": True,
                    "bnb_4bit_quant_type": "nf4",
                    "bnb_4bit_compute_dtype": "torch.bfloat16",
                },
                "tokenizer_kwargs": None,
                "config_kwargs": None,
                "backend": "torch",
                "batch_size": 16,
            },
        }

    def test_from_dict(self):
        data = {
            "type": "haystack.components.rankers.sentence_transformers_similarity.SentenceTransformersSimilarityRanker",
            "init_parameters": {
                "device": ComponentDevice.resolve_device(None).to_dict(),
                "model": "my_model",
                "token": None,
                "top_k": 5,
                "query_prefix": "",
                "document_prefix": "",
                "meta_fields_to_embed": [],
                "embedding_separator": "\n",
                "scale_score": False,
                "score_threshold": 0.01,
                "trust_remote_code": False,
                "model_kwargs": {"torch_dtype": "torch.float16"},
                "tokenizer_kwargs": None,
                "config_kwargs": None,
                "backend": "torch",
                "batch_size": 32,
            },
        }

        component = SentenceTransformersSimilarityRanker.from_dict(data)
        assert component.device == ComponentDevice.resolve_device(None)
        assert component.model == "my_model"
        assert component.token is None
        assert component.top_k == 5
        assert component.query_prefix == ""
        assert component.document_prefix == ""
        assert component.meta_fields_to_embed == []
        assert component.embedding_separator == "\n"
        assert not component.scale_score
        assert component.score_threshold == 0.01
        assert component.trust_remote_code is False
        assert component.model_kwargs == {"torch_dtype": torch.float16}
        assert component.tokenizer_kwargs is None
        assert component.config_kwargs is None
        assert component.batch_size == 32

    def test_from_dict_no_default_parameters(self):
        data = {
            "type": "haystack.components.rankers.sentence_transformers_similarity.SentenceTransformersSimilarityRanker",
            "init_parameters": {},
        }

        component = SentenceTransformersSimilarityRanker.from_dict(data)
        assert component.device == ComponentDevice.resolve_device(None)
        assert component.model == "cross-encoder/ms-marco-MiniLM-L-6-v2"
        assert component.token == Secret.from_env_var(["HF_API_TOKEN", "HF_TOKEN"], strict=False)
        assert component.top_k == 10
        assert component.query_prefix == ""
        assert component.document_prefix == ""
        assert component.meta_fields_to_embed == []
        assert component.embedding_separator == "\n"
        assert component.scale_score
        assert component.score_threshold is None
        assert component.trust_remote_code is False
        assert component.model_kwargs is None
        assert component.tokenizer_kwargs is None
        assert component.config_kwargs is None
        assert component.backend == "torch"
        assert component.batch_size == 16

    @patch("haystack.components.rankers.sentence_transformers_similarity.CrossEncoder")
    def test_warm_up(self, mock_cross_encoder):
        ranker = SentenceTransformersSimilarityRanker()

        mock_cross_encoder.assert_not_called()

        ranker.warm_up()
        mock_cross_encoder.assert_called()

    def test_run_invalid_top_k(self):
        ranker = SentenceTransformersSimilarityRanker()
        ranker._cross_encoder = MagicMock()

        with pytest.raises(ValueError):
            ranker.run(query="test", documents=[Document(content="document")], top_k=-1)

    def test_returns_empty_list_if_no_documents_are_provided(self):
        ranker = SentenceTransformersSimilarityRanker()
        ranker._cross_encoder = MagicMock()

        output = ranker.run(query="City in Germany", documents=[])
        assert not output["documents"]

    def test_raises_component_error_if_model_not_warmed_up(self):
        ranker = SentenceTransformersSimilarityRanker()
        with pytest.raises(RuntimeError):
            ranker.run(query="query", documents=[Document(content="document")])

    def test_embed_meta(self):
        ranker = SentenceTransformersSimilarityRanker(
            model="model", meta_fields_to_embed=["meta_field"], embedding_separator="\n"
        )
        mock_cross_encoder = MagicMock()
        ranker._cross_encoder = mock_cross_encoder

        documents = [Document(content=f"document number {i}", meta={"meta_field": f"meta_value {i}"}) for i in range(5)]

        ranker.run(query="test", documents=documents)

        _, kwargs = mock_cross_encoder.rank.call_args
        assert kwargs["query"] == "test"
        assert kwargs["documents"] == [
            "meta_value 0\ndocument number 0",
            "meta_value 1\ndocument number 1",
            "meta_value 2\ndocument number 2",
            "meta_value 3\ndocument number 3",
            "meta_value 4\ndocument number 4",
        ]
        assert kwargs["batch_size"] == 16
        assert isinstance(kwargs["activation_fn"], torch.nn.Sigmoid)
        assert kwargs["convert_to_numpy"] is True
        assert kwargs["return_documents"] is False

    def test_prefix(self):
        ranker = SentenceTransformersSimilarityRanker(
            model="model", query_prefix="query_instruction: ", document_prefix="document_instruction: "
        )
        mock_cross_encoder = MagicMock()
        ranker._cross_encoder = mock_cross_encoder

        documents = [Document(content=f"document number {i}", meta={"meta_field": f"meta_value {i}"}) for i in range(5)]

        ranker.run(query="test", documents=documents)

        _, kwargs = mock_cross_encoder.rank.call_args
        assert kwargs["query"] == "query_instruction: test"
        assert kwargs["documents"] == [
            "document_instruction: document number 0",
            "document_instruction: document number 1",
            "document_instruction: document number 2",
            "document_instruction: document number 3",
            "document_instruction: document number 4",
        ]
        assert kwargs["batch_size"] == 16
        assert isinstance(kwargs["activation_fn"], torch.nn.Sigmoid)
        assert kwargs["convert_to_numpy"] is True

    def test_scale_score_false(self):
        mock_cross_encoder = MagicMock()
        ranker = SentenceTransformersSimilarityRanker(model="model", scale_score=False)
        ranker._cross_encoder = mock_cross_encoder

        mock_cross_encoder.rank.return_value = [{"score": -10.6859, "corpus_id": 0}, {"score": -8.9874, "corpus_id": 1}]

        documents = [Document(content="document number 0"), Document(content="document number 1")]
        out = ranker.run(query="test", documents=documents)
        assert out["documents"][0].score == pytest.approx(-10.6859, abs=1e-4)
        assert out["documents"][1].score == pytest.approx(-8.9874, abs=1e-4)

    def test_score_threshold(self):
        mock_cross_encoder = MagicMock()
        ranker = SentenceTransformersSimilarityRanker(model="model", scale_score=False, score_threshold=0.1)
        ranker._cross_encoder = mock_cross_encoder

        mock_cross_encoder.rank.return_value = [{"score": 0.955, "corpus_id": 0}, {"score": 0.001, "corpus_id": 1}]

        documents = [Document(content="document number 0"), Document(content="document number 1")]
        out = ranker.run(query="test", documents=documents)
        assert len(out["documents"]) == 1

        documents = [Document(content="document number 0"), Document(content="document number 1")]
        out = ranker.run(query="test", documents=documents)
        assert len(out["documents"]) == 1

    @pytest.mark.integration
    @pytest.mark.slow
    def test_run(self):
        ranker = SentenceTransformersSimilarityRanker(model="cross-encoder/ms-marco-MiniLM-L-6-v2")
        ranker.warm_up()

        query = "City in Bosnia and Herzegovina"
        docs_before_texts = ["Berlin", "Belgrade", "Sarajevo"]
        expected_first_text = "Sarajevo"
        expected_scores = [2.2864143829792738e-05, 0.00012495707778725773, 0.009869757108390331]

        docs_before = [Document(content=text) for text in docs_before_texts]
        output = ranker.run(query=query, documents=docs_before)
        docs_after = output["documents"]

        assert len(docs_after) == 3
        assert docs_after[0].content == expected_first_text

        sorted_scores = sorted(expected_scores, reverse=True)
        assert docs_after[0].score == pytest.approx(sorted_scores[0], abs=1e-6)
        assert docs_after[1].score == pytest.approx(sorted_scores[1], abs=1e-6)
        assert docs_after[2].score == pytest.approx(sorted_scores[2], abs=1e-6)

    @pytest.mark.integration
    @pytest.mark.slow
    def test_run_top_k(self):
        ranker = SentenceTransformersSimilarityRanker(model="cross-encoder/ms-marco-MiniLM-L-6-v2", top_k=2)
        ranker.warm_up()

        query = "City in Bosnia and Herzegovina"
        docs_before_texts = ["Berlin", "Belgrade", "Sarajevo"]
        expected_first_text = "Sarajevo"

        docs_before = [Document(content=text) for text in docs_before_texts]
        output = ranker.run(query=query, documents=docs_before)
        docs_after = output["documents"]

        assert len(docs_after) == 2
        assert docs_after[0].content == expected_first_text

        sorted_scores = sorted([doc.score for doc in docs_after], reverse=True)
        assert [doc.score for doc in docs_after] == sorted_scores

    @pytest.mark.integration
    @pytest.mark.slow
    def test_run_single_document(self):
        ranker = SentenceTransformersSimilarityRanker(model="cross-encoder/ms-marco-MiniLM-L-6-v2", device=None)
        ranker.warm_up()
        docs_before = [Document(content="Berlin")]
        output = ranker.run(query="City in Germany", documents=docs_before)
        docs_after = output["documents"]

        assert len(docs_after) == 1
