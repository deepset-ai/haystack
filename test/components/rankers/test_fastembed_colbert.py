# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from haystack import Document
from haystack.components.rankers.fastembed_colbert import FastembedColBERTRanker
from haystack.utils.device import ComponentDevice


class TestFastembedColBERTRanker:
    def test_init_invalid_top_k(self):
        with pytest.raises(ValueError):
            FastembedColBERTRanker(top_k=-1)

    def test_init_invalid_top_k_zero(self):
        with pytest.raises(ValueError):
            FastembedColBERTRanker(top_k=0)

    @patch("haystack.components.rankers.fastembed_colbert.LateInteractionTextEmbedding")
    def test_warm_up(self, mock_late_interaction):
        ranker = FastembedColBERTRanker(model="colbert-ir/colbertv2.0", device=ComponentDevice.from_str("cpu"))

        mock_late_interaction.assert_not_called()

        ranker.warm_up()
        mock_late_interaction.assert_called_once_with(model_name="colbert-ir/colbertv2.0", device="cpu")

    @patch("haystack.components.rankers.fastembed_colbert.LateInteractionTextEmbedding")
    def test_warm_up_cuda_device(self, mock_late_interaction):
        ranker = FastembedColBERTRanker(model="colbert-ir/colbertv2.0", device=ComponentDevice.from_str("cuda:0"))

        ranker.warm_up()
        mock_late_interaction.assert_called_once_with(model_name="colbert-ir/colbertv2.0", device="cuda")

    def test_to_dict(self):
        component = FastembedColBERTRanker()
        data = component.to_dict()
        assert data == {
            "type": "haystack.components.rankers.fastembed_colbert.FastembedColBERTRanker",
            "init_parameters": {
                "device": ComponentDevice.resolve_device(None).to_dict(),
                "model": "colbert-ir/colbertv2.0",
                "top_k": 10,
                "query_prefix": "",
                "query_suffix": "",
                "document_prefix": "",
                "document_suffix": "",
                "meta_fields_to_embed": [],
                "embedding_separator": "\n",
                "score_threshold": None,
                "batch_size": 32,
            },
        }

    def test_to_dict_with_custom_init_parameters(self):
        component = FastembedColBERTRanker(
            model="answerdotai/answerai-colbert-small-v1",
            device=ComponentDevice.from_str("cuda:0"),
            top_k=5,
            query_prefix="query: ",
            query_suffix="\n",
            document_prefix="passage: ",
            document_suffix="\n",
            meta_fields_to_embed=["title"],
            embedding_separator=" | ",
            score_threshold=0.5,
            batch_size=64,
        )
        data = component.to_dict()
        assert data == {
            "type": "haystack.components.rankers.fastembed_colbert.FastembedColBERTRanker",
            "init_parameters": {
                "device": {"type": "single", "device": "cuda:0"},
                "model": "answerdotai/answerai-colbert-small-v1",
                "top_k": 5,
                "query_prefix": "query: ",
                "query_suffix": "\n",
                "document_prefix": "passage: ",
                "document_suffix": "\n",
                "meta_fields_to_embed": ["title"],
                "embedding_separator": " | ",
                "score_threshold": 0.5,
                "batch_size": 64,
            },
        }

    def test_from_dict(self):
        data = {
            "type": "haystack.components.rankers.fastembed_colbert.FastembedColBERTRanker",
            "init_parameters": {
                "device": ComponentDevice.resolve_device(None).to_dict(),
                "model": "answerdotai/answerai-colbert-small-v1",
                "top_k": 5,
                "query_prefix": "query: ",
                "query_suffix": "",
                "document_prefix": "passage: ",
                "document_suffix": "",
                "meta_fields_to_embed": ["title"],
                "embedding_separator": " | ",
                "score_threshold": 0.5,
                "batch_size": 64,
            },
        }

        component = FastembedColBERTRanker.from_dict(data)
        assert component.device == ComponentDevice.resolve_device(None)
        assert component.model == "answerdotai/answerai-colbert-small-v1"
        assert component.top_k == 5
        assert component.query_prefix == "query: "
        assert component.query_suffix == ""
        assert component.document_prefix == "passage: "
        assert component.document_suffix == ""
        assert component.meta_fields_to_embed == ["title"]
        assert component.embedding_separator == " | "
        assert component.score_threshold == 0.5
        assert component.batch_size == 64

    def test_from_dict_no_default_parameters(self):
        data = {"type": "haystack.components.rankers.fastembed_colbert.FastembedColBERTRanker", "init_parameters": {}}

        component = FastembedColBERTRanker.from_dict(data)
        assert component.device == ComponentDevice.resolve_device(None)
        assert component.model == "colbert-ir/colbertv2.0"
        assert component.top_k == 10
        assert component.query_prefix == ""
        assert component.query_suffix == ""
        assert component.document_prefix == ""
        assert component.document_suffix == ""
        assert component.meta_fields_to_embed == []
        assert component.embedding_separator == "\n"
        assert component.score_threshold is None
        assert component.batch_size == 32

    def test_run_invalid_top_k(self):
        ranker = FastembedColBERTRanker()
        ranker._embedding_model = MagicMock()

        with pytest.raises(ValueError):
            ranker.run(query="test", documents=[Document(content="document")], top_k=-1)

    def test_returns_empty_list_if_no_documents_are_provided(self):
        ranker = FastembedColBERTRanker()
        ranker._embedding_model = MagicMock()

        output = ranker.run(query="City in Germany", documents=[])
        assert not output["documents"]

    def test_embed_meta(self):
        ranker = FastembedColBERTRanker(
            model="colbert-ir/colbertv2.0", meta_fields_to_embed=["meta_field"], embedding_separator="\n"
        )

        mock_model = MagicMock()
        mock_model.query_embed.return_value = [np.random.rand(3, 128)]
        mock_model.embed.return_value = [np.random.rand(5, 128) for _ in range(5)]
        ranker._embedding_model = mock_model

        documents = [Document(content=f"document number {i}", meta={"meta_field": f"meta_value {i}"}) for i in range(5)]

        ranker.run(query="test", documents=documents)

        mock_model.query_embed.assert_called_once_with(["test"])
        mock_model.embed.assert_called_once_with(
            [
                "meta_value 0\ndocument number 0",
                "meta_value 1\ndocument number 1",
                "meta_value 2\ndocument number 2",
                "meta_value 3\ndocument number 3",
                "meta_value 4\ndocument number 4",
            ]
        )

    def test_prefix(self):
        ranker = FastembedColBERTRanker(
            model="colbert-ir/colbertv2.0", query_prefix="query: ", document_prefix="passage: "
        )

        mock_model = MagicMock()
        mock_model.query_embed.return_value = [np.random.rand(3, 128)]
        mock_model.embed.return_value = [np.random.rand(5, 128) for _ in range(5)]
        ranker._embedding_model = mock_model

        documents = [Document(content=f"document number {i}") for i in range(5)]

        ranker.run(query="test", documents=documents)

        mock_model.query_embed.assert_called_once_with(["query: test"])
        mock_model.embed.assert_called_once_with(
            [
                "passage: document number 0",
                "passage: document number 1",
                "passage: document number 2",
                "passage: document number 3",
                "passage: document number 4",
            ]
        )

    def test_suffix(self):
        ranker = FastembedColBERTRanker(
            model="colbert-ir/colbertv2.0", query_suffix="<|endoftext|>", document_suffix="<|endoftext|>"
        )

        mock_model = MagicMock()
        mock_model.query_embed.return_value = [np.random.rand(3, 128)]
        mock_model.embed.return_value = [np.random.rand(5, 128) for _ in range(5)]
        ranker._embedding_model = mock_model

        documents = [Document(content=f"document number {i}") for i in range(5)]

        ranker.run(query="test", documents=documents)

        mock_model.query_embed.assert_called_once_with(["test<|endoftext|>"])
        mock_model.embed.assert_called_once_with(
            [
                "document number 0<|endoftext|>",
                "document number 1<|endoftext|>",
                "document number 2<|endoftext|>",
                "document number 3<|endoftext|>",
                "document number 4<|endoftext|>",
            ]
        )

    def test_score_threshold(self):
        ranker = FastembedColBERTRanker(model="colbert-ir/colbertv2.0", score_threshold=5.0)

        mock_model = MagicMock()
        query_emb = np.random.rand(3, 128)
        mock_model.query_embed.return_value = [query_emb]
        doc_emb_high = np.ones((5, 128)) * 0.5
        doc_emb_low = np.ones((5, 128)) * 0.1
        mock_model.embed.return_value = [doc_emb_high, doc_emb_low]
        ranker._embedding_model = mock_model

        documents = [Document(content="document 0"), Document(content="document 1")]
        out = ranker.run(query="test", documents=documents)

        assert len(out["documents"]) <= 2

        for doc in out["documents"]:
            assert doc.score is not None
            assert doc.score >= 5.0

    def test_top_k_override(self):
        ranker = FastembedColBERTRanker(model="colbert-ir/colbertv2.0", top_k=10)

        mock_model = MagicMock()
        query_emb = np.random.rand(3, 128)
        mock_model.query_embed.return_value = [query_emb]
        mock_model.embed.return_value = [np.random.rand(5, 128) for _ in range(5)]
        ranker._embedding_model = mock_model

        documents = [Document(content=f"document {i}") for i in range(5)]
        out = ranker.run(query="test", documents=documents, top_k=2)

        assert len(out["documents"]) == 2

    def test_compute_maxsim_scores(self):
        ranker = FastembedColBERTRanker()

        query_emb = np.array([[1.0, 0.0], [0.0, 1.0]])
        doc1_emb = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])
        doc2_emb = np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]])

        scores = ranker._compute_maxsim_scores(query_emb, [doc1_emb, doc2_emb])

        assert len(scores) == 2
        assert scores[0] == pytest.approx(2.0, abs=1e-6)
        assert scores[1] == pytest.approx(2.0, abs=1e-6)

    def test_scores_are_python_floats(self):
        ranker = FastembedColBERTRanker()

        mock_model = MagicMock()
        query_emb = np.random.rand(3, 128)
        mock_model.query_embed.return_value = [query_emb]
        mock_model.embed.return_value = [np.random.rand(5, 128) for _ in range(3)]
        ranker._embedding_model = mock_model

        documents = [Document(content=f"doc {i}") for i in range(3)]
        out = ranker.run(query="test", documents=documents)

        for doc in out["documents"]:
            assert isinstance(doc.score, float)

    @pytest.mark.integration
    @pytest.mark.slow
    def test_run(self):
        ranker = FastembedColBERTRanker(model="answerdotai/answerai-colbert-small-v1")

        query = "City in Bosnia and Herzegovina"
        docs_before_texts = ["Berlin", "Belgrade", "Sarajevo"]
        expected_first_text = "Sarajevo"

        docs_before = [Document(content=text) for text in docs_before_texts]
        output = ranker.run(query=query, documents=docs_before)
        docs_after = output["documents"]

        assert len(docs_after) == 3
        assert docs_after[0].content == expected_first_text

        for i in range(len(docs_after) - 1):
            assert docs_after[i].score >= docs_after[i + 1].score

        for doc in docs_after:
            assert isinstance(doc.score, float)

    @pytest.mark.integration
    @pytest.mark.slow
    def test_run_top_k(self):
        ranker = FastembedColBERTRanker(model="answerdotai/answerai-colbert-small-v1", top_k=2)

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

        for doc in docs_after:
            assert isinstance(doc.score, float)

    @pytest.mark.integration
    @pytest.mark.slow
    def test_run_single_document(self):
        ranker = FastembedColBERTRanker(model="answerdotai/answerai-colbert-small-v1")
        docs_before = [Document(content="Berlin")]
        output = ranker.run(query="City in Germany", documents=docs_before)
        docs_after = output["documents"]

        assert len(docs_after) == 1
        assert isinstance(docs_after[0].score, float)
        assert docs_after[0].score is not None
