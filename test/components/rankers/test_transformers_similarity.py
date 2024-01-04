from unittest.mock import MagicMock, patch
import pytest
import torch

from haystack import Document, ComponentError
from haystack.components.rankers.transformers_similarity import TransformersSimilarityRanker


class TestSimilarityRanker:
    def test_to_dict(self):
        component = TransformersSimilarityRanker()
        data = component.to_dict()
        assert data == {
            "type": "haystack.components.rankers.transformers_similarity.TransformersSimilarityRanker",
            "init_parameters": {
                "device": "cpu",
                "top_k": 10,
                "token": None,
                "model_name_or_path": "cross-encoder/ms-marco-MiniLM-L-6-v2",
                "meta_fields_to_embed": [],
                "embedding_separator": "\n",
                "scale_score": True,
                "calibration_factor": 1.0,
                "score_threshold": None,
                "model_kwargs": {},
            },
        }

    def test_to_dict_with_custom_init_parameters(self):
        component = TransformersSimilarityRanker(
            model_name_or_path="my_model",
            device="cuda",
            token="my_token",
            top_k=5,
            scale_score=False,
            calibration_factor=None,
            score_threshold=0.01,
            model_kwargs={"torch_dtype": "auto"},
        )
        data = component.to_dict()
        assert data == {
            "type": "haystack.components.rankers.transformers_similarity.TransformersSimilarityRanker",
            "init_parameters": {
                "device": "cuda",
                "model_name_or_path": "my_model",
                "token": None,  # we don't serialize valid tokens,
                "top_k": 5,
                "meta_fields_to_embed": [],
                "embedding_separator": "\n",
                "scale_score": False,
                "calibration_factor": None,
                "score_threshold": 0.01,
                "model_kwargs": {"torch_dtype": "auto"},
            },
        }

    @patch("torch.sigmoid")
    @patch("torch.sort")
    def test_embed_meta(self, mocked_sort, mocked_sigmoid):
        mocked_sort.return_value = (None, torch.tensor([0]))
        mocked_sigmoid.return_value = torch.tensor([0])
        embedder = TransformersSimilarityRanker(
            model_name_or_path="model", meta_fields_to_embed=["meta_field"], embedding_separator="\n"
        )
        embedder.model = MagicMock()
        embedder.tokenizer = MagicMock()

        documents = [Document(content=f"document number {i}", meta={"meta_field": f"meta_value {i}"}) for i in range(5)]

        embedder.run(query="test", documents=documents)

        embedder.tokenizer.assert_called_once_with(
            [
                ["test", "meta_value 0\ndocument number 0"],
                ["test", "meta_value 1\ndocument number 1"],
                ["test", "meta_value 2\ndocument number 2"],
                ["test", "meta_value 3\ndocument number 3"],
                ["test", "meta_value 4\ndocument number 4"],
            ],
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

    @pytest.mark.integration
    @pytest.mark.parametrize(
        "query,docs_before_texts,expected_first_text,scores",
        [
            (
                "City in Bosnia and Herzegovina",
                ["Berlin", "Belgrade", "Sarajevo"],
                "Sarajevo",
                [2.2864143829792738e-05, 0.00012495707778725773, 0.009869757108390331],
            ),
            (
                "Machine learning",
                ["Python", "Bakery in Paris", "Tesla Giga Berlin"],
                "Python",
                [1.9063229046878405e-05, 1.434577916370472e-05, 1.3049247172602918e-05],
            ),
            (
                "Cubist movement",
                ["Nirvana", "Pablo Picasso", "Coffee"],
                "Pablo Picasso",
                [1.3313065210240893e-05, 9.90335684036836e-05, 1.3518535524781328e-05],
            ),
        ],
    )
    def test_run(self, query, docs_before_texts, expected_first_text, scores):
        """
        Test if the component ranks documents correctly.
        """
        ranker = TransformersSimilarityRanker(model_name_or_path="cross-encoder/ms-marco-MiniLM-L-6-v2")
        ranker.warm_up()
        docs_before = [Document(content=text) for text in docs_before_texts]
        output = ranker.run(query=query, documents=docs_before)
        docs_after = output["documents"]

        assert len(docs_after) == 3
        assert docs_after[0].content == expected_first_text

        sorted_scores = sorted(scores, reverse=True)
        assert docs_after[0].score == pytest.approx(sorted_scores[0])
        assert docs_after[1].score == pytest.approx(sorted_scores[1])
        assert docs_after[2].score == pytest.approx(sorted_scores[2])

    #  Returns an empty list if no documents are provided
    @pytest.mark.integration
    def test_returns_empty_list_if_no_documents_are_provided(self):
        sampler = TransformersSimilarityRanker()
        sampler.warm_up()
        output = sampler.run(query="City in Germany", documents=[])
        assert not output["documents"]

    #  Raises ComponentError if model is not warmed up
    @pytest.mark.integration
    def test_raises_component_error_if_model_not_warmed_up(self):
        sampler = TransformersSimilarityRanker()

        with pytest.raises(ComponentError):
            sampler.run(query="query", documents=[Document(content="document")])

    @pytest.mark.integration
    @pytest.mark.parametrize(
        "query,docs_before_texts,expected_first_text",
        [
            ("City in Bosnia and Herzegovina", ["Berlin", "Belgrade", "Sarajevo"], "Sarajevo"),
            ("Machine learning", ["Python", "Bakery in Paris", "Tesla Giga Berlin"], "Python"),
            ("Cubist movement", ["Nirvana", "Pablo Picasso", "Coffee"], "Pablo Picasso"),
        ],
    )
    def test_run_top_k(self, query, docs_before_texts, expected_first_text):
        """
        Test if the component ranks documents correctly with a custom top_k.
        """
        ranker = TransformersSimilarityRanker(model_name_or_path="cross-encoder/ms-marco-MiniLM-L-6-v2", top_k=2)
        ranker.warm_up()
        docs_before = [Document(content=text) for text in docs_before_texts]
        output = ranker.run(query=query, documents=docs_before)
        docs_after = output["documents"]

        assert len(docs_after) == 2
        assert docs_after[0].content == expected_first_text

        sorted_scores = sorted([doc.score for doc in docs_after], reverse=True)
        assert [doc.score for doc in docs_after] == sorted_scores

    @pytest.mark.integration
    def test_run_single_document(self):
        """
        Test if the component runs with a single document.
        """
        ranker = TransformersSimilarityRanker(model_name_or_path="cross-encoder/ms-marco-MiniLM-L-6-v2", device=None)
        ranker.warm_up()
        docs_before = [Document(content="Berlin")]
        output = ranker.run(query="City in Germany", documents=docs_before)
        docs_after = output["documents"]

        assert len(docs_after) == 1
