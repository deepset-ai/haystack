# pylint: disable=too-many-public-methods
from unittest.mock import patch, Mock

import pytest
import torch

from haystack.preview.components.generators.hugging_face_local import HuggingFaceLocalGenerator, StopWordsCriteria


class TestHuggingFaceLocalGenerator:
    @pytest.mark.unit
    @patch("haystack.preview.components.generators.hugging_face_local.model_info")
    def test_init_default(self, model_info_mock):
        model_info_mock.return_value.pipeline_tag = "text2text-generation"
        generator = HuggingFaceLocalGenerator()

        assert generator.pipeline_kwargs == {
            "model": "google/flan-t5-base",
            "task": "text2text-generation",
            "token": None,
        }
        assert generator.generation_kwargs == {}
        assert generator.pipeline is None

    @pytest.mark.unit
    def test_init_custom_token(self):
        generator = HuggingFaceLocalGenerator(
            model_name_or_path="google/flan-t5-base", task="text2text-generation", token="test-token"
        )

        assert generator.pipeline_kwargs == {
            "model": "google/flan-t5-base",
            "task": "text2text-generation",
            "token": "test-token",
        }

    @pytest.mark.unit
    def test_init_custom_device(self):
        generator = HuggingFaceLocalGenerator(
            model_name_or_path="google/flan-t5-base", task="text2text-generation", device="cuda:0"
        )

        assert generator.pipeline_kwargs == {
            "model": "google/flan-t5-base",
            "task": "text2text-generation",
            "token": None,
            "device": "cuda:0",
        }

    @pytest.mark.unit
    def test_init_task_parameter(self):
        generator = HuggingFaceLocalGenerator(task="text2text-generation")

        assert generator.pipeline_kwargs == {
            "model": "google/flan-t5-base",
            "task": "text2text-generation",
            "token": None,
        }

    @pytest.mark.unit
    def test_init_task_in_pipeline_kwargs(self):
        generator = HuggingFaceLocalGenerator(pipeline_kwargs={"task": "text2text-generation"})

        assert generator.pipeline_kwargs == {
            "model": "google/flan-t5-base",
            "task": "text2text-generation",
            "token": None,
        }

    @pytest.mark.unit
    @patch("haystack.preview.components.generators.hugging_face_local.model_info")
    def test_init_task_inferred_from_model_name(self, model_info_mock):
        model_info_mock.return_value.pipeline_tag = "text2text-generation"
        generator = HuggingFaceLocalGenerator(model_name_or_path="google/flan-t5-base")

        assert generator.pipeline_kwargs == {
            "model": "google/flan-t5-base",
            "task": "text2text-generation",
            "token": None,
        }

    @pytest.mark.unit
    def test_init_invalid_task(self):
        with pytest.raises(ValueError, match="is not supported."):
            HuggingFaceLocalGenerator(task="text-classification")

    @pytest.mark.unit
    def test_init_pipeline_kwargs_override_other_parameters(self):
        """
        pipeline_kwargs represent the main configuration of this component.
        If they are provided, they should override other init parameters.
        """

        pipeline_kwargs = {
            "model": "gpt2",
            "task": "text-generation",
            "device": "cuda:0",
            "token": "another-test-token",
        }

        generator = HuggingFaceLocalGenerator(
            model_name_or_path="google/flan-t5-base",
            task="text2text-generation",
            device="cpu",
            token="test-token",
            pipeline_kwargs=pipeline_kwargs,
        )

        assert generator.pipeline_kwargs == pipeline_kwargs

    @pytest.mark.unit
    def test_init_generation_kwargs(self):
        generator = HuggingFaceLocalGenerator(task="text2text-generation", generation_kwargs={"max_new_tokens": 100})

        assert generator.generation_kwargs == {"max_new_tokens": 100}

    @pytest.mark.unit
    def test_init_set_return_full_text(self):
        """
        if not specified, return_full_text is set to False for text-generation task
        (only generated text is returned, excluding prompt)
        """
        generator = HuggingFaceLocalGenerator(task="text-generation")

        assert generator.generation_kwargs == {"return_full_text": False}

    @pytest.mark.unit
    def test_init_fails_with_both_stopwords_and_stoppingcriteria(self):
        with pytest.raises(
            ValueError,
            match="Found both the `stop_words` init parameter and the `stopping_criteria` key in `generation_kwargs`",
        ):
            HuggingFaceLocalGenerator(
                task="text2text-generation",
                stop_words=["coca", "cola"],
                generation_kwargs={"stopping_criteria": "fake-stopping-criteria"},
            )

    @pytest.mark.unit
    @patch("haystack.preview.components.generators.hugging_face_local.model_info")
    def test_to_dict_default(self, model_info_mock):
        model_info_mock.return_value.pipeline_tag = "text2text-generation"

        component = HuggingFaceLocalGenerator()
        data = component.to_dict()

        assert data == {
            "type": "haystack.preview.components.generators.hugging_face_local.HuggingFaceLocalGenerator",
            "init_parameters": {
                "pipeline_kwargs": {"model": "google/flan-t5-base", "task": "text2text-generation", "token": None},
                "generation_kwargs": {},
                "stop_words": None,
            },
        }

    @pytest.mark.unit
    def test_to_dict_with_parameters(self):
        component = HuggingFaceLocalGenerator(
            model_name_or_path="gpt2",
            task="text-generation",
            device="cuda:0",
            token="test-token",
            generation_kwargs={"max_new_tokens": 100},
            stop_words=["coca", "cola"],
        )
        data = component.to_dict()

        assert data == {
            "type": "haystack.preview.components.generators.hugging_face_local.HuggingFaceLocalGenerator",
            "init_parameters": {
                "pipeline_kwargs": {
                    "model": "gpt2",
                    "task": "text-generation",
                    "token": None,  # we don't want serialize valid tokens
                    "device": "cuda:0",
                },
                "generation_kwargs": {"max_new_tokens": 100, "return_full_text": False},
                "stop_words": ["coca", "cola"],
            },
        }

    @pytest.mark.unit
    @patch("haystack.preview.components.generators.hugging_face_local.pipeline")
    def test_warm_up(self, pipeline_mock):
        generator = HuggingFaceLocalGenerator(
            model_name_or_path="google/flan-t5-base", task="text2text-generation", token="test-token"
        )
        pipeline_mock.assert_not_called()

        generator.warm_up()

        pipeline_mock.assert_called_once_with(
            model="google/flan-t5-base", task="text2text-generation", token="test-token"
        )

    @pytest.mark.unit
    @patch("haystack.preview.components.generators.hugging_face_local.pipeline")
    def test_warm_up_doesn_reload(self, pipeline_mock):
        generator = HuggingFaceLocalGenerator(
            model_name_or_path="google/flan-t5-base", task="text2text-generation", token="test-token"
        )

        pipeline_mock.assert_not_called()

        generator.warm_up()
        generator.warm_up()

        pipeline_mock.assert_called_once()

    @pytest.mark.unit
    def test_run(self):
        generator = HuggingFaceLocalGenerator(
            model_name_or_path="google/flan-t5-base",
            task="text2text-generation",
            generation_kwargs={"max_new_tokens": 100},
        )

        # create the pipeline object (simulating the warm_up)
        generator.pipeline = Mock(return_value=[{"generated_text": "Rome"}])

        results = generator.run(prompt="What's the capital of Italy?")

        generator.pipeline.assert_called_once_with(
            "What's the capital of Italy?", max_new_tokens=100, stopping_criteria=None
        )
        assert results == {"replies": ["Rome"]}

    @pytest.mark.unit
    @patch("haystack.preview.components.generators.hugging_face_local.pipeline")
    def test_run_empty_prompt(self, pipeline_mock):
        generator = HuggingFaceLocalGenerator(
            model_name_or_path="google/flan-t5-base",
            task="text2text-generation",
            generation_kwargs={"max_new_tokens": 100},
        )

        generator.warm_up()

        results = generator.run(prompt="")

        assert results == {"replies": []}

    @pytest.mark.unit
    def test_run_with_generation_kwargs(self):
        generator = HuggingFaceLocalGenerator(
            model_name_or_path="google/flan-t5-base",
            task="text2text-generation",
            generation_kwargs={"max_new_tokens": 100},
        )

        # create the pipeline object (simulating the warm_up)
        generator.pipeline = Mock(return_value=[{"generated_text": "Rome"}])

        generator.run(prompt="irrelevant", generation_kwargs={"max_new_tokens": 200, "temperature": 0.5})

        generator.pipeline.assert_called_once_with(
            "irrelevant", max_new_tokens=200, temperature=0.5, stopping_criteria=None
        )

    @pytest.mark.unit
    def test_run_fails_without_warm_up(self):
        generator = HuggingFaceLocalGenerator(
            model_name_or_path="google/flan-t5-base",
            task="text2text-generation",
            generation_kwargs={"max_new_tokens": 100},
        )

        with pytest.raises(RuntimeError, match="The generation model has not been loaded."):
            generator.run(prompt="irrelevant")

    @pytest.mark.unit
    def test_stop_words_criteria(self):
        """
        Test that StopWordsCriteria will check stop word tokens in a continuous and sequential order
        """
        # input ids for "unambiguously"
        stop_words_id = torch.tensor([[73, 24621, 11937]])

        # input ids for "This is ambiguously, but is unrelated."
        input_ids1 = torch.tensor([[100, 19, 24621, 11937, 6, 68, 19, 73, 3897, 5]])
        # input ids for "This is unambiguously"
        input_ids2 = torch.tensor([[100, 19, 73, 24621, 11937]])

        # We used to implement stop words algorithm using the torch.isin function like this:
        # `all(torch.isin(stop_words_id, input_ids1)[0])`
        # However, this algorithm is not correct as it will return True for presence of "unambiguously" in input_ids1
        # and True for presence of "unambiguously" in input_ids2. This is because the algorithm will check
        # if the stop word tokens are present in the input_ids, but it does not check if the stop word tokens are
        # present in a continuous/sequential order.

        # In "This is ambiguously, but is unrelated." sentence the "un" token comes from "unrelated" and the
        # "ambiguously" token comes from "ambiguously". The algorithm will return True for presence of
        # "unambiguously" in input_ids1 which is not correct.

        stop_words_criteria = StopWordsCriteria(tokenizer=Mock(), stop_words=["mock data"])
        # because we are mocking the tokenizer, we need to set the stop words manually
        stop_words_criteria.stop_ids = stop_words_id

        # this is the correct algorithm to check if the stop word tokens are present in a continuous and sequential order
        # For the input_ids1, the stop word tokens are present BUT not in a continuous order
        present_and_continuous = stop_words_criteria(input_ids1, scores=None)
        assert not present_and_continuous

        # For the input_ids2, the stop word tokens are both present and in a continuous order
        present_and_continuous = stop_words_criteria(input_ids2, scores=None)
        assert present_and_continuous

    @pytest.mark.unit
    @patch("haystack.preview.components.generators.hugging_face_local.pipeline")
    @patch("haystack.preview.components.generators.hugging_face_local.StopWordsCriteria")
    @patch("haystack.preview.components.generators.hugging_face_local.StoppingCriteriaList")
    def test_warm_up_set_stopping_criteria_list(
        self, pipeline_mock, stop_words_criteria_mock, stopping_criteria_list_mock
    ):
        """
        Test that warm_up method sets the `stopping_criteria_list` attribute
        if `stop_words` is provided
        """
        generator = HuggingFaceLocalGenerator(
            model_name_or_path="google/flan-t5-base", task="text2text-generation", stop_words=["coca", "cola"]
        )

        generator.warm_up()

        stop_words_criteria_mock.assert_called_once()
        stopping_criteria_list_mock.assert_called_once()

        assert hasattr(generator, "stopping_criteria_list")

    @pytest.mark.unit
    def test_run_stop_words_removal(self):
        """
        Test that stop words are removed from the generated text
        (does not test stopping text generation)
        """
        generator = HuggingFaceLocalGenerator(
            model_name_or_path="google/flan-t5-base", task="text2text-generation", stop_words=["world"]
        )

        # create the pipeline object (simulating the warm_up)
        generator.pipeline = Mock(return_value=[{"generated_text": "Hello world"}])

        results = generator.run(prompt="irrelevant")

        assert results == {"replies": ["Hello"]}
