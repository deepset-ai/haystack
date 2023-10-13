from unittest.mock import patch, Mock

import pytest

from haystack.preview.components.generators.hugging_face.hugging_face_local import HuggingFaceLocalGenerator


class TestHuggingFaceLocalGenerator:
    @pytest.mark.unit
    @patch("haystack.preview.components.generators.hugging_face.hugging_face_local.model_info")
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
    @patch("haystack.preview.components.generators.hugging_face.hugging_face_local.model_info")
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
    @patch("haystack.preview.components.generators.hugging_face.hugging_face_local.model_info")
    def test_to_dict_default(self, model_info_mock):
        model_info_mock.return_value.pipeline_tag = "text2text-generation"

        component = HuggingFaceLocalGenerator()
        data = component.to_dict()

        assert data == {
            "type": "HuggingFaceLocalGenerator",
            "init_parameters": {
                "pipeline_kwargs": {"model": "google/flan-t5-base", "task": "text2text-generation", "token": None},
                "generation_kwargs": {},
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
        )
        data = component.to_dict()

        assert data == {
            "type": "HuggingFaceLocalGenerator",
            "init_parameters": {
                "pipeline_kwargs": {
                    "model": "gpt2",
                    "task": "text-generation",
                    "token": None,  # we don't want serialize valid tokens
                    "device": "cuda:0",
                },
                "generation_kwargs": {"max_new_tokens": 100, "return_full_text": False},
            },
        }

    @pytest.mark.unit
    def test_from_dict(self):
        data = {
            "type": "HuggingFaceLocalGenerator",
            "init_parameters": {
                "pipeline_kwargs": {
                    "model": "gpt2",
                    "task": "text-generation",
                    "token": "test-token",
                    "device": "cuda:0",
                },
                "generation_kwargs": {"max_new_tokens": 100, "return_full_text": False},
            },
        }

        component = HuggingFaceLocalGenerator.from_dict(data)

        assert component.pipeline_kwargs == {
            "model": "gpt2",
            "task": "text-generation",
            "token": "test-token",
            "device": "cuda:0",
        }
        assert component.generation_kwargs == {"max_new_tokens": 100, "return_full_text": False}

    @pytest.mark.unit
    @patch("haystack.preview.components.generators.hugging_face.hugging_face_local.pipeline")
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
    @patch("haystack.preview.components.generators.hugging_face.hugging_face_local.pipeline")
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

        generator.pipeline.assert_called_once_with("What's the capital of Italy?", max_new_tokens=100)
        assert results == {"replies": ["Rome"]}

    @pytest.mark.unit
    @patch("haystack.preview.components.generators.hugging_face.hugging_face_local.pipeline")
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
    def test_run_fails_without_warm_up(self):
        generator = HuggingFaceLocalGenerator(
            model_name_or_path="google/flan-t5-base",
            task="text2text-generation",
            generation_kwargs={"max_new_tokens": 100},
        )

        with pytest.raises(RuntimeError, match="The generation model has not been loaded."):
            generator.run(prompt="irrelevant")
