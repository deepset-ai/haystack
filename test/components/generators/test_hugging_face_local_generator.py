# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=too-many-public-methods
from unittest.mock import Mock, patch

import pytest
import torch
from transformers import PreTrainedTokenizerFast

from haystack.components.generators.hugging_face_local import HuggingFaceLocalGenerator, StopWordsCriteria
from haystack.utils import ComponentDevice
from haystack.utils.auth import Secret
from haystack.utils.hf import HFTokenStreamingHandler


class TestHuggingFaceLocalGenerator:
    @patch("haystack.utils.hf.model_info")
    def test_init_default(self, model_info_mock, monkeypatch):
        monkeypatch.delenv("HF_API_TOKEN", raising=False)
        monkeypatch.delenv("HF_TOKEN", raising=False)
        model_info_mock.return_value.pipeline_tag = "text2text-generation"
        generator = HuggingFaceLocalGenerator()

        assert generator.huggingface_pipeline_kwargs == {
            "model": "google/flan-t5-base",
            "task": "text2text-generation",
            "token": None,
            "device": ComponentDevice.resolve_device(None).to_hf(),
        }
        assert generator.generation_kwargs == {"max_new_tokens": 512}
        assert generator.pipeline is None

    def test_init_custom_token(self):
        generator = HuggingFaceLocalGenerator(
            model="google/flan-t5-base", task="text2text-generation", token=Secret.from_token("fake-api-token")
        )

        assert generator.huggingface_pipeline_kwargs == {
            "model": "google/flan-t5-base",
            "task": "text2text-generation",
            "token": "fake-api-token",
            "device": ComponentDevice.resolve_device(None).to_hf(),
        }

    def test_init_custom_device(self):
        generator = HuggingFaceLocalGenerator(
            model="google/flan-t5-base",
            task="text2text-generation",
            device=ComponentDevice.from_str("cuda:0"),
            token=Secret.from_token("fake-api-token"),
        )

        assert generator.huggingface_pipeline_kwargs == {
            "model": "google/flan-t5-base",
            "task": "text2text-generation",
            "token": "fake-api-token",
            "device": "cuda:0",
        }

    def test_init_task_parameter(self):
        generator = HuggingFaceLocalGenerator(task="text2text-generation", token=None)

        assert generator.huggingface_pipeline_kwargs == {
            "model": "google/flan-t5-base",
            "task": "text2text-generation",
            "token": None,
            "device": ComponentDevice.resolve_device(None).to_hf(),
        }

    def test_init_task_in_huggingface_pipeline_kwargs(self):
        generator = HuggingFaceLocalGenerator(huggingface_pipeline_kwargs={"task": "text2text-generation"}, token=None)

        assert generator.huggingface_pipeline_kwargs == {
            "model": "google/flan-t5-base",
            "task": "text2text-generation",
            "token": None,
            "device": ComponentDevice.resolve_device(None).to_hf(),
        }

    @patch("haystack.utils.hf.model_info")
    def test_init_task_inferred_from_model_name(self, model_info_mock):
        model_info_mock.return_value.pipeline_tag = "text2text-generation"
        generator = HuggingFaceLocalGenerator(model="google/flan-t5-base", token=None)

        assert generator.huggingface_pipeline_kwargs == {
            "model": "google/flan-t5-base",
            "task": "text2text-generation",
            "token": None,
            "device": ComponentDevice.resolve_device(None).to_hf(),
        }

    def test_init_invalid_task(self):
        with pytest.raises(ValueError, match="is not supported."):
            HuggingFaceLocalGenerator(task="text-classification")

    def test_init_huggingface_pipeline_kwargs_override_other_parameters(self):
        """
        huggingface_pipeline_kwargs represent the main configuration of this component.
        If they are provided, they should override other init parameters.
        """

        huggingface_pipeline_kwargs = {
            "model": "gpt2",
            "task": "text-generation",
            "device": "cuda:0",
            "token": "another-test-token",
        }

        generator = HuggingFaceLocalGenerator(
            model="google/flan-t5-base",
            task="text2text-generation",
            device=ComponentDevice.from_str("cpu"),
            token=None,
            huggingface_pipeline_kwargs=huggingface_pipeline_kwargs,
        )

        assert generator.huggingface_pipeline_kwargs == huggingface_pipeline_kwargs

    def test_init_generation_kwargs(self):
        generator = HuggingFaceLocalGenerator(task="text2text-generation", generation_kwargs={"max_new_tokens": 100})

        assert generator.generation_kwargs == {"max_new_tokens": 100}

    def test_init_set_return_full_text(self):
        """
        if not specified, return_full_text is set to False for text-generation task
        (only generated text is returned, excluding prompt)
        """
        generator = HuggingFaceLocalGenerator(task="text-generation")

        assert generator.generation_kwargs == {"max_new_tokens": 512, "return_full_text": False}

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

    @patch("haystack.utils.hf.model_info")
    def test_to_dict_default(self, model_info_mock):
        model_info_mock.return_value.pipeline_tag = "text2text-generation"

        component = HuggingFaceLocalGenerator()
        data = component.to_dict()

        assert data == {
            "type": "haystack.components.generators.hugging_face_local.HuggingFaceLocalGenerator",
            "init_parameters": {
                "token": {"env_vars": ["HF_API_TOKEN", "HF_TOKEN"], "strict": False, "type": "env_var"},
                "huggingface_pipeline_kwargs": {
                    "model": "google/flan-t5-base",
                    "task": "text2text-generation",
                    "device": ComponentDevice.resolve_device(None).to_hf(),
                },
                "generation_kwargs": {"max_new_tokens": 512},
                "streaming_callback": None,
                "stop_words": None,
            },
        }

    def test_to_dict_with_parameters(self):
        component = HuggingFaceLocalGenerator(
            model="gpt2",
            task="text-generation",
            device=ComponentDevice.from_str("cuda:0"),
            token=Secret.from_env_var("ENV_VAR", strict=False),
            generation_kwargs={"max_new_tokens": 100},
            stop_words=["coca", "cola"],
            huggingface_pipeline_kwargs={
                "model_kwargs": {
                    "load_in_4bit": True,
                    "bnb_4bit_use_double_quant": True,
                    "bnb_4bit_quant_type": "nf4",
                    "bnb_4bit_compute_dtype": torch.bfloat16,
                }
            },
        )
        data = component.to_dict()

        assert data == {
            "type": "haystack.components.generators.hugging_face_local.HuggingFaceLocalGenerator",
            "init_parameters": {
                "token": {"env_vars": ["ENV_VAR"], "strict": False, "type": "env_var"},
                "huggingface_pipeline_kwargs": {
                    "model": "gpt2",
                    "task": "text-generation",
                    "device": "cuda:0",
                    "model_kwargs": {
                        "load_in_4bit": True,
                        "bnb_4bit_use_double_quant": True,
                        "bnb_4bit_quant_type": "nf4",
                        # dtype is correctly serialized
                        "bnb_4bit_compute_dtype": "torch.bfloat16",
                    },
                },
                "generation_kwargs": {"max_new_tokens": 100, "return_full_text": False},
                "streaming_callback": None,
                "stop_words": ["coca", "cola"],
            },
        }

    def test_to_dict_with_quantization_config(self):
        component = HuggingFaceLocalGenerator(
            model="gpt2",
            task="text-generation",
            device=ComponentDevice.from_str("cuda:0"),
            token=None,
            generation_kwargs={"max_new_tokens": 100},
            stop_words=["coca", "cola"],
            huggingface_pipeline_kwargs={
                "model_kwargs": {
                    "quantization_config": {
                        "load_in_4bit": True,
                        "bnb_4bit_use_double_quant": True,
                        "bnb_4bit_quant_type": "nf4",
                        "bnb_4bit_compute_dtype": torch.bfloat16,
                    }
                }
            },
        )
        data = component.to_dict()

        assert data == {
            "type": "haystack.components.generators.hugging_face_local.HuggingFaceLocalGenerator",
            "init_parameters": {
                "token": None,
                "huggingface_pipeline_kwargs": {
                    "model": "gpt2",
                    "task": "text-generation",
                    "device": "cuda:0",
                    "model_kwargs": {
                        "quantization_config": {
                            "load_in_4bit": True,
                            "bnb_4bit_use_double_quant": True,
                            "bnb_4bit_quant_type": "nf4",
                            # dtype is correctly serialized
                            "bnb_4bit_compute_dtype": "torch.bfloat16",
                        }
                    },
                },
                "generation_kwargs": {"max_new_tokens": 100, "return_full_text": False},
                "streaming_callback": None,
                "stop_words": ["coca", "cola"],
            },
        }

    def test_from_dict(self):
        data = {
            "type": "haystack.components.generators.hugging_face_local.HuggingFaceLocalGenerator",
            "init_parameters": {
                "token": None,
                "huggingface_pipeline_kwargs": {
                    "model": "gpt2",
                    "task": "text-generation",
                    "device": "cuda:0",
                    "model_kwargs": {
                        "load_in_4bit": True,
                        "bnb_4bit_use_double_quant": True,
                        "bnb_4bit_quant_type": "nf4",
                        # dtype is correctly serialized
                        "bnb_4bit_compute_dtype": "torch.bfloat16",
                    },
                },
                "generation_kwargs": {"max_new_tokens": 100, "return_full_text": False},
                "stop_words": ["coca", "cola"],
            },
        }

        component = HuggingFaceLocalGenerator.from_dict(data)

        assert component.huggingface_pipeline_kwargs == {
            "model": "gpt2",
            "task": "text-generation",
            "device": "cuda:0",
            "token": None,
            "model_kwargs": {
                "load_in_4bit": True,
                "bnb_4bit_use_double_quant": True,
                "bnb_4bit_quant_type": "nf4",
                # dtype is correctly deserialized
                "bnb_4bit_compute_dtype": torch.bfloat16,
            },
        }
        assert component.generation_kwargs == {"max_new_tokens": 100, "return_full_text": False}
        assert component.stop_words == ["coca", "cola"]

    @patch("haystack.components.generators.hugging_face_local.pipeline")
    def test_warm_up(self, pipeline_mock):
        generator = HuggingFaceLocalGenerator(model="google/flan-t5-base", task="text2text-generation", token=None)
        pipeline_mock.assert_not_called()

        generator.warm_up()

        pipeline_mock.assert_called_once_with(
            model="google/flan-t5-base",
            task="text2text-generation",
            token=None,
            device=ComponentDevice.resolve_device(None).to_hf(),
        )

    @patch("haystack.components.generators.hugging_face_local.pipeline")
    def test_warm_up_doesnt_reload(self, pipeline_mock):
        generator = HuggingFaceLocalGenerator(
            model="google/flan-t5-base", task="text2text-generation", token=Secret.from_token("fake-api-token")
        )

        pipeline_mock.assert_not_called()

        generator.warm_up()
        generator.warm_up()

        pipeline_mock.assert_called_once()

    def test_run(self):
        generator = HuggingFaceLocalGenerator(
            model="google/flan-t5-base", task="text2text-generation", generation_kwargs={"max_new_tokens": 100}
        )

        # create the pipeline object (simulating the warm_up)
        generator.pipeline = Mock(return_value=[{"generated_text": "Rome"}])

        results = generator.run(prompt="What's the capital of Italy?")

        generator.pipeline.assert_called_once_with(
            "What's the capital of Italy?", max_new_tokens=100, stopping_criteria=None
        )
        assert results == {"replies": ["Rome"]}

    @patch("haystack.components.generators.hugging_face_local.pipeline")
    def test_run_empty_prompt(self, pipeline_mock):
        generator = HuggingFaceLocalGenerator(
            model="google/flan-t5-base", task="text2text-generation", generation_kwargs={"max_new_tokens": 100}
        )

        generator.warm_up()

        results = generator.run(prompt="")

        assert results == {"replies": []}

    def test_run_with_generation_kwargs(self):
        generator = HuggingFaceLocalGenerator(
            model="google/flan-t5-base", task="text2text-generation", generation_kwargs={"max_new_tokens": 100}
        )

        # create the pipeline object (simulating the warm_up)
        generator.pipeline = Mock(return_value=[{"generated_text": "Rome"}])

        generator.run(prompt="irrelevant", generation_kwargs={"max_new_tokens": 200, "temperature": 0.5})

        generator.pipeline.assert_called_once_with(
            "irrelevant", max_new_tokens=200, temperature=0.5, stopping_criteria=None
        )

    def test_run_with_streaming(self):
        def streaming_callback_handler(x):
            return x

        generator = HuggingFaceLocalGenerator(
            model="google/flan-t5-base", task="text2text-generation", streaming_callback=streaming_callback_handler
        )

        # create the pipeline object (simulating the warm_up)
        generator.pipeline = Mock(return_value=[{"generated_text": "Rome"}])

        generator.run(prompt="irrelevant")

        # when we use streaming, the pipeline should be called with the `streamer` argument being an instance of
        # ouf our adapter class HFTokenStreamingHandler
        assert isinstance(generator.pipeline.call_args.kwargs["streamer"], HFTokenStreamingHandler)
        streamer = generator.pipeline.call_args.kwargs["streamer"]

        # check that the streaming callback is set
        assert streamer.token_handler == streaming_callback_handler
        # the tokenizer should be set, here it is a mock
        assert streamer.tokenizer

    def test_run_fails_without_warm_up(self):
        generator = HuggingFaceLocalGenerator(
            model="google/flan-t5-base", task="text2text-generation", generation_kwargs={"max_new_tokens": 100}
        )

        with pytest.raises(RuntimeError, match="The component HuggingFaceLocalGenerator was not warmed up"):
            generator.run(prompt="irrelevant")

    def test_stop_words_criteria_with_a_mocked_tokenizer(self):
        """
        Test that StopWordsCriteria will caught stop word tokens in a continuous and sequential order in the input_ids
        """
        stop_words_id = torch.LongTensor([[73, 24621, 11937]])  # "unambiguously"
        # "This is ambiguously, but is unrelated."
        input_ids_one = torch.LongTensor([[100, 19, 24621, 11937, 6, 68, 19, 73, 3897, 5]])
        input_ids_two = torch.LongTensor([[100, 19, 73, 24621, 11937]])  # "This is unambiguously"
        stop_words_criteria = StopWordsCriteria(tokenizer=Mock(spec=PreTrainedTokenizerFast), stop_words=["mock data"])
        stop_words_criteria.stop_ids = stop_words_id
        assert not stop_words_criteria(input_ids_one, scores=None)
        assert stop_words_criteria(input_ids_two, scores=None)

    @patch("haystack.components.generators.hugging_face_local.pipeline")
    @patch("haystack.components.generators.hugging_face_local.StopWordsCriteria")
    @patch("haystack.components.generators.hugging_face_local.StoppingCriteriaList")
    def test_warm_up_set_stopping_criteria_list(
        self, pipeline_mock, stop_words_criteria_mock, stopping_criteria_list_mock
    ):
        """
        Test that warm_up method sets the `stopping_criteria_list` attribute if `stop_words` is provided
        """
        generator = HuggingFaceLocalGenerator(
            model="google/flan-t5-small", task="text2text-generation", stop_words=["coca", "cola"]
        )
        generator.warm_up()
        stop_words_criteria_mock.assert_called_once()
        stopping_criteria_list_mock.assert_called_once()
        assert hasattr(generator, "stopping_criteria_list")

    def test_run_stop_words_removal(self):
        """Test that stop words are removed from the generated text (does not test stopping text generation)"""
        generator = HuggingFaceLocalGenerator(
            model="google/flan-t5-small", task="text2text-generation", stop_words=["world"]
        )
        generator.pipeline = Mock(return_value=[{"generated_text": "Hello world"}])
        generator.stopping_criteria_list = Mock()
        results = generator.run(prompt="irrelevant")
        assert results == {"replies": ["Hello"]}

    @pytest.mark.integration
    def test_stop_words_criteria_using_hf_tokenizer(self):
        """
        Test that StopWordsCriteria catches stop word tokens in a continuous and sequential order in the input_ids
        using a real Huggingface tokenizer.
        """
        from transformers import AutoTokenizer

        model_name = "google/flan-t5-small"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        criteria = StopWordsCriteria(tokenizer=tokenizer, stop_words=["unambiguously"])

        text_one = "This is ambiguously, but is unrelated."
        generated_text_ids = tokenizer.encode(text_one, add_special_tokens=False, return_tensors="pt")
        assert criteria(generated_text_ids, scores=None) is False

        text_two = "This is unambiguously"
        generated_text_ids = tokenizer.encode(text_two, add_special_tokens=False, return_tensors="pt")
        assert criteria(generated_text_ids, scores=None) is True

    @pytest.mark.integration
    def test_hf_pipeline_runs_with_our_criteria(self):
        """Test that creating our own StopWordsCriteria and passing it to a Huggingface pipeline works."""
        generator = HuggingFaceLocalGenerator(
            model="google/flan-t5-small", task="text2text-generation", stop_words=["unambiguously"]
        )
        generator.warm_up()
        results = generator.run(prompt="something that triggers something")
        assert results["replies"] != []
        assert generator.stopping_criteria_list is not None
