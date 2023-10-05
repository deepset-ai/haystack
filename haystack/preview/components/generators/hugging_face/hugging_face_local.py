from typing import Optional, List, Callable, Dict, Any, Union, Literal

import sys
import logging
from collections import defaultdict
from dataclasses import dataclass, asdict

from haystack.preview import component, default_from_dict, default_to_dict, DeserializationError
from haystack.preview.lazy_imports import LazyImport

with LazyImport(message="Run 'pip install torch transformers'") as torch_and_transformers_import:
    import torch
    from transformers import (
        pipeline,
        StoppingCriteriaList,
        StoppingCriteria,
        GenerationConfig,
        PreTrainedTokenizer,
        PreTrainedTokenizerFast,
        PreTrainedModel,
        Pipeline,
        AutoTokenizer,
        AutoConfig,
        TOKENIZER_MAPPING,
    from huggingface_hub import model_info
    )


logger = logging.getLogger(__name__)

@component
class HuggingFaceLocalGenerator:

    def __init__(
        self,
        api_key: str,
        model_name_or_path: str = "google/flan-t5-base",
        use_auth_token: Optional[Union[str, bool]] = None,
        device: str = "cpu",
        task: Optional[Literal["text-generation", "text2text-generation"]] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        pipeline_kwargs: Optional[Dict[str, Any]] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        :param model_name_or_path: Directory of a saved model or the name of a public model e.g. google/t5-small
        :param use_auth_token: Optional API key if the model is hosted on huggingface.co
        :param device: "cpu" or "cuda"
        :param model_kwargs: Optional kwargs that will be passed to the model.
            https://huggingface.co/docs/transformers/en/main_classes/model#transformers.PreTrainedModel.from_pretrained
        :param pipeline_kwargs: Optional kwargs that will be passed to the pipeline.
            https://huggingface.co/docs/transformers/en/main_classes/pipelines#transformers.pipeline.task
        :param generation_kwargs: Optional kwargs that will be passed to the generation method
            https://huggingface.co/docs/transformers/main/en/generation_strategies#customize-text-generation
        """
        torch_and_transformers_import.check()

        self.model_name_or_path = model_name_or_path
        self.use_auth_token = use_auth_token
        self.device = device
        self.model_kwargs = model_kwargs or {}
        self.pipeline_kwargs = pipeline_kwargs or {}
        self.generation_kwargs = generation_kwargs or {}

        # task identification and validation
        if task is None and "task" in self.pipeline_kwargs:
            if 'task' in self.pipeline_kwargs:
                task = self.pipeline_kwargs['task']
            else:
                task = model_info(model_name_or_path, token=use_auth_token).pipeline_tag
            
        if task not in ["text-generation", "text2text-generation"]:
                raise ValueError(
                    f"Task name {task} is not supported. "
                    f"We only support text2text-generation and text-generation tasks."
                )
        self.task = task
        self.pipeline_kwargs["task"] = task

        # self.model = None
        # self.tokenizer = None
        self.pipeline = None

    def warm_up(self):
        self.pipeline = pipeline(**self.pipeline_kwargs, model_kwargs=self.model_kwargs)

    