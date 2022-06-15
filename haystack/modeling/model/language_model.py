# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors,  The HuggingFace Inc. Team and deepset Team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" 
Acknowledgements: Many of the modeling parts here come from the great transformers repository: https://github.com/huggingface/transformers.
Thanks for the great work! 
"""
from typing import Optional, Dict, Any, Union

import re
import json
import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from functools import wraps
import numpy as np
import torch
from torch import nn
import transformers
from transformers import PretrainedConfig, PreTrainedModel
from transformers import AutoModel, AutoConfig
from transformers.modeling_utils import SequenceSummary

from haystack.modeling.model._mappings import (
    HF_PARAMETERS_BY_MODEL, 
    HF_MODEL_TYPES, 
    HF_MODEL_STRINGS_HINTS, 
    KNOWN_LANGUAGE_SPECIFIC_MODELS, 
    KNOWN_LANGUAGES
)


logger = logging.getLogger(__name__)


#: Names of the attributes in various model configs which refer to the number of dimensions in the output vectors
OUTPUT_DIM_NAMES = ["dim", "hidden_size", "d_model"]



def silence_transformers_logs(from_pretrained_func):
    """
    A wrapper that raises the log level of Transformers to
    ERROR to hide some unnecessary warnings.
    """

    @wraps(from_pretrained_func)
    def quiet_from_pretrained_func(cls, *args, **kwargs):

        # Raise the log level of Transformers
        t_logger = logging.getLogger("transformers")
        original_log_level = t_logger.level
        t_logger.setLevel(logging.ERROR)

        result = from_pretrained_func(cls, *args, **kwargs)

        # Restore the log level
        t_logger.setLevel(original_log_level)

        return result

    return quiet_from_pretrained_func



# TODO analyse if LMs can be completely used through HF transformers
class LanguageModel(nn.Module, ABC):
    """
    The parent class for any kind of model that can embed language into a semantic vector space. 
    These models read in tokenized sentences and return vectors that capture the meaning of sentences or of tokens.
    """

    subclasses: dict = {}

    def __init_subclass__(cls, **kwargs):
        """
        This automatically keeps track of all available subclasses.
        Enables generic load() or all specific LanguageModel implementation.
        """
        super().__init_subclass__(**kwargs)
        cls.subclasses[cls.__name__] = cls

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(
        self, 
        input_ids: torch.Tensor, 
        segment_ids: torch.Tensor, 
        padding_mask: torch.Tensor, 
        output_hidden_states: Optional[bool] = None, 
        output_attentions: Optional[bool] = None
    ):
        raise NotImplementedError

    @staticmethod
    def load(
        pretrained_model_name_or_path: Union[Path, str], 
        language: str = None, 
        n_added_tokens: int = 0, 
        language_model_class: Optional[str] = None, 
        auth_token: Optional[str] = None, 
        revision: Optional[str] = None,
        **kwargs
    ):
        """
        Load a pretrained language model by doing one of the following:

        1. Specifying its name and downloading the model.
        2. Pointing to the directory the model is saved in.

        See all supported model variations at: https://huggingface.co/models.

        The appropriate language model class is inferred automatically from model configuration
        or can be manually supplied using `language_model_class`.

        :param pretrained_model_name_or_path: The path of the saved pretrained model or its name.
        :param revision: The version of the model to use from the Hugging Face model hub. This can be a tag name, a branch name, or a commit hash.
        :param language_model_class: (Optional) Name of the language model class to load (for example `Bert`). Unused if the model is local.
        """
        logger.info("LOADING MODEL")
        logger.info("=============")

        config_file = Path(pretrained_model_name_or_path) / "language_model_config.json"

        if os.path.exists(config_file):
            # it's a local directory in Haystack format
            logger.info(f"Model found locally at {pretrained_model_name_or_path}")
            config = json.load(open(config_file))
            language_model_class = config["name"]
        else:
            # It's from the model hub
            logger.info(f"Could not find {pretrained_model_name_or_path} locally.")
            logger.info(f"Looking on Transformers Model Hub (in local cache and online)...")
            if language_model_class is None:
                language_model_class = LanguageModel.get_language_model_class(
                    pretrained_model_name_or_path, auth_token=auth_token, **kwargs
                )
            if  not language_model_class:
                raise Exception(
                    f"Model not found for {pretrained_model_name_or_path}. Either supply the local path for a saved "
                    f"model or one of bert/roberta/xlnet/albert/distilbert models that can be downloaded from remote. "
                    f"Ensure that the model class name can be inferred from the directory name when loading a "
                    f"Transformers' model."
                )
        language_model = LanguageModel.subclasses[language_model_class](
            pretrained_model_name_or_path, 
            auth_token=auth_token, 
            n_added_tokens=n_added_tokens, 
            language=language,
            revision=revision,
            **kwargs
        )
        logger.info(f"Loaded {pretrained_model_name_or_path}")
        return language_model

    @staticmethod
    def get_language_model_class(model_name_or_path, auth_token: Optional[str] = None, revision: Optional[str] = None, **kwargs):
        """
        Given a model name, try to use AutoConfig to understand which model type it is.
        In case it's not successful, tries to infer the type from the name of the model.
        """
        # it's transformers format (either from model hub or local)
        model_name_or_path = str(model_name_or_path)
        config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path=model_name_or_path, 
            use_auth_token=auth_token or False, 
            revision=revision, 
            **kwargs
        )
        language_model_class = HF_MODEL_TYPES.get(config.model_type, None)

        # Handle special cases
        if not language_model_class:

            # DPR
            if config.model_type == "dpr":
                if config.architectures[0] == "DPRQuestionEncoder":
                    language_model_class = "DPRQuestionEncoder"
                elif config.architectures[0] == "DPRContextEncoder":
                    language_model_class = "DPRContextEncoder"
                elif config.archictectures[0] == "DPRReader":
                    raise NotImplementedError("DPRReader models are currently not supported.")

            # Infer from model name if still not found
            else:
                logger.warning("Could not infer the class from config. Trying to infer class from model name.")
                for regex, class_ in HF_MODEL_STRINGS_HINTS.items():
                    if re.match(regex, model_name_or_path):
                        language_model_class = class_
                        break

        # Notes for some models
        if language_model_class == "Roberta" and "mlm" in model_name_or_path.lower():
            raise NotImplementedError("MLM part of codebert is currently not supported in Haystack.")

        return language_model_class

    def get_output_dims(self):
        config = self.model.config
        for odn in OUTPUT_DIM_NAMES:
            if odn in dir(config):
                return getattr(config, odn)
        raise Exception("Could not infer the output dimensions of the language model")

    def save_config(self, save_dir):
        save_filename = Path(save_dir) / "language_model_config.json"
        with open(save_filename, "w") as file:
            setattr(self.model.config, "name", self.__class__.__name__)
            setattr(self.model.config, "language", self.language)
            # For DPR models, transformers overwrites the model_type with the one set in DPRConfig
            # Therefore, we copy the model_type from the model config to DPRConfig
            if self.__class__.__name__ == "DPRQuestionEncoder" or self.__class__.__name__ == "DPRContextEncoder":
                setattr(transformers.DPRConfig, "model_type", self.model.config.model_type)
            string = self.model.config.to_json_string()
            file.write(string)

    def save(self, save_dir: Union[str, Path], state_dict: Dict[Any, Any] = None):
        """
        Save the model `state_dict` and its configuration file so that it can be loaded again.

        :param save_dir: The directory in which the model should be saved.
        :param state_dict: A dictionary containing the whole state of the module, including names of layers. By default, the unchanged state dictionary of the module is used.
        """
        # Save Weights
        save_name = Path(save_dir) / "language_model.bin"
        model_to_save = self.model.module if hasattr(self.model, "module") else self.model  # Only save the model itself

        if not state_dict:
            state_dict = model_to_save.state_dict()
        torch.save(state_dict, save_name)
        self.save_config(save_dir)

    @staticmethod
    def _infer_language_from_name(name: str) -> str:
        language = "english"
        languages = [lang for lang in KNOWN_LANGUAGES if lang in name]
        if len(languages) == 0:
            languages = [lang for model, lang in KNOWN_LANGUAGE_SPECIFIC_MODELS if model in name]
            if len(languages) > 0:
                language = languages[0]
        else:
            language = languages[0]
        logger.info(f"Automatically detected language from model name: {language}")
        return language

    def formatted_preds(self, logits, samples, ignore_first_token=True, padding_mask=None, input_ids=None, **kwargs):
        """
        Extracting vectors from a language model (for example, for extracting sentence embeddings).
        You can use different pooling strategies and layers by specifying them in the object attributes
        `extraction_layer` and `extraction_strategy`. You should set both these attirbutes using the Inferencer:
        Example:  Inferencer(extraction_strategy='cls_token', extraction_layer=-1)

        :param logits: Tuple of (sequence_output, pooled_output) from the language model.
                       Sequence_output: one vector per token, pooled_output: one vector for whole sequence.
        :param samples: For each item in logits, we need additional meta information to format the prediction (for example, input text).
                        This is created by the Processor and passed in here from the Inferencer.
        :param ignore_first_token: When set to `True`, includes the first token for pooling operations (for example, reduce_mean).
                                   Many models use a special token, like [CLS], that you don't want to include in your average of token embeddings.
        :param padding_mask: Mask for the padding tokens. These aren't included in the pooling operations to prevent a bias by the number of padding tokens.
        :param input_ids: IDs of the tokens in the vocabulary.
        :param kwargs: kwargs
        :return: A list of dictionaries containing predictions, for example: [{"context": "some text", "vec": [-0.01, 0.5 ...]}].
        """
        if not hasattr(self, "extraction_layer") or not hasattr(self, "extraction_strategy"):
            raise ValueError(
                "`extraction_layer` or `extraction_strategy` not specified for LM. "
                "Make sure to set both, e.g. via Inferencer(extraction_strategy='cls_token', extraction_layer=-1)`"
            )

        # unpack the tuple from LM forward pass
        sequence_output = logits[0][0]
        pooled_output = logits[0][1]

        # aggregate vectors
        if self.extraction_strategy == "pooled":
            if self.extraction_layer != -1:
                raise ValueError(
                    f"Pooled output only works for the last layer, but got extraction_layer = {self.extraction_layer}. Please set `extraction_layer=-1`.)"
                )
            vecs = pooled_output.cpu().numpy()
        elif self.extraction_strategy == "per_token":
            vecs = sequence_output.cpu().numpy()
        elif self.extraction_strategy == "reduce_mean":
            vecs = self._pool_tokens(
                sequence_output, padding_mask, self.extraction_strategy, ignore_first_token=ignore_first_token
            )
        elif self.extraction_strategy == "reduce_max":
            vecs = self._pool_tokens(
                sequence_output, padding_mask, self.extraction_strategy, ignore_first_token=ignore_first_token
            )
        elif self.extraction_strategy == "cls_token":
            vecs = sequence_output[:, 0, :].cpu().numpy()
        else:
            raise NotImplementedError

        preds = []
        for vec, sample in zip(vecs, samples):
            pred = {}
            pred["context"] = sample.clear_text["text"]
            pred["vec"] = vec
            preds.append(pred)
        return preds

    def _pool_tokens(self, sequence_output, padding_mask, strategy, ignore_first_token):
        token_vecs = sequence_output.cpu().numpy()
        # we only take the aggregated value of non-padding tokens
        padding_mask = padding_mask.cpu().numpy()
        ignore_mask_2d = padding_mask == 0
        # sometimes we want to exclude the CLS token as well from our aggregation operation
        if ignore_first_token:
            ignore_mask_2d[:, 0] = True
        ignore_mask_3d = np.zeros(token_vecs.shape, dtype=bool)
        ignore_mask_3d[:, :, :] = ignore_mask_2d[:, :, np.newaxis]
        if strategy == "reduce_max":
            pooled_vecs = np.ma.array(data=token_vecs, mask=ignore_mask_3d).max(axis=1).data
        if strategy == "reduce_mean":
            pooled_vecs = np.ma.array(data=token_vecs, mask=ignore_mask_3d).mean(axis=1).data

        return pooled_vecs


class HFLanguageModel(LanguageModel):
    """
    A model that wraps Hugging Face's implementation
    (https://github.com/huggingface/transformers) to fit the LanguageModel class.
    """

    @silence_transformers_logs
    def __init__(self, pretrained_model_name_or_path: Union[Path, str], model_type: str, language: str = None, n_added_tokens: int = 0, auth_token: Optional[str] = None, **kwargs):
        """
        Load a pretrained model by supplying one of the following:

        * The name of a remote model on s3 (for example, "bert-base-cased").
        * A local path of a model trained using transformers (for example, "some_dir/huggingface_model").
        * A local path of a model trained using Haystack (for example, "some_dir/haystack_model").

        :param pretrained_model_name_or_path: The path of the saved pretrained model or the name of the model.
        """
        super().__init__()
        self.name = kwargs["haystack_lm_name"] if "haystack_lm_name" in kwargs else pretrained_model_name_or_path

        class_prefix = HF_PARAMETERS_BY_MODEL.get(model_type)["prefix"]
        config_class: PretrainedConfig = getattr(transformers, class_prefix + "Config", None)
        model_class: PreTrainedModel = getattr(transformers, class_prefix + "Model", None)

        # We need to differentiate between loading model using Haystack format and Pytorch-Transformers format
        haystack_lm_config = Path(pretrained_model_name_or_path) / "language_model_config.json"
        if os.path.exists(haystack_lm_config):
            # Haystack style
            haystack_lm_model = Path(pretrained_model_name_or_path) / "language_model.bin"
            model_config = config_class.from_pretrained(haystack_lm_config)
            self.model = model_class.from_pretrained(haystack_lm_model, config=model_config, use_auth_token=auth_token or False, **kwargs)
            self.language = self.model.config.language
        else:
            # Pytorch-transformer Style
            self.model = model_class.from_pretrained(str(pretrained_model_name_or_path), use_auth_token=auth_token or False, **kwargs)
            self.language = language or self._infer_language_from_name(pretrained_model_name_or_path)
        
        # resize embeddings in case of custom vocab
        if n_added_tokens != 0:
            # TODO verify for other models than BERT
            model_emb_size = self.model.resize_token_embeddings(new_num_tokens=None).num_embeddings
            vocab_size = model_emb_size + n_added_tokens
            logger.info(
                f"Resizing embedding layer of LM from {model_emb_size} to {vocab_size} to cope with custom vocab."
            )
            self.model.resize_token_embeddings(vocab_size)
            # verify
            model_emb_size = self.model.resize_token_embeddings(new_num_tokens=None).num_embeddings
            assert vocab_size == model_emb_size

    # @classmethod
    # def from_scratch(cls, vocab_size, name="bert", language="en"):
    #     bert = cls()
    #     bert.name = name
    #     bert.language = language
    #     config = BertConfig(vocab_size=vocab_size)
    #     bert.model = BertModel(config)
    #     return bert

    def forward(
        self,
        input_ids: torch.Tensor,
        segment_ids: torch.Tensor,
        padding_mask: torch.Tensor,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None
    ):
        """
        Perform the forward pass of the model.

        :param input_ids: The IDs of each token in the input sequence. It's a tensor of shape [batch_size, max_seq_len].
        :param segment_ids: The ID of the segment. For example, in next sentence prediction, the tokens in the
           first sentence are marked with 0 and the tokens in the second sentence are marked with 1.
           It is a tensor of shape [batch_size, max_seq_len].
        :param padding_mask: A mask that assigns 1 to valid input tokens and 0 to padding tokens
           of shape [batch_size, max_seq_len].
        :param output_hidden_states: When set to `True`, outputs hidden states in addition to the embeddings.
        :param output_attentions: When set to `True`, outputs attentions in addition to the embeddings.
        :return: Embeddings for each token in the input sequence. Can also return hidden states and attentions if specified using the arguments `output_hidden_states` and `output_attentions`.
        """
        if output_hidden_states is None:
            output_hidden_states = self.model.encoder.config.output_hidden_states
        if output_attentions is None:
            output_attentions = self.model.encoder.config.output_attentions

        output_tuple = self.model(
            input_ids,
            token_type_ids=segment_ids,
            attention_mask=padding_mask,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=False,
        )
        return output_tuple

    def enable_hidden_states_output(self):
        self.model.encoder.config.output_hidden_states = True

    def disable_hidden_states_output(self):
        self.model.encoder.config.output_hidden_states = False


class HFLanguageModelWithPooler(HFLanguageModel):
    """
    A model that wraps Hugging Face's implementation
    (https://github.com/huggingface/transformers) to fit the LanguageModel class,
    with an extra pooler.

    NOTE:
    - Unlike the other BERT variants, these don't output the `pooled_output`. An additional pooler is initialized.
    """

    def __init__(self, pretrained_model_name_or_path: Union[Path, str], language: str = None, n_added_tokens: int = 0, **kwargs):
        """
        Load a pretrained model by supplying one of the following:

        * The name of a remote model on s3 (for example, "distilbert-base-german-cased")
        * A local path of a model trained using transformers (for example, "some_dir/huggingface_model")
        * A local path of a model trained using Haystack (for example, "some_dir/haystack_model")

        :param pretrained_model_name_or_path: The path of the saved pretrained model or its name.
        """
        super().__init__(pretrained_model_name_or_path, language, n_added_tokens, **kwargs)
        self.pooler = None
        config = self.model.config

        # These models do not provide a pooled_output by default. Therefore, we need to initialize an extra pooler.
        # The pooler takes the first hidden representation & feeds it to a dense layer of (hidden_dim x hidden_dim).
        # We don't want a dropout in the end of the pooler, since we do that already in the adaptive model before we
        # feed everything to the prediction head
        sequence_summary_config = HF_PARAMETERS_BY_MODEL.get(self.name)["sequence_summary_config"]
        for key, value in sequence_summary_config.items():
            setattr(config, key, value)

        self.pooler = SequenceSummary(config)
        self.pooler.apply(self.model._init_weights)

    def forward(  # type: ignore
        self,
        input_ids: torch.Tensor,
        segment_ids: torch.Tensor,
        padding_mask: torch.Tensor,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        **kwargs,
    ):
        """
        Perform the forward pass of the model.

        :param input_ids: The IDs of each token in the input sequence. It's a tensor of shape [batch_size, max_seq_len].
        :param padding_mask: A mask that assigns 1 to valid input tokens and 0 to padding tokens
           of shape [batch_size, max_seq_len].
        :param output_hidden_states: When set to `True`, outputs hidden states in addition to the embeddings.
        :param output_attentions: When set to `True`, outputs attentions in addition to the embeddings.
        :return: Embeddings for each token in the input sequence.
        """
        output_tuple = super().forward(
            input_ids=input_ids,
            segment_ids=segment_ids,
            padding_mask=padding_mask,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            **kwargs,
        )
        pooled_output = self.pooler(output_tuple[0])
        return (output_tuple[0], pooled_output) + output_tuple[1:]


class Bert(HFLanguageModel):
    """
    A BERT model that wraps Hugging Face's implementation
    (https://github.com/huggingface/transformers) to fit the LanguageModel class.
    Paper: https://arxiv.org/abs/1810.04805.
    """

    def __init__(self, pretrained_model_name_or_path: Union[Path, str], language: str = None, n_added_tokens: int = 0, **kwargs):
        super().__init__(
            pretrained_model_name_or_path=pretrained_model_name_or_path, 
            language=language, 
            n_added_tokens=n_added_tokens, 
            model_type="bert", 
            **kwargs
        )


class Albert(HFLanguageModel):
    """
    An ALBERT model that wraps the Hugging Face's implementation
    (https://github.com/huggingface/transformers) to fit the LanguageModel class.
    """

    def __init__(self, pretrained_model_name_or_path: Union[Path, str], language: str = None, n_added_tokens: int = 0, **kwargs):
        super().__init__(
            pretrained_model_name_or_path=pretrained_model_name_or_path, 
            language=language, 
            n_added_tokens=n_added_tokens, 
            model_type="albert", 
            **kwargs
        )


class Roberta(HFLanguageModel):
    """
    A roberta model that wraps the Hugging Face's implementation
    (https://github.com/huggingface/transformers) to fit the LanguageModel class.
    Paper: https://arxiv.org/abs/1907.11692
    """

    def __init__(self, pretrained_model_name_or_path: Union[Path, str], language: str = None, n_added_tokens: int = 0, **kwargs):
        super().__init__(
            pretrained_model_name_or_path=pretrained_model_name_or_path, 
            language=language, 
            n_added_tokens=n_added_tokens, 
            model_type="roberta", 
            **kwargs
        )


class XLMRoberta(HFLanguageModel):
    """
    A roberta model that wraps the Hugging Face's implementation
    (https://github.com/huggingface/transformers) to fit the LanguageModel class.
    Paper: https://arxiv.org/abs/1907.11692
    """

    def __init__(self, pretrained_model_name_or_path: Union[Path, str], language: str = None, n_added_tokens: int = 0, **kwargs):
        super().__init__(
            pretrained_model_name_or_path=pretrained_model_name_or_path, 
            language=language, 
            n_added_tokens=n_added_tokens, 
            model_type="xlm-roberta", 
            **kwargs
        )

class DistilBert(HFLanguageModelWithPooler):
    """
    A DistilBERT model that wraps Hugging Face's implementation
    (https://github.com/huggingface/transformers) to fit the LanguageModel class.

    NOTE:
    - DistilBert doesn't have `token_type_ids`, you don't need to indicate which
    token belongs to which segment. Just separate your segments with the separation
    token `tokenizer.sep_token` (or [SEP]).
    - Unlike the other BERT variants, DistilBert does not output the
    `pooled_output`. An additional pooler is initialized.
    """

    def __init__(self, pretrained_model_name_or_path: Union[Path, str], language: str = None, n_added_tokens: int = 0, **kwargs):
        super().__init__(
            pretrained_model_name_or_path=pretrained_model_name_or_path, 
            language=language, 
            n_added_tokens=n_added_tokens, 
            model_type="distilbert", 
            **kwargs
        )

class XLNet(HFLanguageModelWithPooler):
    """
    A XLNet model that wraps the Hugging Face's implementation
    (https://github.com/huggingface/transformers) to fit the LanguageModel class.
    Paper: https://arxiv.org/abs/1906.08237
    """

    def __init__(self, pretrained_model_name_or_path: Union[Path, str], language: str = None, n_added_tokens: int = 0, **kwargs):
        super().__init__(
            pretrained_model_name_or_path=pretrained_model_name_or_path, 
            language=language, 
            n_added_tokens=n_added_tokens, 
            model_type="xlnet", 
            **kwargs
        )

class Electra(HFLanguageModelWithPooler):
    """
    ELECTRA is a new pre-training approach which trains two transformer models:
    the generator and the discriminator. The generator replaces tokens in a sequence,
    and is therefore trained as a masked language model. The discriminator, which is
    the model we're interested in, tries to identify which tokens were replaced by
    the generator in the sequence.

    The ELECTRA model here wraps Hugging Face's implementation
    (https://github.com/huggingface/transformers) to fit the LanguageModel class.

    NOTE:
    - Electra does not output the `pooled_output`. An additional pooler is initialized.
    """

    def __init__(self, pretrained_model_name_or_path: Union[Path, str], language: str = None, n_added_tokens: int = 0, **kwargs):
        super().__init__(
            pretrained_model_name_or_path=pretrained_model_name_or_path, 
            language=language, 
            n_added_tokens=n_added_tokens, 
            model_type="electra", 
            **kwargs
        )

class Camembert(HFLanguageModel):
    """
    A Camembert model that wraps the Hugging Face's implementation
    (https://github.com/huggingface/transformers) to fit the LanguageModel class.
    """

    def __init__(self, pretrained_model_name_or_path: Union[Path, str], language: str = None, n_added_tokens: int = 0, **kwargs):
        super().__init__(
            pretrained_model_name_or_path=pretrained_model_name_or_path, 
            language=language, 
            n_added_tokens=n_added_tokens, 
            model_type="camembert", 
            **kwargs
        )

class BigBird(HFLanguageModel):
    """
    A BERT model that wraps Hugging Face's implementation
    (https://github.com/huggingface/transformers) to fit the LanguageModel class.
    Paper: https://arxiv.org/abs/1810.04805
    """

    def __init__(self, pretrained_model_name_or_path: Union[Path, str], language: str = None, n_added_tokens: int = 0, **kwargs):
        super().__init__(
            pretrained_model_name_or_path=pretrained_model_name_or_path, 
            language=language, 
            n_added_tokens=n_added_tokens, 
            model_type="bigbird", 
            **kwargs
        )

class DebertaV2(HFLanguageModelWithPooler):
    """
    This is a wrapper around the DebertaV2 model from Hugging Face's transformers library.
    It is also compatible with DebertaV3 as DebertaV3 only changes the pretraining procedure.

    NOTE:
    - DebertaV2 does not output the `pooled_output`. An additional pooler is initialized.
    """

    def __init__(self, pretrained_model_name_or_path: Union[Path, str], language: str = None, n_added_tokens: int = 0, **kwargs):
        super().__init__(
            pretrained_model_name_or_path=pretrained_model_name_or_path, 
            language=language, 
            n_added_tokens=n_added_tokens, 
            model_type="deberta-v2", 
            **kwargs
        )


class Data2VecVision(HFLanguageModel):
    """
    A Data2Vec (Vision) model that wraps Hugging Face's implementation
    (https://github.com/huggingface/transformers) to fit the LanguageModel class.
    Paper: https://arxiv.org/abs/1810.04805.
    """

    def __init__(self, pretrained_model_name_or_path: Union[Path, str], language: str = None, n_added_tokens: int = 0, **kwargs):
        super().__init__(
            pretrained_model_name_or_path=pretrained_model_name_or_path, 
            language=language, 
            n_added_tokens=n_added_tokens, 
            model_type="data2vec-vision", 
            **kwargs
        )


class DPRQuestionEncoder(LanguageModel):
    """
    A DPRQuestionEncoder model that wraps Hugging Face's implementation.
    """

    @silence_transformers_logs
    def __init__(
        self,
        pretrained_model_name_or_path: Union[Path, str],
        language: str = None,
        n_added_tokens: int = 0,
        auth_token: Optional[str] = None,
        **kwargs,
    ):
        """
        Load a pretrained model by supplying one of the following:

        * The name of a remote model on s3 (for example, "facebook/dpr-question_encoder-single-nq-base").
        * A local path of a model trained using transformers (for example, "some_dir/huggingface_model").
        * A local path of a model trained using Haystack (for example, "some_dir/haystack_model").

        :param pretrained_model_name_or_path: The path of the base pretrained language model whose weights are used to initialize DPRQuestionEncoder.
        """
        super().__init__()
        if "haystack_lm_name" in kwargs:
            self.name = kwargs["haystack_lm_name"]
        else:
            self.name = pretrained_model_name_or_path

        # We need to differentiate between loading model using Haystack format and Pytorch-Transformers format
        haystack_lm_config = Path(pretrained_model_name_or_path) / "language_model_config.json"
        if os.path.exists(haystack_lm_config):
            # Haystack style
            original_model_config = AutoConfig.from_pretrained(haystack_lm_config)
            haystack_lm_model = Path(pretrained_model_name_or_path) / "language_model.bin"

            if original_model_config.model_type == "dpr":
                dpr_config = transformers.DPRConfig.from_pretrained(haystack_lm_config)
                self.model = transformers.DPRQuestionEncoder.from_pretrained(
                    haystack_lm_model, config=dpr_config, **kwargs
                )
            else:
                if original_model_config.model_type != "bert":
                    logger.warning(
                        f"Using a model of type '{original_model_config.model_type}' which might be incompatible with DPR encoders."
                        f"Bert based encoders are supported that need input_ids,token_type_ids,attention_mask as input tensors."
                    )
                original_config_dict = vars(original_model_config)
                original_config_dict.update(kwargs)
                self.model = transformers.DPRQuestionEncoder(
                    config=transformers.DPRConfig(**original_config_dict)
                )
                language_model_class = DPRQuestionEncoder.get_language_model_class(haystack_lm_config, auth_token or False, **kwargs)
                self.model.base_model.bert_model = (
                    DPRQuestionEncoder.subclasses[language_model_class](str(pretrained_model_name_or_path)).model
                )
            self.language = self.model.config.language
        else:
            original_model_config = AutoConfig.from_pretrained(
                pretrained_model_name_or_path, use_auth_token=auth_token or False
            )
            if original_model_config.model_type == "dpr":
                # "pretrained dpr model": load existing pretrained DPRQuestionEncoder model
                self.model = transformers.DPRQuestionEncoder.from_pretrained(
                    str(pretrained_model_name_or_path), use_auth_token=auth_token or False, **kwargs
                )
            else:
                # "from scratch": load weights from different architecture (e.g. bert) into DPRQuestionEncoder
                # but keep config values from original architecture
                # TODO test for architectures other than BERT, e.g. Electra
                if original_model_config.model_type != "bert":
                    logger.warning(
                        f"Using a model of type '{original_model_config.model_type}' which might be incompatible with DPR encoders."
                        f"Bert based encoders are supported that need input_ids,token_type_ids,attention_mask as input tensors."
                    )
                original_config_dict = vars(original_model_config)
                original_config_dict.update(kwargs)
                self.model = transformers.DPRQuestionEncoder(
                    config=transformers.DPRConfig(**original_config_dict)
                )
                self.model.base_model.bert_model = AutoModel.from_pretrained(
                    str(pretrained_model_name_or_path), use_auth_token=auth_token or False, **original_config_dict
                )
            self.language = language or DPRQuestionEncoder._infer_language_from_name(pretrained_model_name_or_path)


    def save(self, save_dir: Union[str, Path], state_dict: Optional[Dict[Any, Any]] = None):
        """
        Save the model `state_dict` and its configuration file so that it can be loaded again.

        :param save_dir: The directory in which the model should be saved.
        :param state_dict: A dictionary containing the whole state of the module including names of layers.
                           By default, the unchanged state dictionary of the module is used.
        """
        model_to_save = self.model.module if hasattr(self.model, "module") else self.model  # Only save the model itself

        if self.model.config.model_type != "dpr" and model_to_save.base_model_prefix.startswith("question_"):
            state_dict = model_to_save.state_dict()
            if state_dict:
                keys = state_dict.keys()
                for key in list(keys):
                    new_key = key
                    if key.startswith("question_encoder.bert_model.model."):
                        new_key = key.split("_encoder.bert_model.model.", 1)[1]
                    elif key.startswith("question_encoder.bert_model."):
                        new_key = key.split("_encoder.bert_model.", 1)[1]
                    state_dict[new_key] = state_dict.pop(key)

        super().save(save_dir=save_dir, state_dict=state_dict)

    def forward(  # type: ignore
        self,
        query_input_ids: torch.Tensor,
        query_segment_ids: torch.Tensor,
        query_attention_mask: torch.Tensor,
        **kwargs,
    ):
        """
        Perform the forward pass of the DPRQuestionEncoder model.

        :param query_input_ids: The IDs of each token in the input sequence. It's a tensor of shape [batch_size, max_seq_len].
        :param query_segment_ids: The ID of the segment. For example, in next sentence prediction, the tokens in the
           first sentence are marked with 0 and the tokens in the second sentence are marked with 1.
           It is a tensor of shape [batch_size, max_seq_len].
        :param query_attention_mask: A mask that assigns 1 to valid input tokens and 0 to padding tokens
           of shape [batch_size, max_seq_len].
        :return: Embeddings for each token in the input sequence.
        """
        output_tuple = self.model(
            input_ids=query_input_ids,
            token_type_ids=query_segment_ids,
            attention_mask=query_attention_mask,
            return_dict=True,
        )
        if self.model.question_encoder.config.output_hidden_states == True:
            pooled_output, all_hidden_states = output_tuple.pooler_output, output_tuple.hidden_states
            return pooled_output, all_hidden_states
        else:
            pooled_output = output_tuple.pooler_output
            return pooled_output, None

    def enable_hidden_states_output(self):
        self.model.question_encoder.config.output_hidden_states = True

    def disable_hidden_states_output(self):
        self.model.question_encoder.config.output_hidden_states = False


class DPRContextEncoder(LanguageModel):
    """
    A DPRContextEncoder model that wraps Hugging Face's implementation.
    """
    @silence_transformers_logs
    def __init__(
        self,
        pretrained_model_name_or_path: Union[Path, str],
        language: str = None,
        n_added_tokens: int = 0,
        auth_token: Optional[str] = None,
        **kwargs,
    ):
        """
        Load a pretrained model by supplying one of the following:

        * The name of a remote model on s3 (for example, "facebook/dpr-ctx_encoder-single-nq-base").
        * A local path of a model trained using transformers (for example, "some_dir/huggingface_model").
        * A local path of a model trained using Haystack (for example, "some_dir/haystack_model").

        :param pretrained_model_name_or_path: The path of the base pretrained language model whose weights are used to initialize DPRContextEncoder.
        """
        super().__init__()
        if "haystack_lm_name" in kwargs:
            self.name = kwargs["haystack_lm_name"]
        else:
            self.name = pretrained_model_name_or_path
        # We need to differentiate between loading model using Haystack format and Pytorch-Transformers format
        haystack_lm_config = Path(pretrained_model_name_or_path) / "language_model_config.json"

        if os.path.exists(haystack_lm_config):
            # Haystack style
            original_model_config = AutoConfig.from_pretrained(haystack_lm_config)
            haystack_lm_model = Path(pretrained_model_name_or_path) / "language_model.bin"

            if original_model_config.model_type == "dpr":
                dpr_config = transformers.DPRConfig.from_pretrained(haystack_lm_config)
                self.model = transformers.DPRContextEncoder.from_pretrained(
                    haystack_lm_model, config=dpr_config, use_auth_token=auth_token or False, **kwargs
                )
            else:
                if original_model_config.model_type != "bert":
                    logger.warning(
                        f"Using a model of type '{original_model_config.model_type}' which might be incompatible with DPR encoders."
                        f"Bert based encoders are supported that need input_ids,token_type_ids,attention_mask as input tensors."
                    )
                original_config_dict = vars(original_model_config)
                original_config_dict.update(kwargs)
                self.model = transformers.DPRContextEncoder(
                    config=transformers.DPRConfig(**original_config_dict)
                )
                language_model_class = DPRQuestionEncoder.get_language_model_class(haystack_lm_config, **kwargs)
                self.model.base_model.bert_model = (
                    DPRContextEncoder.subclasses[language_model_class](str(pretrained_model_name_or_path), auth_token=auth_token).model
                )
            self.language = self.model.config.language

        else:
            # Pytorch-transformer Style
            original_model_config = AutoConfig.from_pretrained(
                pretrained_model_name_or_path, use_auth_token=auth_token or False
            )
            if original_model_config.model_type == "dpr":
                # "pretrained dpr model": load existing pretrained DPRContextEncoder model
                self.model = transformers.DPRContextEncoder.from_pretrained(
                    str(pretrained_model_name_or_path), use_auth_token=auth_token or False, **kwargs
                )
            else:
                # "from scratch": load weights from different architecture (e.g. bert) into DPRContextEncoder
                # but keep config values from original architecture
                # TODO test for architectures other than BERT, e.g. Electra
                if original_model_config.model_type != "bert":
                    logger.warning(
                        f"Using a model of type '{original_model_config.model_type}' which might be incompatible with DPR encoders."
                        f"Bert based encoders are supported that need input_ids,token_type_ids,attention_mask as input tensors."
                    )
                original_config_dict = vars(original_model_config)
                original_config_dict.update(kwargs)
                self.model = transformers.DPRContextEncoder(
                    config=transformers.DPRConfig(**original_config_dict)
                )
                self.model.base_model.bert_model = AutoModel.from_pretrained(
                    str(pretrained_model_name_or_path), use_auth_token=auth_token or False, **original_config_dict
                )
            self.language = language or DPRContextEncoder._infer_language_from_name(pretrained_model_name_or_path)


    def save(self, save_dir: Union[str, Path], state_dict: Optional[Dict[Any, Any]] = None):
        """
        Save the model `state_dict` and its configuration file so that it can be loaded again.

        :param save_dir: The directory in which the model should be saved.
        :param state_dict: A dictionary containing the whole state of the module including names of layers. By default, the unchanged state dictionary of the module is used.
        """
        model_to_save = (
            self.model.module if hasattr(self.model, "module") else self.model
        )  # Only save the model it-self

        if self.model.config.model_type != "dpr" and model_to_save.base_model_prefix.startswith("ctx_"):
            state_dict = model_to_save.state_dict()
            if state_dict:
                keys = state_dict.keys()
                for key in list(keys):
                    new_key = key
                    if key.startswith("ctx_encoder.bert_model.model."):
                        new_key = key.split("_encoder.bert_model.model.", 1)[1]
                    elif key.startswith("ctx_encoder.bert_model."):
                        new_key = key.split("_encoder.bert_model.", 1)[1]
                    state_dict[new_key] = state_dict.pop(key)

        super().save(save_dir=save_dir, state_dict=state_dict)

    def forward(  # type: ignore
        self,
        passage_input_ids: torch.Tensor,
        passage_segment_ids: torch.Tensor,
        passage_attention_mask: torch.Tensor,
        **kwargs,
    ):
        """
        Perform the forward pass of the DPRContextEncoder model.

        :param passage_input_ids: The IDs of each token in the input sequence. It's a tensor of shape [batch_size, number_of_hard_negative_passages, max_seq_len].
        :param passage_segment_ids: The ID of the segment. For example, in next sentence prediction, the tokens in the
           first sentence are marked with 0 and the tokens in the second sentence are marked with 1.
           It is a tensor of shape [batch_size, number_of_hard_negative_passages, max_seq_len].
        :param passage_attention_mask: A mask that assigns 1 to valid input tokens and 0 to padding tokens
           of shape [batch_size,  number_of_hard_negative_passages, max_seq_len].
        :return: Embeddings for each token in the input sequence.
        """
        max_seq_len = passage_input_ids.shape[-1]
        passage_input_ids = passage_input_ids.view(-1, max_seq_len)
        passage_segment_ids = passage_segment_ids.view(-1, max_seq_len)
        passage_attention_mask = passage_attention_mask.view(-1, max_seq_len)
        output_tuple = self.model(
            input_ids=passage_input_ids,
            token_type_ids=passage_segment_ids,
            attention_mask=passage_attention_mask,
            return_dict=True,
        )
        if self.model.ctx_encoder.config.output_hidden_states == True:
            pooled_output, all_hidden_states = output_tuple.pooler_output, output_tuple.hidden_states
            return pooled_output, all_hidden_states
        else:
            pooled_output = output_tuple.pooler_output
            return pooled_output, None

    def enable_hidden_states_output(self):
        self.model.ctx_encoder.config.output_hidden_states = True

    def disable_hidden_states_output(self):
        self.model.ctx_encoder.config.output_hidden_states = False
