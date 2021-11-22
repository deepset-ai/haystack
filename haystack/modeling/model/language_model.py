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
from __future__ import absolute_import, division, print_function, unicode_literals
from typing import Optional, Dict, Any, Union

import json
import logging
import os
from pathlib import Path
from functools import wraps
import numpy as np
import torch
from torch import nn
import transformers
from transformers import (
    BertModel, BertConfig,
    RobertaModel, RobertaConfig,
    XLNetModel, XLNetConfig,
    AlbertModel, AlbertConfig,
    XLMRobertaModel, XLMRobertaConfig,
    DistilBertModel, DistilBertConfig,
    ElectraModel, ElectraConfig,
    CamembertModel, CamembertConfig,
    BigBirdModel, BigBirdConfig
)
from transformers import AutoModel, AutoConfig
from transformers.modeling_utils import SequenceSummary


logger = logging.getLogger(__name__)


def silence_transformers_logs(from_pretrained_func):
    """
    Wrapper that raises the log level of Transformers to
    ERROR to hide some unnecessary warnings
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


# These are the names of the attributes in various model configs which refer to the number of dimensions
# in the output vectors
OUTPUT_DIM_NAMES = ["dim", "hidden_size", "d_model"]

#TODO analyse if LMs can be completely used through HF transformers
class LanguageModel(nn.Module):
    """
    The parent class for any kind of model that can embed language into a semantic vector space. Practically
    speaking, these models read in tokenized sentences and return vectors that capture the meaning of sentences
    or of tokens.
    """
    subclasses: dict = {}

    def __init_subclass__(cls, **kwargs):
        """ 
        This automatically keeps track of all available subclasses.
        Enables generic load() or all specific LanguageModel implementation.
        """
        super().__init_subclass__(**kwargs)
        cls.subclasses[cls.__name__] = cls

    def forward(
        self,
        input_ids: torch.Tensor,
        segment_ids: torch.Tensor,
        padding_mask: torch.Tensor,
        **kwargs,
    ):
        raise NotImplementedError

    @classmethod
    def load(cls, pretrained_model_name_or_path: Union[Path, str], language: str = None, use_auth_token: Union[bool, str] = None,  **kwargs):
        """
        Load a pretrained language model either by

        1. specifying its name and downloading it
        2. or pointing to the directory it is saved in.

        Available remote models:

        * bert-base-uncased
        * bert-large-uncased
        * bert-base-cased
        * bert-large-cased
        * bert-base-multilingual-uncased
        * bert-base-multilingual-cased
        * bert-base-chinese
        * bert-base-german-cased
        * roberta-base
        * roberta-large
        * xlnet-base-cased
        * xlnet-large-cased
        * xlm-roberta-base
        * xlm-roberta-large
        * albert-base-v2
        * albert-large-v2
        * distilbert-base-german-cased
        * distilbert-base-multilingual-cased
        * google/electra-small-discriminator
        * google/electra-base-discriminator
        * google/electra-large-discriminator
        * facebook/dpr-question_encoder-single-nq-base
        * facebook/dpr-ctx_encoder-single-nq-base

        See all supported model variations here: https://huggingface.co/models

        The appropriate language model class is inferred automatically from model config
        or can be manually supplied via `language_model_class`.

        :param pretrained_model_name_or_path: The path of the saved pretrained model or its name.
        :param revision: The version of model to use from the HuggingFace model hub. Can be tag name, branch name, or commit hash.
        :param language_model_class: (Optional) Name of the language model class to load (e.g. `Bert`)
        """
        n_added_tokens = kwargs.pop("n_added_tokens", 0)
        language_model_class = kwargs.pop("language_model_class", None)
        kwargs["revision"] = kwargs.get("revision", None)
        logger.info("LOADING MODEL")
        logger.info("=============")
        config_file = Path(pretrained_model_name_or_path) / "language_model_config.json"
        if os.path.exists(config_file):
            logger.info(f"Model found locally at {pretrained_model_name_or_path}")
            # it's a local directory in Haystack format
            config = json.load(open(config_file))
            language_model = cls.subclasses[config["name"]].load(pretrained_model_name_or_path)
        else:
            logger.info(f"Could not find {pretrained_model_name_or_path} locally.")
            logger.info(f"Looking on Transformers Model Hub (in local cache and online)...")
            if language_model_class is None:
                language_model_class = cls.get_language_model_class(pretrained_model_name_or_path, use_auth_token=use_auth_token, **kwargs)

            if language_model_class:
                language_model = cls.subclasses[language_model_class].load(pretrained_model_name_or_path, use_auth_token=use_auth_token, **kwargs)
            else:
                language_model = None

        if not language_model:
            raise Exception(
                f"Model not found for {pretrained_model_name_or_path}. Either supply the local path for a saved "
                f"model or one of bert/roberta/xlnet/albert/distilbert models that can be downloaded from remote. "
                f"Ensure that the model class name can be inferred from the directory name when loading a "
                f"Transformers' model."
            )
        else:
            logger.info(f"Loaded {pretrained_model_name_or_path}")

        # resize embeddings in case of custom vocab
        if n_added_tokens != 0:
            # TODO verify for other models than BERT
            model_emb_size = language_model.model.resize_token_embeddings(new_num_tokens=None).num_embeddings
            vocab_size = model_emb_size + n_added_tokens
            logger.info(
                f"Resizing embedding layer of LM from {model_emb_size} to {vocab_size} to cope with custom vocab.")
            language_model.model.resize_token_embeddings(vocab_size)
            # verify
            model_emb_size = language_model.model.resize_token_embeddings(new_num_tokens=None).num_embeddings
            assert vocab_size == model_emb_size

        return language_model

    @staticmethod
    def get_language_model_class(model_name_or_path, use_auth_token: Union[str,bool] = None, **kwargs):
        # it's transformers format (either from model hub or local)
        model_name_or_path = str(model_name_or_path)

        config = AutoConfig.from_pretrained(model_name_or_path, use_auth_token=use_auth_token, **kwargs)
        model_type = config.model_type
        if model_type == "xlm-roberta":
            language_model_class = "XLMRoberta"
        elif model_type == "roberta":
            if "mlm" in model_name_or_path.lower():
                raise NotImplementedError("MLM part of codebert is currently not supported in Haystack")
            language_model_class = "Roberta"
        elif model_type == "camembert":
            language_model_class = "Camembert"
        elif model_type == "albert":
            language_model_class = "Albert"
        elif model_type == "distilbert":
            language_model_class = "DistilBert"
        elif model_type == "bert":
            language_model_class = "Bert"
        elif model_type == "xlnet":
            language_model_class = "XLNet"
        elif model_type == "electra":
            language_model_class = "Electra"
        elif model_type == "dpr":
            if config.architectures[0] == "DPRQuestionEncoder":
                language_model_class = "DPRQuestionEncoder"
            elif config.architectures[0] == "DPRContextEncoder":
                language_model_class = "DPRContextEncoder"
            elif config.archictectures[0] == "DPRReader":
                raise NotImplementedError("DPRReader models are currently not supported.")
        elif model_type == "big_bird":
            language_model_class = "BigBird"
        else:
            # Fall back to inferring type from model name
            logger.warning("Could not infer LanguageModel class from config. Trying to infer "
                           "LanguageModel class from model name.")
            language_model_class = LanguageModel._infer_language_model_class_from_string(model_name_or_path)

        return language_model_class

    @staticmethod
    def _infer_language_model_class_from_string(model_name_or_path):
        # If inferring Language model class from config doesn't succeed,
        # fall back to inferring Language model class from model name.
        if "xlm" in model_name_or_path.lower() and "roberta" in model_name_or_path.lower():
            language_model_class = "XLMRoberta"
        elif "bigbird" in model_name_or_path.lower():
            language_model_class = "BigBird"
        elif "roberta" in model_name_or_path.lower():
            language_model_class = "Roberta"
        elif "codebert" in model_name_or_path.lower():
            if "mlm" in model_name_or_path.lower():
                raise NotImplementedError("MLM part of codebert is currently not supported in Haystack")
            else:
                language_model_class = "Roberta"
        elif "camembert" in model_name_or_path.lower() or "umberto" in model_name_or_path.lower():
            language_model_class = "Camembert"
        elif "albert" in model_name_or_path.lower():
            language_model_class = 'Albert'
        elif "distilbert" in model_name_or_path.lower():
            language_model_class = 'DistilBert'
        elif "bert" in model_name_or_path.lower():
            language_model_class = 'Bert'
        elif "xlnet" in model_name_or_path.lower():
            language_model_class = 'XLNet'
        elif "electra" in model_name_or_path.lower():
            language_model_class = 'Electra'
        elif "word2vec" in model_name_or_path.lower() or "glove" in model_name_or_path.lower():
            language_model_class = 'WordEmbedding_LM'
        elif "minilm" in model_name_or_path.lower():
            language_model_class = "Bert"
        elif "dpr-question_encoder" in model_name_or_path.lower():
            language_model_class = "DPRQuestionEncoder"
        elif "dpr-ctx_encoder" in model_name_or_path.lower():
            language_model_class = "DPRContextEncoder"
        else:
            language_model_class = None

        return language_model_class

    def get_output_dims(self):
        config = self.model.config
        for odn in OUTPUT_DIM_NAMES:
            if odn in dir(config):
                return getattr(config, odn)
        else:
            raise Exception("Could not infer the output dimensions of the language model")

    def freeze(self, layers):
        """ To be implemented"""
        raise NotImplementedError()

    def unfreeze(self):
        """ To be implemented"""
        raise NotImplementedError()

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
        Save the model state_dict and its config file so that it can be loaded again.

        :param save_dir: The directory in which the model should be saved.
        :param state_dict: A dictionary containing a whole state of the module including names of layers. By default, the unchanged state dict of the module is used
        """
        # Save Weights
        save_name = Path(save_dir) / "language_model.bin"
        model_to_save = (
            self.model.module if hasattr(self.model, "module") else self.model
        )  # Only save the model it-self

        if not state_dict:
            state_dict = model_to_save.state_dict()
        torch.save(state_dict, save_name)
        self.save_config(save_dir)

    @classmethod
    def _get_or_infer_language_from_name(cls, language, name):
        if language is not None:
            return language
        else:
            return cls._infer_language_from_name(name)

    @classmethod
    def _infer_language_from_name(cls, name):
        known_languages = (
            "german",
            "english",
            "chinese",
            "indian",
            "french",
            "polish",
            "spanish",
            "multilingual",
        )
        matches = [lang for lang in known_languages if lang in name]
        if "camembert" in name:
            language = "french"
            logger.info(
                f"Automatically detected language from language model name: {language}"
            )
        elif "umberto" in name:
            language = "italian"
            logger.info(
                f"Automatically detected language from language model name: {language}"
            )
        elif len(matches) == 0:
            language = "english"
        elif len(matches) > 1:
            language = matches[0]
        else:
            language = matches[0]
            logger.info(
                f"Automatically detected language from language model name: {language}"
            )

        return language

    def formatted_preds(self, logits, samples, ignore_first_token=True,
                        padding_mask=None, input_ids=None, **kwargs):
        """
        Extracting vectors from language model (e.g. for extracting sentence embeddings).
        Different pooling strategies and layers are available and will be determined from the object attributes
        `extraction_layer` and `extraction_strategy`. Both should be set via the Inferencer:
        Example:  Inferencer(extraction_strategy='cls_token', extraction_layer=-1)

        :param logits: Tuple of (sequence_output, pooled_output) from the language model.
                       Sequence_output: one vector per token, pooled_output: one vector for whole sequence
        :param samples: For each item in logits we need additional meta information to format the prediction (e.g. input text).
                        This is created by the Processor and passed in here from the Inferencer.
        :param ignore_first_token: Whether to include the first token for pooling operations (e.g. reduce_mean).
                                   Many models have here a special token like [CLS] that you don't want to include into your average of token embeddings.
        :param padding_mask: Mask for the padding tokens. Those will also not be included in the pooling operations to prevent a bias by the number of padding tokens.
        :param input_ids: ids of the tokens in the vocab
        :param kwargs: kwargs
        :return: list of dicts containing preds, e.g. [{"context": "some text", "vec": [-0.01, 0.5 ...]}]
        """
        if not hasattr(self, "extraction_layer") or not hasattr(self, "extraction_strategy"):
            raise ValueError("`extraction_layer` or `extraction_strategy` not specified for LM. "
                             "Make sure to set both, e.g. via Inferencer(extraction_strategy='cls_token', extraction_layer=-1)`")

        # unpack the tuple from LM forward pass
        sequence_output = logits[0][0]
        pooled_output = logits[0][1]

        # aggregate vectors
        if self.extraction_strategy == "pooled":
            if self.extraction_layer != -1:
                raise ValueError(f"Pooled output only works for the last layer, but got extraction_layer = {self.extraction_layer}. Please set `extraction_layer=-1`.)")
            vecs = pooled_output.cpu().numpy()
        elif self.extraction_strategy == "per_token":
            vecs = sequence_output.cpu().numpy()
        elif self.extraction_strategy == "reduce_mean":
            vecs = self._pool_tokens(sequence_output, padding_mask, self.extraction_strategy, ignore_first_token=ignore_first_token)
        elif self.extraction_strategy == "reduce_max":
            vecs = self._pool_tokens(sequence_output, padding_mask, self.extraction_strategy, ignore_first_token=ignore_first_token)
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


class Bert(LanguageModel):
    """
    A BERT model that wraps HuggingFace's implementation
    (https://github.com/huggingface/transformers) to fit the LanguageModel class.
    Paper: https://arxiv.org/abs/1810.04805
    """
    def __init__(self):
        super(Bert, self).__init__()
        self.model = None
        self.name = "bert"

    @classmethod
    def from_scratch(cls, vocab_size, name="bert", language="en"):
        bert = cls()
        bert.name = name
        bert.language = language
        config = BertConfig(vocab_size=vocab_size)
        bert.model = BertModel(config)
        return bert

    @classmethod
    @silence_transformers_logs
    def load(cls, pretrained_model_name_or_path: Union[Path, str], language: str = None, **kwargs):
        """
        Load a pretrained model by supplying

        * the name of a remote model on s3 ("bert-base-cased" ...)
        * OR a local path of a model trained via transformers ("some_dir/huggingface_model")
        * OR a local path of a model trained via Haystack ("some_dir/haystack_model")

        :param pretrained_model_name_or_path: The path of the saved pretrained model or its name.
        """
        bert = cls()
        if "haystack_lm_name" in kwargs:
            bert.name = kwargs["haystack_lm_name"]
        else:
            bert.name = pretrained_model_name_or_path
        # We need to differentiate between loading model using Haystack format and Pytorch-Transformers format
        haystack_lm_config = Path(pretrained_model_name_or_path) / "language_model_config.json"
        if os.path.exists(haystack_lm_config):
            # Haystack style
            bert_config = BertConfig.from_pretrained(haystack_lm_config)
            haystack_lm_model = Path(pretrained_model_name_or_path) / "language_model.bin"
            bert.model = BertModel.from_pretrained(haystack_lm_model, config=bert_config, **kwargs)
            bert.language = bert.model.config.language
        else:
            # Pytorch-transformer Style
            bert.model = BertModel.from_pretrained(str(pretrained_model_name_or_path), **kwargs)
            bert.language = cls._get_or_infer_language_from_name(language, pretrained_model_name_or_path)
        return bert

    def forward(
        self,
        input_ids: torch.Tensor,
        segment_ids: torch.Tensor,
        padding_mask: torch.Tensor,
        **kwargs,
    ):
        """
        Perform the forward pass of the BERT model.

        :param input_ids: The ids of each token in the input sequence. Is a tensor of shape [batch_size, max_seq_len]
        :param segment_ids: The id of the segment. For example, in next sentence prediction, the tokens in the
           first sentence are marked with 0 and those in the second are marked with 1.
           It is a tensor of shape [batch_size, max_seq_len]
        :param padding_mask: A mask that assigns a 1 to valid input tokens and 0 to padding tokens
           of shape [batch_size, max_seq_len]
        :return: Embeddings for each token in the input sequence.
        """
        output_tuple = self.model(
            input_ids,
            token_type_ids=segment_ids,
            attention_mask=padding_mask,
        )
        if self.model.encoder.config.output_hidden_states == True:
            sequence_output, pooled_output, all_hidden_states = output_tuple[0], output_tuple[1], output_tuple[2]
            return sequence_output, pooled_output, all_hidden_states
        else:
            sequence_output, pooled_output = output_tuple[0], output_tuple[1]
            return sequence_output, pooled_output

    def enable_hidden_states_output(self):
        self.model.encoder.config.output_hidden_states = True

    def disable_hidden_states_output(self):
        self.model.encoder.config.output_hidden_states = False


class Albert(LanguageModel):
    """
    An ALBERT model that wraps the HuggingFace's implementation
    (https://github.com/huggingface/transformers) to fit the LanguageModel class.
    """
    def __init__(self):
        super(Albert, self).__init__()
        self.model = None
        self.name = "albert"

    @classmethod
    @silence_transformers_logs
    def load(cls, pretrained_model_name_or_path: Union[Path, str], language: str = None, **kwargs):
        """
        Load a language model either by supplying

        * the name of a remote model on s3 ("albert-base" ...)
        * or a local path of a model trained via transformers ("some_dir/huggingface_model")
        * or a local path of a model trained via Haystack ("some_dir/Haystack_model")

        :param pretrained_model_name_or_path: name or path of a model
        :param language: (Optional) Name of language the model was trained for (e.g. "german").
                         If not supplied, Haystack will try to infer it from the model name.
        :return: Language Model
        """
        albert = cls()
        if "haystack_lm_name" in kwargs:
            albert.name = kwargs["haystack_lm_name"]
        else:
            albert.name = pretrained_model_name_or_path
        # We need to differentiate between loading model using Haystack format and Pytorch-Transformers format
        haystack_lm_config = Path(pretrained_model_name_or_path) / "language_model_config.json"
        if os.path.exists(haystack_lm_config):
            # Haystack style
            config = AlbertConfig.from_pretrained(haystack_lm_config)
            haystack_lm_model = Path(pretrained_model_name_or_path) / "language_model.bin"
            albert.model = AlbertModel.from_pretrained(haystack_lm_model, config=config, **kwargs)
            albert.language = albert.model.config.language
        else:
            # Huggingface transformer Style
            albert.model = AlbertModel.from_pretrained(str(pretrained_model_name_or_path), **kwargs)
            albert.language = cls._get_or_infer_language_from_name(language, pretrained_model_name_or_path)
        return albert

    def forward(
        self,
        input_ids: torch.Tensor,
        segment_ids: torch.Tensor,
        padding_mask: torch.Tensor,
        **kwargs,
    ):
        """
        Perform the forward pass of the Albert model.

        :param input_ids: The ids of each token in the input sequence. Is a tensor of shape [batch_size, max_seq_len]
        :param segment_ids: The id of the segment. For example, in next sentence prediction, the tokens in the
           first sentence are marked with 0 and those in the second are marked with 1.
           It is a tensor of shape [batch_size, max_seq_len]
        :param padding_mask: A mask that assigns a 1 to valid input tokens and 0 to padding tokens
           of shape [batch_size, max_seq_len]
        :return: Embeddings for each token in the input sequence.
        """
        output_tuple = self.model(
            input_ids,
            token_type_ids=segment_ids,
            attention_mask=padding_mask,
        )
        if self.model.encoder.config.output_hidden_states == True:
            sequence_output, pooled_output, all_hidden_states = output_tuple[0], output_tuple[1], output_tuple[2]
            return sequence_output, pooled_output, all_hidden_states
        else:
            sequence_output, pooled_output = output_tuple[0], output_tuple[1]
            return sequence_output, pooled_output

    def enable_hidden_states_output(self):
        self.model.encoder.config.output_hidden_states = True

    def disable_hidden_states_output(self):
        self.model.encoder.config.output_hidden_states = False


class Roberta(LanguageModel):
    """
    A roberta model that wraps the HuggingFace's implementation
    (https://github.com/huggingface/transformers) to fit the LanguageModel class.
    Paper: https://arxiv.org/abs/1907.11692
    """
    def __init__(self):
        super(Roberta, self).__init__()
        self.model = None
        self.name = "roberta"

    @classmethod
    @silence_transformers_logs
    def load(cls, pretrained_model_name_or_path: Union[Path, str], language: str = None, **kwargs):
        """
        Load a language model either by supplying

        * the name of a remote model on s3 ("roberta-base" ...)
        * or a local path of a model trained via transformers ("some_dir/huggingface_model")
        * or a local path of a model trained via Haystack ("some_dir/haystack_model")

        :param pretrained_model_name_or_path: name or path of a model
        :param language: (Optional) Name of language the model was trained for (e.g. "german").
                         If not supplied, Haystack will try to infer it from the model name.
        :return: Language Model
        """
        roberta = cls()
        if "haystack_lm_name" in kwargs:
            roberta.name = kwargs["haystack_lm_name"]
        else:
            roberta.name = pretrained_model_name_or_path
        # We need to differentiate between loading model using Haystack format and Pytorch-Transformers format
        haystack_lm_config = Path(pretrained_model_name_or_path) / "language_model_config.json"
        if os.path.exists(haystack_lm_config):
            # Haystack style
            config = RobertaConfig.from_pretrained(haystack_lm_config)
            haystack_lm_model = Path(pretrained_model_name_or_path) / "language_model.bin"
            roberta.model = RobertaModel.from_pretrained(haystack_lm_model, config=config, **kwargs)
            roberta.language = roberta.model.config.language
        else:
            # Huggingface transformer Style
            roberta.model = RobertaModel.from_pretrained(str(pretrained_model_name_or_path), **kwargs)
            roberta.language = cls._get_or_infer_language_from_name(language, pretrained_model_name_or_path)
        return roberta

    def forward(
        self,
        input_ids: torch.Tensor,
        segment_ids: torch.Tensor,
        padding_mask: torch.Tensor,
        **kwargs,
    ):
        """
        Perform the forward pass of the Roberta model.

        :param input_ids: The ids of each token in the input sequence. Is a tensor of shape [batch_size, max_seq_len]
        :param segment_ids: The id of the segment. For example, in next sentence prediction, the tokens in the
           first sentence are marked with 0 and those in the second are marked with 1.
           It is a tensor of shape [batch_size, max_seq_len]
        :param padding_mask: A mask that assigns a 1 to valid input tokens and 0 to padding tokens
           of shape [batch_size, max_seq_len]
        :return: Embeddings for each token in the input sequence.
        """
        output_tuple = self.model(
            input_ids,
            token_type_ids=segment_ids,
            attention_mask=padding_mask,
        )
        if self.model.encoder.config.output_hidden_states == True:
            sequence_output, pooled_output, all_hidden_states = output_tuple[0], output_tuple[1], output_tuple[2]
            return sequence_output, pooled_output, all_hidden_states
        else:
            sequence_output, pooled_output = output_tuple[0], output_tuple[1]
            return sequence_output, pooled_output

    def enable_hidden_states_output(self):
        self.model.encoder.config.output_hidden_states = True

    def disable_hidden_states_output(self):
        self.model.encoder.config.output_hidden_states = False


class XLMRoberta(LanguageModel):
    """
    A roberta model that wraps the HuggingFace's implementation
    (https://github.com/huggingface/transformers) to fit the LanguageModel class.
    Paper: https://arxiv.org/abs/1907.11692
    """
    def __init__(self):
        super(XLMRoberta, self).__init__()
        self.model = None
        self.name = "xlm_roberta"

    @classmethod
    @silence_transformers_logs
    def load(cls, pretrained_model_name_or_path: Union[Path, str], language: str = None, **kwargs):
        """
        Load a language model either by supplying

        * the name of a remote model on s3 ("xlm-roberta-base" ...)
        * or a local path of a model trained via transformers ("some_dir/huggingface_model")
        * or a local path of a model trained via Haystack ("some_dir/haystack_model")

        :param pretrained_model_name_or_path: name or path of a model
        :param language: (Optional) Name of language the model was trained for (e.g. "german").
                         If not supplied, Haystack will try to infer it from the model name.
        :return: Language Model
        """
        xlm_roberta = cls()
        if "haystack_lm_name" in kwargs:
            xlm_roberta.name = kwargs["haystack_lm_name"]
        else:
            xlm_roberta.name = pretrained_model_name_or_path
        # We need to differentiate between loading model using Haystack format and Pytorch-Transformers format
        haystack_lm_config = Path(pretrained_model_name_or_path) / "language_model_config.json"
        if os.path.exists(haystack_lm_config):
            # Haystack style
            config = XLMRobertaConfig.from_pretrained(haystack_lm_config)
            haystack_lm_model = Path(pretrained_model_name_or_path) / "language_model.bin"
            xlm_roberta.model = XLMRobertaModel.from_pretrained(haystack_lm_model, config=config, **kwargs)
            xlm_roberta.language = xlm_roberta.model.config.language
        else:
            # Huggingface transformer Style
            xlm_roberta.model = XLMRobertaModel.from_pretrained(str(pretrained_model_name_or_path), **kwargs)
            xlm_roberta.language = cls._get_or_infer_language_from_name(language, pretrained_model_name_or_path)
        return xlm_roberta

    def forward(
        self,
        input_ids: torch.Tensor,
        segment_ids: torch.Tensor,
        padding_mask: torch.Tensor,
        **kwargs,
    ):
        """
        Perform the forward pass of the XLMRoberta model.

        :param input_ids: The ids of each token in the input sequence. Is a tensor of shape [batch_size, max_seq_len]
        :param segment_ids: The id of the segment. For example, in next sentence prediction, the tokens in the
           first sentence are marked with 0 and those in the second are marked with 1.
           It is a tensor of shape [batch_size, max_seq_len]
        :param padding_mask: A mask that assigns a 1 to valid input tokens and 0 to padding tokens
           of shape [batch_size, max_seq_len]
        :return: Embeddings for each token in the input sequence.
        """
        output_tuple = self.model(
            input_ids,
            token_type_ids=segment_ids,
            attention_mask=padding_mask,
        )
        if self.model.encoder.config.output_hidden_states == True:
            sequence_output, pooled_output, all_hidden_states = output_tuple[0], output_tuple[1], output_tuple[2]
            return sequence_output, pooled_output, all_hidden_states
        else:
            sequence_output, pooled_output = output_tuple[0], output_tuple[1]
            return sequence_output, pooled_output

    def enable_hidden_states_output(self):
        self.model.encoder.config.output_hidden_states = True

    def disable_hidden_states_output(self):
        self.model.encoder.config.output_hidden_states = False


class DistilBert(LanguageModel):
    """
    A DistilBERT model that wraps HuggingFace's implementation
    (https://github.com/huggingface/transformers) to fit the LanguageModel class.

    NOTE:
    - DistilBert doesn’t have token_type_ids, you don’t need to indicate which
    token belongs to which segment. Just separate your segments with the separation
    token tokenizer.sep_token (or [SEP])
    - Unlike the other BERT variants, DistilBert does not output the
    pooled_output. An additional pooler is initialized.
    """
    def __init__(self):
        super(DistilBert, self).__init__()
        self.model = None
        self.name = "distilbert"
        self.pooler = None

    @classmethod
    @silence_transformers_logs
    def load(cls, pretrained_model_name_or_path: Union[Path, str], language: str = None, **kwargs):
        """
        Load a pretrained model by supplying

        * the name of a remote model on s3 ("distilbert-base-german-cased" ...)
        * OR a local path of a model trained via transformers ("some_dir/huggingface_model")
        * OR a local path of a model trained via Haystack ("some_dir/haystack_model")

        :param pretrained_model_name_or_path: The path of the saved pretrained model or its name.
        """
        distilbert = cls()
        if "haystack_lm_name" in kwargs:
            distilbert.name = kwargs["haystack_lm_name"]
        else:
            distilbert.name = pretrained_model_name_or_path
        # We need to differentiate between loading model using Haystack format and Pytorch-Transformers format
        haystack_lm_config = Path(pretrained_model_name_or_path) / "language_model_config.json"
        if os.path.exists(haystack_lm_config):
            # Haystack style
            config = DistilBertConfig.from_pretrained(haystack_lm_config)
            haystack_lm_model = Path(pretrained_model_name_or_path) / "language_model.bin"
            distilbert.model = DistilBertModel.from_pretrained(haystack_lm_model, config=config, **kwargs)
            distilbert.language = distilbert.model.config.language
        else:
            # Pytorch-transformer Style
            distilbert.model = DistilBertModel.from_pretrained(str(pretrained_model_name_or_path), **kwargs)
            distilbert.language = cls._get_or_infer_language_from_name(language, pretrained_model_name_or_path)
        config = distilbert.model.config

        # DistilBERT does not provide a pooled_output by default. Therefore, we need to initialize an extra pooler.
        # The pooler takes the first hidden representation & feeds it to a dense layer of (hidden_dim x hidden_dim).
        # We don't want a dropout in the end of the pooler, since we do that already in the adaptive model before we
        # feed everything to the prediction head
        config.summary_last_dropout = 0
        config.summary_type = 'first'
        config.summary_activation = 'tanh'
        distilbert.pooler = SequenceSummary(config)
        distilbert.pooler.apply(distilbert.model._init_weights)
        return distilbert

    def forward(  # type: ignore
        self,
        input_ids: torch.Tensor,
        padding_mask: torch.Tensor,
        **kwargs,
    ):  
        """
        Perform the forward pass of the DistilBERT model.

        :param input_ids: The ids of each token in the input sequence. Is a tensor of shape [batch_size, max_seq_len]
        :param padding_mask: A mask that assigns a 1 to valid input tokens and 0 to padding tokens
           of shape [batch_size, max_seq_len]
        :return: Embeddings for each token in the input sequence.
        """
        output_tuple = self.model(
            input_ids,
            attention_mask=padding_mask,
        )
        # We need to manually aggregate that to get a pooled output (one vec per seq)
        pooled_output = self.pooler(output_tuple[0])
        if self.model.config.output_hidden_states == True:
            sequence_output, all_hidden_states = output_tuple[0], output_tuple[1]
            return sequence_output, pooled_output
        else:
            sequence_output = output_tuple[0]
            return sequence_output, pooled_output

    def enable_hidden_states_output(self):
        self.model.config.output_hidden_states = True

    def disable_hidden_states_output(self):
        self.model.config.output_hidden_states = False


class XLNet(LanguageModel):
    """
    A XLNet model that wraps the HuggingFace's implementation
    (https://github.com/huggingface/transformers) to fit the LanguageModel class.
    Paper: https://arxiv.org/abs/1906.08237
    """
    def __init__(self):
        super(XLNet, self).__init__()
        self.model = None
        self.name = "xlnet"
        self.pooler = None

    @classmethod
    @silence_transformers_logs
    def load(cls, pretrained_model_name_or_path: Union[Path, str], language: str = None, **kwargs):
        """
        Load a language model either by supplying

        * the name of a remote model on s3 ("xlnet-base-cased" ...)
        * or a local path of a model trained via transformers ("some_dir/huggingface_model")
        * or a local path of a model trained via Haystack ("some_dir/haystack_model")

        :param pretrained_model_name_or_path: name or path of a model
        :param language: (Optional) Name of language the model was trained for (e.g. "german").
                         If not supplied, Haystack will try to infer it from the model name.
        :return: Language Model
        """
        xlnet = cls()
        if "haystack_lm_name" in kwargs:
            xlnet.name = kwargs["haystack_lm_name"]
        else:
            xlnet.name = pretrained_model_name_or_path
        # We need to differentiate between loading model using Haystack format and Pytorch-Transformers format
        haystack_lm_config = Path(pretrained_model_name_or_path) / "language_model_config.json"
        if os.path.exists(haystack_lm_config):
            # Haystack style
            config = XLNetConfig.from_pretrained(haystack_lm_config)
            haystack_lm_model = Path(pretrained_model_name_or_path) / "language_model.bin"
            xlnet.model = XLNetModel.from_pretrained(haystack_lm_model, config=config, **kwargs)
            xlnet.language = xlnet.model.config.language
        else:
            # Pytorch-transformer Style
            xlnet.model = XLNetModel.from_pretrained(str(pretrained_model_name_or_path), **kwargs)
            xlnet.language = cls._get_or_infer_language_from_name(language, pretrained_model_name_or_path)
            config = xlnet.model.config
        # XLNet does not provide a pooled_output by default. Therefore, we need to initialize an extra pooler.
        # The pooler takes the last hidden representation & feeds it to a dense layer of (hidden_dim x hidden_dim).
        # We don't want a dropout in the end of the pooler, since we do that already in the adaptive model before we
        # feed everything to the prediction head
        config.summary_last_dropout = 0
        xlnet.pooler = SequenceSummary(config)
        xlnet.pooler.apply(xlnet.model._init_weights)
        return xlnet

    def forward(
        self,
        input_ids: torch.Tensor,
        segment_ids: torch.Tensor,
        padding_mask: torch.Tensor,
        **kwargs,
    ):
        """
        Perform the forward pass of the XLNet model.

        :param input_ids: The ids of each token in the input sequence. Is a tensor of shape [batch_size, max_seq_len]
        :param segment_ids: The id of the segment. For example, in next sentence prediction, the tokens in the
           first sentence are marked with 0 and those in the second are marked with 1.
           It is a tensor of shape [batch_size, max_seq_len]
        :param padding_mask: A mask that assigns a 1 to valid input tokens and 0 to padding tokens
           of shape [batch_size, max_seq_len]
        :return: Embeddings for each token in the input sequence.
        """
        # Note: XLNet has a couple of special input tensors for pretraining / text generation  (perm_mask, target_mapping ...)
        # We will need to implement them, if we wanna support LM adaptation
        output_tuple = self.model(
            input_ids,
            token_type_ids=segment_ids,
            attention_mask=padding_mask,
        )
        # XLNet also only returns the sequence_output (one vec per token)
        # We need to manually aggregate that to get a pooled output (one vec per seq)
        # TODO verify that this is really doing correct pooling
        pooled_output = self.pooler(output_tuple[0])

        if self.model.output_hidden_states == True:
            sequence_output, all_hidden_states = output_tuple[0], output_tuple[1]
            return sequence_output, pooled_output, all_hidden_states
        else:
            sequence_output = output_tuple[0]
            return sequence_output, pooled_output

    def enable_hidden_states_output(self):
        self.model.output_hidden_states = True

    def disable_hidden_states_output(self):
        self.model.output_hidden_states = False


class Electra(LanguageModel):
    """
    ELECTRA is a new pre-training approach which trains two transformer models:
    the generator and the discriminator. The generator replaces tokens in a sequence,
    and is therefore trained as a masked language model. The discriminator, which is
    the model we're interested in, tries to identify which tokens were replaced by
    the generator in the sequence.

    The ELECTRA model here wraps HuggingFace's implementation
    (https://github.com/huggingface/transformers) to fit the LanguageModel class.

    NOTE:
    - Electra does not output the pooled_output. An additional pooler is initialized.
    """

    def __init__(self):
        super(Electra, self).__init__()
        self.model = None
        self.name = "electra"
        self.pooler = None

    @classmethod
    @silence_transformers_logs
    def load(cls, pretrained_model_name_or_path: Union[Path, str], language: str = None, **kwargs):
        """
        Load a pretrained model by supplying

        * the name of a remote model on s3 ("google/electra-base-discriminator" ...)
        * OR a local path of a model trained via transformers ("some_dir/huggingface_model")
        * OR a local path of a model trained via Haystack ("some_dir/haystack_model")

        :param pretrained_model_name_or_path: The path of the saved pretrained model or its name.
        """
        electra = cls()
        if "haystack_lm_name" in kwargs:
            electra.name = kwargs["haystack_lm_name"]
        else:
            electra.name = pretrained_model_name_or_path
        # We need to differentiate between loading model using Haystack format and Transformers format
        haystack_lm_config = Path(pretrained_model_name_or_path) / "language_model_config.json"
        if os.path.exists(haystack_lm_config):
            # Haystack style
            config = ElectraConfig.from_pretrained(haystack_lm_config)
            haystack_lm_model = Path(pretrained_model_name_or_path) / "language_model.bin"
            electra.model = ElectraModel.from_pretrained(haystack_lm_model, config=config, **kwargs)
            electra.language = electra.model.config.language
        else:
            # Transformers Style
            electra.model = ElectraModel.from_pretrained(str(pretrained_model_name_or_path), **kwargs)
            electra.language = cls._get_or_infer_language_from_name(language, pretrained_model_name_or_path)
        config = electra.model.config

        # ELECTRA does not provide a pooled_output by default. Therefore, we need to initialize an extra pooler.
        # The pooler takes the first hidden representation & feeds it to a dense layer of (hidden_dim x hidden_dim).
        # We don't want a dropout in the end of the pooler, since we do that already in the adaptive model before we
        # feed everything to the prediction head.
        # Note: ELECTRA uses gelu as activation (BERT uses tanh instead)
        config.summary_last_dropout = 0
        config.summary_type = 'first'
        config.summary_activation = 'gelu'
        config.summary_use_proj = False
        electra.pooler = SequenceSummary(config)
        electra.pooler.apply(electra.model._init_weights)
        return electra

    def forward(
        self,
        input_ids: torch.Tensor,
        segment_ids: torch.Tensor,
        padding_mask: torch.Tensor,
        **kwargs,
    ):
        """
        Perform the forward pass of the ELECTRA model.

        :param input_ids: The ids of each token in the input sequence. Is a tensor of shape [batch_size, max_seq_len]
        :param padding_mask: A mask that assigns a 1 to valid input tokens and 0 to padding tokens
           of shape [batch_size, max_seq_len]
        :return: Embeddings for each token in the input sequence.
        """
        output_tuple = self.model(
            input_ids,
            token_type_ids=segment_ids,
            attention_mask=padding_mask,
        )

        # We need to manually aggregate that to get a pooled output (one vec per seq)
        pooled_output = self.pooler(output_tuple[0])

        if self.model.config.output_hidden_states == True:
            sequence_output, all_hidden_states = output_tuple[0], output_tuple[1]
            return sequence_output, pooled_output
        else:
            sequence_output = output_tuple[0]
            return sequence_output, pooled_output

    def enable_hidden_states_output(self):
        self.model.config.output_hidden_states = True

    def disable_hidden_states_output(self):
        self.model.config.output_hidden_states = False


class Camembert(Roberta):
    """
    A Camembert model that wraps the HuggingFace's implementation
    (https://github.com/huggingface/transformers) to fit the LanguageModel class.
    """
    def __init__(self):
        super(Camembert, self).__init__()
        self.model = None
        self.name = "camembert"

    @classmethod
    @silence_transformers_logs
    def load(cls, pretrained_model_name_or_path: Union[Path, str], language: str = None, **kwargs):
        """
        Load a language model either by supplying

        * the name of a remote model on s3 ("camembert-base" ...)
        * or a local path of a model trained via transformers ("some_dir/huggingface_model")
        * or a local path of a model trained via Haystack ("some_dir/haystack_model")

        :param pretrained_model_name_or_path: name or path of a model
        :param language: (Optional) Name of language the model was trained for (e.g. "german").
                         If not supplied, Haystack will try to infer it from the model name.
        :return: Language Model
        """
        camembert = cls()
        if "haystack_lm_name" in kwargs:
            camembert.name = kwargs["haystack_lm_name"]
        else:
            camembert.name = pretrained_model_name_or_path
        # We need to differentiate between loading model using Haystack format and Pytorch-Transformers format
        haystack_lm_config = Path(pretrained_model_name_or_path) / "language_model_config.json"
        if os.path.exists(haystack_lm_config):
            # Haystack style
            config = CamembertConfig.from_pretrained(haystack_lm_config)
            haystack_lm_model = Path(pretrained_model_name_or_path) / "language_model.bin"
            camembert.model = CamembertModel.from_pretrained(haystack_lm_model, config=config, **kwargs)
            camembert.language = camembert.model.config.language
        else:
            # Huggingface transformer Style
            camembert.model = CamembertModel.from_pretrained(str(pretrained_model_name_or_path), **kwargs)
            camembert.language = cls._get_or_infer_language_from_name(language, pretrained_model_name_or_path)
        return camembert


class DPRQuestionEncoder(LanguageModel):
    """
    A DPRQuestionEncoder model that wraps HuggingFace's implementation
    """
    def __init__(self):
        super(DPRQuestionEncoder, self).__init__()
        self.model = None
        self.name = "dpr_question_encoder"

    @classmethod
    @silence_transformers_logs
    def load(cls, pretrained_model_name_or_path: Union[Path, str], language: str = None, use_auth_token: Union[str,bool] = None, **kwargs):
        """
        Load a pretrained model by supplying

        * the name of a remote model on s3 ("facebook/dpr-question_encoder-single-nq-base" ...)
        * OR a local path of a model trained via transformers ("some_dir/huggingface_model")
        * OR a local path of a model trained via Haystack ("some_dir/haystack_model")

        :param pretrained_model_name_or_path: The path of the base pretrained language model whose weights are used to initialize DPRQuestionEncoder
        """
        dpr_question_encoder = cls()
        if "haystack_lm_name" in kwargs:
            dpr_question_encoder.name = kwargs["haystack_lm_name"]
        else:
            dpr_question_encoder.name = pretrained_model_name_or_path

        # We need to differentiate between loading model using Haystack format and Pytorch-Transformers format
        haystack_lm_config = Path(pretrained_model_name_or_path) / "language_model_config.json"
        if os.path.exists(haystack_lm_config):
            # Haystack style
            original_model_config = AutoConfig.from_pretrained(haystack_lm_config)
            haystack_lm_model = Path(pretrained_model_name_or_path) / "language_model.bin"

            if original_model_config.model_type == "dpr":
                dpr_config = transformers.DPRConfig.from_pretrained(haystack_lm_config)
                dpr_question_encoder.model = transformers.DPRQuestionEncoder.from_pretrained(haystack_lm_model, config=dpr_config, **kwargs)
            else:
                if original_model_config.model_type != "bert":
                    logger.warning(f"Using a model of type '{original_model_config.model_type}' which might be incompatible with DPR encoders."
                                   f"Bert based encoders are supported that need input_ids,token_type_ids,attention_mask as input tensors.")
                original_config_dict = vars(original_model_config)
                original_config_dict.update(kwargs)
                dpr_question_encoder.model = transformers.DPRQuestionEncoder(config=transformers.DPRConfig(**original_config_dict))
                language_model_class = cls.get_language_model_class(haystack_lm_config, use_auth_token, **kwargs)
                dpr_question_encoder.model.base_model.bert_model = cls.subclasses[language_model_class].load(str(pretrained_model_name_or_path)).model
            dpr_question_encoder.language = dpr_question_encoder.model.config.language
        else:
            original_model_config = AutoConfig.from_pretrained(pretrained_model_name_or_path, use_auth_token=use_auth_token)
            if original_model_config.model_type == "dpr":
                # "pretrained dpr model": load existing pretrained DPRQuestionEncoder model
                dpr_question_encoder.model = transformers.DPRQuestionEncoder.from_pretrained(
                    str(pretrained_model_name_or_path), use_auth_token=use_auth_token, **kwargs)
            else:
                # "from scratch": load weights from different architecture (e.g. bert) into DPRQuestionEncoder
                # but keep config values from original architecture
                # TODO test for architectures other than BERT, e.g. Electra
                if original_model_config.model_type != "bert":
                    logger.warning(f"Using a model of type '{original_model_config.model_type}' which might be incompatible with DPR encoders."
                                   f"Bert based encoders are supported that need input_ids,token_type_ids,attention_mask as input tensors.")
                original_config_dict = vars(original_model_config)
                original_config_dict.update(kwargs)
                dpr_question_encoder.model = transformers.DPRQuestionEncoder(config=transformers.DPRConfig(**original_config_dict))
                dpr_question_encoder.model.base_model.bert_model = AutoModel.from_pretrained(
                    str(pretrained_model_name_or_path), use_auth_token=use_auth_token, **original_config_dict)
            dpr_question_encoder.language = cls._get_or_infer_language_from_name(language, pretrained_model_name_or_path)

        return dpr_question_encoder

    def save(self, save_dir: Union[str, Path], state_dict: Optional[Dict[Any, Any]] = None):
        """
        Save the model state_dict and its config file so that it can be loaded again.

        :param save_dir: The directory in which the model should be saved.
        :param state_dict: A dictionary containing a whole state of the module including names of layers. 
                           By default, the unchanged state dict of the module is used
        """
        model_to_save = (
            self.model.module if hasattr(self.model, "module") else self.model
        )  # Only save the model itself

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

        super(DPRQuestionEncoder, self).save(save_dir=save_dir, state_dict=state_dict)

    def forward(  # type: ignore
        self,
        query_input_ids: torch.Tensor,
        query_segment_ids: torch.Tensor,
        query_attention_mask: torch.Tensor,
        **kwargs,
    ):
        """
        Perform the forward pass of the DPRQuestionEncoder model.

        :param query_input_ids: The ids of each token in the input sequence. Is a tensor of shape [batch_size, max_seq_len]
        :param query_segment_ids: The id of the segment. For example, in next sentence prediction, the tokens in the
           first sentence are marked with 0 and those in the second are marked with 1.
           It is a tensor of shape [batch_size, max_seq_len]
        :param query_attention_mask: A mask that assigns a 1 to valid input tokens and 0 to padding tokens
           of shape [batch_size, max_seq_len]
        :return: Embeddings for each token in the input sequence.
        """
        output_tuple = self.model(
            input_ids=query_input_ids,
            token_type_ids=query_segment_ids,
            attention_mask=query_attention_mask,
            return_dict=True
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
    A DPRContextEncoder model that wraps HuggingFace's implementation
    """
    def __init__(self):
        super(DPRContextEncoder, self).__init__()
        self.model = None
        self.name = "dpr_context_encoder"

    @classmethod
    @silence_transformers_logs
    def load(cls, pretrained_model_name_or_path: Union[Path, str], language: str = None, use_auth_token: Union[str,bool] = None, **kwargs):
        """
        Load a pretrained model by supplying

        * the name of a remote model on s3 ("facebook/dpr-ctx_encoder-single-nq-base" ...)
        * OR a local path of a model trained via transformers ("some_dir/huggingface_model")
        * OR a local path of a model trained via Haystack ("some_dir/haystack_model")

        :param pretrained_model_name_or_path: The path of the base pretrained language model whose weights are used to initialize DPRContextEncoder
        """
        dpr_context_encoder = cls()
        if "haystack_lm_name" in kwargs:
            dpr_context_encoder.name = kwargs["haystack_lm_name"]
        else:
            dpr_context_encoder.name = pretrained_model_name_or_path
        # We need to differentiate between loading model using Haystack format and Pytorch-Transformers format
        haystack_lm_config = Path(pretrained_model_name_or_path) / "language_model_config.json"

        if os.path.exists(haystack_lm_config):
            # Haystack style
            original_model_config = AutoConfig.from_pretrained(haystack_lm_config)
            haystack_lm_model = Path(pretrained_model_name_or_path) / "language_model.bin"

            if original_model_config.model_type == "dpr":
                dpr_config = transformers.DPRConfig.from_pretrained(haystack_lm_config)
                dpr_context_encoder.model = transformers.DPRContextEncoder.from_pretrained(haystack_lm_model,config=dpr_config, use_auth_token=use_auth_token, **kwargs)
            else:
                if original_model_config.model_type != "bert":
                    logger.warning(
                        f"Using a model of type '{original_model_config.model_type}' which might be incompatible with DPR encoders."
                        f"Bert based encoders are supported that need input_ids,token_type_ids,attention_mask as input tensors.")
                original_config_dict = vars(original_model_config)
                original_config_dict.update(kwargs)
                dpr_context_encoder.model = transformers.DPRContextEncoder(config=transformers.DPRConfig(**original_config_dict))
                language_model_class = cls.get_language_model_class(haystack_lm_config, **kwargs)
                dpr_context_encoder.model.base_model.bert_model = cls.subclasses[language_model_class].load(
                    str(pretrained_model_name_or_path), use_auth_token=use_auth_token).model
            dpr_context_encoder.language = dpr_context_encoder.model.config.language

        else:
            # Pytorch-transformer Style
            original_model_config = AutoConfig.from_pretrained(pretrained_model_name_or_path, use_auth_token=use_auth_token)
            if original_model_config.model_type == "dpr":
                # "pretrained dpr model": load existing pretrained DPRContextEncoder model
                dpr_context_encoder.model = transformers.DPRContextEncoder.from_pretrained(
                    str(pretrained_model_name_or_path), use_auth_token=use_auth_token, **kwargs)
            else:
                # "from scratch": load weights from different architecture (e.g. bert) into DPRContextEncoder
                # but keep config values from original architecture
                # TODO test for architectures other than BERT, e.g. Electra
                if original_model_config.model_type != "bert":
                    logger.warning(
                        f"Using a model of type '{original_model_config.model_type}' which might be incompatible with DPR encoders."
                        f"Bert based encoders are supported that need input_ids,token_type_ids,attention_mask as input tensors.")
                original_config_dict = vars(original_model_config)
                original_config_dict.update(kwargs)
                dpr_context_encoder.model = transformers.DPRContextEncoder(
                    config=transformers.DPRConfig(**original_config_dict))
                dpr_context_encoder.model.base_model.bert_model = AutoModel.from_pretrained(
                    str(pretrained_model_name_or_path), use_auth_token=use_auth_token, **original_config_dict)
            dpr_context_encoder.language = cls._get_or_infer_language_from_name(language, pretrained_model_name_or_path)

        return dpr_context_encoder

    def save(self, save_dir: Union[str, Path], state_dict: Optional[Dict[Any, Any]] = None):
        """
        Save the model state_dict and its config file so that it can be loaded again.

        :param save_dir: The directory in which the model should be saved.
        :param state_dict: A dictionary containing a whole state of the module including names of layers. By default, the unchanged state dict of the module is used
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

        super(DPRContextEncoder, self).save(save_dir=save_dir, state_dict=state_dict)

    def forward(  # type: ignore
        self,
        passage_input_ids: torch.Tensor,
        passage_segment_ids: torch.Tensor,
        passage_attention_mask: torch.Tensor,
        **kwargs,
    ):
        """
        Perform the forward pass of the DPRContextEncoder model.

        :param passage_input_ids: The ids of each token in the input sequence. Is a tensor of shape [batch_size, number_of_hard_negative_passages, max_seq_len]
        :param passage_segment_ids: The id of the segment. For example, in next sentence prediction, the tokens in the
           first sentence are marked with 0 and those in the second are marked with 1.
           It is a tensor of shape [batch_size, number_of_hard_negative_passages, max_seq_len]
        :param passage_attention_mask: A mask that assigns a 1 to valid input tokens and 0 to padding tokens
           of shape [batch_size,  number_of_hard_negative_passages, max_seq_len]
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
            return_dict=True
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


class BigBird(LanguageModel):
    """
    A BERT model that wraps HuggingFace's implementation
    (https://github.com/huggingface/transformers) to fit the LanguageModel class.
    Paper: https://arxiv.org/abs/1810.04805
    """
    def __init__(self):
        super(BigBird, self).__init__()
        self.model = None
        self.name = "big_bird"

    @classmethod
    def from_scratch(cls, vocab_size, name="big_bird", language="en"):
        big_bird = cls()
        big_bird.name = name
        big_bird.language = language
        config = BigBirdConfig(vocab_size=vocab_size)
        big_bird.model = BigBirdModel(config)
        return big_bird

    @classmethod
    @silence_transformers_logs
    def load(cls, pretrained_model_name_or_path: Union[Path, str], language: str = None, **kwargs):
        """
        Load a pretrained model by supplying

        * the name of a remote model on s3 ("bert-base-cased" ...)
        * OR a local path of a model trained via transformers ("some_dir/huggingface_model")
        * OR a local path of a model trained via Haystack ("some_dir/haystack_model")

        :param pretrained_model_name_or_path: The path of the saved pretrained model or its name.
        """
        big_bird = cls()
        if "haystack_lm_name" in kwargs:
            big_bird.name = kwargs["haystack_lm_name"]
        else:
            big_bird.name = pretrained_model_name_or_path
        # We need to differentiate between loading model using Haystack format and Pytorch-Transformers format
        haystack_lm_config = Path(pretrained_model_name_or_path) / "language_model_config.json"
        if os.path.exists(haystack_lm_config):
            # Haystack style
            big_bird_config = BigBirdConfig.from_pretrained(haystack_lm_config)
            haystack_lm_model = Path(pretrained_model_name_or_path) / "language_model.bin"
            big_bird.model = BigBirdModel.from_pretrained(haystack_lm_model, config=big_bird_config, **kwargs)
            big_bird.language = big_bird.model.config.language
        else:
            # Pytorch-transformer Style
            big_bird.model = BigBirdModel.from_pretrained(str(pretrained_model_name_or_path), **kwargs)
            big_bird.language = cls._get_or_infer_language_from_name(language, pretrained_model_name_or_path)
        return big_bird

    def forward(
        self,
        input_ids: torch.Tensor,
        segment_ids: torch.Tensor,
        padding_mask: torch.Tensor,
        **kwargs,
    ):
        """
        Perform the forward pass of the BERT model.

        :param input_ids: The ids of each token in the input sequence. Is a tensor of shape [batch_size, max_seq_len]
        :param segment_ids: The id of the segment. For example, in next sentence prediction, the tokens in the
           first sentence are marked with 0 and those in the second are marked with 1.
           It is a tensor of shape [batch_size, max_seq_len]
        :param padding_mask: A mask that assigns a 1 to valid input tokens and 0 to padding tokens
           of shape [batch_size, max_seq_len]
        :return: Embeddings for each token in the input sequence.
        """
        output_tuple = self.model(
            input_ids,
            token_type_ids=segment_ids,
            attention_mask=padding_mask,
        )
        if self.model.encoder.config.output_hidden_states == True:
            sequence_output, pooled_output, all_hidden_states = output_tuple[0], output_tuple[1], output_tuple[2]
            return sequence_output, pooled_output, all_hidden_states
        else:
            sequence_output, pooled_output = output_tuple[0], output_tuple[1]
            return sequence_output, pooled_output

    def enable_hidden_states_output(self):
        self.model.encoder.config.output_hidden_states = True

    def disable_hidden_states_output(self):
        self.model.encoder.config.output_hidden_states = False
