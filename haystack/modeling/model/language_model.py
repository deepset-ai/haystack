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

from typing import Type, Optional, Dict, Any, Union, List
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal  # type: ignore

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

from haystack.errors import ModelingError

LANGUAGE_HINTS = (
    ("german", "german"), 
    ("english", "english"), 
    ("chinese", "chinese"), 
    ("indian", "indian"), 
    ("french", "french"), 
    ("camembert", "french"), 
    ("polish", "polish"), 
    ("spanish", "spanish"), 
    ("umberto", "italian"),
    ("multilingual", "multilingual"),
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

    def __init__(self, name: str):
        super().__init__()
        self._output_dims = None 
        self.name = name

    @property
    def encoder(self):
        return self.model.encoder

    @abstractmethod
    def forward(
        self, 
        input_ids: torch.Tensor, 
        segment_ids: torch.Tensor, 
        padding_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None, 
        output_attentions: Optional[bool] = None
    ):
        raise NotImplementedError

    @property
    def output_hidden_states(self):
        """
        Controls whether the model outputs the hidden states or not
        """
        self.encoder.config.output_hidden_states = True

    @output_hidden_states.setter
    def output_hidden_states(self, value: bool):
        """
        Sets the model to output the hidden states or not
        """
        self.encoder.config.output_hidden_states = value

    @property
    def output_dims(self):
        """
        The output dimension of this language model
        """
        if self._output_dims:
            return self._output_dims

        for odn in OUTPUT_DIM_NAMES:
            try:
                value = getattr(self.model.config, odn, None)
                if value:
                    self._output_dims = value
                    return value
            except AttributeError as e:
                raise ModelingError("Can't get the output dimension before loading the model.")

        raise ModelingError("Could not infer the output dimensions of the language model.")

    def save_config(self, save_dir: Union[Path, str]):
        """
        Save the configuration of the language model in Haystack format.
        """
        save_filename = Path(save_dir) / "language_model_config.json"
        setattr(self.model.config, "name", self.name)
        setattr(self.model.config, "language", self.language)

        string = self.model.config.to_json_string()
        with open(save_filename, "w") as file:
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

    def formatted_preds(
        self, 
        logits, 
        samples, 
        ignore_first_token: bool = True, 
        padding_mask: torch.Tensor = None
    ) -> List[Dict[str, Any]]:
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
            raise ModelingError(
                "`extraction_layer` or `extraction_strategy` not specified for LM. "
                "Make sure to set both, e.g. via Inferencer(extraction_strategy='cls_token', extraction_layer=-1)`"
            )

        # unpack the tuple from LM forward pass
        sequence_output = logits[0][0]
        pooled_output = logits[0][1]

        # aggregate vectors
        if self.extraction_strategy == "pooled":
            if self.extraction_layer != -1:
                raise ModelingError(
                    f"Pooled output only works for the last layer, but got extraction_layer={self.extraction_layer}. "
                    "Please set `extraction_layer=-1`"
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
            raise NotImplementedError(f"This extraction strategy ({self.extraction_strategy}) is not supported by Haystack.")

        preds = []
        for vec, sample in zip(vecs, samples):
            pred = {}
            pred["context"] = sample.clear_text["text"]
            pred["vec"] = vec
            preds.append(pred)
        return preds

    def _pool_tokens(self, sequence_output: torch.Tensor, padding_mask: torch.Tensor, strategy: str, ignore_first_token: bool):
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
    def __init__(
        self, 
        pretrained_model_name_or_path: Union[Path, str], 
        model_type: str,
        language: str = None, 
        n_added_tokens: int = 0, 
        auth_token: Optional[str] = None, 
        transformers_args: Optional[Dict[str, Any]] = None
    ):
        """
        Load a pretrained model by supplying one of the following:

        * The name of a remote model on s3 (for example, "bert-base-cased").
        * A local path of a model trained using transformers (for example, "some_dir/huggingface_model").
        * A local path of a model trained using Haystack (for example, "some_dir/haystack_model").

        You can also use `get_language_model()` for a uniform interface across different model types.

        :param pretrained_model_name_or_path: The path of the saved pretrained model or the name of the model.
        :param model_type: the HuggingFace class name prefix (for example 'Bert', 'Roberta', etc...)
        :param language: the model's language ('multilingual' is also accepted)
        :param auth_token: the HF token, if necessary
        """
        super().__init__(name=model_type)
        
        config_class: PretrainedConfig = getattr(transformers, model_type + "Config", None)
        model_class: PreTrainedModel = getattr(transformers, model_type + "Model", None)

        # We need to differentiate between loading model using Haystack format and Pytorch-Transformers format
        haystack_lm_config = Path(pretrained_model_name_or_path) / "language_model_config.json"
        if os.path.exists(haystack_lm_config):
            # Haystack style
            haystack_lm_model = Path(pretrained_model_name_or_path) / "language_model.bin"
            model_config = config_class.from_pretrained(haystack_lm_config)
            self.model = model_class.from_pretrained(haystack_lm_model, config=model_config, use_auth_token=auth_token or False, **(transformers_args or {}))
            self.language = self.model.config.language
        else:
            # Pytorch-transformer Style
            self.model = model_class.from_pretrained(str(pretrained_model_name_or_path), use_auth_token=auth_token or False, **(transformers_args or {}))
            self.language = language or _guess_language(pretrained_model_name_or_path)
        
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

    def forward(
        self,
        input_ids: torch.Tensor,
        segment_ids: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None
    ):
        """
        Perform the forward pass of the model.

        :param input_ids: The IDs of each token in the input sequence. It's a tensor of shape [batch_size, max_seq_len].
        :param segment_ids: The ID of the segment. For example, in next sentence prediction, the tokens in the
           first sentence are marked with 0 and the tokens in the second sentence are marked with 1.
           It is a tensor of shape [batch_size, max_seq_len].
        :param padding_mask/attention_mask: A mask that assigns 1 to valid input tokens and 0 to padding tokens
           of shape [batch_size, max_seq_len]. Different models call this parameter differently (padding/attention mask).
        :param output_hidden_states: When set to `True`, outputs hidden states in addition to the embeddings.
        :param output_attentions: When set to `True`, outputs attentions in addition to the embeddings.
        :return: Embeddings for each token in the input sequence. Can also return hidden states and attentions if specified using the arguments `output_hidden_states` and `output_attentions`.
        """
        mask = {}
        if padding_mask is not None:
            mask["padding_mask"] = padding_mask
        else:
            mask["attention_mask"] = attention_mask
        return self.model(
            input_ids,
            token_type_ids=segment_ids,
            output_hidden_states=output_hidden_states or self.encoder.config.output_hidden_states,
            output_attentions=output_attentions or self.encoder.config.output_attentions,
            return_dict=False,
            **mask
        )


class HFLanguageModelWithPooler(HFLanguageModel):
    """
    A model that wraps Hugging Face's implementation
    (https://github.com/huggingface/transformers) to fit the LanguageModel class,
    with an extra pooler.

    NOTE:
    - Unlike the other BERT variants, these don't output the `pooled_output`. An additional pooler is initialized.
    """

    def __init__(self, pretrained_model_name_or_path: Union[Path, str], language: str = None, n_added_tokens: int = 0, transformers_args: Optional[Dict[str, Any]] = None):
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
        sequence_summary_config = PARAMETERS_BY_MODEL.get(self.name.lower())
        for key, value in sequence_summary_config.items():
            setattr(config, key, value)

        self.pooler = SequenceSummary(config)
        self.pooler.apply(self.model._init_weights)

    def forward(  # type: ignore
        self,
        input_ids: torch.Tensor,
        segment_ids: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        **kwargs,
    ):
        """
        Perform the forward pass of the model.

        :param input_ids: The IDs of each token in the input sequence. It's a tensor of shape [batch_size, max_seq_len].
        :param padding_mask/attention_mask: A mask that assigns 1 to valid input tokens and 0 to padding tokens
           of shape [batch_size, max_seq_len]. Different models call this parameter differently (padding/attention mask).
        :param output_hidden_states: When set to `True`, outputs hidden states in addition to the embeddings.
        :param output_attentions: When set to `True`, outputs attentions in addition to the embeddings.
        :return: Embeddings for each token in the input sequence.
        """

        output_tuple = super().forward(
            input_ids=input_ids,
            segment_ids=segment_ids,
            padding_mask=padding_mask,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            **kwargs,
        )
        pooled_output = self.pooler(output_tuple[0])
        return (output_tuple[0], pooled_output) + output_tuple[1:]


class DPREncoder(LanguageModel):
    """
    A DPREncoder model that wraps Hugging Face's implementation.
    """
    @silence_transformers_logs
    def __init__(
        self,
        pretrained_model_name_or_path: Union[Path, str],
        model_type: str,
        language: str = None,
        auth_token: Optional[str] = None,
        transformers_kwargs: Optional[Dict[str, Any]] = None
    ):
        """
        Load a pretrained model by supplying one of the following:
        * The name of a remote model on s3 (for example, "facebook/dpr-question_encoder-single-nq-base").
        * A local path of a model trained using transformers (for example, "some_dir/huggingface_model").
        * A local path of a model trained using Haystack (for example, "some_dir/haystack_model").
        :param pretrained_model_name_or_path: The path of the base pretrained language model whose weights are used to initialize DPRQuestionEncoder.
        """
        super().__init__(name=model_type)
        self.role = "question" if "question" in model_type.lower() else "context"
        self._encoder = None

        kwargs = transformers_kwargs or {}
        model_classname = f"DPR{self.role.capitalize()}Encoder"
        model_class: Type[PreTrainedModel] = getattr(transformers, model_classname, None)

        # We need to differentiate between loading model using Haystack format and Pytorch-Transformers format
        haystack_lm_config = Path(pretrained_model_name_or_path) / "language_model_config.json"
        if os.path.exists(haystack_lm_config):
            # Haystack style
            original_model_config = AutoConfig.from_pretrained(haystack_lm_config)
            haystack_lm_model = Path(pretrained_model_name_or_path) / "language_model.bin"

            if original_model_config.model_type == "dpr":
                dpr_config = transformers.DPRConfig.from_pretrained(haystack_lm_config)
                self.model = model_class.from_pretrained(
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
                self.model = model_class(
                    config=transformers.DPRConfig(**original_config_dict)
                )

                language_model_type = _get_model_type(haystack_lm_config, auth_token=auth_token, **kwargs)
                # Find the class corresponding to this model type
                language_model_class: Type[LanguageModel] = HUGGINGFACE_TO_HAYSTACK.get(language_model_type, None)
                if not language_model_class:
                    raise ValueError(
                        f"The type of model supplied ({language_model_type}) is not supported by Haystack. "
                        f"Supported model categories are: {', '.join(HUGGINGFACE_TO_HAYSTACK.keys())}")

                # Instantiate the class for this model
                self.model.base_model.bert_model = language_model_class(
                    pretrained_model_name_or_path,
                    model_type=language_model_type,
                    **kwargs
                ).model

            self.language = self.model.config.language
        else:
            original_model_config = AutoConfig.from_pretrained(
                pretrained_model_name_or_path, use_auth_token=auth_token or False
            )
            if original_model_config.model_type == "dpr":
                # "pretrained dpr model": load existing pretrained DPRQuestionEncoder model
                self.model = model_class.from_pretrained(
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
                self.model = model_class(
                    config=transformers.DPRConfig(**original_config_dict)
                )
                self.model.base_model.bert_model = AutoModel.from_pretrained(
                    str(pretrained_model_name_or_path), use_auth_token=auth_token or False, **original_config_dict
                )
            self.language = language or _guess_language(pretrained_model_name_or_path)

    @property
    def encoder(self):
        if not self._encoder:
            self._encoder = self.model.question_encoder if self.role == "question" else self.model.ctx_encoder
        return self._encoder

    def save_config(self, save_dir: Union[Path, str]):
        """
        Save the configuration of the language model in Haystack format.
        """
        # For DPR models, transformers overwrites the model_type with the one set in DPRConfig
        # Therefore, we copy the model_type from the model config to DPRConfig
        setattr(transformers.DPRConfig, "model_type", self.model.config.model_type)
        super().save_config(save_dir=save_dir)
        
    def save(self, save_dir: Union[str, Path], state_dict: Optional[Dict[Any, Any]] = None):
        """
        Save the model `state_dict` and its configuration file so that it can be loaded again.

        :param save_dir: The directory in which the model should be saved.
        :param state_dict: A dictionary containing the whole state of the module including names of layers. By default, the unchanged state dictionary of the module is used.
        """
        model_to_save = (
            self.model.module if hasattr(self.model, "module") else self.model
        )  # Only save the model itself

        if "dpr" not in self.model.config.model_type.lower():
            if model_to_save.base_model_prefix.startswith("ctx_"):
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

            elif model_to_save.base_model_prefix.startswith("question_"):
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
        input_ids: torch.Tensor,
        segment_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ):
        """
        Perform the forward pass of the DPR encoder model.

        :param input_ids: The IDs of each token in the input sequence. It's a tensor of shape [batch_size, number_of_hard_negative, max_seq_len].
        :param segment_ids: The ID of the segment. For example, in next sentence prediction, the tokens in the
           first sentence are marked with 0 and the tokens in the second sentence are marked with 1.
           It is a tensor of shape [batch_size, number_of_hard_negative_passages, max_seq_len].
        :param attention_mask: A mask that assigns 1 to valid input tokens and 0 to padding tokens
           of shape [batch_size,  number_of_hard_negative_passages, max_seq_len].
        :return: Embeddings for each token in the input sequence.
        """
        if not self.role == "question":
            max_seq_len = input_ids.shape[-1]
            input_ids = input_ids.view(-1, max_seq_len)
            segment_ids = segment_ids.view(-1, max_seq_len)
            attention_mask = attention_mask.view(-1, max_seq_len)

        output_tuple = self.model(
            input_ids=input_ids,
            token_type_ids=segment_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        if self.encoder.config.output_hidden_states == True:
            pooled_output, all_hidden_states = output_tuple.pooler_output, output_tuple.hidden_states
            return pooled_output, all_hidden_states
        else:
            pooled_output = output_tuple.pooler_output
            return pooled_output, None


HUGGINGFACE_TO_HAYSTACK = {
    "Albert": HFLanguageModel,
    "Bert": HFLanguageModel,
    "BigBird": HFLanguageModel,
    "Camembert": HFLanguageModel,
    "Codebert": HFLanguageModel,
    "Data2VecVision": HFLanguageModel,
    "DebertaV2": HFLanguageModelWithPooler,
    "DistilBert": HFLanguageModelWithPooler,
    "DPRContextEncoder": DPREncoder,
    "DPRQuestionEncoder": DPREncoder,
    "Electra": HFLanguageModelWithPooler,
    "GloVe": HFLanguageModel,
    "MiniLM": HFLanguageModel,
    "Roberta": HFLanguageModel,
    "Umberto": HFLanguageModel,
    "Word2Vec": HFLanguageModel,
    "WordEmbedding_LM": HFLanguageModel,
    "XLMRoberta": HFLanguageModel,
    "XLNet": HFLanguageModelWithPooler,
    
}
NAME_HINTS = {
    "xlm.*roberta": "XLMRoberta",
    "roberta.*xml": "XLMRoberta",
    "codebert.*mlm": "Roberta",
    "mlm.*codebert": "Roberta",
    "dpr.*question.*encoder": "DPRQuestionEncoder",
    "dpr.*context.*encoder": "DPRContextEncoder",
    "dpr.*ctx.*encoder": "DPRContextEncoder",
    "mlm.*codebert": "Roberta",
    "deberta-v2": "DebertaV2",
    "data2vec-vision": "Data2VecVision",
}
PARAMETERS_BY_MODEL = {
    "DistilBert": {"summary_last_dropout": 0, "summary_type": "first", "summary_activation": "tanh"},
    "XLNet": {"summary_last_dropout": 0},
    "Electra": {
        "summary_last_dropout": 0,
        "summary_type": "first",
        "summary_activation": "gelu",
        "summary_use_proj": False,
    },
    "DebertaV2": {
        "summary_last_dropout": 0,
        "summary_type": "first",
        "summary_activati": "tanh",
        "summary_use_proj": False,
    },
}

def get_language_model(
    pretrained_model_name_or_path: Union[Path, str], 
    language_model_type: Optional[str] = None, 
    auth_token: Optional[str] = None, 
    revision: Optional[str] = None, 
    autoconfig_kwargs: Optional[Dict[str, Any]] = None,
    model_kwargs: Optional[Dict[str, Any]] = None
) -> LanguageModel:
    """
    Load a pretrained language model by doing one of the following:

    1. Specifying its name and downloading the model.
    2. Pointing to the directory the model is saved in.

    See all supported model variations at: https://huggingface.co/models.

    The appropriate language model class is inferred automatically from model configuration
    or can be manually supplied using `language_model_class`.

    :param pretrained_model_name_or_path: The path of the saved pretrained model or its name.
    :param revision: The version of the model to use from the Hugging Face model hub. This can be a tag name, a branch name, or a commit hash.
    :param language_model_type: (Optional) Name of the language model class to load (for example `Bert`). Overrides any other discovered value.
    """
    logger.info(f" * LOADING MODEL: '{pretrained_model_name_or_path}'")

    config_file = Path(pretrained_model_name_or_path) / "language_model_config.json"

    if language_model_type is None:

        if os.path.exists(config_file):
            # it's a local directory in Haystack format
            logger.info(f"Model found locally at {pretrained_model_name_or_path}")
            config = json.load(open(config_file))
            language_model_type = config["name"]

        else:
            # It's from the model hub
            logger.info(f"Could not find '{pretrained_model_name_or_path}' locally.")
            logger.info(f"Looking on Transformers Model Hub (in local cache and online)...")
            language_model_type = _get_model_type(
                pretrained_model_name_or_path, auth_token=auth_token, revision=revision, autoconfig_kwargs=autoconfig_kwargs
            )
            if  not language_model_type:
                raise Exception(
                    f"Model not found for '{pretrained_model_name_or_path}'. Either supply the local path for a saved "
                    f"model or one of bert/roberta/xlnet/albert/distilbert models that can be downloaded from remote. "
                    f"Ensure that the model class name can be inferred from the directory name when loading a "
                    f"Transformers' model."
                )

    # Find the class corresponding to this model type
    language_model_class: Type[LanguageModel] = HUGGINGFACE_TO_HAYSTACK.get(language_model_type, None)
    if not language_model_class:
        raise ValueError(
            f"The type of model supplied ({language_model_type}) is not supported by Haystack. "
            f"Supported model categories are: {', '.join(HUGGINGFACE_TO_HAYSTACK.keys())}")

    # Instantiate the class for this model
    language_model = language_model_class(
        pretrained_model_name_or_path,
        model_type=language_model_type,
        auth_token=auth_token,
        transformers_args=model_kwargs
    )
    logger.info(f"Loaded '{pretrained_model_name_or_path}' ({language_model_type} model)")
    return language_model


def _get_model_type(
    model_name_or_path: Union[str, Path], 
    auth_token: Optional[str] = None, 
    revision: Optional[str] = None, 
    autoconfig_kwargs: Optional[Dict[str, Any]] = None
) -> str:
    """
    Given a model name, try to use AutoConfig to understand which model type it is.
    In case it's not successful, tries to infer the type from the name of the model.
    """
    model_name_or_path = str(model_name_or_path)

    if autoconfig_kwargs and "use_auth_token" in autoconfig_kwargs:
        auth_token = autoconfig_kwargs["use_auth_token"]
        del autoconfig_kwargs["use_auth_token"]

    model_type: Optional[Type[LanguageModel]] = None
    # Use AutoConfig to understand the model class
    try:
        config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path=model_name_or_path, 
            use_auth_token=auth_token or False, 
            revision=revision, 
            **(autoconfig_kwargs or {})
        )
        # Find if this mode is present in MODEL_TYPE_BY_NAME.keys() even with a different capitalization
        model_type = {key.lower(): key for key in HUGGINGFACE_TO_HAYSTACK.keys()}.get(config.model_type.lower(), None)

    except Exception as e:
        logger.exception(
            f"AutoConfig failed to load on '{model_name_or_path}'. "
        )

    if not model_type:
        logger.warning("Could not infer the model type from its config. Looking for clues in the model name.")

        # Look for other patterns and variation that hints at the model type
        for regex, model_name in NAME_HINTS.items():
            if re.match(f".*{regex}.*", model_name_or_path):
                model_type = model_name
                break

    if model_type and model_type.lower() == "roberta" and "mlm" in model_name_or_path.lower():
        logging.error(f"MLM part of codebert is currently not supported in Haystack: '{model_name_or_path}' may crash later.")

    return model_type


def _guess_language(name: str) -> str:
    """
    Looks for clues about the model language in the model name.
    """
    languages = [lang for hint, lang in LANGUAGE_HINTS if hint.lower() in name.lower()]
    if len(languages) > 0:
        language = languages[0]
    else:
        language = "english"
    logger.info(f"Auto-detected model language: {language}")
    return language