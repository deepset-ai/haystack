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

import re
import json
import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np
import torch
from torch import nn
import transformers
from transformers import PretrainedConfig, PreTrainedModel
from transformers import AutoModel, AutoConfig
from transformers.modeling_utils import SequenceSummary

from haystack.errors import ModelingError
from haystack.modeling.utils import silence_transformers_logs


logger = logging.getLogger(__name__)


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

#: Names of the attributes in various model configs which refer to the number of dimensions in the output vectors
OUTPUT_DIM_NAMES = ["dim", "hidden_size", "d_model"]


# TODO analyse if LMs can be completely used through HF transformers
class LanguageModel(nn.Module, ABC):
    """
    The parent class for any kind of model that can embed language into a semantic vector space.
    These models read in tokenized sentences and return vectors that capture the meaning of sentences or of tokens.
    """

    def __init__(self, model_type: str):
        super().__init__()
        self._output_dims = None
        self.name = model_type

    @property
    def encoder(self):
        return self.model.encoder

    @abstractmethod
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        segment_ids: Optional[torch.Tensor],  # DistilBERT does not use them, see DistilBERTLanguageModel
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: bool = False,
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
            except AttributeError:
                raise ModelingError("Can't get the output dimension before loading the model.")

        raise ModelingError("Could not infer the output dimensions of the language model.")

    def save_config(self, save_dir: Union[Path, str]):
        """
        Save the configuration of the language model in Haystack format.
        """
        save_filename = Path(save_dir) / "language_model_config.json"
        setattr(self.model.config, "name", self.name)  # type: ignore [union-attr]
        setattr(self.model.config, "language", self.language)  # type: ignore [union-attr]

        string = self.model.config.to_json_string()  # type: ignore [union-attr,operator]
        with open(save_filename, "w") as file:
            file.write(string)

    def save(self, save_dir: Union[str, Path], state_dict: Optional[Dict[Any, Any]] = None):
        """
        Save the model `state_dict` and its configuration file so that it can be loaded again.

        :param save_dir: The directory in which the model should be saved.
        :param state_dict: A dictionary containing the whole state of the module, including names of layers. By default, the unchanged state dictionary of the module is used.
        """
        # Save Weights
        save_name = Path(save_dir) / "language_model.bin"
        model_to_save = self.model.module if hasattr(self.model, "module") else self.model  # Only save the model itself

        if not state_dict:
            state_dict = model_to_save.state_dict()  # type: ignore [union-attr]
        torch.save(state_dict, save_name)
        self.save_config(save_dir)

    def formatted_preds(
        self, logits, samples, ignore_first_token: bool = True, padding_mask: Optional[torch.Tensor] = None
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
                sequence_output, padding_mask, self.extraction_strategy, ignore_first_token=ignore_first_token  # type: ignore [arg-type]   # type: ignore [arg-type]
            )
        elif self.extraction_strategy == "reduce_max":
            vecs = self._pool_tokens(
                sequence_output, padding_mask, self.extraction_strategy, ignore_first_token=ignore_first_token  # type: ignore [arg-type]   # type: ignore [arg-type]
            )
        elif self.extraction_strategy == "cls_token":
            vecs = sequence_output[:, 0, :].cpu().numpy()
        else:
            raise NotImplementedError(
                f"This extraction strategy ({self.extraction_strategy}) is not supported by Haystack."
            )

        preds = []
        for vec, sample in zip(vecs, samples):
            pred = {}
            pred["context"] = sample.clear_text["text"]
            pred["vec"] = vec
            preds.append(pred)
        return preds

    def _pool_tokens(
        self, sequence_output: torch.Tensor, padding_mask: torch.Tensor, strategy: str, ignore_first_token: bool
    ):
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
        language: Optional[str] = None,
        n_added_tokens: int = 0,
        use_auth_token: Optional[Union[str, bool]] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
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
        :param use_auth_token: The API token used to download private models from Huggingface.
                               If this parameter is set to `True`, then the token generated when running
                               `transformers-cli login` (stored in ~/.huggingface) will be used.
                               Additional information can be found here
                               https://huggingface.co/transformers/main_classes/model.html#transformers.PreTrainedModel.from_pretrained
        """
        super().__init__(model_type=model_type)

        config_class: PretrainedConfig = getattr(transformers, model_type + "Config", None)
        model_class: PreTrainedModel = getattr(transformers, model_type + "Model", None)

        haystack_lm_config = Path(pretrained_model_name_or_path) / "language_model_config.json"
        if os.path.exists(haystack_lm_config):
            # Haystack style
            haystack_lm_model = Path(pretrained_model_name_or_path) / "language_model.bin"
            model_config = config_class.from_pretrained(haystack_lm_config, use_auth_token=use_auth_token)
            self.model = model_class.from_pretrained(
                haystack_lm_model, config=model_config, use_auth_token=use_auth_token, **(model_kwargs or {})
            )
            self.language = self.model.config.language
        else:
            # Pytorch-transformer Style
            self.model = model_class.from_pretrained(
                str(pretrained_model_name_or_path), use_auth_token=use_auth_token, **(model_kwargs or {})
            )
            self.language = language or _guess_language(str(pretrained_model_name_or_path))

        # resize embeddings in case of custom vocab
        if n_added_tokens != 0:
            # TODO verify for other models than BERT
            model_emb_size = self.model.resize_token_embeddings(new_num_tokens=None).num_embeddings
            vocab_size = model_emb_size + n_added_tokens
            logger.info(
                "Resizing embedding layer of LM from %s to %s to cope with custom vocab.", model_emb_size, vocab_size
            )
            self.model.resize_token_embeddings(vocab_size)
            # verify
            model_emb_size = self.model.resize_token_embeddings(new_num_tokens=None).num_embeddings
            assert vocab_size == model_emb_size

    def forward(  # type: ignore [override]
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        segment_ids: torch.Tensor,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: bool = False,
    ):
        """
        Perform the forward pass of the model.

        :param input_ids: The IDs of each token in the input sequence. It's a tensor of shape [batch_size, max_seq_len].
        :param segment_ids: The ID of the segment. For example, in next sentence prediction, the tokens in the
           first sentence are marked with 0 and the tokens in the second sentence are marked with 1.
           It is a tensor of shape [batch_size, max_seq_len].
        :param attention_mask: A mask that assigns 1 to valid input tokens and 0 to padding tokens
           of shape [batch_size, max_seq_len]. Different models call this parameter differently (padding/attention mask).
        :param output_hidden_states: When set to `True`, outputs hidden states in addition to the embeddings.
        :param output_attentions: When set to `True`, outputs attentions in addition to the embeddings.
        :return: Embeddings for each token in the input sequence. Can also return hidden states and attentions if specified using the arguments `output_hidden_states` and `output_attentions`.
        """
        if hasattr(self, "encoder"):  # Not all models have an encoder
            if output_hidden_states is None:
                output_hidden_states = self.model.encoder.config.output_hidden_states
            if output_attentions is None:
                output_attentions = self.model.encoder.config.output_attentions

        params = {}
        if input_ids is not None:
            params["input_ids"] = input_ids
        if segment_ids is not None:
            # Some models don't take this (see DistilBERT)
            params["token_type_ids"] = segment_ids
        if attention_mask is not None:
            params["attention_mask"] = attention_mask
        if output_hidden_states:
            params["output_hidden_states"] = output_hidden_states  # type: ignore [assignment]
        if output_attentions:
            params["output_attentions"] = output_attentions  # type: ignore [assignment]

        return self.model(**params, return_dict=return_dict)


class HFLanguageModelWithPooler(HFLanguageModel):
    """
    A model that wraps Hugging Face's implementation
    (https://github.com/huggingface/transformers) to fit the LanguageModel class,
    with an extra pooler.

    NOTE:
    - Unlike the other BERT variants, these don't output the `pooled_output`. An additional pooler is initialized.
    """

    def __init__(
        self,
        pretrained_model_name_or_path: Union[Path, str],
        model_type: str,
        language: Optional[str] = None,
        n_added_tokens: int = 0,
        use_auth_token: Optional[Union[str, bool]] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Load a pretrained model by supplying one of the following:

        * The name of a remote model on s3 (for example, "distilbert-base-german-cased")
        * A local path of a model trained using transformers (for example, "some_dir/huggingface_model")
        * A local path of a model trained using Haystack (for example, "some_dir/haystack_model")

        :param pretrained_model_name_or_path: The path of the saved pretrained model or its name.
        :param model_type: the HuggingFace class name prefix (for example 'DebertaV2', 'Electra', etc...)
        :param language: the model's language ('multilingual' is also accepted)
        :param use_auth_token: The API token used to download private models from Huggingface.
                               If this parameter is set to `True`, then the token generated when running
                               `transformers-cli login` (stored in ~/.huggingface) will be used.
                               Additional information can be found here
                               https://huggingface.co/transformers/main_classes/model.html#transformers.PreTrainedModel.from_pretrained
        """
        super().__init__(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            model_type=model_type,
            language=language,
            n_added_tokens=n_added_tokens,
            use_auth_token=use_auth_token,
            model_kwargs=model_kwargs,
        )
        config = self.model.config

        # These models do not provide a pooled_output by default. Therefore, we need to initialize an extra pooler.
        # The pooler takes the first hidden representation & feeds it to a dense layer of (hidden_dim x hidden_dim).
        # We don't want a dropout in the end of the pooler, since we do that already in the adaptive model before we
        # feed everything to the prediction head
        sequence_summary_config = POOLER_PARAMETERS.get(self.name.lower(), {})
        for key, value in sequence_summary_config.items():
            setattr(config, key, value)

        self.pooler = SequenceSummary(config)
        self.pooler.apply(self.model._init_weights)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        segment_ids: Optional[torch.Tensor],
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: bool = False,
    ):
        """
        Perform the forward pass of the model.

        :param input_ids: The IDs of each token in the input sequence. It's a tensor of shape [batch_size, max_seq_len].
        :param segment_ids: The ID of the segment. For example, in next sentence prediction, the tokens in the
           first sentence are marked with 0 and the tokens in the second sentence are marked with 1.
           It is a tensor of shape [batch_size, max_seq_len]. Optional, some models don't need it (DistilBERT for example)
        :param padding_mask/attention_mask: A mask that assigns 1 to valid input tokens and 0 to padding tokens
           of shape [batch_size, max_seq_len]. Different models call this parameter differently (padding/attention mask).
        :param output_hidden_states: When set to `True`, outputs hidden states in addition to the embeddings.
        :param output_attentions: When set to `True`, outputs attentions in addition to the embeddings.
        :return: Embeddings for each token in the input sequence.
        """
        output_tuple = super().forward(
            input_ids=input_ids,
            segment_ids=segment_ids,  # type: ignore [arg-type]
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=return_dict,
        )
        pooled_output = self.pooler(output_tuple[0])
        return (output_tuple[0], pooled_output) + output_tuple[1:]


class HFLanguageModelNoSegmentIds(HFLanguageModelWithPooler):
    """
    A model that wraps Hugging Face's implementation of a model that does not need segment ids.
    (https://github.com/huggingface/transformers) to fit the LanguageModel class.

    These are for now kept in a separate subclass to show a proper warning.
    """

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        segment_ids: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: bool = False,
    ):
        """
        Perform the forward pass of the model.

        :param input_ids: The IDs of each token in the input sequence. It's a tensor of shape [batch_size, max_seq_len].
        :param attention_mask: A mask that assigns 1 to valid input tokens and 0 to padding tokens
           of shape [batch_size, max_seq_len]. Different models call this parameter differently (padding/attention mask).
        :param segment_ids: Unused. See DistilBERT documentation.
        :param output_hidden_states: When set to `True`, outputs hidden states in addition to the embeddings.
        :param output_attentions: When set to `True`, outputs attentions in addition to the embeddings.
        :return: Embeddings for each token in the input sequence. Can also return hidden states and attentions if
            specified using the arguments `output_hidden_states` and `output_attentions`.
        """
        if segment_ids is not None:
            logger.warning("'segment_ids' is not None, but %s does not use them. They will be ignored.", self.name)

        return super().forward(
            input_ids=input_ids,
            segment_ids=None,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=return_dict,
        )


class DPREncoder(LanguageModel):
    """
    A DPREncoder model that wraps Hugging Face's implementation.
    """

    @silence_transformers_logs
    def __init__(
        self,
        pretrained_model_name_or_path: Union[Path, str],
        model_type: str,
        language: Optional[str] = None,
        n_added_tokens: int = 0,
        use_auth_token: Optional[Union[str, bool]] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Load a pretrained model by supplying one of the following:
        * The name of a remote model on s3 (for example, "facebook/dpr-question_encoder-single-nq-base").
        * A local path of a model trained using transformers (for example, "some_dir/huggingface_model").
        * A local path of a model trained using Haystack (for example, "some_dir/haystack_model").

        :param pretrained_model_name_or_path: The path of the base pretrained language model whose weights are used to initialize DPRQuestionEncoder.
        :param model_type: the type of model (see `HUGGINGFACE_TO_HAYSTACK`)
        :param language: the model's language. If not given, it will be inferred. Defaults to english.
        :param n_added_tokens: unused for `DPREncoder`
        :param use_auth_token: The API token used to download private models from Huggingface.
                               If this parameter is set to `True`, then the token generated when running
                               `transformers-cli login` (stored in ~/.huggingface) will be used.
                               Additional information can be found here
                               https://huggingface.co/transformers/main_classes/model.html#transformers.PreTrainedModel.from_pretrained
        :param model_kwargs: any kwarg to pass to the model at init
        """
        super().__init__(model_type=model_type)
        self.role = "question" if "question" in model_type.lower() else "context"
        self._encoder = None

        model_classname = f"DPR{self.role.capitalize()}Encoder"
        try:
            model_class: Type[PreTrainedModel] = getattr(transformers, model_classname)
        except AttributeError:
            raise ModelingError(f"Model class of type '{model_classname}' not found.")

        haystack_lm_config = Path(pretrained_model_name_or_path) / "language_model_config.json"
        if os.path.exists(haystack_lm_config):
            self._init_model_haystack_style(
                haystack_lm_config=haystack_lm_config,
                model_name_or_path=pretrained_model_name_or_path,
                model_class=model_class,
                model_kwargs=model_kwargs or {},
                use_auth_token=use_auth_token,
            )
        else:
            self._init_model_transformers_style(
                model_name_or_path=pretrained_model_name_or_path,
                model_class=model_class,
                model_kwargs=model_kwargs or {},
                use_auth_token=use_auth_token,
                language=language,
            )

    def _init_model_haystack_style(
        self,
        haystack_lm_config: Path,
        model_name_or_path: Union[str, Path],
        model_class: Type[PreTrainedModel],
        model_kwargs: Dict[str, Any],
        use_auth_token: Optional[Union[str, bool]] = None,
    ):
        """
        Init a Haystack-style DPR model.

        :param haystack_lm_config: path to the language model config file
        :param model_name_or_path: name or path of the model to load
        :param model_class: The HuggingFace model class name
        :param model_kwargs: any kwarg to pass to the model at init
        :param use_auth_token: The API token used to download private models from Huggingface.
                               If this parameter is set to `True`, then the token generated when running
                               `transformers-cli login` (stored in ~/.huggingface) will be used.
                               Additional information can be found here
                               https://huggingface.co/transformers/main_classes/model.html#transformers.PreTrainedModel.from_pretrained
        """
        original_model_config = AutoConfig.from_pretrained(haystack_lm_config)
        haystack_lm_model = Path(model_name_or_path) / "language_model.bin"

        original_model_type = original_model_config.model_type
        if original_model_type and "dpr" in original_model_type.lower():
            dpr_config = transformers.DPRConfig.from_pretrained(haystack_lm_config, use_auth_token=use_auth_token)
            self.model = model_class.from_pretrained(
                haystack_lm_model, config=dpr_config, use_auth_token=use_auth_token, **model_kwargs
            )

        else:
            self.model = self._init_model_through_config(
                model_config=original_model_config, model_class=model_class, model_kwargs=model_kwargs
            )
            original_model_type = capitalize_model_type(original_model_type)
            language_model_class = get_language_model_class(original_model_type)
            if not language_model_class:
                raise ValueError(
                    f"The type of model supplied ({model_name_or_path} , "
                    f"({original_model_type}) is not supported by Haystack. "
                    f"Supported model categories are: {', '.join(HUGGINGFACE_TO_HAYSTACK.keys())}"
                )
            # Instantiate the class for this model
            self.model.base_model.bert_model = language_model_class(
                pretrained_model_name_or_path=model_name_or_path,
                model_type=original_model_type,
                use_auth_token=use_auth_token,
                **model_kwargs,
            ).model

        self.language = self.model.config.language

    def _init_model_transformers_style(
        self,
        model_name_or_path: Union[str, Path],
        model_class: Type[PreTrainedModel],
        model_kwargs: Dict[str, Any],
        use_auth_token: Optional[Union[str, bool]] = None,
        language: Optional[str] = None,
    ):
        """
        Init a Transformers-style DPR model.

        :param model_name_or_path: name or path of the model to load
        :param model_class: The HuggingFace model class name
        :param model_kwargs: any kwarg to pass to the model at init
        :param use_auth_token: The API token used to download private models from Huggingface.
                               If this parameter is set to `True`, then the token generated when running
                               `transformers-cli login` (stored in ~/.huggingface) will be used.
                               Additional information can be found here
                               https://huggingface.co/transformers/main_classes/model.html#transformers.PreTrainedModel.from_pretrained
        :param language: the model's language. If not given, it will be inferred. Defaults to english.
        """
        original_model_config = AutoConfig.from_pretrained(model_name_or_path, use_auth_token=use_auth_token)
        if "dpr" in original_model_config.model_type.lower():
            # "pretrained dpr model": load existing pretrained DPRQuestionEncoder model
            self.model = model_class.from_pretrained(
                str(model_name_or_path), use_auth_token=use_auth_token, **model_kwargs
            )
        else:
            # "from scratch": load weights from different architecture (e.g. bert) into DPRQuestionEncoder
            # but keep config values from original architecture
            # TODO test for architectures other than BERT, e.g. Electra
            self.model = self._init_model_through_config(
                model_config=original_model_config, model_class=model_class, model_kwargs=model_kwargs
            )
            self.model.base_model.bert_model = AutoModel.from_pretrained(
                str(model_name_or_path), use_auth_token=use_auth_token, **vars(original_model_config)
            )
        self.language = language or _guess_language(str(model_name_or_path))

    def _init_model_through_config(
        self, model_config: AutoConfig, model_class: Type[PreTrainedModel], model_kwargs: Optional[Dict[str, Any]]
    ):
        """
        Init a DPR model using a config object.
        """
        if model_config.model_type.lower() != "bert":
            logger.warning(
                "Using a model of type '%s' which might be incompatible with DPR encoders. "
                "Only Bert-based encoders are supported. They need input_ids, token_type_ids, attention_mask as input tensors.",
                model_config.model_type,
            )
        config_dict = vars(model_config)
        if model_kwargs:
            config_dict.update(model_kwargs)
        return model_class(config=transformers.DPRConfig(**config_dict))

    @property
    def encoder(self):
        if not self._encoder:
            self._encoder = self.model.question_encoder if self.role == "question" else self.model.ctx_encoder
        return self._encoder

    def save_config(self, save_dir: Union[Path, str]) -> None:
        """
        Save the configuration of the language model in Haystack format.

        :param save_dir: the path to save the model at
        """
        # For DPR models, transformers overwrites the model_type with the one set in DPRConfig
        # Therefore, we copy the model_type from the model config to DPRConfig
        setattr(transformers.DPRConfig, "model_type", self.model.config.model_type)
        super().save_config(save_dir=save_dir)

    def save(self, save_dir: Union[str, Path], state_dict: Optional[Dict[Any, Any]] = None) -> None:
        """
        Save the model `state_dict` and its configuration file so that it can be loaded again.

        :param save_dir: The directory in which the model should be saved.
        :param state_dict: A dictionary containing the whole state of the module including names of layers.
                           By default, the unchanged state dictionary of the module is used.
        """
        model_to_save = self.model.module if hasattr(self.model, "module") else self.model  # Only save the model itself

        if "dpr" not in self.model.config.model_type.lower():
            prefix = "question" if self.role == "question" else "ctx"

            state_dict = model_to_save.state_dict()
            if state_dict:
                for key in list(state_dict.keys()):  # list() here performs a copy and allows editing the dict
                    new_key = key

                    if key.startswith(f"{prefix}_encoder.bert_model.model."):
                        new_key = key.split("_encoder.bert_model.model.", 1)[1]

                    elif key.startswith(f"{prefix}_encoder.bert_model."):
                        new_key = key.split("_encoder.bert_model.", 1)[1]

                    state_dict[new_key] = state_dict.pop(key)

        super().save(save_dir=save_dir, state_dict=state_dict)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        segment_ids: Optional[torch.Tensor],
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: bool = True,
    ):
        """
        Perform the forward pass of the DPR encoder model.

        :param input_ids: The IDs of each token in the input sequence. It's a tensor of shape [batch_size, number_of_hard_negative, max_seq_len].
        :param segment_ids: The ID of the segment. For example, in next sentence prediction, the tokens in the
           first sentence are marked with 0 and the tokens in the second sentence are marked with 1.
           It is a tensor of shape [batch_size, max_seq_len].
        :param attention_mask: A mask that assigns 1 to valid input tokens and 0 to padding tokens
           of shape [batch_size, max_seq_len].
        :param output_hidden_states: whether to add the hidden states along with the pooled output
        :param output_attentions: unused
        :return: Embeddings for each token in the input sequence.
        """
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.encoder.config.output_hidden_states
        )

        model_output = self.model(
            input_ids=input_ids,
            token_type_ids=segment_ids,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            output_attentions=False,
            return_dict=return_dict,
        )

        if output_hidden_states:
            return model_output.pooler_output, model_output.hidden_states
        return model_output.pooler_output, None


#: Match the name of the HuggingFace Model class to the corresponding Haystack wrapper
HUGGINGFACE_TO_HAYSTACK: Dict[str, Union[Type[HFLanguageModel], Type[DPREncoder]]] = {
    "Auto": HFLanguageModel,
    "Albert": HFLanguageModel,
    "Bert": HFLanguageModel,
    "BigBird": HFLanguageModel,
    "Camembert": HFLanguageModel,
    "Codebert": HFLanguageModel,
    "DebertaV2": HFLanguageModelWithPooler,
    "DistilBert": HFLanguageModelNoSegmentIds,
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
#: HF Capitalization pairs
HUGGINGFACE_CAPITALIZE = {
    "xlm-roberta": "XLMRoberta",
    "deberta-v2": "DebertaV2",
    **{k.lower(): k for k in HUGGINGFACE_TO_HAYSTACK.keys()},
}

#: Regex to match variants of the HF class name, to enhance our mode type guessing abilities.
NAME_HINTS: Dict[str, str] = {
    "xlm.*roberta": "XLMRoberta",
    "roberta.*xml": "XLMRoberta",
    "codebert.*mlm": "Roberta",
    "mlm.*codebert": "Roberta",
    "[dpr]?.*question.*encoder": "DPRQuestionEncoder",
    "[dpr]?.*query.*encoder": "DPRQuestionEncoder",
    "[dpr]?.*passage.*encoder": "DPRContextEncoder",
    "[dpr]?.*context.*encoder": "DPRContextEncoder",
    "[dpr]?.*ctx.*encoder": "DPRContextEncoder",
    "deberta-v2": "DebertaV2",
}

#: Parameters or the pooler of models that don't have their own pooler
POOLER_PARAMETERS: Dict[str, Dict[str, Any]] = {
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
        "summary_activation": "tanh",
        "summary_use_proj": False,
    },
}


def capitalize_model_type(model_type: str) -> str:
    """
    Returns the proper capitalized version of the model type, that can be used to
    retrieve the model class from transformers.
    :param model_type: the model_type as found in the config file
    :return: the capitalized version of the model type, or the original name of not found.
    """
    return HUGGINGFACE_CAPITALIZE.get(model_type.lower(), model_type)


def is_supported_model(model_type: Optional[str]):
    """
    Returns whether the model type is supported by Haystack
    :param model_type: the model_type as found in the config file
    :return: whether the model type is supported by the Haystack
    """
    return model_type and model_type.lower() in HUGGINGFACE_CAPITALIZE


def get_language_model_class(model_type: str) -> Optional[Type[Union[HFLanguageModel, DPREncoder]]]:
    """
    Returns the corresponding Haystack LanguageModel subclass.
    :param model_type: the model_type , properly capitalized (see `capitalize_model_type()`)
    :return: the wrapper class, or `None` if `model_type` was `None` or was not recognized.
        Lower case model_type values will return `None` as well
    """
    return HUGGINGFACE_TO_HAYSTACK.get(model_type)


def get_language_model(
    pretrained_model_name_or_path: Union[Path, str],
    language: Optional[str] = None,
    n_added_tokens: int = 0,
    use_auth_token: Optional[Union[str, bool]] = None,
    revision: Optional[str] = None,
    autoconfig_kwargs: Optional[Dict[str, Any]] = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
) -> LanguageModel:
    """
    Load a pretrained language model by doing one of the following:

    1. Specifying its name and downloading the model.
    2. Pointing to the directory the model is saved in.

    See all supported model variations at: https://huggingface.co/models.

    The appropriate language model class is inferred automatically from model configuration.

    :param pretrained_model_name_or_path: The path of the saved pretrained model or its name.
    :param language: The language of the model (i.e english etc).
    :param n_added_tokens: The number of added tokens to the model.
    :param use_auth_token: The API token used to download private models from Huggingface.
                           If this parameter is set to `True`, then the token generated when running
                           `transformers-cli login` (stored in ~/.huggingface) will be used.
                           Additional information can be found here
                           https://huggingface.co/transformers/main_classes/model.html#transformers.PreTrainedModel.from_pretrained
    :param revision: The version of the model to use from the Hugging Face model hub. This can be a tag name,
    a branch name, or a commit hash.
    :param autoconfig_kwargs: Additional keyword arguments to pass to the autoconfig function.
    :param model_kwargs: Additional keyword arguments to pass to the lamguage model constructor.
    """

    if not pretrained_model_name_or_path or not isinstance(pretrained_model_name_or_path, (str, Path)):
        raise ValueError(f"{pretrained_model_name_or_path} is not a valid pretrained_model_name_or_path parameter")

    config_file = Path(pretrained_model_name_or_path) / "language_model_config.json"

    model_type = None
    config_file_exists = os.path.exists(config_file)
    if config_file_exists:
        # it's a local directory in Haystack format
        with open(config_file) as f:
            config = json.load(f)
        model_type = config["name"]

    if not model_type:
        model_type = _get_model_type(
            pretrained_model_name_or_path,
            use_auth_token=use_auth_token,
            revision=revision,
            autoconfig_kwargs=autoconfig_kwargs,
        )

    if not model_type:
        logger.error(
            "Model type not understood for '%s' (%s). "
            "Either supply the local path for a saved model, "
            "or the name of a model that can be downloaded from the Model Hub. "
            "Ensure that the model class name can be inferred from the directory name "
            "when loading a Transformers model.",
            pretrained_model_name_or_path,
            model_type if model_type else "model_type not set",
        )
        logger.error("Using the AutoModel class for '%s'. This can cause crashes!", pretrained_model_name_or_path)
        model_type = "Auto"

    # Find the class corresponding to this model type
    model_type = capitalize_model_type(model_type)
    language_model_class = get_language_model_class(model_type)
    if not language_model_class:
        raise ValueError(
            f"The type of model supplied ({model_type}) is not supported by Haystack or was not correctly identified. "
            f"Supported model types are: {', '.join(HUGGINGFACE_TO_HAYSTACK.keys())}"
        )

    logger.info(
        " * LOADING MODEL: '%s' %s", pretrained_model_name_or_path, "(" + model_type + ")" if model_type else ""
    )

    # Instantiate the class for this model
    language_model = language_model_class(
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        model_type=model_type,
        language=language,
        n_added_tokens=n_added_tokens,
        use_auth_token=use_auth_token,
        model_kwargs=model_kwargs,
    )
    logger.info(
        "Loaded '%s' (%s model) from %s.",
        pretrained_model_name_or_path,
        model_type,
        "local file system" if config_file_exists else "model hub",
    )
    return language_model


def _get_model_type(
    model_name_or_path: Union[str, Path],
    use_auth_token: Optional[Union[str, bool]] = None,
    revision: Optional[str] = None,
    autoconfig_kwargs: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """
    Given a model name, try to use AutoConfig to understand which model type it is.
    In case it's not successful, tries to infer the type from the name of the model.
    """
    model_name_or_path = str(model_name_or_path)

    model_type: Optional[str] = None
    # Use AutoConfig to understand the model class
    try:
        config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path=model_name_or_path,
            use_auth_token=use_auth_token,
            revision=revision,
            **(autoconfig_kwargs or {}),
        )
        model_type = config.model_type
        # if unsupported model, try to infer from config.architectures
        if not is_supported_model(model_type) and config.architectures:
            model_type = config.architectures[0] if is_supported_model(config.architectures[0]) else None

    except Exception as e:
        logger.error("AutoConfig failed to load on '%s': %s", model_name_or_path, e)

    if not model_type:
        logger.warning("Could not infer the model type from its config. Looking for clues in the model name.")

        # Look for other patterns and variation that hints at the model type
        for regex, model_name in NAME_HINTS.items():
            if re.match(f".*{regex}.*", model_name_or_path):
                model_type = model_name
                break

    if model_type and model_type.lower() == "roberta" and "mlm" in model_name_or_path.lower():
        logger.error(
            "MLM part of codebert is currently not supported in Haystack: '%s' may crash later.", model_name_or_path
        )

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
    logger.info("Auto-detected model language: %s", language)
    return language
