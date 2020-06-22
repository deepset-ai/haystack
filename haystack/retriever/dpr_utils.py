#!/usr/bin/env python3
# Building upon the code (https://github.com/facebookresearch/DPR) published by Facebook research together with this Publication:
# Karpukhin, Vladimir, et al. "Dense Passage Retrieval for Open-Domain Question Answering." arXiv preprint arXiv:2004.04906 (2020).
# (https://arxiv.org/abs/2004.04906)

import logging
from typing import List, Tuple

import torch
from torch import nn
from torch import Tensor as T
from torch.serialization import default_restore_location
import collections

from transformers.tokenization_bert import BertTokenizer
from transformers.modeling_bert import BertModel, BertConfig

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

CheckpointState = collections.namedtuple("CheckpointState",
                                         ['model_dict', 'optimizer_dict', 'scheduler_dict', 'offset', 'epoch',
                                          'encoder_params'])

# UTILS
def load_states_from_checkpoint(model_file: str) -> CheckpointState:
    logger.info('Reading saved model from %s', model_file)
    state_dict = torch.load(model_file, map_location=lambda s, l: default_restore_location(s, 'cpu'))
    logger.info('model_state_dict keys %s', state_dict.keys())
    return CheckpointState(**state_dict)


# CLASSES
class HFBertEncoder(BertModel):

    def __init__(self, config, project_dim: int = 0):
        BertModel.__init__(self, config)
        assert config.hidden_size > 0, 'Encoder hidden_size can\'t be zero'
        self.encode_proj = nn.Linear(config.hidden_size, project_dim) if project_dim != 0 else None
        self.init_weights()

    @classmethod
    def init_encoder(cls, cfg_name: str, projection_dim: int = 0, dropout: float = 0.1, **kwargs) -> BertModel:
        cfg = BertConfig.from_pretrained(cfg_name if cfg_name else 'bert-base-uncased')
        if dropout != 0:
            cfg.attention_probs_dropout_prob = dropout
            cfg.hidden_dropout_prob = dropout
        return cls.from_pretrained(cfg_name, config=cfg, project_dim=projection_dim, **kwargs)

    def forward(self, input_ids: torch.Tensor, token_type_ids: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        if self.config.output_hidden_states:
            sequence_output, pooled_output, hidden_states = super().forward(input_ids=input_ids,
                                                                            token_type_ids=token_type_ids,
                                                                            attention_mask=attention_mask)
        else:
            hidden_states = None
            sequence_output, pooled_output = super().forward(input_ids=input_ids, token_type_ids=token_type_ids,
                                                             attention_mask=attention_mask)

        pooled_output = sequence_output[:, 0, :]
        if self.encode_proj:
            pooled_output = self.encode_proj(pooled_output)
        return sequence_output, pooled_output, hidden_states

    def get_out_size(self):
        if self.encode_proj:
            return self.encode_proj.out_features
        return self.config.hidden_size


class Tensorizer(object):
    """
    Component for all text to model input data conversions and related utility methods
    """

    # Note: title, if present, is supposed to be put before text (i.e. optional title + document body)
    def text_to_tensor(self, text: str, title: str = None, add_special_tokens: bool = True):
        raise NotImplementedError

    def get_pair_separator_ids(self) -> T:
        raise NotImplementedError

    def get_pad_id(self) -> int:
        raise NotImplementedError

    def get_attn_mask(self, tokens_tensor: T):
        raise NotImplementedError

    def is_sub_word_id(self, token_id: int):
        raise NotImplementedError

    def to_string(self, token_ids, skip_special_tokens=True):
        raise NotImplementedError

    def set_pad_to_max(self, pad: bool):
        raise NotImplementedError


class BertTensorizer(Tensorizer):
    def __init__(self, tokenizer: BertTokenizer, max_length: int, pad_to_max: bool = True):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_to_max = pad_to_max

    def text_to_tensor(self, text: str, title: str = None, add_special_tokens: bool = True):
        text = text.strip()

        # tokenizer automatic padding is explicitly disabled since its inconsistent behavior
        if title:
            token_ids = self.tokenizer.encode(title, text_pair=text, add_special_tokens=add_special_tokens,
                                              max_length=self.max_length,
                                              pad_to_max_length=False)
        else:
            token_ids = self.tokenizer.encode(text, add_special_tokens=add_special_tokens, max_length=self.max_length,
                                              pad_to_max_length=False)

        seq_len = self.max_length
        if self.pad_to_max and len(token_ids) < seq_len:
            token_ids = token_ids + [self.tokenizer.pad_token_id] * (seq_len - len(token_ids))
        if len(token_ids) > seq_len:
            token_ids = token_ids[0:seq_len]
            token_ids[-1] = self.tokenizer.sep_token_id

        return torch.tensor(token_ids)

    def get_pair_separator_ids(self) -> T:
        return torch.tensor([self.tokenizer.sep_token_id])

    def get_pad_id(self) -> int:
        return self.tokenizer.pad_token_type_id

    def get_attn_mask(self, tokens_tensor: T) -> T:
        return tokens_tensor != self.get_pad_id()

    def is_sub_word_id(self, token_id: int):
        token = self.tokenizer.convert_ids_to_tokens([token_id])[0]
        return token.startswith("##") or token.startswith(" ##")

    def to_string(self, token_ids, skip_special_tokens=True):
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def set_pad_to_max(self, do_pad: bool):
        self.pad_to_max = do_pad
