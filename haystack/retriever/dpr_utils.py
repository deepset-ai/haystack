#!/usr/bin/env python3
# Utilility functions and classes required for DensePassageRetriever
#
# Building upon the code (https://github.com/facebookresearch/DPR) published by Facebook research under Creative Commons License (https://github.com/facebookresearch/DPR/blob/master/LICENSE)
# It is based on the following research work:
# Karpukhin, Vladimir, et al. "Dense Passage Retrieval for Open-Domain Question Answering." arXiv preprint arXiv:2004.04906 (2020).
# (https://arxiv.org/abs/2004.04906)

import logging
from typing import Tuple, Union, List

import gzip
import os
import pathlib
import wget

import torch
from torch import nn
from torch import Tensor as T
from torch.serialization import default_restore_location
import collections
from farm.file_utils import http_get

from transformers.tokenization_bert import BertTokenizer
from transformers.modeling_bert import BertModel, BertConfig

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

CheckpointState = collections.namedtuple("CheckpointState",
                                         ['model_dict', 'optimizer_dict', 'scheduler_dict', 'offset', 'epoch',
                                          'encoder_params'])


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


# UTILS
def load_states_from_checkpoint(model_file: str) -> CheckpointState:
    logger.info('Reading saved model from %s', model_file)
    state_dict = torch.load(model_file, map_location=lambda s, l: default_restore_location(s, 'cpu'))
    logger.info('model_state_dict keys %s', state_dict.keys())
    return CheckpointState(**state_dict)


def move_to_device(sample, device):
    if len(sample) == 0:
        return {}

    def _move_to_device(maybe_tensor, device):
        if torch.is_tensor(maybe_tensor):
            return maybe_tensor.to(device)
        elif isinstance(maybe_tensor, dict):
            return {
                key: _move_to_device(value, device)
                for key, value in maybe_tensor.items()
            }
        elif isinstance(maybe_tensor, list):
            return [_move_to_device(x, device) for x in maybe_tensor]
        elif isinstance(maybe_tensor, tuple):
            return [_move_to_device(x, device) for x in maybe_tensor]
        else:
            return maybe_tensor

    return _move_to_device(sample, device)

# DOWNLOADS
NQ_LICENSE_FILES = [
    'https://dl.fbaipublicfiles.com/dpr/nq_license/LICENSE',
    'https://dl.fbaipublicfiles.com/dpr/nq_license/README',
]


RESOURCES_MAP = {
    'checkpoint.retriever.single.nq.bert-base-encoder': {
        's3_url': 'https://dl.fbaipublicfiles.com/dpr/checkpoint/retriever/single/nq/hf_bert_base.cp',
        'original_ext': '.cp',
        'compressed': False,
        'desc': 'Biencoder weights trained on NQ data and HF bert-base-uncased model'
    },

    'checkpoint.retriever.multiset.bert-base-encoder': {
        's3_url': 'https://dl.fbaipublicfiles.com/dpr/checkpoint/retriver/multiset/hf_bert_base.cp',
        'original_ext': '.cp',
        'compressed': False,
        'desc': 'Biencoder weights trained on multi set data and HF bert-base-uncased model'
    },
    'data.wikipedia_split.psgs_w100': {
        's3_url': 'https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz',
        'original_ext': '.tsv',
        'compressed': True,
        'desc': 'Entire wikipedia passages set obtain by splitting all pages into 100-word segments (no overlap)'
    },
    'data.retriever.nq-dev': {
        's3_url': 'https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-nq-dev.json.gz',
        'original_ext': '.json',
        'compressed': True,
        'desc': 'NQ dev subset with passages pools for the Retriever train time validation',
        'license_files': NQ_LICENSE_FILES,
    },

    'data.retriever.nq-train': {
        's3_url': 'https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-nq-train.json.gz',
        'original_ext': '.json',
        'compressed': True,
        'desc': 'NQ train subset with passages pools for the Retriever training',
        'license_files': NQ_LICENSE_FILES,
    },

    'data.retriever.trivia-dev': {
        's3_url': 'https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-trivia-dev.json.gz',
        'original_ext': '.json',
        'compressed': True,
        'desc': 'TriviaQA dev subset with passages pools for the Retriever train time validation'
    },

    'data.retriever.trivia-train': {
        's3_url': 'https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-trivia-train.json.gz',
        'original_ext': '.json',
        'compressed': True,
        'desc': 'TriviaQA train subset with passages pools for the Retriever training'
    },

    'data.retriever.qas.nq-dev': {
        's3_url': 'https://dl.fbaipublicfiles.com/dpr/data/retriever/nq-dev.qa.csv',
        'original_ext': '.csv',
        'compressed': False,
        'desc': 'NQ dev subset for Retriever validation and IR results generation',
        'license_files': NQ_LICENSE_FILES,
    },

    'data.retriever.qas.nq-test': {
        's3_url': 'https://dl.fbaipublicfiles.com/dpr/data/retriever/nq-test.qa.csv',
        'original_ext': '.csv',
        'compressed': False,
        'desc': 'NQ test subset for Retriever validation and IR results generation',
        'license_files': NQ_LICENSE_FILES,
    },

    'data.retriever.qas.nq-train': {
        's3_url': 'https://dl.fbaipublicfiles.com/dpr/data/retriever/nq-train.qa.csv',
        'original_ext': '.csv',
        'compressed': False,
        'desc': 'NQ train subset for Retriever validation and IR results generation',
        'license_files': NQ_LICENSE_FILES,
    },
    'data.retriever.qas.trivia-dev': {
        's3_url': 'https://dl.fbaipublicfiles.com/dpr/data/retriever/trivia-dev.qa.csv.gz',
        'original_ext': '.csv',
        'compressed': True,
        'desc': 'Trivia dev subset for Retriever validation and IR results generation'
    },

    'data.retriever.qas.trivia-test': {
        's3_url': 'https://dl.fbaipublicfiles.com/dpr/data/retriever/trivia-test.qa.csv.gz',
        'original_ext': '.csv',
        'compressed': True,
        'desc': 'Trivia test subset for Retriever validation and IR results generation'
    },

    'data.retriever.qas.trivia-train': {
        's3_url': 'https://dl.fbaipublicfiles.com/dpr/data/retriever/trivia-train.qa.csv.gz',
        'original_ext': '.csv',
        'compressed': True,
        'desc': 'Trivia train subset for Retriever validation and IR results generation'
    },

    'data.gold_passages_info.nq_train': {
        's3_url': 'https://dl.fbaipublicfiles.com/dpr/data/nq_gold_info/nq-train_gold_info.json.gz',
        'original_ext': '.json',
        'compressed': True,
        'desc': 'Original NQ (our train subset) gold positive passages and alternative question tokenization',
        'license_files': NQ_LICENSE_FILES,
    },

    'data.gold_passages_info.nq_dev': {
        's3_url': 'https://dl.fbaipublicfiles.com/dpr/data/nq_gold_info/nq-dev_gold_info.json.gz',
        'original_ext': '.json',
        'compressed': True,
        'desc': 'Original NQ (our dev subset) gold positive passages and alternative question tokenization',
        'license_files': NQ_LICENSE_FILES,
    },

    'data.gold_passages_info.nq_test': {
        's3_url': 'https://dl.fbaipublicfiles.com/dpr/data/nq_gold_info/nq-test_gold_info.json.gz',
        'original_ext': '.json',
        'compressed': True,
        'desc': 'Original NQ (our test, original dev subset) gold positive passages and alternative question '
                'tokenization',
        'license_files': NQ_LICENSE_FILES,
    },
    'data.retriever_results.nq.single.test': {
        's3_url': 'https://dl.fbaipublicfiles.com/dpr/data/retriever_results/single/nq-test.json.gz',
        'original_ext': '.json',
        'compressed': True,
        'desc': 'Retrieval results of NQ test dataset for the encoder trained on NQ',
        'license_files': NQ_LICENSE_FILES,
    },
    'data.retriever_results.nq.single.dev': {
        's3_url': 'https://dl.fbaipublicfiles.com/dpr/data/retriever_results/single/nq-dev.json.gz',
        'original_ext': '.json',
        'compressed': True,
        'desc': 'Retrieval results of NQ dev dataset for the encoder trained on NQ',
        'license_files': NQ_LICENSE_FILES,
    },
    'data.retriever_results.nq.single.train': {
        's3_url': 'https://dl.fbaipublicfiles.com/dpr/data/retriever_results/single/nq-train.json.gz',
        'original_ext': '.json',
        'compressed': True,
        'desc': 'Retrieval results of NQ train dataset for the encoder trained on NQ',
        'license_files': NQ_LICENSE_FILES,
    }
}


def unpack(gzip_file: str, out_file: str):
    print('Uncompressing ', gzip_file)
    input = gzip.GzipFile(gzip_file, 'rb')
    s = input.read()
    input.close()
    output = open(out_file, 'wb')
    output.write(s)
    output.close()
    print('Saved to ', out_file)


def download_resource(s3_url: str, original_ext: str, compressed: bool, resource_key: str, out_dir: str) -> str:
    print('Loading from ', s3_url)

    # create local dir
    path_names = resource_key.split('.')

    root_dir = out_dir if out_dir else './'
    save_root = os.path.join(root_dir, *path_names[:-1])  # last segment is for file name

    pathlib.Path(save_root).mkdir(parents=True, exist_ok=True)

    local_file = os.path.join(save_root, path_names[-1] + ('.tmp' if compressed else original_ext))

    if os.path.exists(local_file):
        print('File already exist ', local_file)
        return save_root

    wget.download(s3_url, out=local_file)

    print('Saved to ', local_file)

    if compressed:
        uncompressed_file = os.path.join(save_root, path_names[-1] + original_ext)
        unpack(local_file, uncompressed_file)
        os.remove(local_file)
    return save_root


def download_file(s3_url: str, out_dir: str, file_name: str):
    print('Loading from ', s3_url)
    local_file = os.path.join(out_dir, file_name)

    if os.path.exists(local_file):
        print('File already exist ', local_file)
        return
    with open(local_file, "w") as file:
        http_get(s3_url, temp_file=file)
    wget.download(s3_url, out=local_file)
    print('Saved to ', local_file)


def download_dpr(resource_key: str, out_dir: str):
    if resource_key not in RESOURCES_MAP:
        # match by prefix
        resources = [k for k in RESOURCES_MAP.keys() if k.startswith(resource_key)]
        if resources:
            for key in resources:
                download_dpr(key, out_dir)
        else:
            print('no resources found for specified key')
        return
    download_info = RESOURCES_MAP[resource_key]

    s3_url: Union[str, List[str]] = download_info['s3_url'] # type: ignore
    original_ext: str = download_info['original_ext'] # type: ignore
    compressed: bool = download_info['compressed'] # type: ignore

    save_root_dir = None
    if isinstance(s3_url, list):
        for i, url in enumerate(s3_url):
            save_root_dir = download_resource(url, original_ext, compressed,
                                              '{}_{}'.format(resource_key, i), out_dir)
    else:
        save_root_dir = download_resource(s3_url, original_ext, compressed,
                                          resource_key, out_dir)

    license_files = download_info.get('license_files', None)
    if not license_files:
        return

    download_file(license_files[0], save_root_dir, 'LICENSE')  # type: ignore
    download_file(license_files[1], save_root_dir, 'README') # type: ignore