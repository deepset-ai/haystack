# coding=utf-8
# Copyright 2018 deepset team.
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

from typing import Dict, Any, Union, Tuple, Optional, List

import re
import os
import json
import logging
from pathlib import Path

import numpy as np
import transformers
from transformers import PreTrainedTokenizer, RobertaTokenizer, AutoConfig, AutoFeatureExtractor, AutoTokenizer

# NOTE: These two constants are internals of HF. Keep in mind that they might be renamed or removed at any time.
from transformers.models.auto.feature_extraction_auto import FEATURE_EXTRACTOR_MAPPING_NAMES
from transformers.models.auto.tokenization_auto import TOKENIZER_MAPPING_NAMES

from haystack.errors import ModelingError
from haystack.modeling.data_handler.samples import SampleBasket


logger = logging.getLogger(__name__)


#: Special characters used by the different tokenizers to indicate start of word / whitespace
SPECIAL_TOKENIZER_CHARS = r"^(##|Ġ|▁)"


FEATURE_EXTRACTORS = {
    **{key: AutoTokenizer for key in TOKENIZER_MAPPING_NAMES.keys()},
    **{key: AutoFeatureExtractor for key in FEATURE_EXTRACTOR_MAPPING_NAMES.keys()},
}


DEFAULT_EXTRACTION_PARAMS = {
    AutoTokenizer: {
        "max_length": 256,
        "add_special_tokens": True,
        "truncation": True,
        "truncation_strategy": "longest_first",
        "padding": "max_length",
        "return_token_type_ids": True,
    },
    AutoFeatureExtractor: {"return_tensors": "pt"},
}


class FeatureExtractor:
    def __init__(
        self,
        pretrained_model_name_or_path: Union[str, Path],
        revision: Optional[str] = None,
        use_fast: bool = True,
        use_auth_token: Optional[Union[str, bool]] = None,
        **kwargs,
    ):
        """
        Enables loading of different feature extractors, including tokenizers, with a uniform interface.

        Use `FeatureExtractor.extract_features()` to convert your input queries, documents, images, and tables
        into vectors that you can pass to the language model.

        :param pretrained_model_name_or_path:  The path of the saved pretrained model or its name (for example, `bert-base-uncased`)
        :param revision: The version of the model to use from the Hugging Face model hub. It can be tag name, branch name, or commit hash.
        :param use_fast: Indicate if Haystack should try to load the fast version of the tokenizer (True) or use the Python one (False). Defaults to True.
        :param use_auth_token: The API token used to download private models from Hugging Face.
                            If this parameter is set to `True`, then the token generated when running
                            `transformers-cli login` (stored in ~/.huggingface) is used.
                            For more information, see
                            [Hugging Face documentation](https://huggingface.co/transformers/main_classes/model.html#transformers.PreTrainedModel.from_pretrained)
        :param kwargs: Other kwargs you want to pass on to `PretrainedTokenizer.from_pretrained()`
        """
        model_name_or_path = str(pretrained_model_name_or_path)
        model_type = None

        config_file = Path(pretrained_model_name_or_path) / "tokenizer_config.json"
        if os.path.exists(config_file):
            # it's a local directory
            with open(config_file) as f:
                config = json.load(f)
            feature_extractor_classname = config["tokenizer_class"]
            logger.debug("⛏️ Selected feature extractor: %s (from %s)", feature_extractor_classname, config_file)
            # Use FastTokenizers as much as possible
            try:
                feature_extractor_class = getattr(transformers, feature_extractor_classname + "Fast")
                logger.debug(
                    "Fast version of this tokenizer exists. Loaded class: %s",
                    feature_extractor_class.__class__.__name__,
                )
            except AttributeError:
                logger.debug("Fast version could not be loaded. Falling back to base version.")
                feature_extractor_class = getattr(transformers, feature_extractor_classname)

        else:
            # it's a HF Hub identifier
            config = AutoConfig.from_pretrained(
                pretrained_model_name_or_path=model_name_or_path, use_auth_token=use_auth_token, revision=revision
            )
            model_type = config.model_type
            try:
                feature_extractor_class = FEATURE_EXTRACTORS[model_type]
            except KeyError as e:
                raise ModelingError(
                    f"'{pretrained_model_name_or_path}' has no known feature extractor. "
                    "Haystack can assign tokenizers to the following model types: "
                    # Using chr(10) instead of \n due to f-string limitation
                    # https://peps.python.org/pep-0498/#specification: "Backslashes may not appear anywhere within expressions"
                    f"\n- {f'{chr(10)}- '.join(FEATURE_EXTRACTORS.keys())}"
                ) from e
            logger.debug(
                "⛏️ Selected feature extractor: %s (for model type '%s')", feature_extractor_class.__name__, model_type
            )

        self.default_params = DEFAULT_EXTRACTION_PARAMS.get(feature_extractor_class, {})

        self.feature_extractor = feature_extractor_class.from_pretrained(
            pretrained_model_name_or_path=model_name_or_path,
            revision=revision,
            use_fast=use_fast,
            use_auth_token=use_auth_token,
            **kwargs,
        )

    def __call__(self, **kwargs):
        params = {**self.default_params, **(kwargs or {})}
        return self.feature_extractor(**params)


def tokenize_batch_question_answering(
    pre_baskets: List[Dict[str, Any]], tokenizer: PreTrainedTokenizer, indices: List[Any]
) -> List[SampleBasket]:
    """
    Tokenizes text data for question answering tasks. Tokenization means splitting words into subwords, depending on the
    tokenizer's vocabulary.

    - We first tokenize all documents in batch mode. (When using FastTokenizers Rust multithreading can be enabled by TODO add how to enable rust mt)
    - Then we tokenize each question individually
    - We construct dicts with question and corresponding document text + tokens + offsets + ids

    :param pre_baskets: input dicts with QA info #TODO change to input objects
    :param tokenizer: tokenizer to be used
    :param indices: indices used during multiprocessing so that IDs assigned to our baskets are unique
    :return: baskets, list containing question and corresponding document information
    """
    if not len(indices) == len(pre_baskets):
        raise ValueError("indices and pre_baskets must have the same length")

    if not tokenizer.is_fast:
        raise ModelingError(
            "Processing QA data is only supported with fast tokenizers for now."
            "Please load Tokenizers with 'use_fast=True' option."
        )

    baskets = []
    # # Tokenize texts in batch mode
    texts = [d["context"] for d in pre_baskets]
    tokenized_docs_batch = tokenizer(
        text=texts,
        return_offsets_mapping=True,
        return_special_tokens_mask=True,
        add_special_tokens=False,
        verbose=False,
    )

    # Extract relevant data
    tokenids_batch = tokenized_docs_batch["input_ids"]
    offsets_batch = []
    for o in tokenized_docs_batch["offset_mapping"]:
        offsets_batch.append(np.asarray([x[0] for x in o], dtype="int16"))
    start_of_words_batch = []
    for e in tokenized_docs_batch.encodings:
        start_of_words_batch.append(_get_start_of_word_QA(e.word_ids))

    for i_doc, d in enumerate(pre_baskets):
        document_text = d["context"]
        # # Tokenize questions one by one
        for i_q, q in enumerate(d["qas"]):
            question_text = q["question"]
            tokenized_q = tokenizer(
                question_text, return_offsets_mapping=True, return_special_tokens_mask=True, add_special_tokens=False
            )

            # Extract relevant data
            question_tokenids = tokenized_q["input_ids"]
            question_offsets = [x[0] for x in tokenized_q["offset_mapping"]]
            question_sow = _get_start_of_word_QA(tokenized_q.encodings[0].word_ids)

            external_id = q["id"]
            # The internal_id depends on unique ids created for each process before forking
            internal_id = f"{indices[i_doc]}-{i_q}"
            raw = {
                "document_text": document_text,
                "document_tokens": tokenids_batch[i_doc],
                "document_offsets": offsets_batch[i_doc],
                "document_start_of_word": start_of_words_batch[i_doc],
                "question_text": question_text,
                "question_tokens": question_tokenids,
                "question_offsets": question_offsets,
                "question_start_of_word": question_sow,
                "answers": q["answers"],
            }
            # TODO add only during debug mode (need to create debug mode)
            raw["document_tokens_strings"] = tokenized_docs_batch.encodings[i_doc].tokens
            raw["question_tokens_strings"] = tokenized_q.encodings[0].tokens

            baskets.append(SampleBasket(raw=raw, id_internal=internal_id, id_external=external_id, samples=None))
    return baskets


def _get_start_of_word_QA(word_ids):
    return [1] + list(np.ediff1d(np.asarray(word_ids, dtype="int16")))


def truncate_sequences(
    seq_a: list,
    seq_b: Optional[list],
    tokenizer: AutoTokenizer,
    max_seq_len: int,
    truncation_strategy: str = "longest_first",
    with_special_tokens: bool = True,
    stride: int = 0,
) -> Tuple[List[Any], Optional[List[Any]], List[Any]]:
    """
    Reduces a single sequence or a pair of sequences to a maximum sequence length.
    The sequences can contain tokens or any other elements (offsets, masks ...).
    If `with_special_tokens` is enabled, it'll remove some additional tokens to have exactly
    enough space for later adding special tokens (CLS, SEP etc.)

    Supported truncation strategies:

    - longest_first: (default) Iteratively reduce the inputs sequence until the input is under
        max_length starting from the longest one at each token (when there is a pair of input sequences).
        Overflowing tokens only contains overflow from the first sequence.
    - only_first: Only truncate the first sequence. raise an error if the first sequence is
        shorter or equal to than num_tokens_to_remove.
    - only_second: Only truncate the second sequence
    - do_not_truncate: Does not truncate (raise an error if the input sequence is longer than max_length)

    :param seq_a: First sequence of tokens/offsets/...
    :param seq_b: Optional second sequence of tokens/offsets/...
    :param tokenizer: Tokenizer (e.g. from get_tokenizer))
    :param max_seq_len:
    :param truncation_strategy: how the sequence(s) should be truncated down.
        Default: "longest_first" (see above for other options).
    :param with_special_tokens: If true, it'll remove some additional tokens to have exactly enough space
        for later adding special tokens (CLS, SEP etc.)
    :param stride: optional stride of the window during truncation
    :return: truncated seq_a, truncated seq_b, overflowing tokens
    """
    pair = seq_b is not None
    len_a = len(seq_a)
    len_b = len(seq_b) if seq_b is not None else 0
    num_special_tokens = tokenizer.num_special_tokens_to_add(pair=pair) if with_special_tokens else 0
    total_len = len_a + len_b + num_special_tokens
    overflowing_tokens = []

    if max_seq_len and total_len > max_seq_len:
        seq_a, seq_b, overflowing_tokens = tokenizer.truncate_sequences(
            seq_a,
            pair_ids=seq_b,
            num_tokens_to_remove=total_len - max_seq_len,
            truncation_strategy=truncation_strategy,
            stride=stride,
        )
    return (seq_a, seq_b, overflowing_tokens)


#
# FIXME this is a relic from FARM. If there's the occasion, remove it!
#
def tokenize_with_metadata(text: str, tokenizer: PreTrainedTokenizer) -> Dict[str, Any]:
    """
    Performing tokenization while storing some important metadata for each token:

    * offsets: (int) Character index where the token begins in the original text
    * start_of_word: (bool) If the token is the start of a word. Particularly helpful for NER and QA tasks.

    We do this by first doing whitespace tokenization and then applying the model specific tokenizer to each "word".

    .. note::  We don't assume to preserve exact whitespaces in the tokens!
               This means: tabs, new lines, multiple whitespace etc will all resolve to a single " ".
               This doesn't make a difference for BERT + XLNet but it does for RoBERTa.
               For RoBERTa it has the positive effect of a shorter sequence length, but some information about whitespace
               type is lost which might be helpful for certain NLP tasks ( e.g tab for tables).

    :param text: Text to tokenize
    :param tokenizer: Tokenizer (e.g. from get_tokenizer))
    :return: Dictionary with "tokens", "offsets" and "start_of_word"
    """
    # normalize all other whitespace characters to " "
    # Note: using text.split() directly would destroy the offset,
    # since \n\n\n would be treated similarly as a single \n
    text = re.sub(r"\s", " ", text)

    words: Union[List[str], np.ndarray] = []
    word_offsets: Union[List[int], np.ndarray] = []
    start_of_word: List[Union[int, bool]] = []

    # Fast Tokenizers return offsets, so we don't need to calculate them ourselves
    if tokenizer.is_fast:
        # tokenized = tokenizer(text, return_offsets_mapping=True, return_special_tokens_mask=True)
        tokenized = tokenizer(text, return_offsets_mapping=True, return_special_tokens_mask=True)

        tokens = tokenized["input_ids"]
        offsets = np.array([x[0] for x in tokenized["offset_mapping"]])
        # offsets2 = [x[0] for x in tokenized2["offset_mapping"]]
        words = np.array(tokenized.encodings[0].words)

        # TODO check for validity for all tokenizer and special token types
        words[0] = -1
        words[-1] = words[-2]
        words += 1
        start_of_word = [0] + list(np.ediff1d(words))
        return {"tokens": tokens, "offsets": offsets, "start_of_word": start_of_word}

    # split text into "words" (here: simple whitespace tokenizer).
    words = text.split(" ")
    cumulated = 0
    for word in words:
        word_offsets.append(cumulated)  # type: ignore [union-attr]
        cumulated += len(word) + 1  # 1 because we so far have whitespace tokenizer

    # split "words" into "subword tokens"
    tokens, offsets, start_of_word = _words_to_tokens(words, word_offsets, tokenizer)  # type: ignore
    return {"tokens": tokens, "offsets": offsets, "start_of_word": start_of_word}


# Note: only used by tokenize_with_metadata()
def _words_to_tokens(
    words: List[str], word_offsets: List[int], tokenizer: PreTrainedTokenizer
) -> Tuple[List[str], List[int], List[bool]]:
    """
    Tokenize "words" into subword tokens while keeping track of offsets and if a token is the start of a word.
    :param words: list of words.
    :param word_offsets: Character indices where each word begins in the original text
    :param tokenizer: Tokenizer (e.g. from get_tokenizer))
    :return: Tuple of (tokens, offsets, start_of_word)
    """
    tokens: List[str] = []
    token_offsets: List[int] = []
    start_of_word: List[bool] = []
    index = 0
    for index, (word, word_offset) in enumerate(zip(words, word_offsets)):
        if index % 500000 == 0:
            logger.info(index)
        # Get (subword) tokens of single word.

        # empty / pure whitespace
        if len(word) == 0:
            continue
        # For the first word of a text: we just call the regular tokenize function.
        # For later words: we need to call it with add_prefix_space=True to get the same results with roberta / gpt2 tokenizer
        # see discussion here. https://github.com/huggingface/transformers/issues/1196
        if len(tokens) == 0:
            tokens_word = tokenizer.tokenize(word)
        else:
            if type(tokenizer) == RobertaTokenizer:
                tokens_word = tokenizer.tokenize(word, add_prefix_space=True)
            else:
                tokens_word = tokenizer.tokenize(word)
        # Sometimes the tokenizer returns no tokens
        if len(tokens_word) == 0:
            continue
        tokens += tokens_word

        # get global offset for each token in word + save marker for first tokens of a word
        first_token = True
        for token in tokens_word:
            token_offsets.append(word_offset)
            # Depending on the tokenizer type special chars are added to distinguish tokens with preceeding
            # whitespace (=> "start of a word"). We need to get rid of these to calculate the original length of the token
            original_token = re.sub(SPECIAL_TOKENIZER_CHARS, "", token)
            # Don't use length of unk token for offset calculation
            if original_token == tokenizer.special_tokens_map["unk_token"]:
                word_offset += 1
            else:
                word_offset += len(original_token)
            if first_token:
                start_of_word.append(True)
                first_token = False
            else:
                start_of_word.append(False)

    return tokens, token_offsets, start_of_word
