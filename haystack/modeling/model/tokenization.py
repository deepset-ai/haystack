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
"""
Tokenization classes.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
from typing import Dict, Any, Tuple, Optional, List, Union

import re
import logging
import numpy as np
from transformers import (
    AlbertTokenizer,
    AlbertTokenizerFast,
    BertTokenizer,
    BertTokenizerFast,
    DistilBertTokenizer,
    DistilBertTokenizerFast,
    ElectraTokenizer,
    ElectraTokenizerFast,
    RobertaTokenizer,
    RobertaTokenizerFast,
    XLMRobertaTokenizer,
    XLMRobertaTokenizerFast,
    XLNetTokenizer,
    XLNetTokenizerFast,
    CamembertTokenizer,
    CamembertTokenizerFast,
    DPRContextEncoderTokenizer,
    DPRContextEncoderTokenizerFast,
    DPRQuestionEncoderTokenizer,
    DPRQuestionEncoderTokenizerFast,
    BigBirdTokenizer,
    BigBirdTokenizerFast,
    DebertaV2Tokenizer,
    DebertaV2TokenizerFast,
)
from transformers import AutoConfig

from haystack.modeling.data_handler.samples import SampleBasket


logger = logging.getLogger(__name__)


# Special characters used by the different tokenizers to indicate start of word / whitespace
SPECIAL_TOKENIZER_CHARS = r"^(##|Ġ|▁)"

# TODO analyse if tokenizers can be completely used through HF transformers
class Tokenizer:
    """
    Simple Wrapper for Tokenizers from the transformers package. Enables loading of different Tokenizer classes with a uniform interface.
    """

    @classmethod
    def load(
        cls,
        pretrained_model_name_or_path,
        revision=None,
        tokenizer_class=None,
        use_fast=True,
        use_auth_token: Union[bool, str] = None,
        **kwargs,
    ):
        """
        Enables loading of different Tokenizer classes with a uniform interface. Either infer the class from
        model config or define it manually via `tokenizer_class`.

        :param pretrained_model_name_or_path:  The path of the saved pretrained model or its name (e.g. `bert-base-uncased`)
        :type pretrained_model_name_or_path: str
        :param revision: The version of model to use from the HuggingFace model hub. Can be tag name, branch name, or commit hash.
        :type revision: str
        :param tokenizer_class: (Optional) Name of the tokenizer class to load (e.g. `BertTokenizer`)
        :type tokenizer_class: str
        :param use_fast: (Optional, False by default) Indicate if Haystack should try to load the fast version of the tokenizer (True) or
            use the Python one (False).
            Only DistilBERT, BERT and Electra fast tokenizers are supported.
        :type use_fast: bool
        :param kwargs:
        :return: Tokenizer
        """
        pretrained_model_name_or_path = str(pretrained_model_name_or_path)
        kwargs["revision"] = revision

        if tokenizer_class is None:
            tokenizer_class = cls._infer_tokenizer_class(pretrained_model_name_or_path, use_auth_token=use_auth_token)

        logger.debug(f"Loading tokenizer of type '{tokenizer_class}'")
        # return appropriate tokenizer object
        ret = None
        if "AlbertTokenizer" in tokenizer_class:
            if use_fast:
                ret = AlbertTokenizerFast.from_pretrained(
                    pretrained_model_name_or_path, keep_accents=True, use_auth_token=use_auth_token, **kwargs
                )
            else:
                ret = AlbertTokenizer.from_pretrained(
                    pretrained_model_name_or_path, keep_accents=True, use_auth_token=use_auth_token, **kwargs
                )
        elif "XLMRobertaTokenizer" in tokenizer_class:
            if use_fast:
                ret = XLMRobertaTokenizerFast.from_pretrained(
                    pretrained_model_name_or_path, use_auth_token=use_auth_token, **kwargs
                )
            else:
                ret = XLMRobertaTokenizer.from_pretrained(
                    pretrained_model_name_or_path, use_auth_token=use_auth_token, **kwargs
                )
        elif "RobertaTokenizer" in tokenizer_class:
            if use_fast:
                ret = RobertaTokenizerFast.from_pretrained(
                    pretrained_model_name_or_path, use_auth_token=use_auth_token, **kwargs
                )
            else:
                ret = RobertaTokenizer.from_pretrained(
                    pretrained_model_name_or_path, use_auth_token=use_auth_token, **kwargs
                )
        elif "DistilBertTokenizer" in tokenizer_class:
            if use_fast:
                ret = DistilBertTokenizerFast.from_pretrained(
                    pretrained_model_name_or_path, use_auth_token=use_auth_token, **kwargs
                )
            else:
                ret = DistilBertTokenizer.from_pretrained(
                    pretrained_model_name_or_path, use_auth_token=use_auth_token, **kwargs
                )
        elif "BertTokenizer" in tokenizer_class:
            if use_fast:
                ret = BertTokenizerFast.from_pretrained(
                    pretrained_model_name_or_path, use_auth_token=use_auth_token, **kwargs
                )
            else:
                ret = BertTokenizer.from_pretrained(
                    pretrained_model_name_or_path, use_auth_token=use_auth_token, **kwargs
                )
        elif "XLNetTokenizer" in tokenizer_class:
            if use_fast:
                ret = XLNetTokenizerFast.from_pretrained(
                    pretrained_model_name_or_path, keep_accents=True, use_auth_token=use_auth_token, **kwargs
                )
            else:
                ret = XLNetTokenizer.from_pretrained(
                    pretrained_model_name_or_path, keep_accents=True, use_auth_token=use_auth_token, **kwargs
                )
        elif "ElectraTokenizer" in tokenizer_class:
            if use_fast:
                ret = ElectraTokenizerFast.from_pretrained(
                    pretrained_model_name_or_path, use_auth_token=use_auth_token, **kwargs
                )
            else:
                ret = ElectraTokenizer.from_pretrained(
                    pretrained_model_name_or_path, use_auth_token=use_auth_token, **kwargs
                )
        elif "CamembertTokenizer" in tokenizer_class:
            if use_fast:
                ret = CamembertTokenizerFast.from_pretrained(
                    pretrained_model_name_or_path, use_auth_token=use_auth_token, **kwargs
                )
            else:
                ret = CamembertTokenizer.from_pretrained(
                    pretrained_model_name_or_path, use_auth_token=use_auth_token, **kwargs
                )
        elif "DPRQuestionEncoderTokenizer" in tokenizer_class:
            if use_fast:
                ret = DPRQuestionEncoderTokenizerFast.from_pretrained(
                    pretrained_model_name_or_path, use_auth_token=use_auth_token, **kwargs
                )
            else:
                ret = DPRQuestionEncoderTokenizer.from_pretrained(
                    pretrained_model_name_or_path, use_auth_token=use_auth_token, **kwargs
                )
        elif "DPRContextEncoderTokenizer" in tokenizer_class:
            if use_fast:
                ret = DPRContextEncoderTokenizerFast.from_pretrained(
                    pretrained_model_name_or_path, use_auth_token=use_auth_token, **kwargs
                )
            else:
                ret = DPRContextEncoderTokenizer.from_pretrained(
                    pretrained_model_name_or_path, use_auth_token=use_auth_token, **kwargs
                )
        elif "BigBirdTokenizer" in tokenizer_class:
            if use_fast:
                ret = BigBirdTokenizerFast.from_pretrained(
                    pretrained_model_name_or_path, use_auth_token=use_auth_token, **kwargs
                )
            else:
                ret = BigBirdTokenizer.from_pretrained(
                    pretrained_model_name_or_path, use_auth_token=use_auth_token, **kwargs
                )
        elif "DebertaV2Tokenizer" in tokenizer_class:
            if use_fast:
                ret = DebertaV2TokenizerFast.from_pretrained(
                    pretrained_model_name_or_path, use_auth_token=use_auth_token, **kwargs
                )
            else:
                ret = DebertaV2Tokenizer.from_pretrained(
                    pretrained_model_name_or_path, use_auth_token=use_auth_token, **kwargs
                )
        if ret is None:
            raise Exception("Unable to load tokenizer")
        return ret

    @staticmethod
    def _infer_tokenizer_class(pretrained_model_name_or_path, use_auth_token: Union[bool, str] = None):
        # Infer Tokenizer from model type in config
        try:
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path, use_auth_token=use_auth_token)
        except OSError:
            # Haystack model (no 'config.json' file)
            try:
                config = AutoConfig.from_pretrained(
                    pretrained_model_name_or_path + "/language_model_config.json", use_auth_token=use_auth_token
                )
            except Exception as e:
                logger.warning("No config file found. Trying to infer Tokenizer type from model name")
                tokenizer_class = Tokenizer._infer_tokenizer_class_from_string(pretrained_model_name_or_path)
                return tokenizer_class

        model_type = config.model_type

        if model_type == "xlm-roberta":
            tokenizer_class = "XLMRobertaTokenizer"
        elif model_type == "roberta":
            if "mlm" in pretrained_model_name_or_path.lower():
                raise NotImplementedError("MLM part of codebert is currently not supported in Haystack")
            tokenizer_class = "RobertaTokenizer"
        elif model_type == "camembert":
            tokenizer_class = "CamembertTokenizer"
        elif model_type == "albert":
            tokenizer_class = "AlbertTokenizer"
        elif model_type == "distilbert":
            tokenizer_class = "DistilBertTokenizer"
        elif model_type == "bert":
            tokenizer_class = "BertTokenizer"
        elif model_type == "xlnet":
            tokenizer_class = "XLNetTokenizer"
        elif model_type == "electra":
            tokenizer_class = "ElectraTokenizer"
        elif model_type == "dpr":
            if config.architectures[0] == "DPRQuestionEncoder":
                tokenizer_class = "DPRQuestionEncoderTokenizer"
            elif config.architectures[0] == "DPRContextEncoder":
                tokenizer_class = "DPRContextEncoderTokenizer"
            elif config.architectures[0] == "DPRReader":
                raise NotImplementedError("DPRReader models are currently not supported.")
        elif model_type == "big_bird":
            tokenizer_class = "BigBirdTokenizer"
        elif model_type == "deberta-v2":
            tokenizer_class = "DebertaV2Tokenizer"
        else:
            # Fall back to inferring type from model name
            logger.warning(
                "Could not infer Tokenizer type from config. Trying to infer Tokenizer type from model name."
            )
            tokenizer_class = Tokenizer._infer_tokenizer_class_from_string(pretrained_model_name_or_path)

        return tokenizer_class

    @staticmethod
    def _infer_tokenizer_class_from_string(pretrained_model_name_or_path):
        # If inferring tokenizer class from config doesn't succeed,
        # fall back to inferring tokenizer class from model name.
        if "albert" in pretrained_model_name_or_path.lower():
            tokenizer_class = "AlbertTokenizer"
        elif "bigbird" in pretrained_model_name_or_path.lower():
            tokenizer_class = "BigBirdTokenizer"
        elif "xlm-roberta" in pretrained_model_name_or_path.lower():
            tokenizer_class = "XLMRobertaTokenizer"
        elif "roberta" in pretrained_model_name_or_path.lower():
            tokenizer_class = "RobertaTokenizer"
        elif "codebert" in pretrained_model_name_or_path.lower():
            if "mlm" in pretrained_model_name_or_path.lower():
                raise NotImplementedError("MLM part of codebert is currently not supported in Haystack")
            tokenizer_class = "RobertaTokenizer"
        elif "camembert" in pretrained_model_name_or_path.lower() or "umberto" in pretrained_model_name_or_path.lower():
            tokenizer_class = "CamembertTokenizer"
        elif "distilbert" in pretrained_model_name_or_path.lower():
            tokenizer_class = "DistilBertTokenizer"
        elif (
            "debertav2" in pretrained_model_name_or_path.lower() or "debertav3" in pretrained_model_name_or_path.lower()
        ):
            tokenizer_class = "DebertaV2Tokenizer"
        elif "bert" in pretrained_model_name_or_path.lower():
            tokenizer_class = "BertTokenizer"
        elif "xlnet" in pretrained_model_name_or_path.lower():
            tokenizer_class = "XLNetTokenizer"
        elif "electra" in pretrained_model_name_or_path.lower():
            tokenizer_class = "ElectraTokenizer"
        elif "minilm" in pretrained_model_name_or_path.lower():
            tokenizer_class = "BertTokenizer"
        elif "dpr-question_encoder" in pretrained_model_name_or_path.lower():
            tokenizer_class = "DPRQuestionEncoderTokenizer"
        elif "dpr-ctx_encoder" in pretrained_model_name_or_path.lower():
            tokenizer_class = "DPRContextEncoderTokenizer"
        else:
            raise ValueError(
                f"Could not infer tokenizer_class from model config or "
                f"name '{pretrained_model_name_or_path}'. Set arg `tokenizer_class` "
                f"in Tokenizer.load() to one of: AlbertTokenizer, XLMRobertaTokenizer, "
                f"RobertaTokenizer, DistilBertTokenizer, BertTokenizer, XLNetTokenizer, "
                f"CamembertTokenizer, ElectraTokenizer, DPRQuestionEncoderTokenizer,"
                f"DPRContextEncoderTokenizer."
            )

        return tokenizer_class


def tokenize_batch_question_answering(pre_baskets, tokenizer, indices):
    """
    Tokenizes text data for question answering tasks. Tokenization means splitting words into subwords, depending on the
    tokenizer's vocabulary.

    - We first tokenize all documents in batch mode. (When using FastTokenizers Rust multithreading can be enabled by TODO add how to enable rust mt)
    - Then we tokenize each question individually
    - We construct dicts with question and corresponding document text + tokens + offsets + ids

    :param pre_baskets: input dicts with QA info #todo change to input objects
    :param tokenizer: tokenizer to be used
    :param indices: list, indices used during multiprocessing so that IDs assigned to our baskets are unique
    :return: baskets, list containing question and corresponding document information
    """
    assert len(indices) == len(pre_baskets)
    assert tokenizer.is_fast, (
        "Processing QA data is only supported with fast tokenizers for now.\n"
        "Please load Tokenizers with 'use_fast=True' option."
    )
    baskets = []
    # # Tokenize texts in batch mode
    texts = [d["context"] for d in pre_baskets]
    tokenized_docs_batch = tokenizer.batch_encode_plus(
        texts, return_offsets_mapping=True, return_special_tokens_mask=True, add_special_tokens=False, verbose=False
    )

    # Extract relevant data
    tokenids_batch = tokenized_docs_batch["input_ids"]
    offsets_batch = []
    for o in tokenized_docs_batch["offset_mapping"]:
        offsets_batch.append(np.array([x[0] for x in o]))
    start_of_words_batch = []
    for e in tokenized_docs_batch.encodings:
        start_of_words_batch.append(_get_start_of_word_QA(e.words))

    for i_doc, d in enumerate(pre_baskets):
        document_text = d["context"]
        # # Tokenize questions one by one
        for i_q, q in enumerate(d["qas"]):
            question_text = q["question"]
            tokenized_q = tokenizer.encode_plus(
                question_text, return_offsets_mapping=True, return_special_tokens_mask=True, add_special_tokens=False
            )

            # Extract relevant data
            question_tokenids = tokenized_q["input_ids"]
            question_offsets = [x[0] for x in tokenized_q["offset_mapping"]]
            question_sow = _get_start_of_word_QA(tokenized_q.encodings[0].words)

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
    words = np.array(word_ids)
    start_of_word_single = [1] + list(np.ediff1d(words))
    return start_of_word_single


def tokenize_with_metadata(text: str, tokenizer) -> Dict[str, Any]:
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
    :param tokenizer: Tokenizer (e.g. from Tokenizer.load())
    :return: Dictionary with "tokens", "offsets" and "start_of_word"
    """
    # normalize all other whitespace characters to " "
    # Note: using text.split() directly would destroy the offset,
    # since \n\n\n would be treated similarly as a single \n
    text = re.sub(r"\s", " ", text)
    # Fast Tokenizers return offsets, so we don't need to calculate them ourselves
    if tokenizer.is_fast:
        # tokenized = tokenizer(text, return_offsets_mapping=True, return_special_tokens_mask=True)
        tokenized2 = tokenizer.encode_plus(text, return_offsets_mapping=True, return_special_tokens_mask=True)

        tokens2 = tokenized2["input_ids"]
        offsets2 = np.array([x[0] for x in tokenized2["offset_mapping"]])
        # offsets2 = [x[0] for x in tokenized2["offset_mapping"]]
        words = np.array(tokenized2.encodings[0].words)

        # TODO check for validity for all tokenizer and special token types
        words[0] = -1
        words[-1] = words[-2]
        words += 1
        start_of_word2 = [0] + list(np.ediff1d(words))
        #######

        # start_of_word3 = []
        # last_word = -1
        # for word_id in tokenized2.encodings[0].words:
        #     if word_id is None or word_id == last_word:
        #         start_of_word3.append(0)
        #     else:
        #         start_of_word3.append(1)
        #         last_word = word_id

        tokenized_dict = {"tokens": tokens2, "offsets": offsets2, "start_of_word": start_of_word2}
    else:
        # split text into "words" (here: simple whitespace tokenizer).
        words = text.split(" ")
        word_offsets = []
        cumulated = 0
        for idx, word in enumerate(words):
            word_offsets.append(cumulated)
            cumulated += len(word) + 1  # 1 because we so far have whitespace tokenizer

        # split "words" into "subword tokens"
        tokens, offsets, start_of_word = _words_to_tokens(words, word_offsets, tokenizer)
        tokenized_dict = {"tokens": tokens, "offsets": offsets, "start_of_word": start_of_word}
    return tokenized_dict


def truncate_sequences(
    seq_a: list,
    seq_b: Optional[list],
    tokenizer,
    max_seq_len: int,
    truncation_strategy: str = "longest_first",
    with_special_tokens: bool = True,
    stride: int = 0,
) -> Tuple[List[Any], Optional[List[Any]], List[Any]]:
    """
    Reduces a single sequence or a pair of sequences to a maximum sequence length.
    The sequences can contain tokens or any other elements (offsets, masks ...).
    If `with_special_tokens` is enabled, it'll remove some additional tokens to have exactly enough space for later adding special tokens (CLS, SEP etc.)

    Supported truncation strategies:

    - longest_first: (default) Iteratively reduce the inputs sequence until the input is under max_length starting from the longest one at each token (when there is a pair of input sequences). Overflowing tokens only contains overflow from the first sequence.
    - only_first: Only truncate the first sequence. raise an error if the first sequence is shorter or equal to than num_tokens_to_remove.
    - only_second: Only truncate the second sequence
    - do_not_truncate: Does not truncate (raise an error if the input sequence is longer than max_length)

    :param seq_a: First sequence of tokens/offsets/...
    :param seq_b: Optional second sequence of tokens/offsets/...
    :param tokenizer: Tokenizer (e.g. from Tokenizer.load())
    :param max_seq_len:
    :param truncation_strategy: how the sequence(s) should be truncated down. Default: "longest_first" (see above for other options).
    :param with_special_tokens: If true, it'll remove some additional tokens to have exactly enough space for later adding special tokens (CLS, SEP etc.)
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


def _words_to_tokens(words, word_offsets, tokenizer):
    """
    Tokenize "words" into subword tokens while keeping track of offsets and if a token is the start of a word.
    :param words: list of words.
    :type words: list
    :param word_offsets: Character indices where each word begins in the original text
    :type word_offsets: list
    :param tokenizer: Tokenizer (e.g. from Tokenizer.load())
    :return: tokens, offsets, start_of_word
    """
    tokens = []
    token_offsets = []
    start_of_word = []
    idx = 0
    for w, w_off in zip(words, word_offsets):
        idx += 1
        if idx % 500000 == 0:
            logger.info(idx)
        # Get (subword) tokens of single word.

        # empty / pure whitespace
        if len(w) == 0:
            continue
        # For the first word of a text: we just call the regular tokenize function.
        # For later words: we need to call it with add_prefix_space=True to get the same results with roberta / gpt2 tokenizer
        # see discussion here. https://github.com/huggingface/transformers/issues/1196
        if len(tokens) == 0:
            tokens_word = tokenizer.tokenize(w)
        else:
            if type(tokenizer) == RobertaTokenizer:
                tokens_word = tokenizer.tokenize(w, add_prefix_space=True)
            else:
                tokens_word = tokenizer.tokenize(w)
        # Sometimes the tokenizer returns no tokens
        if len(tokens_word) == 0:
            continue
        tokens += tokens_word

        # get global offset for each token in word + save marker for first tokens of a word
        first_tok = True
        for tok in tokens_word:
            token_offsets.append(w_off)
            # Depending on the tokenizer type special chars are added to distinguish tokens with preceeding
            # whitespace (=> "start of a word"). We need to get rid of these to calculate the original length of the token
            orig_tok = re.sub(SPECIAL_TOKENIZER_CHARS, "", tok)
            # Don't use length of unk token for offset calculation
            if orig_tok == tokenizer.special_tokens_map["unk_token"]:
                w_off += 1
            else:
                w_off += len(orig_tok)
            if first_tok:
                start_of_word.append(True)
                first_tok = False
            else:
                start_of_word.append(False)

    return tokens, token_offsets, start_of_word
