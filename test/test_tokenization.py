import logging
import pytest
import re
from transformers import BertTokenizer, BertTokenizerFast, RobertaTokenizer, RobertaTokenizerFast, \
    XLNetTokenizer, XLNetTokenizerFast, ElectraTokenizerFast

from tokenizers.pre_tokenizers import WhitespaceSplit

from haystack.modeling.model.tokenization import Tokenizer

import numpy as np


TEXTS = [
    "This is a sentence",
    "Der entscheidende Pass",
    "This      is a sentence with multiple spaces",
    "力加勝北区ᴵᴺᵀᵃছজটডণত",
    "Thiso text is included tolod makelio sure Unicodeel is handled properly:",
    "This is a sentence...",
    "Let's see all on this text and. !23# neverseenwordspossible",
    """This is a sentence.
    With linebreak""",
    """Sentence with multiple
    newlines
    """,
    "and another one\n\n\nwithout space",
    "This is a sentence	with tab",
    "This is a sentence			with multiple tabs",
]


def test_basic_loading(caplog):
    caplog.set_level(logging.CRITICAL)
    # slow tokenizers
    tokenizer = Tokenizer.load(
        pretrained_model_name_or_path="bert-base-cased",
        do_lower_case=True,
        use_fast=False,
        )
    assert type(tokenizer) == BertTokenizer
    assert tokenizer.basic_tokenizer.do_lower_case == True

    tokenizer = Tokenizer.load(
        pretrained_model_name_or_path="xlnet-base-cased",
        do_lower_case=True,
        use_fast=False
        )
    assert type(tokenizer) == XLNetTokenizer
    assert tokenizer.do_lower_case == True

    tokenizer = Tokenizer.load(
        pretrained_model_name_or_path="roberta-base",
        use_fast=False
        )
    assert type(tokenizer) == RobertaTokenizer

    # fast tokenizers
    tokenizer = Tokenizer.load(
        pretrained_model_name_or_path="bert-base-cased",
        do_lower_case=True
    )
    assert type(tokenizer) == BertTokenizerFast
    assert tokenizer.do_lower_case == True

    tokenizer = Tokenizer.load(
        pretrained_model_name_or_path="xlnet-base-cased",
        do_lower_case=True
    )
    assert type(tokenizer) == XLNetTokenizerFast
    assert tokenizer.do_lower_case == True

    tokenizer = Tokenizer.load(
        pretrained_model_name_or_path="roberta-base"
    )
    assert type(tokenizer) == RobertaTokenizerFast


def test_bert_tokenizer_all_meta(caplog):
    caplog.set_level(logging.CRITICAL)

    lang_model = "bert-base-cased"

    tokenizer = Tokenizer.load(
        pretrained_model_name_or_path=lang_model,
        do_lower_case=False
        )

    basic_text = "Some Text with neverseentokens plus !215?#. and a combined-token_with/chars"

    tokenized = tokenizer.tokenize(basic_text)
    assert tokenized == ['Some', 'Text', 'with', 'never', '##see', '##nto', '##ken', '##s', 'plus', '!', '215', '?', '#', '.', 'and', 'a', 'combined', '-', 'token', '_', 'with', '/', 'ch', '##ars']

    encoded_batch = tokenizer.encode_plus(basic_text)
    encoded = encoded_batch.encodings[0]
    words = np.array(encoded.words)
    words[words == None] = -1
    start_of_word_single = [False] + list(np.ediff1d(words) > 0)
    assert encoded.tokens == ['[CLS]', 'Some', 'Text', 'with', 'never', '##see', '##nto', '##ken', '##s', 'plus', '!', '215', '?', '#', '.', 'and', 'a', 'combined', '-', 'token', '_', 'with', '/', 'ch', '##ars', '[SEP]']
    assert [x[0] for x in encoded.offsets] == [0, 0, 5, 10, 15, 20, 23, 26, 29, 31, 36, 37, 40, 41, 42, 44, 48, 50, 58, 59, 64, 65, 69, 70, 72, 0]
    assert start_of_word_single == [False, True, True, True, True, False, False, False, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, False]

def test_save_load(caplog):
    caplog.set_level(logging.CRITICAL)

    lang_names = ["bert-base-cased", "roberta-base", "xlnet-base-cased"]
    tokenizers = []
    for lang_name in lang_names:
        if "xlnet" in lang_name.lower():
            t = Tokenizer.load(lang_name, lower_case=False, use_fast=True, from_slow=True)
        else:
            t = Tokenizer.load(lang_name, lower_case=False)
        t.add_tokens(new_tokens=["neverseentokens"])
        tokenizers.append(t)

    basic_text = "Some Text with neverseentokens plus !215?#. and a combined-token_with/chars"

    for tokenizer in tokenizers:
        tokenizer_type = tokenizer.__class__.__name__
        save_dir = f"testsave/{tokenizer_type}"
        tokenizer.save_pretrained(save_dir)
        tokenizer_loaded = Tokenizer.load(save_dir, tokenizer_class=tokenizer_type)
        encoded_before = tokenizer.encode_plus(basic_text).encodings[0]
        encoded_after = tokenizer_loaded.encode_plus(basic_text).encodings[0]
        data_before = {"tokens": encoded_before.tokens,
                       "offsets": encoded_before.offsets,
                       "words": encoded_before.words}
        data_after = {"tokens": encoded_after.tokens,
                       "offsets": encoded_after.offsets,
                       "words": encoded_after.words}
        assert data_before == data_after


@pytest.mark.parametrize("model_name", ["bert-base-german-cased",
                         "google/electra-small-discriminator",
                         ])
def test_fast_tokenizer_with_examples(caplog, model_name):
    fast_tokenizer = Tokenizer.load(model_name, lower_case=False, use_fast=True)
    tokenizer = Tokenizer.load(model_name, lower_case=False, use_fast=False)

    for text in TEXTS:
            # plain tokenize function
            tokenized = tokenizer.tokenize(text)
            fast_tokenized = fast_tokenizer.tokenize(text)

            assert tokenized == fast_tokenized


def test_all_tokenizer_on_special_cases(caplog):
    caplog.set_level(logging.CRITICAL)

    lang_names = ["bert-base-cased", "roberta-base", "xlnet-base-cased"]

    tokenizers = []
    for lang_name in lang_names:
        if "roberta" in lang_name:
            add_prefix_space = True
        else:
            add_prefix_space = False
        t = Tokenizer.load(lang_name, lower_case=False, add_prefix_space=add_prefix_space)
        tokenizers.append(t)

    texts = [
     "This is a sentence",
     "Der entscheidende Pass",
     "力加勝北区ᴵᴺᵀᵃছজটডণত",
     "Thiso text is included tolod makelio sure Unicodeel is handled properly:",
     "This is a sentence...",
     "Let's see all on this text and. !23# neverseenwordspossible"
     "This      is a sentence with multiple spaces",
      """This is a sentence.
      With linebreak""",
      """Sentence with multiple
      newlines
      """,
      "and another one\n\n\nwithout space",
      "This is a sentence			with multiple tabs",
    ]

    expected_to_fail = [(1, 1), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (2, 1), (2, 5)]

    for i_tok, tokenizer in enumerate(tokenizers):
        for i_text, text in enumerate(texts):
            # Important: we don't assume to preserve whitespaces after tokenization.
            # This means: \t, \n " " etc will all resolve to a single " ".
            # This doesn't make a difference for BERT + XLNet but it does for roBERTa

            test_passed = True

            # 1. original tokenize function from transformer repo on full sentence
            standardized_whitespace_text = ' '.join(text.split()) # remove multiple whitespaces
            tokenized = tokenizer.tokenize(standardized_whitespace_text)

            # 2. Our tokenization method using a pretokenizer which can normalize multiple white spaces
            # This approach is used in NER
            pre_tokenizer = WhitespaceSplit()
            words_and_spans = pre_tokenizer.pre_tokenize_str(text)
            words = [x[0] for x in words_and_spans]
            word_spans = [x[1] for x in words_and_spans]

            encoded = tokenizer.encode_plus(words, is_split_into_words=True, add_special_tokens=False).encodings[0]

            # verify that tokenization on full sequence is the same as the one on "whitespace tokenized words"
            if encoded.tokens != tokenized:
                test_passed = False

            # token offsets are originally relative to the beginning of the word
            # These lines convert them so they are relative to the beginning of the sentence
            token_offsets = []
            for (start, end), w_index, in zip(encoded.offsets, encoded.words):
                word_start_ch = word_spans[w_index][0]
                token_offsets.append((start + word_start_ch, end + word_start_ch))
            if getattr(tokenizer, "add_prefix_space", None):
                token_offsets = [(start-1, end) for start, end in token_offsets]

            # verify that offsets align back to original text
            if text == "力加勝北区ᴵᴺᵀᵃছজটডণত":
                # contains [UNK] that are impossible to match back to original text space
                continue
            for tok, (start, end) in zip(encoded.tokens, token_offsets):
                #subword-tokens have special chars depending on model type. In order to align with original text we need to get rid of them
                tok = re.sub(r"^(##|Ġ|▁)", "", tok)
                #tok = tokenizer.decode(tokenizer.convert_tokens_to_ids(tok))
                original_tok = text[start: end]
                if tok != original_tok:
                    test_passed = False
            if (i_tok, i_text) in expected_to_fail:
                assert not test_passed, f"Behaviour of {tokenizer.__class__.__name__} has changed on text {text}'"
            else:
                assert test_passed, f"Behaviour of {tokenizer.__class__.__name__} has changed on text {text}'"


def test_bert_custom_vocab(caplog):
    caplog.set_level(logging.CRITICAL)

    lang_model = "bert-base-cased"

    tokenizer = Tokenizer.load(
        pretrained_model_name_or_path=lang_model,
        do_lower_case=False
        )

    #deprecated: tokenizer.add_custom_vocab("samples/tokenizer/custom_vocab.txt")
    tokenizer.add_tokens(new_tokens=["neverseentokens"])

    basic_text = "Some Text with neverseentokens plus !215?#. and a combined-token_with/chars"

    # original tokenizer from transformer repo
    tokenized = tokenizer.tokenize(basic_text)
    assert tokenized == ['Some', 'Text', 'with', 'neverseentokens', 'plus', '!', '215', '?', '#', '.', 'and', 'a', 'combined', '-', 'token', '_', 'with', '/', 'ch', '##ars']

    # ours with metadata
    encoded = tokenizer.encode_plus(basic_text, add_special_tokens=False).encodings[0]
    offsets = [x[0] for x in encoded.offsets]
    start_of_word_single = [True] + list(np.ediff1d(encoded.words) > 0)
    assert encoded.tokens == tokenized
    assert offsets == [0, 5, 10, 15, 31, 36, 37, 40, 41, 42, 44, 48, 50, 58, 59, 64, 65, 69, 70, 72]
    assert start_of_word_single == [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False]


def test_fast_bert_custom_vocab(caplog):
    caplog.set_level(logging.CRITICAL)

    lang_model = "bert-base-cased"

    tokenizer = Tokenizer.load(
        pretrained_model_name_or_path=lang_model,
        do_lower_case=False, use_fast=True
        )

    #deprecated: tokenizer.add_custom_vocab("samples/tokenizer/custom_vocab.txt")
    tokenizer.add_tokens(new_tokens=["neverseentokens"])

    basic_text = "Some Text with neverseentokens plus !215?#. and a combined-token_with/chars"

    # original tokenizer from transformer repo
    tokenized = tokenizer.tokenize(basic_text)
    assert tokenized == ['Some', 'Text', 'with', 'neverseentokens', 'plus', '!', '215', '?', '#', '.', 'and', 'a', 'combined', '-', 'token', '_', 'with', '/', 'ch', '##ars']

    # ours with metadata
    encoded = tokenizer.encode_plus(basic_text, add_special_tokens=False).encodings[0]
    offsets = [x[0] for x in encoded.offsets]
    start_of_word_single = [True] + list(np.ediff1d(encoded.words) > 0)
    assert encoded.tokens == tokenized
    assert offsets == [0, 5, 10, 15, 31, 36, 37, 40, 41, 42, 44, 48, 50, 58, 59, 64, 65, 69, 70, 72]
    assert start_of_word_single == [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False]


@pytest.mark.parametrize("model_name, tokenizer_type", [
                         ("bert-base-german-cased", BertTokenizerFast),
                         ("google/electra-small-discriminator", ElectraTokenizerFast),
                         ])
def test_fast_tokenizer_type(caplog, model_name, tokenizer_type):
    caplog.set_level(logging.CRITICAL)

    tokenizer = Tokenizer.load(model_name, use_fast=True)
    assert type(tokenizer) is tokenizer_type

# See discussion in https://github.com/deepset-ai/FARM/pull/624 for reason to remove the test
# def test_fast_bert_tokenizer_strip_accents(caplog):
#     caplog.set_level(logging.CRITICAL)
#
#     tokenizer = Tokenizer.load("dbmdz/bert-base-german-uncased",
#                                use_fast=True,
#                                strip_accents=False)
#     assert type(tokenizer) is BertTokenizerFast
#     assert tokenizer.do_lower_case
#     assert tokenizer._tokenizer._parameters['strip_accents'] is False


def test_fast_electra_tokenizer(caplog):
    caplog.set_level(logging.CRITICAL)

    tokenizer = Tokenizer.load("dbmdz/electra-base-german-europeana-cased-discriminator",
                               use_fast=True)
    assert type(tokenizer) is ElectraTokenizerFast


@pytest.mark.parametrize("model_name", ["bert-base-cased", "distilbert-base-uncased", "deepset/electra-base-squad2"])
def test_detokenization_in_fast_tokenizers(model_name):
    tokenizer = Tokenizer.load(
        pretrained_model_name_or_path=model_name,
        use_fast=True
    )
    for text in TEXTS:
        encoded = tokenizer.encode_plus(text, add_special_tokens=False).encodings[0]

        detokenized = " ".join(encoded.tokens)
        detokenized = re.sub(r"(^|\s+)(##)", "", detokenized)

        detokenized_ids = tokenizer(detokenized, add_special_tokens=False)["input_ids"]
        detokenized_tokens = [tokenizer.decode([tok_id]).strip() for tok_id in detokenized_ids]

        assert encoded.tokens == detokenized_tokens


if __name__ == "__main__":
    test_all_tokenizer_on_special_cases()