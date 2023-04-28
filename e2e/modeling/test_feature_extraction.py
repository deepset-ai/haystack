import re

import pytest
import numpy as np
from tokenizers.pre_tokenizers import WhitespaceSplit

from haystack.modeling.model.feature_extraction import FeatureExtractor


BERT = "bert-base-cased"
ROBERTA = "roberta-base"
XLNET = "xlnet-base-cased"

TOKENIZERS_TO_TEST = [BERT, ROBERTA, XLNET]
TOKENIZERS_TO_TEST_WITH_TOKEN_MARKER = [(BERT, "##"), (ROBERTA, "Ġ"), (XLNET, "▁")]

REGULAR_SENTENCE = "This is a sentence"
GERMAN_SENTENCE = "Der entscheidende Pass"
OTHER_ALPHABETS = "力加勝北区ᴵᴺᵀᵃছজটডণত"
GIBBERISH_SENTENCE = "Thiso text is included tolod makelio sure Unicodeel is handled properly:"
SENTENCE_WITH_ELLIPSIS = "This is a sentence..."
SENTENCE_WITH_LINEBREAK_1 = "and another one\n\n\nwithout space"
SENTENCE_WITH_LINEBREAK_2 = """This is a sentence.
    With linebreak"""
SENTENCE_WITH_LINEBREAKS = """Sentence
    with
    multiple
    newlines
    """
SENTENCE_WITH_EXCESS_WHITESPACE = "This      is a sentence with multiple spaces"
SENTENCE_WITH_TABS = "This is a sentence			with multiple tabs"
SENTENCE_WITH_CUSTOM_TOKEN = "Let's see all on this text and. !23# neverseenwordspossible"


def convert_offset_from_word_reference_to_text_reference(offsets, words, word_spans):
    """
    Token offsets are originally relative to the beginning of the word
    We make them relative to the beginning of the sentence.

    Not a fixture, just a utility.
    """
    token_offsets = []
    for (start, end), word_index in zip(offsets, words):
        word_start = word_spans[word_index][0]
        token_offsets.append((start + word_start, end + word_start))
    return token_offsets


@pytest.mark.parametrize("model_name", TOKENIZERS_TO_TEST)
def test_save_load(tmp_path, model_name: str):
    tokenizer = FeatureExtractor(pretrained_model_name_or_path=model_name, do_lower_case=False)
    text = "Some Text with neverseentokens plus !215?#. and a combined-token_with/chars"

    tokenizer.feature_extractor.add_tokens(new_tokens=["neverseentokens"])
    original_encoding = tokenizer.feature_extractor(text)

    save_dir = tmp_path / "saved_tokenizer"
    tokenizer.feature_extractor.save_pretrained(save_dir)

    tokenizer_loaded = FeatureExtractor(pretrained_model_name_or_path=save_dir)
    new_encoding = tokenizer_loaded.feature_extractor(text)

    assert original_encoding == new_encoding


@pytest.mark.parametrize(
    "edge_case",
    [
        REGULAR_SENTENCE,
        OTHER_ALPHABETS,
        GIBBERISH_SENTENCE,
        SENTENCE_WITH_ELLIPSIS,
        SENTENCE_WITH_LINEBREAK_1,
        SENTENCE_WITH_LINEBREAK_2,
        SENTENCE_WITH_LINEBREAKS,
        SENTENCE_WITH_EXCESS_WHITESPACE,
        SENTENCE_WITH_TABS,
    ],
)
@pytest.mark.parametrize("model_name", TOKENIZERS_TO_TEST)
def test_tokenization_on_edge_cases_full_sequence_tokenization(model_name: str, edge_case: str):
    """
    Verify that tokenization on full sequence is the same as the one on "whitespace tokenized words"
    """
    tokenizer = FeatureExtractor(pretrained_model_name_or_path=model_name, do_lower_case=False, add_prefix_space=True)

    pre_tokenizer = WhitespaceSplit()
    words_and_spans = pre_tokenizer.pre_tokenize_str(edge_case)
    words = [x[0] for x in words_and_spans]

    encoded = tokenizer.feature_extractor(words, is_split_into_words=True, add_special_tokens=False).encodings[0]
    expected_tokenization = tokenizer.feature_extractor.tokenize(
        " ".join(edge_case.split())
    )  # remove multiple whitespaces

    assert encoded.tokens == expected_tokenization


@pytest.mark.parametrize("edge_case", [SENTENCE_WITH_CUSTOM_TOKEN, GERMAN_SENTENCE])
@pytest.mark.parametrize("model_name", [t for t in TOKENIZERS_TO_TEST if t != ROBERTA])
def test_tokenization_on_edge_cases_full_sequence_tokenization_roberta_exceptions(model_name: str, edge_case: str):
    """
    Verify that tokenization on full sequence is the same as the one on "whitespace tokenized words".
    These test cases work for all tokenizers under test except for RoBERTa.
    """
    tokenizer = FeatureExtractor(pretrained_model_name_or_path=model_name, do_lower_case=False, add_prefix_space=True)

    pre_tokenizer = WhitespaceSplit()
    words_and_spans = pre_tokenizer.pre_tokenize_str(edge_case)
    words = [x[0] for x in words_and_spans]

    encoded = tokenizer.feature_extractor(words, is_split_into_words=True, add_special_tokens=False).encodings[0]
    expected_tokenization = tokenizer.feature_extractor.tokenize(
        " ".join(edge_case.split())
    )  # remove multiple whitespaces

    assert encoded.tokens == expected_tokenization


@pytest.mark.parametrize(
    "edge_case",
    [
        REGULAR_SENTENCE,
        # OTHER_ALPHABETS,  # contains [UNK] that are impossible to match back to original text space
        GIBBERISH_SENTENCE,
        SENTENCE_WITH_ELLIPSIS,
        SENTENCE_WITH_LINEBREAK_1,
        SENTENCE_WITH_LINEBREAK_2,
        SENTENCE_WITH_LINEBREAKS,
        SENTENCE_WITH_EXCESS_WHITESPACE,
        SENTENCE_WITH_TABS,
    ],
)
@pytest.mark.parametrize("model_name,marker", TOKENIZERS_TO_TEST_WITH_TOKEN_MARKER)
def test_tokenization_on_edge_cases_full_sequence_verify_spans(model_name: str, marker: str, edge_case: str):
    tokenizer = FeatureExtractor(pretrained_model_name_or_path=model_name, do_lower_case=False, add_prefix_space=True)

    pre_tokenizer = WhitespaceSplit()
    words_and_spans = pre_tokenizer.pre_tokenize_str(edge_case)
    words = [x[0] for x in words_and_spans]
    word_spans = [x[1] for x in words_and_spans]

    encoded = tokenizer.feature_extractor(words, is_split_into_words=True, add_special_tokens=False).encodings[0]

    # subword-tokens have special chars depending on model type. To align with original text we get rid of them
    tokens = [token.replace(marker, "") for token in encoded.tokens]
    token_offsets = convert_offset_from_word_reference_to_text_reference(encoded.offsets, encoded.words, word_spans)

    for token, (start, end) in zip(tokens, token_offsets):
        assert token == edge_case[start:end]


@pytest.mark.parametrize(
    "edge_case",
    [
        REGULAR_SENTENCE,
        GERMAN_SENTENCE,
        SENTENCE_WITH_EXCESS_WHITESPACE,
        OTHER_ALPHABETS,
        GIBBERISH_SENTENCE,
        SENTENCE_WITH_ELLIPSIS,
        SENTENCE_WITH_CUSTOM_TOKEN,
        SENTENCE_WITH_LINEBREAK_1,
        SENTENCE_WITH_LINEBREAK_2,
        SENTENCE_WITH_LINEBREAKS,
        SENTENCE_WITH_TABS,
    ],
)
def test_detokenization_for_bert(edge_case):
    tokenizer = FeatureExtractor(pretrained_model_name_or_path=BERT, do_lower_case=False)

    encoded = tokenizer(edge_case, add_special_tokens=False).encodings[0]

    detokenized = " ".join(encoded.tokens)
    detokenized = re.sub(r"(^|\s+)(##)", "", detokenized)

    detokenized_ids = tokenizer(detokenized, add_special_tokens=False)["input_ids"]
    detokenized_tokens = [tokenizer.decode([tok_id]).strip() for tok_id in detokenized_ids]

    assert encoded.tokens == detokenized_tokens


def test_encode_plus_for_bert():
    tokenizer = FeatureExtractor(pretrained_model_name_or_path=BERT, do_lower_case=False)
    text = "Some Text with neverseentokens plus !215?#. and a combined-token_with/chars"

    encoded_batch = tokenizer(text)
    encoded = encoded_batch.encodings[0]

    words = np.array(encoded.words)
    words[0] = -1
    words[-1] = -1

    print(words.tolist())

    tokens = encoded.tokens
    offsets = [x[0] for x in encoded.offsets]
    start_of_word = [False] + list(np.ediff1d(words) > 0)

    assert list(zip(tokens, offsets, start_of_word)) == [
        ("[CLS]", 0, False),
        ("Some", 0, True),
        ("Text", 5, True),
        ("with", 10, True),
        ("never", 15, True),
        ("##see", 20, False),
        ("##nto", 23, False),
        ("##ken", 26, False),
        ("##s", 29, False),
        ("plus", 31, True),
        ("!", 36, True),
        ("215", 37, True),
        ("?", 40, True),
        ("#", 41, True),
        (".", 42, True),
        ("and", 44, True),
        ("a", 48, True),
        ("combined", 50, True),
        ("-", 58, True),
        ("token", 59, True),
        ("_", 64, True),
        ("with", 65, True),
        ("/", 69, True),
        ("ch", 70, True),
        ("##ars", 72, False),
        ("[SEP]", 0, False),
    ]


def test_tokenize_custom_vocab_bert():
    tokenizer = FeatureExtractor(pretrained_model_name_or_path=BERT, do_lower_case=False)

    tokenizer.feature_extractor.add_tokens(new_tokens=["neverseentokens"])
    text = "Some Text with neverseentokens plus !215?#. and a combined-token_with/chars"

    tokenized = tokenizer.feature_extractor.tokenize(text)
    assert (
        tokenized == "Some Text with neverseentokens plus ! 215 ? # . and a combined - token _ with / ch ##ars".split()
    )
    encoded = tokenizer.feature_extractor(text, add_special_tokens=False).encodings[0]
    offsets = [x[0] for x in encoded.offsets]
    start_of_word_single = [True] + list(np.ediff1d(encoded.words) > 0)

    assert encoded.tokens == tokenized
    assert offsets == [0, 5, 10, 15, 31, 36, 37, 40, 41, 42, 44, 48, 50, 58, 59, 64, 65, 69, 70, 72]
    assert start_of_word_single == [True] * 19 + [False]
