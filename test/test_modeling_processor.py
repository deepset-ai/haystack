import logging

from transformers import AutoTokenizer

from haystack.modeling.data_handler.processor import SquadProcessor
from haystack.modeling.model.tokenization import Tokenizer


# during inference (parameter return_baskets = False) we do not convert labels
def test_dataset_from_dicts_qa_inference(caplog=None):
    if caplog:
        caplog.set_level(logging.CRITICAL)

    models = [
        "deepset/roberta-base-squad2",
        "deepset/bert-base-cased-squad2",
        "deepset/xlm-roberta-large-squad2",
        "deepset/minilm-uncased-squad2",
        "deepset/electra-base-squad2",
        ]
    sample_types = ["answer-wrong", "answer-offset-wrong", "noanswer", "vanilla"]

    for model in models:
        tokenizer = Tokenizer.load(pretrained_model_name_or_path=model, use_fast=True)
        processor = SquadProcessor(tokenizer, max_seq_len=256, data_dir=None)

        for sample_type in sample_types:
            dicts = processor.file_to_dicts(f"samples/qa/{sample_type}.json")
            dataset, tensor_names, problematic_sample_ids, baskets = processor.dataset_from_dicts(dicts, indices=[1], return_baskets=True)
            assert tensor_names == ['input_ids', 'padding_mask', 'segment_ids', 'passage_start_t', 'start_of_word', 'labels', 'id', 'seq_2_start_t', 'span_mask'], f"Processing for {model} has changed."
            assert len(problematic_sample_ids) == 0, f"Processing for {model} has changed."
            assert baskets[0].id_external == '5ad3d560604f3c001a3ff2c8', f"Processing for {model} has changed."
            assert baskets[0].id_internal == '1-0', f"Processing for {model} has changed."

            # roberta
            if model == "deepset/roberta-base-squad2":
                assert len(baskets[0].samples[0].tokenized["passage_tokens"]) == 6, f"Processing for {model} has changed."
                assert len(baskets[0].samples[0].tokenized["question_tokens"]) == 7, f"Processing for {model} has changed."
                if sample_type == "noanswer":
                    assert baskets[0].samples[0].features[0]["input_ids"][:13] == \
                           [0, 6179, 171, 82, 697, 11, 2201, 116, 2, 2, 26795, 2614, 34], \
                        f"Processing for {model} and {sample_type}-testsample has changed."
                else:
                    assert baskets[0].samples[0].features[0]["input_ids"][:13] == \
                           [0, 6179, 171, 82, 697, 11, 5459, 116, 2, 2, 26795, 2614, 34], \
                        f"Processing for {model} and {sample_type}-testsample has changed."

            # bert
            if model == "deepset/bert-base-cased-squad2":
                assert len(baskets[0].samples[0].tokenized["passage_tokens"]) == 5, f"Processing for {model} has changed."
                assert len(baskets[0].samples[0].tokenized["question_tokens"]) == 7, f"Processing for {model} has changed."
                if sample_type == "noanswer":
                    assert baskets[0].samples[0].features[0]["input_ids"][:10] == \
                           [101, 1731, 1242, 1234, 1686, 1107, 2123, 136, 102, 3206], \
                        f"Processing for {model} and {sample_type}-testsample has changed."
                else:
                    assert baskets[0].samples[0].features[0]["input_ids"][:10] == \
                           [101, 1731, 1242, 1234, 1686, 1107, 3206, 136, 102, 3206], \
                        f"Processing for {model} and {sample_type}-testsample has changed."

            # xlm-roberta
            if model ==  "deepset/xlm-roberta-large-squad2":
                assert len(baskets[0].samples[0].tokenized["passage_tokens"]) == 7, f"Processing for {model} has changed."
                assert len(baskets[0].samples[0].tokenized["question_tokens"]) == 7, f"Processing for {model} has changed."
                if sample_type == "noanswer":
                    assert baskets[0].samples[0].features[0]["input_ids"][:12] == \
                           [0, 11249, 5941, 3395, 6867, 23, 7270, 32, 2, 2, 10271, 1556], \
                        f"Processing for {model} and {sample_type}-testsample has changed."
                else:
                    assert baskets[0].samples[0].features[0]["input_ids"][:12] == \
                           [0, 11249, 5941, 3395, 6867, 23, 10271, 32, 2, 2, 10271, 1556], \
                        f"Processing for {model} and {sample_type}-testsample has changed."

            # minilm and electra have same vocab + tokenizer
            if model == "deepset/minilm-uncased-squad2" or model == "deepset/electra-base-squad2":
                assert len(baskets[0].samples[0].tokenized["passage_tokens"]) == 5, f"Processing for {model} has changed."
                assert len(baskets[0].samples[0].tokenized["question_tokens"]) == 7, f"Processing for {model} has changed."
                if sample_type == "noanswer":
                    assert baskets[0].samples[0].features[0]["input_ids"][:10] == \
                           [101, 2129, 2116, 2111, 2444, 1999, 3000, 1029, 102, 4068], \
                        f"Processing for {model} and {sample_type}-testsample has changed."
                else:
                    assert baskets[0].samples[0].features[0]["input_ids"][:10] == \
                           [101, 2129, 2116, 2111, 2444, 1999, 4068, 1029, 102, 4068], \
                        f"Processing for {model} and {sample_type}-testsample has changed."


def test_batch_encoding_flatten_rename():
    from haystack.modeling.data_handler.dataset import flatten_rename

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    batch_sentences = ["Hello I'm a single sentence", "And another sentence", "And the very very last one"]
    encoded_inputs = tokenizer(batch_sentences, padding=True, truncation=True)

    keys = ["input_ids", "token_type_ids", "attention_mask"]
    rename_keys = ["input_ids", "segment_ids", "padding_mask"]
    features_flat = flatten_rename(encoded_inputs, keys, rename_keys)

    assert len(features_flat) == 3, "should have three elements in the feature dict list"
    for e in features_flat:
        for k in rename_keys:
            assert k in e, f"feature dict list item {e} in a list should have a key {k}"

    # rename no keys/rename keys
    features_flat = flatten_rename(encoded_inputs)
    assert len(features_flat) == 3, "should have three elements in the feature dict list"
    for e in features_flat:
        for k in keys:
            assert k in e, f"feature dict list item {e} in a list should have a key {k}"

    # empty input keys
    flatten_rename(encoded_inputs, [])

    # empty keys and rename keys
    flatten_rename(encoded_inputs, [], [])

    # no encoding_batch provided
    flatten_rename(None, [], [])

    # keys and renamed_keys have different sizes
    try:
        flatten_rename(encoded_inputs, [], ["blah"])
    except AssertionError:
        pass


def test_dataset_from_dicts_qa_labelconversion(caplog=None):
    if caplog:
        caplog.set_level(logging.CRITICAL)

    models = [
        "deepset/roberta-base-squad2",
        "deepset/bert-base-cased-squad2",
        "deepset/xlm-roberta-large-squad2",
        "deepset/minilm-uncased-squad2",
        "deepset/electra-base-squad2",
        ]
    sample_types = ["answer-wrong", "answer-offset-wrong", "noanswer", "vanilla"]

    for model in models:
        tokenizer = Tokenizer.load(pretrained_model_name_or_path=model, use_fast=True)
        processor = SquadProcessor(tokenizer, max_seq_len=256, data_dir=None)

        for sample_type in sample_types:
            dicts = processor.file_to_dicts(f"samples/qa/{sample_type}.json")
            dataset, tensor_names, problematic_sample_ids = processor.dataset_from_dicts(dicts, indices=[1], return_baskets=False)

            if sample_type == "answer-wrong" or sample_type == "answer-offset-wrong":
                assert len(problematic_sample_ids) == 1, f"Processing labels for {model} has changed."

            if sample_type == "noanswer":
                assert list(dataset.tensors[tensor_names.index("labels")].numpy()[0, 0, :]) == [0, 0], f"Processing labels for {model} has changed."
                assert list(dataset.tensors[tensor_names.index("labels")].numpy()[0, 1, :]) == [-1, -1], f"Processing labels for {model} has changed."

            if sample_type == "vanilla":
                # roberta
                if model == "deepset/roberta-base-squad2":
                    assert list(dataset.tensors[tensor_names.index("labels")].numpy()[0,0,:]) == [13, 13], f"Processing labels for {model} has changed."
                    assert list(dataset.tensors[tensor_names.index("labels")].numpy()[0,1,:]) == [13, 14], f"Processing labels for {model} has changed."
                # bert, minilm, electra
                if model == "deepset/bert-base-cased-squad2" or model == "deepset/minilm-uncased-squad2" or model == "deepset/electra-base-squad2":
                    assert list(dataset.tensors[tensor_names.index("labels")].numpy()[0,0,:]) == [11, 11], f"Processing labels for {model} has changed."
                # xlm-roberta
                if model ==  "deepset/xlm-roberta-large-squad2":
                    assert list(dataset.tensors[tensor_names.index("labels")].numpy()[0,0,:]) == [12, 12], f"Processing labels for {model} has changed."


if(__name__=="__main__"):
    test_dataset_from_dicts_qa_labelconversion()