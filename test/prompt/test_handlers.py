import pytest

from haystack.nodes.prompt.invocation_layer.handlers import DefaultPromptResizer


@pytest.mark.unit
def test_gpt2_prompt_resizer():
    # test gpt2 BPE based tokenizer
    resizer = DefaultPromptResizer(model_name_or_path="gpt2", model_max_length=20, max_length=10)

    # test no resize
    assert resizer("This is a test") == {
        "prompt_length": 4,
        "resized_prompt": "This is a test",
        "max_length": 10,
        "model_max_length": 20,
        "new_prompt_length": 4,
    }

    # test resize
    assert resizer("This is a prompt that will be resized because it is longer than allowed") == {
        "prompt_length": 15,
        "resized_prompt": "This is a prompt that will be resized because",
        "max_length": 10,
        "model_max_length": 20,
        "new_prompt_length": 10,
    }


@pytest.mark.unit
def test_flan_prompt_resizer():
    # test google/flan-t5-xxl tokenizer
    resizer = DefaultPromptResizer(model_name_or_path="google/flan-t5-xxl", model_max_length=20, max_length=10)

    # test no resize
    assert resizer("This is a test") == {
        "prompt_length": 5,
        "resized_prompt": "This is a test",
        "max_length": 10,
        "model_max_length": 20,
        "new_prompt_length": 5,
    }

    # test resize
    assert resizer("This is a prompt that will be resized because it is longer than allowed") == {
        "prompt_length": 17,
        "resized_prompt": "This is a prompt that will be re",
        "max_length": 10,
        "model_max_length": 20,
        "new_prompt_length": 10,
    }
