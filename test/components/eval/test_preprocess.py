from haystack.components.eval.preprocess import _preprocess_text


def test_preprocess_text_default_parameters():
    """
    Test preprocess_text with default parameters.
    There should be no changes to the input text.
    """
    texts = ["Test, Output-1!", "Test, Output-2!"]
    expected_output = ["Test, Output-1!", "Test, Output-2!"]
    actual_output = _preprocess_text(texts)

    assert actual_output == expected_output


def test_preprocess_text_ignore_case():
    """
    Test preprocess_text with ignore_case=True.

    """
    texts = ["Test, Output-1!"]
    expected_output = ["test, output-1!"]

    actual_output = _preprocess_text(texts, ignore_case=True)

    assert actual_output == expected_output


def test_preprocess_text_ignore_punctuation():
    """
    Test preprocess_text with ignore_punctuation=True.
    """
    texts = ["Test, Output-1!"]
    expected_output = ["Test Output1"]

    actual_output = _preprocess_text(texts, ignore_punctuation=True)

    assert actual_output == expected_output


# Preprocess text with ignore_numbers=True.
def test_preprocess_text_ignore_numbers():
    """
    Test preprocess_text with ignore_numbers=True. It should be able to remove numbers from the input.
    """
    texts = ["Test, Output-1!"]
    expected_output = ["Test, Output-!"]

    actual_output = _preprocess_text(texts, ignore_numbers=True)

    assert actual_output == expected_output


def test_preprocess_text_regexes_to_ignore():
    """
    Test preprocess_text with a list of regex patterns to ignore.
    """
    texts = ["Test, Output-1!"]
    expected_output = ["Test Output"]

    # Use regex patterns to remove digits and non-alphanumeric characters
    actual_output = _preprocess_text(texts, regexes_to_ignore=[r"\d", r"[^\w\s]"])

    assert actual_output == expected_output


def test_preprocess_text_empty_list():
    """
    Test preprocess_text with empty list of texts.
    """
    texts = []
    expected_output = []

    actual_output = _preprocess_text(texts)

    assert actual_output == expected_output


def test_preprocess_text_all_ignore_parameters():
    """
    Test preprocess_text with all ignore parameters set to True.
    """
    texts = ["Test, Output-1!"]
    expected_output = ["test output"]

    actual_output = _preprocess_text(texts, ignore_case=True, ignore_punctuation=True, ignore_numbers=True)

    assert actual_output == expected_output


def test_preprocess_text_regexes_to_ignore_empty_string():
    """
    Test preprocess_text with regexes_to_ignore=[""].
    """
    texts = ["Test, Output-1!"]
    expected_output = ["Test, Output-1!"]

    actual_output = _preprocess_text(texts, regexes_to_ignore=[""])

    assert actual_output == expected_output


# Preprocess text with regexes_to_ignore=[".*"].
def test_preprocess_text_regexes_to_ignore_dot_star():
    """
    Test preprocess_text with regexes_to_ignore=[".*"].
    """
    texts = ["Test, Output-1!"]
    expected_output = [""]

    actual_output = _preprocess_text(texts, regexes_to_ignore=[".*"])

    assert actual_output == expected_output


def test_preprocess_text_regexes_to_ignore_same_substring():
    """
    Test preprocess_text with regexes_to_ignore where all the regex patterns match the same substring.
    """
    texts = ["Test, Output-1!"]
    expected_output = ["Test, Output-!"]

    actual_output = _preprocess_text(texts, regexes_to_ignore=[r"\d", r"\d"])

    assert actual_output == expected_output
