import pytest
from haystack.modeling.evaluation.squad import normalize_answer


def test_normalize_answer():
    assert normalize_answer("100 - 150") == "100 150"
    assert normalize_answer("100 – 150") == "100 150"
    assert normalize_answer("100 % 150") == "100 150"
    assert normalize_answer("100 + 150") == "100 150"
    assert normalize_answer("100 / 150") == "100 150"
    assert normalize_answer("a test") == "test"
    assert normalize_answer("the test") == "test"
    assert normalize_answer("an experiment") == "experiment"
    assert normalize_answer("normalization test") == "normalization test"
    assert normalize_answer("NorMaliZation teSt") == "normalization test"
    assert normalize_answer("   normalization   test  ") == "normalization test"
    assert normalize_answer("  the normalization test") == "normalization test"
