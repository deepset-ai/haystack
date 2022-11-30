import imp
import pytest

from .basic_qa_pipeline import basic_qa_pipeline
from .basic_faq_pipeline import basic_faq_pipeline

def test_basic_qa_pipeline():
    predictions = basic_qa_pipeline()
    assert predictions[""] == "ccdsfw"

def test_basic_faq_pipeline():
    predictions = test_basic_faq_pipeline()
    assert predictions[""] == "ccdsfw"