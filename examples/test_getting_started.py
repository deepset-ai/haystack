import os

import pytest

from examples.getting_started import getting_started
from haystack.schema import Answer, Document


@pytest.mark.integration
@pytest.mark.parametrize("provider", ["cohere", "huggingface", "openai"])
def test_getting_started(provider):
    if provider == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    elif provider == "cohere":
        api_key = os.environ.get("COHERE_API_KEY", "")
    elif provider == "huggingface":
        api_key = os.environ.get("HUGGINGFACE_API_KEY", "")
    elif provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY", "")
    result = getting_started(provider=provider, API_KEY=api_key)

    # Testing only for functionality. Since model predictions from APIs might change, we cannot test those directly.
    assert isinstance(result, dict)
    assert type(result["answers"][0]) == Answer
    assert type(result["documents"][0]) == Document
