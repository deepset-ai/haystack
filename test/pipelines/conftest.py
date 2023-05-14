import pytest

from haystack.nodes import FARMReader


@pytest.fixture
def reader_without_normalized_scores():
    return FARMReader(
        model_name_or_path="deepset/bert-medium-squad2-distilled",
        use_gpu=False,
        top_k_per_sample=5,
        num_processes=0,
        use_confidence_scores=False,
    )
