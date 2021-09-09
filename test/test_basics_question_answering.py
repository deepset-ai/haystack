import logging

from haystack.basics.data_handler.processor import SquadProcessor
from haystack.basics.modeling.adaptive_model import AdaptiveModel


def test_training(distilbert_squad, caplog=None):
    if caplog:
        caplog.set_level(logging.CRITICAL)

    model, processor = distilbert_squad
    assert type(model) == AdaptiveModel
    assert type(processor) == SquadProcessor


if __name__ == "__main__":
    test_training()
