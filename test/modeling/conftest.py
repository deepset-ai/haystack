import logging

import pytest
import psutil

from haystack.modeling.infer import Inferencer, QAInferencer


@pytest.fixture
def adaptive_model_qa(num_processes):
    """
    PyTest Fixture for a Question Answering Inferencer based on PyTorch.
    """

    model = Inferencer.load(
        "deepset/bert-medium-squad2-distilled",
        task_type="question_answering",
        batch_size=16,
        num_processes=num_processes,
        gpu=False,
    )
    yield model

    # check if all workers (sub processes) are closed
    current_process = psutil.Process()
    children = current_process.children()
    if len(children) != 0:
        logging.error("Not all the subprocesses are closed! %s are still running.", len(children))


@pytest.fixture
def bert_base_squad2(request):
    model = QAInferencer.load(
        "deepset/minilm-uncased-squad2",
        task_type="question_answering",
        batch_size=4,
        num_processes=0,
        multithreading_rust=False,
        use_fast=True,  # TODO parametrize this to test slow as well
    )
    return model
