import pytest

import torch

from haystack.modeling.data_handler.dataloader import NamedDataLoader


@pytest.fixture
def named_dataloader():
    tensor_names = ["input_ids", "labels"]
    return NamedDataLoader(None, 1, tensor_names=tensor_names)


@pytest.fixture
def batch():
    # batch containing tensors of different lengths
    return [
        (torch.tensor([1, 2, 3]), torch.tensor([[0, 0], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1]])),
        (torch.tensor([4, 5, 6]), torch.tensor([[0, 0], [-1, -1], [-1, -1]])),
        (torch.tensor([7, 8, 9]), torch.tensor([[0, 0], [-1, -1]])),
    ]


@pytest.mark.unit
def test_compute_max_number_of_labels(named_dataloader, batch):
    tensor_names = ["input_ids", "labels"]
    max_num_labels = named_dataloader._compute_max_number_of_labels(batch, tensor_names)
    assert max_num_labels == 6


@pytest.mark.unit
def test_collate_fn(named_dataloader, batch):
    collated_batch = named_dataloader.collate_fn(batch)

    expected_collated_batch = {
        "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
        "labels": torch.tensor(
            [
                [[0, 0], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1]],
                [[0, 0], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1]],
                [[0, 0], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1]],
            ]
        ),
    }

    for key in collated_batch:
        assert torch.equal(collated_batch[key], expected_collated_batch[key])
