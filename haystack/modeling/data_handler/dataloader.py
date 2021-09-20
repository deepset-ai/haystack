from math import ceil

import torch
from torch.utils.data import DataLoader, Dataset, Sampler


class NamedDataLoader(DataLoader):
    """
    A modified version of the PyTorch DataLoader that returns a dictionary where the key is
    the name of the tensor and the value is the tensor itself.
    """

    def __init__(self, dataset, batch_size, sampler=None, tensor_names=None, num_workers=0, pin_memory=False):
        """
        :param dataset: The dataset that will be wrapped by this NamedDataLoader
        :type dataset: Dataset
        :param sampler: The sampler used by the NamedDataLoader to choose which samples to include in the batch
        :type sampler: Sampler
        :param batch_size: The size of the batch to be returned by the NamedDataLoader
        :type batch_size: int
        :param tensor_names: The names of the tensor, in the order that the dataset returns them in.
        :type tensor_names: list
        :param num_workers: number of workers to use for the DataLoader
        :type num_workers: int
        :param pin_memory: argument for Data Loader to use page-locked memory for faster transfer of data to GPU
        :type pin_memory: bool
        """

        def collate_fn(batch):
            """
            A custom collate function that formats the batch as a dictionary where the key is
            the name of the tensor and the value is the tensor itself
            """

            if type(dataset).__name__ == "_StreamingDataSet":
                _tensor_names = dataset.tensor_names
            else:
                _tensor_names = tensor_names

            if type(batch[0]) == list:
                batch = batch[0]

            assert len(batch[0]) == len(
                _tensor_names
            ), "Dataset contains {} tensors while there are {} tensor names supplied: {}".format(
                len(batch[0]), len(_tensor_names), _tensor_names
            )
            lists_temp = [[] for _ in range(len(_tensor_names))]
            ret = dict(zip(_tensor_names, lists_temp))

            for example in batch:
                for name, tensor in zip(_tensor_names, example):
                    ret[name].append(tensor)

            for key in ret:
                ret[key] = torch.stack(ret[key])

            return ret

        super(NamedDataLoader, self).__init__(
            dataset=dataset,
            sampler=sampler,
            batch_size=batch_size,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            num_workers=num_workers,
        )

    def __len__(self):
        if type(self.dataset).__name__ == "_StreamingDataSet":
            num_samples = len(self.dataset)
            num_batches = ceil(num_samples / self.dataset.batch_size)
            return num_batches
        else:
            return super().__len__()


def covert_dataset_to_dataloader(dataset, sampler, batch_size):
    """
    Wraps a PyTorch Dataset with a DataLoader.

    :param dataset: Dataset to be wrapped.
    :type dataset: Dataset
    :param sampler: PyTorch sampler used to pick samples in a batch.
    :type sampler: Sampler
    :param batch_size: Number of samples in the batch.
    :return: A DataLoader that wraps the input Dataset.
    """
    sampler_initialized = sampler(dataset)
    data_loader = DataLoader(
        dataset, sampler=sampler_initialized, batch_size=batch_size
    )
    return data_loader
