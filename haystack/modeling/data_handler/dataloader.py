from typing import Optional, List

from math import ceil

import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from haystack.errors import ModelingError


class NamedDataLoader(DataLoader):
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        sampler: Optional[Sampler] = None,
        tensor_names: Optional[List[str]] = None,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        """
        A modified version of the PyTorch DataLoader that returns a dictionary where the key is
        the name of the tensor and the value is the tensor itself.

        :param dataset: The dataset that will be wrapped by this NamedDataLoader
        :param sampler: The sampler used by the NamedDataLoader to choose which samples to include in the batch
        :param batch_size: The size of the batch to be returned by the NamedDataLoader
        :param tensor_names: The names of the tensor, in the order that the dataset returns them in.
        :param num_workers: number of workers to use for the DataLoader
        :param pin_memory: argument for Data Loader to use page-locked memory for faster transfer of data to GPU
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

            if len(batch[0]) != len(_tensor_names):
                raise ModelingError(
                    f"Dataset contains {len(batch[0])} tensors while there are {len(_tensor_names)} tensor names supplied: {_tensor_names}"
                )
            lists_temp = [[] for _ in range(len(_tensor_names))]
            ret = dict(zip(_tensor_names, lists_temp))

            for example in batch:
                for name, tensor in zip(_tensor_names, example):
                    ret[name].append(tensor)

            for key in ret:
                ret[key] = torch.stack(ret[key])

            return ret

        super().__init__(
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
