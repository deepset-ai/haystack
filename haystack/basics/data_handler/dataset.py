import logging
import numbers
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset, TensorDataset

from haystack.basics.utils import flatten_list

logger = logging.getLogger(__name__)


def convert_features_to_dataset(features):
    """
    Converts a list of feature dictionaries (one for each sample) into a PyTorch Dataset.

    :param features: A list of dictionaries. Each dictionary corresponds to one sample. Its keys are the
                     names of the type of feature and the keys are the features themselves.
    :Return: a Pytorch dataset and a list of tensor names.
    """
    # features can be an empty list in cases where down sampling occurs
    if len(features) == 0:
        return None, None
    tensor_names = list(features[0].keys())
    all_tensors = []
    for t_name in tensor_names:
        try:
            # Checking weather a non-integer will be silently converted to torch.long
            check = features[0][t_name]
            if isinstance(check, numbers.Number):
                base = check
            # extract a base variable from a nested lists or tuples
            elif isinstance(check, list):
                base = list(flatten_list(check))[0]
            # extract a base variable from numpy arrays
            else:
                base = check.ravel()[0]
            if not np.issubdtype(type(base), np.integer):
                logger.warning(f"Problem during conversion to torch tensors:\n"
                               f"A non-integer value for feature '{t_name}' with a value of: "
                               f"'{base}' will be converted to a torch tensor of dtype long.")
        except:
            logger.warning(f"Could not determine type for feature '{t_name}'. "
                           "Converting now to a tensor of default type long.")

        # Convert all remaining python objects to torch long tensors
        cur_tensor = torch.tensor([sample[t_name] for sample in features], dtype=torch.long)

        all_tensors.append(cur_tensor)

    dataset = TensorDataset(*all_tensors)
    return dataset, tensor_names


class ConcatTensorDataset(ConcatDataset):
    r"""ConcatDataset of only TensorDatasets which supports getting slices.

    This dataset allows the use of slices, e.g. ds[2:4] if all concatenated
    datasets are either TensorDatasets or Subset or other ConcatTensorDataset instances
    which eventually contain only TensorDataset instances. If no slicing is needed,
    this class works exactly like torch.utils.data.ConcatDataset and can concatenate arbitrary
    (not just TensorDataset) datasets.

    Args:
        datasets (sequence): List of datasets to be concatenated
    """
    def __init__(self, datasets: Iterable[Dataset]) -> None:
        super(ConcatTensorDataset, self).__init__(datasets)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            rows = [super(ConcatTensorDataset, self).__getitem__(i) for i in range(self.__len__())[idx]]
            return tuple(map(torch.stack, zip(*rows)))
        elif isinstance(idx, (list, np.ndarray)):
            rows = [super(ConcatTensorDataset, self).__getitem__(i) for i in idx]
            return tuple(map(torch.stack, zip(*rows)))
        else:
            return super(ConcatTensorDataset, self).__getitem__(idx)
