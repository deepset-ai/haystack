import logging
import numbers
from typing import Optional, List

import numpy as np
import torch
from torch.utils.data import ConcatDataset, TensorDataset
from transformers import BatchEncoding

from haystack.modeling.utils import flatten_list

logger = logging.getLogger(__name__)


def flatten_rename(
    encoded_batch: BatchEncoding, keys: Optional[List[str]] = None, renamed_keys: Optional[List[str]] = None
):
    if encoded_batch is None:
        return []
    if not keys:
        keys = list(encoded_batch.keys())
    if not renamed_keys:
        renamed_keys = keys
    assert len(keys) == len(
        renamed_keys
    ), f"keys and renamed_keys have different size {len(keys)} != {len(renamed_keys)}"
    assert any(key in encoded_batch for key in keys), f"one of the keys {keys} is not in batch {encoded_batch.keys()}"
    features_flat = []
    for item in range(len(encoded_batch[keys[0]])):
        feat_dict = {k: v for k, v in zip(renamed_keys, [encoded_batch[k][item] for k in keys])}
        features_flat.append(feat_dict)
    return features_flat


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
            # Checking whether a non-integer will be silently converted to torch.long
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
                logger.warning(
                    "Problem during conversion to torch tensors:\n"
                    "A non-integer value for feature '%s' with a value of: "
                    "'%s' will be converted to a torch tensor of dtype long.",
                    t_name,
                    base,
                )
        except:
            logger.debug(
                "Could not determine type for feature '%s'. Converting now to a tensor of default type long.", t_name
            )

        # Convert all remaining python objects to torch long tensors
        cur_tensor = torch.as_tensor(np.array([sample[t_name] for sample in features]), dtype=torch.long)

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

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            rows = [super(ConcatTensorDataset, self).__getitem__(i) for i in range(self.__len__())[idx]]
            return tuple(map(torch.stack, zip(*rows)))
        elif isinstance(idx, (list, np.ndarray)):
            rows = [super(ConcatTensorDataset, self).__getitem__(i) for i in idx]
            return tuple(map(torch.stack, zip(*rows)))
        else:
            return super(ConcatTensorDataset, self).__getitem__(idx)
