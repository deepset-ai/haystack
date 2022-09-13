from typing import Union

import torch
from torch.utils.data import Dataset


class ListDataset(Dataset):
    def __init__(self, original_list):
        self.original_list = original_list

    def __len__(self):
        return len(self.original_list)

    def __getitem__(self, i):
        return self.original_list[i]


def ensure_tensor_on_device(inputs: Union[dict, list, tuple, torch.Tensor], device: torch.device):
    """Utility function to check that all torch tensors present in `inputs` are sent to the correct device.

    :param inputs: Contains the torch tensors that will be sent to `device`.
    :param device: The torch device to send the tensors to.
    """
    if isinstance(inputs, dict):
        return {name: ensure_tensor_on_device(tensor, device) for name, tensor in inputs.items()}
    elif isinstance(inputs, list):
        return [ensure_tensor_on_device(item, device) for item in inputs]
    elif isinstance(inputs, tuple):
        return tuple(ensure_tensor_on_device(item, device) for item in inputs)
    elif isinstance(inputs, torch.Tensor):
        if device == torch.device("cpu") and inputs.dtype in {torch.float16, torch.bfloat16}:
            inputs = inputs.float()
        return inputs.to(device)
    else:
        return inputs
