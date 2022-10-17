from typing import Optional, List, Union

import torch
from torch.utils.data import Dataset


class ListDataset(Dataset):
    def __init__(self, original_list):
        self.original_list = original_list

    def __len__(self):
        return len(self.original_list)

    def __getitem__(self, i):
        return self.original_list[i]


def get_devices(devices: Optional[List[Union[str, torch.device]]]) -> List[torch.device]:
    """
    Convert a list of device names into a list of Torch devices,
    depending on the system's configuration and hardware.
    """
    if devices is not None:
        return [torch.device(device) for device in devices]
    elif torch.cuda.is_available():
        return [torch.device(device) for device in range(torch.cuda.device_count())]
    return [torch.device("cpu")]
