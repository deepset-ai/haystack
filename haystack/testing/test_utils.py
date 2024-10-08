# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
import random

from haystack import logging

logger = logging.getLogger(__name__)


def set_all_seeds(seed: int, deterministic_cudnn: bool = False) -> None:
    """
    Setting multiple seeds to make runs reproducible.

    Important: Enabling `deterministic_cudnn` gives you full reproducibility with CUDA,
    but might slow down your training (see https://pytorch.org/docs/stable/notes/randomness.html#cudnn) !

    :param seed:number to use as seed
    :param deterministic_cudnn: Enable for full reproducibility when using CUDA. Caution: might slow down training.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic_cudnn:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    except (ImportError, ModuleNotFoundError) as exc:
        logger.info("Could not set PyTorch seed because torch is not installed. Exception: {exception}", exception=exc)
