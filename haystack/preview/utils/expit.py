import numpy as np


def expit(x: float) -> float:
    return 1 / (1 + np.exp(-x))
