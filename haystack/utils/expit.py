import numpy as np


def expit(x: float) -> float:
    """Compute logistic sigmoid function. Maps input values to a range between 0 and 1"""
    return 1 / (1 + np.exp(-x))
