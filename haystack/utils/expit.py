# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from math import exp


def expit(x: float) -> float:
    """Compute logistic sigmoid function. Maps input values to a range between 0 and 1"""
    return 1 / (1 + exp(-x))
