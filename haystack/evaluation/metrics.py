import json
from enum import Enum
from pathlib import Path
from typing import Union


class Metric(Enum):
    """
    Contains a list of standard metrics available.
    """

    RECALL = "Recall"
    MRR = "Mean Reciprocal Rank"
    MAP = "Mean Average Precision"
    F1 = "F1"
    EM = "Exact Match"
    SAS = "Semantic Answer Similarity"


class MetricsResult(dict):
    """
    Stores the metric values computed during the evaluation.
    """

    def save(self, file: Union[str, Path]):
        """
        Save the metrics stored in the MetricsResult to a json file.

        :param file: The file path or file name to save the data.
        """
        with open(file, "w") as outfile:
            json.dump(self, outfile, indent=4)
