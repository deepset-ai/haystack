import json
from enum import Enum
from pathlib import Path
from typing import Union


class Metric(Enum):
    """
    Contains a list of standard metrics available.
    """

    ACCURACY = "Accuracy"
    RECALL = "Recall"
    PRECISION = "Precision"
    MRR = "Mean Reciprocal Rank"
    MAP = "Mean Average Precision"
    EM = "Exact Match"
    F1 = "F1"
    SAS = "SemanticAnswerSimilarity"


class MetricsResult(dict):
    def save(self, file: Union[str, Path]):
        """
        Save the metrics stored in the MetricsResult to a json file.

        :param file: The file path or file name to save the data.
        """
        with open(file, "w") as outfile:
            json.dump(self, outfile, indent=4)
