from typing import Optional, Tuple, List, Dict

import logging


logger = logging.getLogger(__name__)


class EarlyStopping:
    """
    An object that can be used to control early stopping with a Node's `train()` method or a Trainer class. A custom EarlyStopping class can be used instead as long
    as it implements the method check_stopping and provides the attribute save_dir
    """

    def __init__(
        self,
        head: int = 0,
        metric: str = "loss",
        save_dir: Optional[str] = None,
        mode: str = "min",
        patience: int = 0,
        min_delta: float = 0.001,
        min_evals: int = 0,
    ):
        """
        :param head: The index of the prediction head that you are evaluating using the metric. 
                               In Haystack, the large majority of the models are trained from the one loss signal from a single prediction head and so default of 0 should work in most cases.
        :param save_dir: The directory where to save the final best model. If you set it to None, the model will not be saved.
        :param metric: The name of dev set metric to monitor (default: loss) which is extracted from the 0th prediction
                       head, or a function that extracts a value from the trainer dev evaluation result.
                       NOTE: This is different from the metric that is specified in the Processor which defines how
                       to calculate one or more evaluation metric values from the prediction and target sets. The
                       metric variable in this function specifies the name of one particular metric value, or it is a
                       method to calculate that value from the result returned by the Processor metric.
        :param mode: When set to "min", training stops if the metric does not continue to decrease. When set to "max", training stops if the metric does not continue to increase.
        :param patience: How many evaluations with no improvement to perform before stopping training.
        :param min_delta: Minimum difference to a previous best value to count as an improvement.
        :param min_evals: Minimum number of evaluations to go perform before using eval value.
        """
        self.head = head
        self.metric = metric
        self.save_dir = save_dir
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        self.min_evals = min_evals
        # for more complex modes
        self.eval_values = []  # type: List
        self.n_since_best = None  # type: Optional[int]
        if mode == "min":
            self.best_so_far = 1.0e99
        elif mode == "max":
            self.best_so_far = -1.0e99
        else:
            raise Exception("Mode must be 'min' or 'max'")

    def check_stopping(self, eval_result: List[Dict]) -> Tuple[bool, bool, float]:
        """
        Provides the evaluation value for the current evaluation. Returns true if stopping should occur.
        This will save the model, if necessary.

        :param eval_result: The current evaluation result which consists of a list of dictionaries, one for each
                            prediction head. Each dictionary contains the metrics and reports generated during
                            evaluation.
        :return: A tuple (stopprocessing, savemodel, eval_value) indicating if processing should be stopped
                 and if the current model should get saved and the evaluation value used.
        """
        if isinstance(self.metric, str):
            eval_value = float(eval_result[self.head][self.metric])
        else:
            eval_value = float(self.metric(eval_result))
        self.eval_values.append(eval_value)
        stopprocessing, savemodel = False, False
        if len(self.eval_values) <= self.min_evals:
            return stopprocessing, savemodel, eval_value
        if self.mode == "min":
            delta = self.best_so_far - eval_value
        else:
            delta = eval_value - self.best_so_far
        if delta > self.min_delta:
            self.best_so_far = eval_value
            self.n_since_best = 0
            if self.save_dir:
                savemodel = True
        else:
            self.n_since_best += 1  # type: ignore
        if self.n_since_best > self.patience:
            stopprocessing = True
        return stopprocessing, savemodel, eval_value
