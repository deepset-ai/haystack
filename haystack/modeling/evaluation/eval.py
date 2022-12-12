from typing import Dict, List, Optional, Any

import logging
import numbers
import torch
from torch.nn import DataParallel
import numpy as np
from tqdm.auto import tqdm

from haystack.modeling.evaluation.metrics import compute_metrics, compute_report_metrics
from haystack.modeling.model.adaptive_model import AdaptiveModel
from haystack.modeling.model.biadaptive_model import BiAdaptiveModel
from haystack.modeling.model.optimization import WrappedDataParallel
from haystack.utils.experiment_tracking import Tracker as tracker
from haystack.modeling.visual import BUSH_SEP


logger = logging.getLogger(__name__)


class Evaluator:
    """
    Handles evaluation of a given model over a specified dataset.
    """

    def __init__(self, data_loader: torch.utils.data.DataLoader, tasks, device: torch.device, report: bool = True):
        """
        :param data_loader: The PyTorch DataLoader that will return batches of data from the evaluation dataset
        :param tesks:
        :param device: The device on which the tensors should be processed. Choose from torch.device("cpu") and torch.device("cuda").
        :param report: Whether an eval report should be generated (e.g. classification report per class).
        """
        self.data_loader = data_loader
        self.tasks = tasks
        self.device = device
        self.report = report

    def eval(
        self,
        model: AdaptiveModel,
        return_preds_and_labels: bool = False,
        calibrate_conf_scores: bool = False,
        use_confidence_scores_for_ranking: bool = True,
        use_no_answer_legacy_confidence: bool = False,
    ) -> List[Dict]:
        """
        Performs evaluation on a given model.

        :param model: The model on which to perform evaluation
        :param return_preds_and_labels: Whether to add preds and labels in the returned dicts of the
        :param calibrate_conf_scores: Whether to calibrate the temperature for scaling of the confidence scores.
        :param use_confidence_scores_for_ranking: Whether to sort answers by confidence score (normalized between 0 and 1)(default) or by standard score (unbounded).
        :param use_no_answer_legacy_confidence: Whether to use the legacy confidence definition for no_answer: difference
                                                between the best overall answer confidence and the no_answer gap confidence.
                                                Otherwise, we use the no_answer score normalized to a range of [0,1] by
                                                an expit function (default).
        :return: all_results: A list of dictionaries, one for each prediction head. Each dictionary contains the metrics
                             and reports generated during evaluation.
        """
        model.prediction_heads[0].use_confidence_scores_for_ranking = use_confidence_scores_for_ranking
        model.prediction_heads[0].use_no_answer_legacy_confidence = use_no_answer_legacy_confidence
        model.eval()

        # init empty lists per prediction head
        loss_all: List = [0 for _ in model.prediction_heads]
        preds_all: List = [[] for _ in model.prediction_heads]
        label_all: List = [[] for _ in model.prediction_heads]
        ids_all: List = [[] for _ in model.prediction_heads]
        passage_start_t_all: List = [[] for _ in model.prediction_heads]
        logits_all: List = [[] for _ in model.prediction_heads]

        for step, batch in enumerate(tqdm(self.data_loader, desc="Evaluating", mininterval=10)):
            batch = {key: batch[key].to(self.device) for key in batch}

            if isinstance(model, (DataParallel, WrappedDataParallel)):
                module = model.module
            else:
                module = model

            with torch.inference_mode():
                if isinstance(module, AdaptiveModel):
                    logits = model.forward(
                        input_ids=batch.get("input_ids", None),
                        segment_ids=batch.get("segment_ids", None),
                        padding_mask=batch.get("padding_mask", None),
                        output_hidden_states=batch.get("output_hidden_states", False),
                        output_attentions=batch.get("output_attentions", False),
                    )
                elif isinstance(module, BiAdaptiveModel):
                    logits = model.forward(
                        query_input_ids=batch.get("query_input_ids", None),
                        query_segment_ids=batch.get("query_segment_ids", None),
                        query_attention_mask=batch.get("query_attention_mask", None),
                        passage_input_ids=batch.get("passage_input_ids", None),
                        passage_segment_ids=batch.get("passage_segment_ids", None),
                        passage_attention_mask=batch.get("passage_attention_mask", None),
                    )
                else:
                    logits = model.forward(**batch)

                losses_per_head = model.logits_to_loss_per_head(logits=logits, **batch)
                preds = model.logits_to_preds(logits=logits, **batch)
                labels = model.prepare_labels(**batch)

            # stack results of all batches per prediction head
            for head_num, head in enumerate(model.prediction_heads):
                loss_all[head_num] += np.sum(_to_numpy(losses_per_head[head_num]))
                preds_all[head_num] += list(_to_numpy(preds[head_num]))
                label_all[head_num] += list(_to_numpy(labels[head_num]))
                if head.model_type == "span_classification":
                    ids_all[head_num] += list(_to_numpy(batch["id"]))
                    passage_start_t_all[head_num] += list(_to_numpy(batch["passage_start_t"]))
                    if calibrate_conf_scores:
                        logits_all[head_num] += list(_to_numpy(logits))

        # Evaluate per prediction head
        all_results = []
        for head_num, head in enumerate(model.prediction_heads):
            if head.model_type == "span_classification" and calibrate_conf_scores:
                temperature_previous = head.temperature_for_confidence.item()
                logger.info("temperature used for confidence scores before calibration: %s", temperature_previous)
                head.calibrate_conf(logits_all[head_num], label_all[head_num])
                temperature_current = head.temperature_for_confidence.item()
                logger.info("temperature used for confidence scores after calibration: %s", temperature_current)
                temperature_change = (abs(temperature_current - temperature_previous) / temperature_previous) * 100.0
                if temperature_change > 50:
                    logger.warning(
                        f"temperature used for calibration of confidence scores changed by more than {temperature_change} percent"
                    )
            if hasattr(head, "aggregate_preds"):
                # Needed to convert NQ ids from np arrays to strings
                ids_all_str = [x.astype(str) for x in ids_all[head_num]]
                ids_all_list = [list(x) for x in ids_all_str]
                head_ids = ["-".join(x) for x in ids_all_list]
                preds_all[head_num], label_all[head_num] = head.aggregate_preds(
                    preds=preds_all[head_num],
                    labels=label_all[head_num],
                    passage_start_t=passage_start_t_all[head_num],
                    ids=head_ids,
                )
            result = {"loss": loss_all[head_num] / len(self.data_loader.dataset), "task_name": head.task_name}
            result.update(compute_metrics(metric=head.metric, preds=preds_all[head_num], labels=label_all[head_num]))
            # Select type of report depending on prediction head output type
            if self.report:
                try:
                    result["report"] = compute_report_metrics(head, preds_all[head_num], label_all[head_num])
                except:
                    logger.error(
                        f"Couldn't create eval report for head {head_num} with following preds and labels:"
                        f"\n Preds: {preds_all[head_num]} \n Labels: {label_all[head_num]}"
                    )
                    result["report"] = "Error"

            if return_preds_and_labels:
                result["preds"] = preds_all[head_num]
                result["labels"] = label_all[head_num]

            all_results.append(result)

        return all_results

    @staticmethod
    def log_results(
        results: List[Any],
        dataset_name: str,
        steps: int,
        logging: bool = True,
        print: bool = True,
        num_fold: Optional[int] = None,
    ):
        # Print a header
        header = "\n\n"
        header += BUSH_SEP + "\n"
        header += "***************************************************\n"
        if num_fold:
            header += (
                f"***** EVALUATION | FOLD: {num_fold} | {dataset_name.upper()} SET | AFTER {steps} BATCHES *****\n"
            )
        else:
            header += f"***** EVALUATION | {dataset_name.upper()} SET | AFTER {steps} BATCHES *****\n"
        header += "***************************************************\n"
        header += BUSH_SEP + "\n"
        logger.info(header)

        for head_num, head in enumerate(results):
            logger.info("\n _________ {} _________".format(head["task_name"]))
            for metric_name, metric_val in head.items():
                # log with experiment tracking framework (e.g. Mlflow)
                if logging:
                    if not metric_name in ["preds", "labels"] and not metric_name.startswith("_"):
                        if isinstance(metric_val, numbers.Number):
                            tracker.track_metrics(
                                metrics={f"{dataset_name}_{metric_name}_{head['task_name']}": metric_val}, step=steps
                            )
                # print via standard python logger
                if print:
                    if metric_name == "report":
                        if isinstance(metric_val, str) and len(metric_val) > 8000:
                            metric_val = metric_val[:7500] + "\n ............................. \n" + metric_val[-500:]
                        logger.info("{}: \n {}".format(metric_name, metric_val))
                    else:
                        if not metric_name in ["preds", "labels"] and not metric_name.startswith("_"):
                            logger.info("{}: {}".format(metric_name, metric_val))


def _to_numpy(container):
    try:
        return container.cpu().numpy()
    except AttributeError:
        return container
